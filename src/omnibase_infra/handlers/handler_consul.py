# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""HashiCorp Consul Handler - MVP implementation using python-consul client.

Supports service discovery operations with configurable retry logic and
circuit breaker pattern for fault tolerance.

Security Features:
    - SecretStr protection for ACL tokens (prevents accidental logging)
    - Sanitized error messages (never expose tokens in logs)
    - Token handling follows security best practices

Supported Operations:
    - consul.kv_get: Retrieve value from KV store
    - consul.kv_put: Store value in KV store
    - consul.register: Register service with Consul agent
    - consul.deregister: Deregister service from Consul agent
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, TypeVar
from uuid import UUID, uuid4

import consul
from omnibase_core.models.dispatch import ModelHandlerOutput
from pydantic import SecretStr, ValidationError

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraAuthenticationError,
    InfraConnectionError,
    InfraTimeoutError,
    ModelInfraErrorContext,
    ProtocolConfigurationError,
    RuntimeHostError,
)
from omnibase_infra.handlers.model_consul_handler_config import ModelConsulHandlerConfig
from omnibase_infra.handlers.models import (
    ModelOperationContext,
    ModelRetryState,
)
from omnibase_infra.handlers.models.consul import (
    ConsulPayload,
    ModelConsulDeregisterPayload,
    ModelConsulHandlerPayload,
    ModelConsulKVGetFoundPayload,
    ModelConsulKVGetNotFoundPayload,
    ModelConsulKVGetRecursePayload,
    ModelConsulKVItem,
    ModelConsulKVPutPayload,
    ModelConsulRegisterPayload,
)
from omnibase_infra.handlers.models.model_consul_handler_response import (
    ModelConsulHandlerResponse,
)
from omnibase_infra.mixins import (
    EnumRetryErrorCategory,
    MixinAsyncCircuitBreaker,
    MixinEnvelopeExtraction,
    MixinRetryExecution,
    RetryErrorClassification,
)

if TYPE_CHECKING:
    from omnibase_core.types import JsonValue

T = TypeVar("T")

logger = logging.getLogger(__name__)

# Consul handler type constant for registry compatibility
# Note: EnumHandlerType.CONSUL will be added to omnibase_core in future
HANDLER_TYPE_CONSUL: str = "consul"

# Handler ID for ModelHandlerOutput
HANDLER_ID_CONSUL: str = "consul-handler"

SUPPORTED_OPERATIONS: frozenset[str] = frozenset(
    {
        "consul.kv_get",
        "consul.kv_put",
        "consul.register",
        "consul.deregister",
    }
)


class ConsulHandler(
    MixinAsyncCircuitBreaker, MixinRetryExecution, MixinEnvelopeExtraction
):
    """HashiCorp Consul handler using python-consul client (MVP: KV, service registration).

    Security Policy - Token Handling:
        The Consul ACL token contains sensitive credentials and is treated as a secret
        throughout this handler. The following security measures are enforced:

        1. Token is stored as SecretStr in config (never logged or exposed)
        2. All error messages use generic descriptions without exposing token
        3. The describe() method returns capabilities without credentials

        See CLAUDE.md "Error Sanitization Guidelines" for the full security policy
        on what information is safe vs unsafe to include in errors and logs.

    Thread Pool Management (Production-Grade):
        - Bounded ThreadPoolExecutor prevents resource exhaustion
        - Configurable max_concurrent_operations (default: 10, max: 100)
        - Thread pool gracefully shutdown on handler.shutdown()
        - All consul (synchronous) operations run in dedicated thread pool

        Queue Size Management (MVP Behavior):
            ThreadPoolExecutor uses an unbounded queue by default. The max_queue_size
            parameter is calculated (max_workers * multiplier) for monitoring purposes,
            but is NOT enforced by the executor.

            Why unbounded is acceptable for MVP:
                - Consul operations are typically short-lived (KV get/put)
                - Circuit breaker provides backpressure when Consul is unavailable
                - Thread pool size limits concurrent execution (default: 10 workers)
                - Memory exhaustion from queue growth is unlikely in normal operation

            Future Enhancement Path:
                For production deployments with strict resource controls, implement a
                custom executor with bounded queue using queue.Queue(maxsize=N):

                    from queue import Queue
                    from concurrent.futures import ThreadPoolExecutor

                    class BoundedThreadPoolExecutor(ThreadPoolExecutor):
                        def __init__(self, max_workers, max_queue_size):
                            super().__init__(max_workers)
                            self._work_queue = Queue(maxsize=max_queue_size)

                This would reject tasks when queue is full, enabling explicit backpressure.

    Circuit Breaker Pattern (Production-Grade):
        - Uses MixinAsyncCircuitBreaker for consistent circuit breaker implementation
        - Prevents cascading failures to Consul service
        - Three states: CLOSED (normal), OPEN (blocking), HALF_OPEN (testing)
        - Configurable failure_threshold (default: 5 consecutive failures)
        - Configurable reset_timeout (default: 30 seconds)
        - Raises InfraUnavailableError when circuit is OPEN

    Retry Logic:
        - All operations use exponential backoff retry logic
        - Retry configuration from ModelConsulRetryConfig
        - Backoff calculation: initial_delay * (exponential_base ** attempt)
        - Max backoff capped at max_delay_seconds
        - Circuit breaker checked before retry execution

    Error Context Design:
        Error contexts use static target_name="consul_handler" for consistency with
        VaultHandler and other infrastructure handlers. This provides predictable
        error categorization and log filtering across all Consul operations.

        For multi-DC deployments, datacenter differentiation is achieved via:
        - Circuit breaker service_name (e.g., "consul.dc1", "consul.dc2")
        - Structured logging with datacenter field in extra dict
        - Correlation IDs that can be traced across datacenters

        This design keeps error aggregation unified (all Consul errors grouped under
        "consul_handler") while still providing operational visibility per-datacenter
        through circuit breaker metrics and structured logs.

        Future Enhancement: If error differentiation per-DC becomes a requirement
        (e.g., for DC-specific alerting), target_name could be made dynamic:
        target_name=f"consul.{self._config.datacenter or 'default'}"
    """

    def __init__(self) -> None:
        """Initialize ConsulHandler in uninitialized state.

        Note: Circuit breaker is initialized during initialize() call when
        configuration is available. The mixin's _init_circuit_breaker() method
        is called there with the actual config values.
        """
        self._client: consul.Consul | None = None
        self._config: ModelConsulHandlerConfig | None = None
        self._initialized: bool = False
        self._executor: ThreadPoolExecutor | None = None
        self._max_workers: int = 0
        self._max_queue_size: int = 0
        # Circuit breaker initialized flag - set after _init_circuit_breaker called
        self._circuit_breaker_initialized: bool = False
        # Retry state tracking for inter-method communication
        self._last_retry_state: ModelRetryState = ModelRetryState()

    @property
    def handler_type(self) -> str:
        """Return handler type identifier for Consul.

        Returns:
            String "consul" for registry compatibility.

        Note:
            Will return EnumHandlerType.CONSUL once that enum value is added
            to omnibase_core.
        """
        return HANDLER_TYPE_CONSUL

    @property
    def max_workers(self) -> int:
        """Return thread pool max workers (public API for tests)."""
        return self._max_workers

    @property
    def max_queue_size(self) -> int:
        """Return maximum queue size (public API for tests)."""
        return self._max_queue_size

    # MixinRetryExecution abstract method implementations

    def _get_transport_type(self) -> EnumInfraTransportType:
        """Return transport type for error context."""
        return EnumInfraTransportType.CONSUL

    def _get_target_name(self) -> str:
        """Return target name for error context."""
        return "consul_handler"

    def _classify_error(
        self, error: Exception, operation: str
    ) -> RetryErrorClassification:
        """Classify Consul-specific exceptions for retry handling.

        Args:
            error: The exception to classify.
            operation: The operation name for context.

        Returns:
            RetryErrorClassification with retry decision and error details.
        """
        if isinstance(error, TimeoutError):
            return RetryErrorClassification(
                category=EnumRetryErrorCategory.TIMEOUT,
                should_retry=True,
                record_circuit_failure=True,
                error_message=f"Timeout: {type(error).__name__}",
            )

        if isinstance(error, consul.ACLPermissionDenied):
            return RetryErrorClassification(
                category=EnumRetryErrorCategory.AUTHENTICATION,
                should_retry=False,
                record_circuit_failure=True,
                error_message="Consul ACL permission denied - check token permissions",
            )

        if isinstance(error, consul.Timeout):
            return RetryErrorClassification(
                category=EnumRetryErrorCategory.TIMEOUT,
                should_retry=True,
                record_circuit_failure=True,
                error_message=f"Consul timeout: {type(error).__name__}",
            )

        if isinstance(error, consul.ConsulException):
            return RetryErrorClassification(
                category=EnumRetryErrorCategory.CONNECTION,
                should_retry=True,
                record_circuit_failure=True,
                error_message=f"Consul error: {type(error).__name__}",
            )

        # Unknown error - retry eligible
        return RetryErrorClassification(
            category=EnumRetryErrorCategory.UNKNOWN,
            should_retry=True,
            record_circuit_failure=True,
            error_message=f"Unexpected error: {type(error).__name__}",
        )

    # Initialization helper methods

    def _validate_consul_config(
        self, config: dict[str, JsonValue], correlation_id: UUID
    ) -> ModelConsulHandlerConfig:
        """Validate and parse Consul configuration.

        Args:
            config: Raw configuration dictionary.
            correlation_id: Correlation ID for error context.

        Returns:
            Validated ModelConsulHandlerConfig.

        Raises:
            ProtocolConfigurationError: If validation fails.
            RuntimeHostError: If parsing fails unexpectedly.
        """
        try:
            # Handle SecretStr token conversion
            token_raw = config.get("token")
            if isinstance(token_raw, str):
                config = dict(config)  # Make mutable copy
                config["token"] = SecretStr(token_raw)

            return ModelConsulHandlerConfig.model_validate(config)
        except ValidationError as e:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.CONSUL,
                operation="initialize",
                target_name="consul_handler",
                correlation_id=correlation_id,
            )
            sanitized_fields = [err.get("loc", ("unknown",))[-1] for err in e.errors()]
            raise ProtocolConfigurationError(
                f"Invalid Consul configuration - validation failed for fields: {sanitized_fields}",
                context=ctx,
            ) from e
        except Exception as e:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.CONSUL,
                operation="initialize",
                target_name="consul_handler",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                f"Configuration parsing failed: {type(e).__name__}",
                context=ctx,
            ) from e

    def _setup_consul_client(
        self, consul_config: ModelConsulHandlerConfig
    ) -> consul.Consul:
        """Create and configure the Consul client.

        Args:
            consul_config: Validated Consul configuration.

        Returns:
            Configured consul.Consul client instance.
        """
        token_value: str | None = None
        if consul_config.token is not None:
            token_value = consul_config.token.get_secret_value()

        return consul.Consul(
            host=consul_config.host,
            port=consul_config.port,
            scheme=consul_config.scheme,
            token=token_value,
            dc=consul_config.datacenter,
        )

    def _verify_consul_connection(
        self, client: consul.Consul, correlation_id: UUID
    ) -> None:
        """Verify connectivity to Consul by checking leader status.

        Args:
            client: The Consul client to verify.
            correlation_id: Correlation ID for error context.

        Raises:
            InfraConnectionError: If verification fails.
        """
        try:
            leader = client.status.leader()
            if not leader:
                ctx = ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.CONSUL,
                    operation="initialize",
                    target_name="consul_handler",
                    correlation_id=correlation_id,
                )
                raise InfraConnectionError(
                    "Consul cluster has no leader - cluster may be unavailable",
                    context=ctx,
                )
        except consul.ConsulException as e:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.CONSUL,
                operation="initialize",
                target_name="consul_handler",
                correlation_id=correlation_id,
            )
            raise InfraConnectionError(
                "Consul connectivity verification failed",
                context=ctx,
            ) from e

    def _setup_thread_pool(self, consul_config: ModelConsulHandlerConfig) -> None:
        """Set up the thread pool executor for async operations.

        Args:
            consul_config: Validated Consul configuration.
        """
        self._max_workers = consul_config.max_concurrent_operations
        self._executor = ThreadPoolExecutor(
            max_workers=self._max_workers,
            thread_name_prefix="consul_handler_",
        )
        self._max_queue_size = (
            self._max_workers * consul_config.max_queue_size_multiplier
        )

    def _setup_circuit_breaker(self, consul_config: ModelConsulHandlerConfig) -> None:
        """Initialize circuit breaker if enabled.

        Args:
            consul_config: Validated Consul configuration.
        """
        if consul_config.circuit_breaker_enabled:
            self._init_circuit_breaker(
                threshold=consul_config.circuit_breaker_failure_threshold,
                reset_timeout=consul_config.circuit_breaker_reset_timeout_seconds,
                service_name=f"consul.{consul_config.datacenter or 'default'}",
                transport_type=EnumInfraTransportType.CONSUL,
            )
            self._circuit_breaker_initialized = True

    async def initialize(self, config: dict[str, JsonValue]) -> None:
        """Initialize Consul client with configuration.

        Args:
            config: Configuration dict containing:
                - host: Consul server hostname (default: "localhost")
                - port: Consul server port (default: 8500)
                - scheme: HTTP scheme "http" or "https" (default: "http")
                - token: Optional Consul ACL token
                - timeout_seconds: Optional timeout (default 30.0)
                - datacenter: Optional datacenter for multi-DC deployments

        Raises:
            ProtocolConfigurationError: If configuration validation fails.
            InfraAuthenticationError: If token authentication fails.
            InfraConnectionError: If connection to Consul server fails.
            RuntimeHostError: If client initialization fails for other reasons.

        Security:
            Token must be provided via environment variable, not hardcoded in config.
            Use SecretStr for token to prevent accidental logging.
        """
        init_correlation_id = uuid4()

        logger.info(
            "Initializing %s",
            self.__class__.__name__,
            extra={
                "handler": self.__class__.__name__,
                "correlation_id": str(init_correlation_id),
            },
        )

        # Validate configuration
        self._config = self._validate_consul_config(config, init_correlation_id)

        # Set up client and infrastructure
        try:
            self._client = self._setup_consul_client(self._config)
            self._verify_consul_connection(self._client, init_correlation_id)
            self._setup_thread_pool(self._config)
            self._setup_circuit_breaker(self._config)

            self._initialized = True
            self._log_initialization_success(init_correlation_id)

        except (InfraConnectionError, InfraAuthenticationError):
            raise
        except consul.ACLPermissionDenied as e:
            self._raise_auth_error(init_correlation_id, e)
        except consul.ConsulException as e:
            self._raise_connection_error(init_correlation_id, e)
        except Exception as e:
            self._raise_runtime_error(init_correlation_id, e)

    def _log_initialization_success(self, correlation_id: UUID) -> None:
        """Log successful initialization with handler details."""
        if self._config is None:
            return
        logger.info(
            "%s initialized successfully",
            self.__class__.__name__,
            extra={
                "handler": self.__class__.__name__,
                "host": self._config.host,
                "port": self._config.port,
                "scheme": self._config.scheme,
                "datacenter": self._config.datacenter,
                "timeout_seconds": self._config.timeout_seconds,
                "thread_pool_max_workers": self._max_workers,
                "thread_pool_max_queue_size": self._max_queue_size,
                "circuit_breaker_enabled": self._circuit_breaker_initialized,
                "correlation_id": str(correlation_id),
            },
        )

    def _raise_auth_error(
        self, correlation_id: UUID, original_error: Exception
    ) -> None:
        """Raise InfraAuthenticationError with context."""
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.CONSUL,
            operation="initialize",
            target_name="consul_handler",
            correlation_id=correlation_id,
        )
        raise InfraAuthenticationError(
            "Consul ACL permission denied - check token validity and permissions",
            context=ctx,
        ) from original_error

    def _raise_connection_error(
        self, correlation_id: UUID, original_error: Exception
    ) -> None:
        """Raise InfraConnectionError with context."""
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.CONSUL,
            operation="initialize",
            target_name="consul_handler",
            correlation_id=correlation_id,
        )
        raise InfraConnectionError(
            f"Consul connection failed: {type(original_error).__name__}",
            context=ctx,
        ) from original_error

    def _raise_runtime_error(
        self, correlation_id: UUID, original_error: Exception
    ) -> None:
        """Raise RuntimeHostError with context."""
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.CONSUL,
            operation="initialize",
            target_name="consul_handler",
            correlation_id=correlation_id,
        )
        raise RuntimeHostError(
            f"Consul client initialization failed: {type(original_error).__name__}",
            context=ctx,
        ) from original_error

    async def shutdown(self) -> None:
        """Close Consul client and release resources.

        Cleanup includes:
            - Shutting down thread pool executor (waits for pending tasks)
            - Clearing Consul client connection
            - Resetting circuit breaker state (thread-safe via mixin)
        """
        shutdown_correlation_id = uuid4()

        if self._executor is not None:
            # Shutdown thread pool gracefully (wait for pending tasks)
            self._executor.shutdown(wait=True)
            self._executor = None

        if self._client is not None:
            # python-consul.Client doesn't have close method, just clear reference
            self._client = None

        # Reset circuit breaker state using mixin (thread-safe)
        if self._circuit_breaker_initialized:
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

        self._initialized = False
        self._config = None
        self._circuit_breaker_initialized = False
        logger.info(
            "ConsulHandler shutdown complete",
            extra={
                "correlation_id": str(shutdown_correlation_id),
            },
        )

    def _build_response(
        self,
        typed_payload: ConsulPayload,
        correlation_id: UUID,
        input_envelope_id: UUID,
    ) -> ModelHandlerOutput[ModelConsulHandlerResponse]:
        """Build standardized ModelConsulHandlerResponse wrapped in ModelHandlerOutput.

        This helper method ensures consistent response formatting across all
        Consul operations, matching the pattern used by DbHandler.

        Args:
            typed_payload: Strongly-typed payload from the discriminated union.
            correlation_id: Correlation ID for tracing.
            input_envelope_id: Input envelope ID for causality tracking.

        Returns:
            ModelHandlerOutput wrapping ModelConsulHandlerResponse.
        """
        response = ModelConsulHandlerResponse(
            status="success",
            payload=ModelConsulHandlerPayload(data=typed_payload),
            correlation_id=correlation_id,
        )
        return ModelHandlerOutput.for_compute(
            input_envelope_id=input_envelope_id,
            correlation_id=correlation_id,
            handler_id=HANDLER_ID_CONSUL,
            result=response,
        )

    async def execute(
        self, envelope: dict[str, JsonValue]
    ) -> ModelHandlerOutput[ModelConsulHandlerResponse]:
        """Execute Consul operation from envelope.

        Args:
            envelope: Request envelope containing:
                - operation: Consul operation (consul.kv_get, consul.kv_put, etc.)
                - payload: dict with operation-specific parameters
                - correlation_id: Optional correlation ID for tracing
                - envelope_id: Optional envelope ID for causality tracking

        Returns:
            ModelHandlerOutput wrapping the operation result with correlation tracking

        Raises:
            RuntimeHostError: If handler not initialized or invalid input.
            InfraConnectionError: If Consul connection fails.
            InfraAuthenticationError: If authentication fails.
            InfraUnavailableError: If circuit breaker is open.
        """
        correlation_id = self._extract_correlation_id(envelope)
        input_envelope_id = self._extract_envelope_id(envelope)

        if not self._initialized or self._client is None or self._config is None:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.CONSUL,
                operation="execute",
                target_name="consul_handler",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                "ConsulHandler not initialized. Call initialize() first.",
                context=ctx,
            )

        operation = envelope.get("operation")
        if not isinstance(operation, str):
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.CONSUL,
                operation="execute",
                target_name="consul_handler",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                "Missing or invalid 'operation' in envelope",
                context=ctx,
            )

        if operation not in SUPPORTED_OPERATIONS:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.CONSUL,
                operation=operation,
                target_name="consul_handler",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                f"Operation '{operation}' not supported in MVP. "
                f"Available: {', '.join(sorted(SUPPORTED_OPERATIONS))}",
                context=ctx,
            )

        payload = envelope.get("payload")
        if not isinstance(payload, dict):
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.CONSUL,
                operation=operation,
                target_name="consul_handler",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                "Missing or invalid 'payload' in envelope",
                context=ctx,
            )

        # Route to appropriate handler
        if operation == "consul.kv_get":
            return await self._kv_get(payload, correlation_id, input_envelope_id)
        elif operation == "consul.kv_put":
            return await self._kv_put(payload, correlation_id, input_envelope_id)
        elif operation == "consul.register":
            return await self._register_service(
                payload, correlation_id, input_envelope_id
            )
        else:  # consul.deregister - validated above, guaranteed to be deregister
            return await self._deregister_service(
                payload, correlation_id, input_envelope_id
            )

    async def _execute_with_retry(
        self,
        operation: str,
        func: Callable[[], T],
        correlation_id: UUID,
    ) -> T:
        """Execute operation with exponential backoff retry logic and circuit breaker.

        Thread Pool Integration:
            All consul operations (which are synchronous) are executed in a dedicated
            thread pool via loop.run_in_executor(). This prevents blocking the async
            event loop and allows concurrent Consul operations up to max_workers limit.

        Circuit breaker integration (via MixinAsyncCircuitBreaker):
            - Checks circuit state before execution (raises if OPEN)
            - Records success/failure for circuit state management
            - Allows test request in HALF_OPEN state

        Args:
            operation: Operation name for logging
            func: Callable to execute (synchronous consul method)
            correlation_id: Correlation ID for tracing

        Returns:
            Result from func()

        Raises:
            InfraTimeoutError: If all retries exhausted or operation times out
            InfraConnectionError: If connection fails
            InfraAuthenticationError: If authentication fails
            InfraUnavailableError: If circuit breaker is OPEN
        """
        if self._config is None:
            raise RuntimeError("Config not initialized")

        await self._check_circuit_if_enabled(operation, correlation_id)

        retry_state, op_context = self._init_retry_context(operation, correlation_id)

        while retry_state.is_retriable():
            result = await self._try_execute_operation(
                func, retry_state, op_context, operation, correlation_id
            )
            if result is not None:
                return result[0]  # Unpack the result tuple

            # Get updated retry state after error handling
            retry_state = self._last_retry_state

            await self._log_retry_attempt(operation, retry_state, correlation_id)
            await asyncio.sleep(retry_state.delay_seconds)

        # Should never reach here, but satisfy type checker
        if retry_state.last_error is not None:
            raise RuntimeError(f"Retry exhausted: {retry_state.last_error}")
        raise RuntimeError("Retry loop completed without result")

    def _init_retry_context(
        self, operation: str, correlation_id: UUID
    ) -> tuple[ModelRetryState, ModelOperationContext]:
        """Initialize retry state and operation context.

        Args:
            operation: Operation name for context.
            correlation_id: Correlation ID for tracing.

        Returns:
            Tuple of (ModelRetryState, ModelOperationContext).
        """
        if self._config is None:
            raise RuntimeError("Config not initialized")

        retry_config = self._config.retry
        retry_state = ModelRetryState(
            attempt=0,
            max_attempts=retry_config.max_attempts,
            delay_seconds=retry_config.initial_delay_seconds,
            backoff_multiplier=retry_config.exponential_base,
        )

        op_context = ModelOperationContext.create(
            operation_name=operation,
            correlation_id=correlation_id,
            timeout_seconds=self._config.timeout_seconds,
        )

        return retry_state, op_context

    async def _try_execute_operation(
        self,
        func: Callable[[], T],
        retry_state: ModelRetryState,
        op_context: ModelOperationContext,
        operation: str,
        correlation_id: UUID,
    ) -> tuple[T] | None:
        """Try to execute an operation once with error handling.

        Args:
            func: Callable to execute.
            retry_state: Current retry state.
            op_context: Operation context.
            operation: Operation name.
            correlation_id: Correlation ID.

        Returns:
            Tuple containing result if successful, None if should retry.

        Raises:
            InfraTimeoutError, InfraConnectionError, InfraAuthenticationError:
                If error is not retriable or retries are exhausted.
        """
        if self._config is None:
            raise RuntimeError("Config not initialized")

        retry_config = self._config.retry
        try:
            loop = asyncio.get_running_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(self._executor, func),
                timeout=op_context.timeout_seconds,
            )

            await self._reset_circuit_if_enabled()
            return (result,)

        except Exception as e:
            classification = self._classify_error(e, operation)

            if not classification.should_retry:
                error = await self._handle_non_retriable_error(
                    classification, operation, correlation_id, e
                )
                raise error from e

            new_state, error_to_raise = await self._handle_retriable_error(
                classification,
                retry_state,
                retry_config.max_delay_seconds,
                operation,
                correlation_id,
                op_context,
                e,
            )

            self._last_retry_state = new_state

            if error_to_raise is not None:
                raise error_to_raise from e

            return None

    async def _kv_get(
        self,
        payload: dict[str, JsonValue],
        correlation_id: UUID,
        input_envelope_id: UUID,
    ) -> ModelHandlerOutput[ModelConsulHandlerResponse]:
        """Get value from Consul KV store.

        Args:
            payload: dict containing:
                - key: KV key path (required)
                - recurse: Optional bool to get all keys under prefix (default: False)
            correlation_id: Correlation ID for tracing
            input_envelope_id: Input envelope ID for causality tracking

        Returns:
            ModelHandlerOutput wrapping the KV data with correlation tracking
        """
        key = payload.get("key")
        if not isinstance(key, str) or not key:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.CONSUL,
                operation="consul.kv_get",
                target_name="consul_handler",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                "Missing or invalid 'key' in payload",
                context=ctx,
            )

        recurse = payload.get("recurse", False)
        recurse_bool = recurse is True or recurse == "true"

        if self._client is None:
            raise RuntimeError("Client not initialized")

        def get_func() -> tuple[
            int, list[dict[str, JsonValue]] | dict[str, JsonValue] | None
        ]:
            if self._client is None:
                raise RuntimeError("Client not initialized")
            index, data = self._client.kv.get(key, recurse=recurse_bool)
            return index, data

        index, data = await self._execute_with_retry(
            "consul.kv_get",
            get_func,
            correlation_id,
        )

        # Handle response - data can be None if key doesn't exist
        if data is None:
            typed_payload = ModelConsulKVGetNotFoundPayload(
                key=key,
                index=index,
            )
            return self._build_response(
                typed_payload, correlation_id, input_envelope_id
            )

        # Handle single key or recurse results
        if isinstance(data, list):
            # Recurse mode - multiple keys
            items: list[ModelConsulKVItem] = []
            for item in data:
                value = item.get("Value")
                decoded_value = (
                    value.decode("utf-8") if isinstance(value, bytes) else value
                )
                item_key = item.get("Key")
                items.append(
                    ModelConsulKVItem(
                        key=item_key if isinstance(item_key, str) else "",
                        value=decoded_value if isinstance(decoded_value, str) else None,
                        flags=item.get("Flags")
                        if isinstance(item.get("Flags"), int)
                        else None,
                        modify_index=item.get("ModifyIndex")
                        if isinstance(item.get("ModifyIndex"), int)
                        else None,
                    )
                )
            typed_payload_recurse = ModelConsulKVGetRecursePayload(
                found=len(items) > 0,
                items=items,
                count=len(items),
                index=index,
            )
            return self._build_response(
                typed_payload_recurse, correlation_id, input_envelope_id
            )
        else:
            # Single key mode
            value = data.get("Value")
            decoded_value = value.decode("utf-8") if isinstance(value, bytes) else value
            data_key = data.get("Key")
            typed_payload_found = ModelConsulKVGetFoundPayload(
                key=data_key if isinstance(data_key, str) else key,
                value=decoded_value if isinstance(decoded_value, str) else None,
                flags=data.get("Flags") if isinstance(data.get("Flags"), int) else None,
                modify_index=data.get("ModifyIndex")
                if isinstance(data.get("ModifyIndex"), int)
                else None,
                index=index,
            )
            return self._build_response(
                typed_payload_found, correlation_id, input_envelope_id
            )

    async def _kv_put(
        self,
        payload: dict[str, JsonValue],
        correlation_id: UUID,
        input_envelope_id: UUID,
    ) -> ModelHandlerOutput[ModelConsulHandlerResponse]:
        """Put value to Consul KV store.

        Args:
            payload: dict containing:
                - key: KV key path (required)
                - value: Value to store (required, string)
                - flags: Optional integer flags
                - cas: Optional check-and-set index for optimistic locking
            correlation_id: Correlation ID for tracing
            input_envelope_id: Input envelope ID for causality tracking

        Returns:
            ModelHandlerOutput wrapping the operation result with correlation tracking
        """
        key = payload.get("key")
        if not isinstance(key, str) or not key:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.CONSUL,
                operation="consul.kv_put",
                target_name="consul_handler",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                "Missing or invalid 'key' in payload",
                context=ctx,
            )

        value = payload.get("value")
        if not isinstance(value, str):
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.CONSUL,
                operation="consul.kv_put",
                target_name="consul_handler",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                "Missing or invalid 'value' in payload - must be a string",
                context=ctx,
            )

        flags = payload.get("flags")
        flags_int: int | None = flags if isinstance(flags, int) else None

        cas = payload.get("cas")
        cas_int: int | None = cas if isinstance(cas, int) else None

        if self._client is None:
            raise RuntimeError("Client not initialized")

        def put_func() -> bool:
            if self._client is None:
                raise RuntimeError("Client not initialized")
            result: bool = self._client.kv.put(key, value, flags=flags_int, cas=cas_int)
            return result

        success = await self._execute_with_retry(
            "consul.kv_put",
            put_func,
            correlation_id,
        )

        typed_payload = ModelConsulKVPutPayload(
            success=success,
            key=key,
        )
        return self._build_response(typed_payload, correlation_id, input_envelope_id)

    async def _register_service(
        self,
        payload: dict[str, JsonValue],
        correlation_id: UUID,
        input_envelope_id: UUID,
    ) -> ModelHandlerOutput[ModelConsulHandlerResponse]:
        """Register service with Consul agent.

        Args:
            payload: dict containing:
                - name: Service name (required)
                - service_id: Optional unique service ID (defaults to name)
                - address: Optional service address
                - port: Optional service port
                - tags: Optional list of tags
                - check: Optional health check configuration dict
            correlation_id: Correlation ID for tracing
            input_envelope_id: Input envelope ID for causality tracking

        Returns:
            ModelHandlerOutput wrapping the registration result with correlation tracking
        """
        name = payload.get("name")
        if not isinstance(name, str) or not name:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.CONSUL,
                operation="consul.register",
                target_name="consul_handler",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                "Missing or invalid 'name' in payload",
                context=ctx,
            )

        service_id = payload.get("service_id")
        service_id_str: str | None = service_id if isinstance(service_id, str) else None

        address = payload.get("address")
        address_str: str | None = address if isinstance(address, str) else None

        port = payload.get("port")
        port_int: int | None = port if isinstance(port, int) else None

        tags = payload.get("tags")
        tags_list: list[str] | None = None
        if isinstance(tags, list):
            tags_list = [str(t) for t in tags]

        check = payload.get("check")
        check_dict: dict[str, JsonValue] | None = (
            check if isinstance(check, dict) else None
        )

        if self._client is None:
            raise RuntimeError("Client not initialized")

        def register_func() -> bool:
            if self._client is None:
                raise RuntimeError("Client not initialized")
            self._client.agent.service.register(
                name=name,
                service_id=service_id_str,
                address=address_str,
                port=port_int,
                tags=tags_list,
                check=check_dict,
            )
            return True

        await self._execute_with_retry(
            "consul.register",
            register_func,
            correlation_id,
        )

        typed_payload = ModelConsulRegisterPayload(
            registered=True,
            name=name,
            consul_service_id=service_id_str or name,
        )
        return self._build_response(typed_payload, correlation_id, input_envelope_id)

    async def _deregister_service(
        self,
        payload: dict[str, JsonValue],
        correlation_id: UUID,
        input_envelope_id: UUID,
    ) -> ModelHandlerOutput[ModelConsulHandlerResponse]:
        """Deregister service from Consul agent.

        Args:
            payload: dict containing:
                - service_id: Service ID to deregister (required)
            correlation_id: Correlation ID for tracing
            input_envelope_id: Input envelope ID for causality tracking

        Returns:
            ModelHandlerOutput wrapping the deregistration result with correlation tracking
        """
        service_id = payload.get("service_id")
        if not isinstance(service_id, str) or not service_id:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.CONSUL,
                operation="consul.deregister",
                target_name="consul_handler",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                "Missing or invalid 'service_id' in payload",
                context=ctx,
            )

        if self._client is None:
            raise RuntimeError("Client not initialized")

        def deregister_func() -> bool:
            if self._client is None:
                raise RuntimeError("Client not initialized")
            self._client.agent.service.deregister(service_id)
            return True

        await self._execute_with_retry(
            "consul.deregister",
            deregister_func,
            correlation_id,
        )

        typed_payload = ModelConsulDeregisterPayload(
            deregistered=True,
            consul_service_id=service_id,
        )
        return self._build_response(typed_payload, correlation_id, input_envelope_id)

    def describe(self) -> dict[str, JsonValue]:
        """Return handler metadata and capabilities.

        Returns:
            Handler description with supported operations and configuration
        """
        return {
            "handler_type": self.handler_type,
            "supported_operations": sorted(SUPPORTED_OPERATIONS),
            "timeout_seconds": self._config.timeout_seconds if self._config else 30.0,
            "initialized": self._initialized,
            "version": "0.1.0-mvp",
        }


__all__: list[str] = ["ConsulHandler"]
