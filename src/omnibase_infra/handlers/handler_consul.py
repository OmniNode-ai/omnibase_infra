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

Envelope-Based Routing:
    This handler uses envelope-based operation routing. See CLAUDE.md section
    "Intent Model Architecture > Envelope-Based Handler Routing" for the full
    design pattern and how orchestrators translate intents to handler envelopes.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple, TypeVar
from uuid import UUID, uuid4

import consul

from omnibase_core.container import ModelONEXContainer
from omnibase_core.models.dispatch import ModelHandlerOutput
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
    EnumResponseStatus,
)
from omnibase_infra.errors import (
    InfraAuthenticationError,
    InfraConnectionError,
    InfraConsulError,
    ModelInfraErrorContext,
    RuntimeHostError,
)
from omnibase_infra.handlers.mixins import (
    MixinConsulInitialization,
    MixinConsulKV,
    MixinConsulService,
    MixinConsulTopicIndex,
)
from omnibase_infra.handlers.models import (
    ModelOperationContext,
    ModelRetryState,
)
from omnibase_infra.handlers.models.consul import (
    ConsulPayload,
    ModelConsulHandlerConfig,
    ModelConsulHandlerPayload,
)
from omnibase_infra.handlers.models.model_consul_handler_response import (
    ModelConsulHandlerResponse,
)
from omnibase_infra.mixins import (
    EnumRetryErrorCategory,
    MixinAsyncCircuitBreaker,
    MixinEnvelopeExtraction,
    MixinRetryExecution,
    ModelRetryErrorClassification,
)

T = TypeVar("T")

logger = logging.getLogger(__name__)

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


class RetryContext(NamedTuple):
    """Context for retry operations containing state and operation metadata.

    This named tuple provides clear field access for retry initialization,
    improving code clarity over plain tuples.

    Attributes:
        retry_state: Current retry state including attempt count and delays.
        operation_context: Operation context with correlation ID and timeout.
    """

    retry_state: ModelRetryState
    operation_context: ModelOperationContext


class HandlerConsul(
    MixinAsyncCircuitBreaker,
    MixinRetryExecution,
    MixinEnvelopeExtraction,
    MixinConsulInitialization,
    MixinConsulKV,
    MixinConsulTopicIndex,  # Must come before MixinConsulService (MRO order)
    MixinConsulService,
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
        HandlerVault and other infrastructure handlers. This provides predictable
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

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize HandlerConsul in uninitialized state.

        Args:
            container: ONEX container for dependency injection. Required for
                consistent handler initialization pattern across all handlers.

        Note: Circuit breaker is initialized during initialize() call when
        configuration is available. The mixin's _init_circuit_breaker() method
        is called there with the actual config values.
        """
        self._container = container
        self._client: consul.Consul | None = None
        self._config: ModelConsulHandlerConfig | None = None
        self._initialized: bool = False
        self._executor: ThreadPoolExecutor | None = None
        self._max_workers: int = 0
        self._max_queue_size: int = 0
        # Circuit breaker initialized flag - set after _init_circuit_breaker called
        self._circuit_breaker_initialized: bool = False

    @property
    def handler_type(self) -> EnumHandlerType:
        """Return the architectural role of this handler.

        Returns:
            EnumHandlerType.INFRA_HANDLER - This handler is an infrastructure
            protocol/transport handler (as opposed to NODE_HANDLER for event
            processing, PROJECTION_HANDLER for read models, or COMPUTE_HANDLER
            for pure computation).

        Note:
            handler_type determines lifecycle, protocol selection, and runtime
            invocation patterns. It answers "what is this handler in the architecture?"

        See Also:
            - handler_category: Behavioral classification (EFFECT/COMPUTE)
            - docs/architecture/HANDLER_PROTOCOL_DRIVEN_ARCHITECTURE.md
        """
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Return the behavioral classification of this handler.

        Returns:
            EnumHandlerTypeCategory.EFFECT - This handler performs side-effecting
            I/O operations (Consul KV store and service registry). EFFECT handlers
            are not deterministic and interact with external systems.

        Note:
            handler_category determines security rules, determinism guarantees,
            replay safety, and permissions. It answers "how does this handler
            behave at runtime?"

            Categories:
            - COMPUTE: Pure, deterministic transformations (no side effects)
            - EFFECT: Side-effecting I/O (database, HTTP, service calls)
            - NONDETERMINISTIC_COMPUTE: Pure but not deterministic (UUID, random)

        See Also:
            - handler_type: Architectural role (INFRA_HANDLER/NODE_HANDLER/etc.)
            - docs/architecture/HANDLER_PROTOCOL_DRIVEN_ARCHITECTURE.md
        """
        return EnumHandlerTypeCategory.EFFECT

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
    ) -> ModelRetryErrorClassification:
        """Classify Consul-specific exceptions for retry handling.

        Args:
            error: The exception to classify.
            operation: The operation name for context.

        Returns:
            ModelRetryErrorClassification with retry decision and error details.
        """
        if isinstance(error, TimeoutError):
            return ModelRetryErrorClassification(
                category=EnumRetryErrorCategory.TIMEOUT,
                should_retry=True,
                record_circuit_failure=True,
                error_message=f"Timeout: {type(error).__name__}",
            )

        if isinstance(error, consul.ACLPermissionDenied):
            return ModelRetryErrorClassification(
                category=EnumRetryErrorCategory.AUTHENTICATION,
                should_retry=False,
                record_circuit_failure=True,
                error_message="Consul ACL permission denied - check token permissions",
            )

        if isinstance(error, consul.Timeout):
            return ModelRetryErrorClassification(
                category=EnumRetryErrorCategory.TIMEOUT,
                should_retry=True,
                record_circuit_failure=True,
                error_message=f"Consul timeout: {type(error).__name__}",
            )

        if isinstance(error, consul.ConsulException):
            return ModelRetryErrorClassification(
                category=EnumRetryErrorCategory.CONNECTION,
                should_retry=True,
                record_circuit_failure=True,
                error_message=f"Consul error: {type(error).__name__}",
            )

        # Unknown error - retry eligible
        return ModelRetryErrorClassification(
            category=EnumRetryErrorCategory.UNKNOWN,
            should_retry=True,
            record_circuit_failure=True,
            error_message=f"Unexpected error: {type(error).__name__}",
        )

    async def initialize(self, config: dict[str, object]) -> None:
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
            self._log_initialization_success(self._config, init_correlation_id)

        except (InfraConnectionError, InfraAuthenticationError):
            raise
        except consul.ACLPermissionDenied as e:
            self._raise_auth_error(init_correlation_id, e)
        except consul.ConsulException as e:
            self._raise_connection_error(init_correlation_id, e)
        except Exception as e:
            self._raise_runtime_error(init_correlation_id, e)

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
            "HandlerConsul shutdown complete",
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
        Consul operations, matching the pattern used by HandlerDb.

        Args:
            typed_payload: Strongly-typed payload from the discriminated union.
            correlation_id: Correlation ID for tracing.
            input_envelope_id: Input envelope ID for causality tracking.

        Returns:
            ModelHandlerOutput wrapping ModelConsulHandlerResponse.
        """
        response = ModelConsulHandlerResponse(
            status=EnumResponseStatus.SUCCESS,
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
        self, envelope: dict[str, object]
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
                "Consul client not initialized for operation 'execute'. "
                "Call initialize() first.",
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

        Thread-Safety:
            This method is concurrency-safe. Each call maintains its own retry
            state stack, with no shared mutable state between concurrent operations.
            This allows multiple operations to execute in parallel without
            interfering with each other's retry logic.

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
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.CONSUL,
                operation=operation,
                target_name="consul_handler",
                correlation_id=correlation_id,
            )
            raise InfraConsulError("Consul config not initialized", context=context)

        await self._check_circuit_if_enabled(operation, correlation_id)

        ctx = self._init_retry_context(operation, correlation_id)
        retry_state = ctx.retry_state
        op_context = ctx.operation_context

        # Track last exception for proper error chaining when retries exhaust
        last_exception: Exception | None = None

        while retry_state.is_retriable():
            (
                result_tuple,
                retry_state,
                caught_exception,
            ) = await self._try_execute_operation(
                func, retry_state, op_context, operation, correlation_id
            )
            if result_tuple is not None:
                return result_tuple[0]  # Unpack the result tuple

            # Track the last exception for error chaining
            last_exception = caught_exception

            await self._log_retry_attempt(operation, retry_state, correlation_id)
            await asyncio.sleep(retry_state.delay_seconds)

        # Should never reach here, but satisfy type checker
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.CONSUL,
            operation=operation,
            target_name="consul_handler",
            correlation_id=correlation_id,
        )
        if retry_state.last_error is not None:
            # Chain to original exception to preserve root cause for debugging
            raise InfraConsulError(
                f"Retry exhausted: {retry_state.last_error}",
                context=context,
            ) from last_exception
        raise InfraConsulError(
            "Retry loop completed without result",
            context=context,
        ) from last_exception

    def _init_retry_context(self, operation: str, correlation_id: UUID) -> RetryContext:
        """Initialize retry state and operation context.

        Args:
            operation: Operation name for context.
            correlation_id: Correlation ID for tracing.

        Returns:
            RetryContext with initialized retry_state and operation_context.
        """
        if self._config is None:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.CONSUL,
                operation=operation,
                target_name="consul_handler",
                correlation_id=correlation_id,
            )
            raise InfraConsulError("Consul config not initialized", context=context)

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

        return RetryContext(
            retry_state=retry_state,
            operation_context=op_context,
        )

    async def _try_execute_operation(
        self,
        func: Callable[[], T],
        retry_state: ModelRetryState,
        op_context: ModelOperationContext,
        operation: str,
        correlation_id: UUID,
    ) -> (
        tuple[tuple[T], ModelRetryState, None] | tuple[None, ModelRetryState, Exception]
    ):
        """Try to execute an operation once with error handling.

        Thread-Safety:
            This method is concurrency-safe. All retry state is passed explicitly
            as parameters and returned as values. No shared mutable state is used.
            This allows multiple concurrent operations to execute independently.

        Args:
            func: Callable to execute.
            retry_state: Current retry state.
            op_context: Operation context.
            operation: Operation name.
            correlation_id: Correlation ID.

        Returns:
            Tuple of ((result,), retry_state, None) if successful.
            Tuple of (None, updated_retry_state, exception) if should retry.
            The exception is returned to enable proper error chaining when
            retries are eventually exhausted.

        Raises:
            InfraTimeoutError, InfraConnectionError, InfraAuthenticationError:
                If error is not retriable or retries are exhausted.
        """
        if self._config is None:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.CONSUL,
                operation=operation,
                target_name="consul_handler",
                correlation_id=correlation_id,
            )
            raise InfraConsulError("Consul config not initialized", context=context)

        retry_config = self._config.retry
        try:
            loop = asyncio.get_running_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(self._executor, func),
                timeout=op_context.timeout_seconds,
            )

            await self._reset_circuit_if_enabled()
            return (result,), retry_state, None

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

            if error_to_raise is not None:
                raise error_to_raise from e

            return None, new_state, e

    def describe(self) -> dict[str, object]:
        """Return handler metadata and capabilities for introspection.

        This method exposes the handler's type classification along with its
        operational configuration and capabilities.

        Returns:
            dict containing:
                - handler_type: Architectural role from handler_type property
                  (e.g., "infra_handler"). See EnumHandlerType for valid values.
                - handler_category: Behavioral classification from handler_category
                  property (e.g., "effect"). See EnumHandlerTypeCategory for valid values.
                - supported_operations: List of supported operations
                - timeout_seconds: Request timeout in seconds
                - initialized: Whether the handler is initialized
                - version: Handler version string

        Note:
            The handler_type and handler_category fields form the handler
            classification system:

            1. handler_type (architectural role): Determines lifecycle and invocation
               patterns. This handler is INFRA_HANDLER (protocol/transport handler).

            2. handler_category (behavioral classification): Determines security rules
               and replay safety. This handler is EFFECT (side-effecting I/O).

            The transport type for this handler is CONSUL (service discovery).

        See Also:
            - handler_type property: Full documentation of architectural role
            - handler_category property: Full documentation of behavioral classification
            - docs/architecture/HANDLER_PROTOCOL_DRIVEN_ARCHITECTURE.md
        """
        return {
            "handler_type": self.handler_type.value,
            "handler_category": self.handler_category.value,
            "supported_operations": sorted(SUPPORTED_OPERATIONS),
            "timeout_seconds": self._config.timeout_seconds if self._config else 30.0,
            "initialized": self._initialized,
            "version": "0.1.0-mvp",
        }


__all__: list[str] = ["HandlerConsul", "RetryContext"]
