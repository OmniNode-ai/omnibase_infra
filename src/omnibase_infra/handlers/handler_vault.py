# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""HashiCorp Vault Handler - MVP implementation using hvac async client.

Supports secret management operations with configurable retry logic and
automatic token renewal management.

Security Features:
    - SecretStr protection for tokens (prevents accidental logging)
    - Sanitized error messages (never expose secrets in logs)
    - SSL verification enabled by default
    - Token auto-renewal management

All secret operations MUST use proper authentication and authorization.

Return Type:
    All operations return ModelHandlerOutput[dict[str, JsonValue]] per OMN-975.
    Uses ModelHandlerOutput.for_compute() since handlers return synchronous results
    rather than emitting events to the event bus.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, TypeVar
from uuid import UUID, uuid4

import hvac
from omnibase_core.enums.enum_handler_type import EnumHandlerType
from omnibase_core.models.dispatch import ModelHandlerOutput
from pydantic import SecretStr, ValidationError

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraAuthenticationError,
    InfraConnectionError,
    InfraTimeoutError,
    InfraUnavailableError,
    ModelInfraErrorContext,
    ProtocolConfigurationError,
    RuntimeHostError,
    SecretResolutionError,
)
from omnibase_infra.handlers.model_vault_handler_config import ModelVaultHandlerConfig
from omnibase_infra.mixins import MixinAsyncCircuitBreaker, MixinEnvelopeExtraction

if TYPE_CHECKING:
    from omnibase_core.types import JsonValue

T = TypeVar("T")

logger = logging.getLogger(__name__)

# Handler ID for ModelHandlerOutput
HANDLER_ID_VAULT: str = "vault-handler"

DEFAULT_MOUNT_POINT: str = "secret"
SUPPORTED_OPERATIONS: frozenset[str] = frozenset(
    {
        "vault.read_secret",
        "vault.write_secret",
        "vault.delete_secret",
        "vault.list_secrets",
        "vault.renew_token",
        "vault.health_check",
    }
)


class VaultHandler(MixinAsyncCircuitBreaker, MixinEnvelopeExtraction):
    """HashiCorp Vault handler using hvac client (MVP: KV v2 secrets engine).

    Security Policy - Token Handling:
        The Vault token contains sensitive credentials and is treated as a secret
        throughout this handler. The following security measures are enforced:

        1. Token is stored as SecretStr in config (never logged or exposed)
        2. All error messages use generic descriptions without exposing token
        3. Health check responses exclude token information
        4. Token renewal is automatic when TTL falls below threshold
        5. The describe() method returns capabilities without credentials

        See CLAUDE.md "Error Sanitization Guidelines" for the full security policy
        on what information is safe vs unsafe to include in errors and logs.

    Token Renewal Management:
        - Tokens are automatically renewed when TTL < token_renewal_threshold_seconds
        - Token renewal is checked before each operation
        - Failed renewal raises InfraAuthenticationError
        - Token expiration tracking uses self._token_expires_at

    Thread Pool Management (Production-Grade):
        - Bounded ThreadPoolExecutor prevents resource exhaustion
        - Configurable max_concurrent_operations (default: 10, max: 100)
        - Thread pool gracefully shutdown on handler.shutdown()
        - All hvac (synchronous) operations run in dedicated thread pool

    Circuit Breaker Pattern (Production-Grade):
        - Uses MixinAsyncCircuitBreaker for consistent circuit breaker implementation
        - Prevents cascading failures to Vault service
        - Three states: CLOSED (normal), OPEN (blocking), HALF_OPEN (testing)
        - Configurable failure_threshold (default: 5 consecutive failures)
        - Configurable reset_timeout (default: 30 seconds)
        - Raises InfraUnavailableError when circuit is OPEN
        - Can be disabled via circuit_breaker_enabled=False

    Retry Logic:
        - All operations use exponential backoff retry logic
        - Retry configuration from ModelVaultRetryConfig
        - Backoff calculation: initial_backoff * (exponential_base ** attempt)
        - Max backoff capped at max_backoff_seconds
        - Circuit breaker checked before retry execution
    """

    def __init__(self) -> None:
        """Initialize VaultHandler in uninitialized state.

        Note: Circuit breaker is initialized during initialize() call when
        configuration is available. The mixin's _init_circuit_breaker() method
        is called there with the actual config values.
        """
        self._client: hvac.Client | None = None
        self._config: ModelVaultHandlerConfig | None = None
        self._initialized: bool = False
        self._token_expires_at: float = 0.0
        self._executor: ThreadPoolExecutor | None = None
        self._max_workers: int = 0
        self._max_queue_size: int = 0
        # Circuit breaker initialized flag - set after _init_circuit_breaker called
        self._circuit_breaker_initialized: bool = False

    @property
    def handler_type(self) -> EnumHandlerType:
        """Return EnumHandlerType.VAULT."""
        return EnumHandlerType.VAULT

    @property
    def max_workers(self) -> int:
        """Return thread pool max workers (public API for tests)."""
        return self._max_workers

    @property
    def max_queue_size(self) -> int:
        """Return maximum queue size (public API for tests)."""
        return self._max_queue_size

    async def initialize(self, config: dict[str, JsonValue]) -> None:
        """Initialize Vault client with configuration.

        Args:
            config: Configuration dict containing:
                - url: Vault server URL (required) - must be a valid URL format
                  (e.g., "http://localhost:8200" or "https://vault.example.com:8200")
                - token: Vault authentication token (required)
                - namespace: Optional Vault namespace for Enterprise
                - timeout_seconds: Optional timeout (default 30.0)
                - verify_ssl: Optional SSL verification (default True)
                - token_renewal_threshold_seconds: Optional renewal threshold (default 300.0)
                - retry: Optional retry configuration dict

        Raises:
            ProtocolConfigurationError: If configuration validation fails. This includes:
                - Missing URL: URL is a required field (Pydantic validation)
                - Empty URL: URL cannot be empty string (field_validator)
                - Invalid URL format: Must start with http:// or https:// (field_validator)
                - Missing token: Token is a required field (post-Pydantic defensive check)
                - Other Pydantic validation failures (e.g., timeout out of range)

                All URL-related configuration errors raise ProtocolConfigurationError because
                they represent invalid configuration, not runtime connectivity issues.
                Error message will contain Pydantic validation details.

            RuntimeHostError: If client initialization fails for non-auth/non-connection
                reasons (e.g., unexpected exception during hvac.Client creation).

            InfraAuthenticationError: If token authentication fails (token rejected by Vault).
                This occurs when the Vault server is reachable but rejects the provided token.

            InfraConnectionError: If connection to Vault server fails (network/DNS issues).
                This occurs when the Vault server is unreachable at the specified URL.
                Use this error type when the URL format is valid but the server cannot be reached.

        Error Type Decision Guide:
            - URL missing/empty/malformed -> ProtocolConfigurationError (config validation)
            - URL valid but server unreachable -> InfraConnectionError (runtime connectivity)
            - URL valid, server reachable, auth fails -> InfraAuthenticationError (auth issue)

        Security:
            Token must be provided via environment variable, not hardcoded in config.
            Use SecretStr for token to prevent accidental logging.

        Example Error Scenarios:
            >>> # Missing URL - raises ProtocolConfigurationError
            >>> await handler.initialize({"token": "s.xxx"})
            ProtocolConfigurationError: Invalid Vault configuration: ... url ... required

            >>> # Empty URL - raises ProtocolConfigurationError
            >>> await handler.initialize({"url": "", "token": "s.xxx"})
            ProtocolConfigurationError: Invalid Vault configuration: ... URL cannot be empty

            >>> # Invalid URL format - raises ProtocolConfigurationError
            >>> await handler.initialize({"url": "not-a-valid-url", "token": "s.xxx"})
            ProtocolConfigurationError: Invalid Vault configuration: ... must start with http://

            >>> # Valid URL but server unreachable - raises InfraConnectionError
            >>> await handler.initialize({"url": "http://unreachable:8200", "token": "s.xxx"})
            InfraConnectionError: Failed to connect to Vault: VaultError

            >>> # Valid URL, server reachable, invalid token - raises InfraAuthenticationError
            >>> await handler.initialize({"url": "http://localhost:8200", "token": "bad"})
            InfraAuthenticationError: Vault authentication failed - check token validity
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

        # Parse configuration using Pydantic model
        try:
            # Handle SecretStr token conversion
            token_raw = config.get("token")
            if isinstance(token_raw, str):
                config = dict(config)  # Make mutable copy
                config["token"] = SecretStr(token_raw)

            # Type ignore for dict unpacking - Pydantic handles validation
            self._config = ModelVaultHandlerConfig(**config)  # type: ignore[arg-type]
        except ValidationError as e:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.VAULT,
                operation="initialize",
                target_name="vault_handler",
                correlation_id=init_correlation_id,
                namespace=None,  # Config not initialized yet
            )
            raise ProtocolConfigurationError(
                f"Invalid Vault configuration: {e}",
                context=ctx,
            ) from e
        except Exception as e:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.VAULT,
                operation="initialize",
                target_name="vault_handler",
                correlation_id=init_correlation_id,
                namespace=None,  # Config not initialized yet
            )
            raise RuntimeHostError(
                f"Configuration parsing failed: {type(e).__name__}",
                context=ctx,
            ) from e

        # Defensive validation for required fields
        # Note: These checks are defensive programming since Pydantic validation
        # should catch missing/empty URL and missing token. However, we keep them
        # to ensure consistent error handling if the Pydantic model changes.
        # All configuration validation errors use ProtocolConfigurationError per
        # ONEX error handling patterns (config issues != runtime connectivity issues).
        if not self._config.url:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.VAULT,
                operation="initialize",
                target_name="vault_handler",
                correlation_id=init_correlation_id,
                namespace=self._config.namespace if self._config else None,
            )
            raise ProtocolConfigurationError(
                "Missing 'url' in config - Vault server URL required",
                context=ctx,
            )

        if self._config.token is None:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.VAULT,
                operation="initialize",
                target_name="vault_handler",
                correlation_id=init_correlation_id,
                namespace=self._config.namespace if self._config else None,
            )
            raise ProtocolConfigurationError(
                "Missing 'token' in config - Vault authentication token required",
                context=ctx,
            )

        try:
            # Initialize hvac client (synchronous)
            self._client = hvac.Client(
                url=self._config.url,
                token=self._config.token.get_secret_value(),
                namespace=self._config.namespace,
                verify=self._config.verify_ssl,
                timeout=self._config.timeout_seconds,
            )

            # Verify authentication
            if not self._client.is_authenticated():
                ctx = ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.VAULT,
                    operation="initialize",
                    target_name="vault_handler",
                    correlation_id=init_correlation_id,
                    namespace=self._config.namespace if self._config else None,
                )
                raise InfraAuthenticationError(
                    "Vault authentication failed - check token validity",
                    context=ctx,
                )

            # Initialize token expiration tracking by querying actual TTL from Vault
            try:
                token_info = self._client.auth.token.lookup_self()
                token_data = token_info.get("data", {})
                token_ttl = None

                if isinstance(token_data, dict):
                    # TTL is in seconds, use it if available
                    ttl_seconds = token_data.get("ttl")
                    if isinstance(ttl_seconds, int) and ttl_seconds > 0:
                        token_ttl = ttl_seconds

                if token_ttl is None:
                    # Fallback to config or safe default
                    token_ttl = self._config.default_token_ttl
                    logger.warning(
                        "Token TTL not in Vault response, using fallback",
                        extra={
                            "ttl": token_ttl,
                            "correlation_id": str(init_correlation_id),
                        },
                    )

                self._token_expires_at = time.time() + token_ttl

                logger.info(
                    "Token TTL initialized",
                    extra={
                        "ttl_seconds": token_ttl,
                        "correlation_id": str(init_correlation_id),
                    },
                )
            except Exception as e:
                # Fallback to config default TTL if lookup fails
                token_ttl = self._config.default_token_ttl
                logger.warning(
                    "Failed to query token TTL, using fallback",
                    extra={
                        "error_type": type(e).__name__,
                        "default_ttl_seconds": token_ttl,
                        "correlation_id": str(init_correlation_id),
                    },
                )
                self._token_expires_at = time.time() + token_ttl

            # Create bounded thread pool executor for production safety
            self._max_workers = self._config.max_concurrent_operations
            self._executor = ThreadPoolExecutor(
                max_workers=self._max_workers,
                thread_name_prefix="vault_handler_",
            )
            # Calculate max queue size
            self._max_queue_size = (
                self._max_workers * self._config.max_queue_size_multiplier
            )

            # Initialize circuit breaker using mixin (if enabled)
            if self._config.circuit_breaker_enabled:
                self._init_circuit_breaker(
                    threshold=self._config.circuit_breaker_failure_threshold,
                    reset_timeout=self._config.circuit_breaker_reset_timeout_seconds,
                    service_name=f"vault.{self._config.namespace or 'default'}",
                    transport_type=EnumInfraTransportType.VAULT,
                )
                self._circuit_breaker_initialized = True

            self._initialized = True
            logger.info(
                "%s initialized successfully",
                self.__class__.__name__,
                extra={
                    "handler": self.__class__.__name__,
                    "url": self._config.url,
                    "namespace": self._config.namespace,
                    "timeout_seconds": self._config.timeout_seconds,
                    "verify_ssl": self._config.verify_ssl,
                    "thread_pool_max_workers": self._max_workers,
                    "thread_pool_max_queue_size": self._max_queue_size,
                    "circuit_breaker_enabled": self._config.circuit_breaker_enabled,
                    "correlation_id": str(init_correlation_id),
                },
            )

        except InfraAuthenticationError:
            # Re-raise our own authentication errors without wrapping
            raise
        except hvac.exceptions.InvalidRequest as e:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.VAULT,
                operation="initialize",
                target_name="vault_handler",
                correlation_id=init_correlation_id,
                namespace=self._config.namespace if self._config else None,
            )
            raise InfraAuthenticationError(
                "Vault authentication failed - invalid token or permissions",
                context=ctx,
            ) from e
        except hvac.exceptions.VaultError as e:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.VAULT,
                operation="initialize",
                target_name="vault_handler",
                correlation_id=init_correlation_id,
                namespace=self._config.namespace if self._config else None,
            )
            raise InfraConnectionError(
                f"Failed to connect to Vault: {type(e).__name__}",
                context=ctx,
            ) from e
        except Exception as e:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.VAULT,
                operation="initialize",
                target_name="vault_handler",
                correlation_id=init_correlation_id,
                namespace=self._config.namespace if self._config else None,
            )
            raise RuntimeHostError(
                f"Failed to initialize Vault client: {type(e).__name__}",
                context=ctx,
            ) from e

    async def shutdown(self) -> None:
        """Close Vault client and release resources.

        Cleanup includes:
            - Shutting down thread pool executor (waits for pending tasks)
            - Clearing Vault client connection
            - Resetting circuit breaker state (thread-safe via mixin)
        """
        if self._executor is not None:
            # Shutdown thread pool gracefully (wait for pending tasks)
            self._executor.shutdown(wait=True)
            self._executor = None
        if self._client is not None:
            # hvac.Client doesn't have async close, just clear reference
            self._client = None

        # Reset circuit breaker state using mixin (thread-safe)
        if self._circuit_breaker_initialized:
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

        self._initialized = False
        self._config = None
        self._circuit_breaker_initialized = False
        logger.info("VaultHandler shutdown complete")

    async def execute(
        self, envelope: dict[str, JsonValue]
    ) -> ModelHandlerOutput[dict[str, JsonValue]]:
        """Execute Vault operation from envelope.

        Args:
            envelope: Request envelope containing:
                - operation: Vault operation (vault.read_secret, vault.write_secret, etc.)
                - payload: dict with operation-specific parameters
                - correlation_id: Optional correlation ID for tracing
                - envelope_id: Optional envelope ID for causality tracking

        Returns:
            ModelHandlerOutput[dict[str, JsonValue]] with status, payload, and correlation_id
            per OMN-975 handler output standardization.

        Raises:
            RuntimeHostError: If handler not initialized or invalid input.
            InfraConnectionError: If Vault connection fails.
            InfraAuthenticationError: If authentication fails.
            SecretResolutionError: If secret resolution fails.
        """
        correlation_id = self._extract_correlation_id(envelope)
        input_envelope_id = self._extract_envelope_id(envelope)

        if not self._initialized or self._client is None or self._config is None:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.VAULT,
                operation="execute",
                target_name="vault_handler",
                correlation_id=correlation_id,
                namespace=self._config.namespace if self._config else None,
            )
            raise RuntimeHostError(
                "VaultHandler not initialized. Call initialize() first.",
                context=ctx,
            )

        operation = envelope.get("operation")
        if not isinstance(operation, str):
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.VAULT,
                operation="execute",
                target_name="vault_handler",
                correlation_id=correlation_id,
                namespace=self._config.namespace if self._config else None,
            )
            raise RuntimeHostError(
                "Missing or invalid 'operation' in envelope",
                context=ctx,
            )

        if operation not in SUPPORTED_OPERATIONS:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.VAULT,
                operation=operation,
                target_name="vault_handler",
                correlation_id=correlation_id,
                namespace=self._config.namespace if self._config else None,
            )
            raise RuntimeHostError(
                f"Operation '{operation}' not supported in MVP. "
                f"Available: {', '.join(sorted(SUPPORTED_OPERATIONS))}",
                context=ctx,
            )

        payload = envelope.get("payload")
        if not isinstance(payload, dict):
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.VAULT,
                operation=operation,
                target_name="vault_handler",
                correlation_id=correlation_id,
                namespace=self._config.namespace if self._config else None,
            )
            raise RuntimeHostError(
                "Missing or invalid 'payload' in envelope",
                context=ctx,
            )

        # Check token renewal before operation
        await self._check_token_renewal(correlation_id)

        # Route to appropriate handler
        if operation == "vault.read_secret":
            return await self._read_secret(payload, correlation_id, input_envelope_id)
        elif operation == "vault.write_secret":
            return await self._write_secret(payload, correlation_id, input_envelope_id)
        elif operation == "vault.delete_secret":
            return await self._delete_secret(payload, correlation_id, input_envelope_id)
        elif operation == "vault.list_secrets":
            return await self._list_secrets(payload, correlation_id, input_envelope_id)
        elif operation == "vault.renew_token":
            return await self._renew_token_operation(correlation_id, input_envelope_id)
        else:  # vault.health_check
            return await self._health_check_operation(correlation_id, input_envelope_id)

    async def _check_token_renewal(self, correlation_id: UUID) -> None:
        """Check if token needs renewal and renew if necessary.

        Args:
            correlation_id: Correlation ID for tracing

        Raises:
            InfraAuthenticationError: If token renewal fails
        """
        if self._config is None or self._client is None:
            logger.debug(
                "Token renewal check skipped - handler not initialized",
                extra={
                    "config_initialized": self._config is not None,
                    "client_initialized": self._client is not None,
                    "correlation_id": str(correlation_id),
                },
            )
            return

        current_time = time.time()
        time_until_expiry = self._token_expires_at - current_time
        threshold = self._config.token_renewal_threshold_seconds
        needs_renewal = time_until_expiry < threshold

        # Log edge case when expiry time exactly equals threshold
        # This helps troubleshoot boundary condition behavior
        is_edge_case = abs(time_until_expiry - threshold) < 0.001  # Within 1ms

        logger.debug(
            "Token renewal check",
            extra={
                "current_time": current_time,
                "token_expires_at": self._token_expires_at,
                "time_until_expiry_seconds": time_until_expiry,
                "threshold_seconds": threshold,
                "needs_renewal": needs_renewal,
                "is_threshold_edge_case": is_edge_case,
                "correlation_id": str(correlation_id),
            },
        )

        if is_edge_case:
            logger.debug(
                "Token renewal edge case detected - expiry equals threshold",
                extra={
                    "time_until_expiry_seconds": time_until_expiry,
                    "threshold_seconds": threshold,
                    "difference_ms": abs(time_until_expiry - threshold) * 1000,
                    "will_renew": needs_renewal,
                    "correlation_id": str(correlation_id),
                },
            )

        if needs_renewal:
            logger.info(
                "Token approaching expiration, renewing",
                extra={
                    "time_until_expiry_seconds": time_until_expiry,
                    "threshold_seconds": threshold,
                    "correlation_id": str(correlation_id),
                },
            )
            await self.renew_token(correlation_id=correlation_id)
            logger.debug(
                "Token renewal completed successfully",
                extra={
                    "new_expires_at": self._token_expires_at,
                    "new_time_until_expiry_seconds": self._token_expires_at
                    - time.time(),
                    "correlation_id": str(correlation_id),
                },
            )
        else:
            logger.debug(
                "Token renewal skipped - token still valid",
                extra={
                    "time_until_expiry_seconds": time_until_expiry,
                    "threshold_seconds": threshold,
                    "margin_seconds": time_until_expiry - threshold,
                    "correlation_id": str(correlation_id),
                },
            )

    async def _execute_with_retry(
        self,
        operation: str,
        func: Callable[[], T],
        correlation_id: UUID,
    ) -> T:
        """Execute operation with exponential backoff retry logic and circuit breaker.

        Thread Pool Integration:
            All hvac operations (which are synchronous) are executed in a dedicated
            thread pool via loop.run_in_executor(). This prevents blocking the async
            event loop and allows concurrent Vault operations up to max_workers limit.

        Circuit breaker integration (via MixinAsyncCircuitBreaker):
            - Checks circuit state before execution (raises if OPEN)
            - Records success/failure for circuit state management
            - Allows test request in HALF_OPEN state

        Args:
            operation: Operation name for logging
            func: Callable to execute (synchronous hvac method)
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

        # Check circuit breaker before execution (async mixin pattern)
        if self._circuit_breaker_initialized:
            async with self._circuit_breaker_lock:
                await self._check_circuit_breaker(operation, correlation_id)

        retry_config = self._config.retry
        last_exception: Exception | None = None

        for attempt in range(retry_config.max_attempts):
            try:
                # hvac is synchronous, wrap in custom thread executor
                loop = asyncio.get_running_loop()

                result = await asyncio.wait_for(
                    loop.run_in_executor(self._executor, func),
                    timeout=self._config.timeout_seconds,
                )

                # Record success for circuit breaker (async mixin pattern)
                if self._circuit_breaker_initialized:
                    async with self._circuit_breaker_lock:
                        await self._reset_circuit_breaker()

                return result

            except TimeoutError as e:
                last_exception = e
                # Only record circuit failure on final retry attempt
                if attempt == retry_config.max_attempts - 1:
                    if self._circuit_breaker_initialized:
                        async with self._circuit_breaker_lock:
                            await self._record_circuit_failure(
                                operation, correlation_id
                            )
                    ctx = ModelInfraErrorContext(
                        transport_type=EnumInfraTransportType.VAULT,
                        operation=operation,
                        target_name="vault_handler",
                        correlation_id=correlation_id,
                        namespace=self._config.namespace if self._config else None,
                    )
                    raise InfraTimeoutError(
                        f"Vault operation timed out after {self._config.timeout_seconds}s",
                        context=ctx,
                        timeout_seconds=self._config.timeout_seconds,
                    ) from e

            except hvac.exceptions.Forbidden as e:
                # Don't retry authentication failures, record for circuit breaker
                if self._circuit_breaker_initialized:
                    async with self._circuit_breaker_lock:
                        await self._record_circuit_failure(operation, correlation_id)
                ctx = ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.VAULT,
                    operation=operation,
                    target_name="vault_handler",
                    correlation_id=correlation_id,
                    namespace=self._config.namespace if self._config else None,
                )
                raise InfraAuthenticationError(
                    "Vault operation forbidden - check token permissions",
                    context=ctx,
                ) from e

            except hvac.exceptions.InvalidPath as e:
                # Don't retry invalid path errors (not a circuit breaker failure)
                ctx = ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.VAULT,
                    operation=operation,
                    target_name="vault_handler",
                    correlation_id=correlation_id,
                    namespace=self._config.namespace if self._config else None,
                )
                raise SecretResolutionError(
                    "Secret path not found or invalid",
                    context=ctx,
                ) from e

            except hvac.exceptions.VaultDown as e:
                last_exception = e
                # Only record circuit failure on final retry attempt
                if attempt == retry_config.max_attempts - 1:
                    if self._circuit_breaker_initialized:
                        async with self._circuit_breaker_lock:
                            await self._record_circuit_failure(
                                operation, correlation_id
                            )
                    ctx = ModelInfraErrorContext(
                        transport_type=EnumInfraTransportType.VAULT,
                        operation=operation,
                        target_name="vault_handler",
                        correlation_id=correlation_id,
                        namespace=self._config.namespace if self._config else None,
                    )
                    raise InfraUnavailableError(
                        "Vault server is unavailable",
                        context=ctx,
                    ) from e

            except Exception as e:
                last_exception = e
                # Only record circuit failure on final retry attempt
                if attempt == retry_config.max_attempts - 1:
                    if self._circuit_breaker_initialized:
                        async with self._circuit_breaker_lock:
                            await self._record_circuit_failure(
                                operation, correlation_id
                            )
                    ctx = ModelInfraErrorContext(
                        transport_type=EnumInfraTransportType.VAULT,
                        operation=operation,
                        target_name="vault_handler",
                        correlation_id=correlation_id,
                        namespace=self._config.namespace if self._config else None,
                    )
                    raise InfraConnectionError(
                        f"Vault operation failed: {type(e).__name__}",
                        context=ctx,
                    ) from e

            # Calculate exponential backoff
            backoff = min(
                retry_config.initial_backoff_seconds
                * (retry_config.exponential_base**attempt),
                retry_config.max_backoff_seconds,
            )

            logger.debug(
                "Retrying Vault operation",
                extra={
                    "operation": operation,
                    "attempt": attempt + 1,
                    "max_attempts": retry_config.max_attempts,
                    "backoff_seconds": backoff,
                    "correlation_id": str(correlation_id),
                },
            )

            await asyncio.sleep(backoff)

        # Should never reach here, but satisfy type checker
        if last_exception is not None:
            raise last_exception
        raise RuntimeError("Retry loop completed without result")

    async def _read_secret(
        self,
        payload: dict[str, JsonValue],
        correlation_id: UUID,
        input_envelope_id: UUID,
    ) -> ModelHandlerOutput[dict[str, JsonValue]]:
        """Read secret from Vault KV v2 secrets engine.

        Args:
            payload: dict containing:
                - path: Secret path (required)
                - mount_point: KV mount point (default: "secret")
            correlation_id: Correlation ID for tracing
            input_envelope_id: Input envelope ID for causality tracking

        Returns:
            ModelHandlerOutput with secret data
        """
        path = payload.get("path")
        if not isinstance(path, str) or not path:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.VAULT,
                operation="vault.read_secret",
                target_name="vault_handler",
                correlation_id=correlation_id,
                namespace=self._config.namespace if self._config else None,
            )
            raise RuntimeHostError(
                "Missing or invalid 'path' in payload",
                context=ctx,
            )

        mount_point = payload.get("mount_point", DEFAULT_MOUNT_POINT)
        if not isinstance(mount_point, str):
            mount_point = DEFAULT_MOUNT_POINT

        if self._client is None:
            raise RuntimeError("Client not initialized")

        def read_func() -> dict[str, JsonValue]:
            if self._client is None:
                raise RuntimeError("Client not initialized")
            result: dict[str, JsonValue] = (
                self._client.secrets.kv.v2.read_secret_version(
                    path=path,
                    mount_point=mount_point,
                )
            )
            return result

        result = await self._execute_with_retry(
            "vault.read_secret",
            read_func,
            correlation_id,
        )

        # Extract nested data with type checking
        data_obj = result.get("data", {})
        data_dict = data_obj if isinstance(data_obj, dict) else {}
        secret_data = data_dict.get("data", {})
        metadata = data_dict.get("metadata", {})

        return ModelHandlerOutput.for_compute(
            input_envelope_id=input_envelope_id,
            correlation_id=correlation_id,
            handler_id=HANDLER_ID_VAULT,
            result={
                "status": "success",
                "payload": {
                    "data": secret_data if isinstance(secret_data, dict) else {},
                    "metadata": metadata if isinstance(metadata, dict) else {},
                },
                "correlation_id": str(correlation_id),
            },
        )

    async def _write_secret(
        self,
        payload: dict[str, JsonValue],
        correlation_id: UUID,
        input_envelope_id: UUID,
    ) -> ModelHandlerOutput[dict[str, JsonValue]]:
        """Write secret to Vault KV v2 secrets engine.

        Args:
            payload: dict containing:
                - path: Secret path (required)
                - data: Secret data dict (required)
                - mount_point: KV mount point (default: "secret")
            correlation_id: Correlation ID for tracing
            input_envelope_id: Input envelope ID for causality tracking

        Returns:
            ModelHandlerOutput with write confirmation
        """
        path = payload.get("path")
        if not isinstance(path, str) or not path:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.VAULT,
                operation="vault.write_secret",
                target_name="vault_handler",
                correlation_id=correlation_id,
                namespace=self._config.namespace if self._config else None,
            )
            raise RuntimeHostError(
                "Missing or invalid 'path' in payload",
                context=ctx,
            )

        data = payload.get("data")
        if not isinstance(data, dict):
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.VAULT,
                operation="vault.write_secret",
                target_name="vault_handler",
                correlation_id=correlation_id,
                namespace=self._config.namespace if self._config else None,
            )
            raise RuntimeHostError(
                "Missing or invalid 'data' in payload - must be a dict",
                context=ctx,
            )

        mount_point = payload.get("mount_point", DEFAULT_MOUNT_POINT)
        if not isinstance(mount_point, str):
            mount_point = DEFAULT_MOUNT_POINT

        if self._client is None:
            raise RuntimeError("Client not initialized")

        def write_func() -> dict[str, JsonValue]:
            if self._client is None:
                raise RuntimeError("Client not initialized")
            result: dict[str, JsonValue] = (
                self._client.secrets.kv.v2.create_or_update_secret(
                    path=path,
                    secret=data,
                    mount_point=mount_point,
                )
            )
            return result

        result = await self._execute_with_retry(
            "vault.write_secret",
            write_func,
            correlation_id,
        )

        # Extract nested data with type checking
        data_obj = result.get("data", {})
        data_dict = data_obj if isinstance(data_obj, dict) else {}

        return ModelHandlerOutput.for_compute(
            input_envelope_id=input_envelope_id,
            correlation_id=correlation_id,
            handler_id=HANDLER_ID_VAULT,
            result={
                "status": "success",
                "payload": {
                    "version": data_dict.get("version"),
                    "created_time": data_dict.get("created_time"),
                },
                "correlation_id": str(correlation_id),
            },
        )

    async def _delete_secret(
        self,
        payload: dict[str, JsonValue],
        correlation_id: UUID,
        input_envelope_id: UUID,
    ) -> ModelHandlerOutput[dict[str, JsonValue]]:
        """Delete secret from Vault KV v2 secrets engine.

        Args:
            payload: dict containing:
                - path: Secret path (required)
                - mount_point: KV mount point (default: "secret")
            correlation_id: Correlation ID for tracing
            input_envelope_id: Input envelope ID for causality tracking

        Returns:
            ModelHandlerOutput with deletion confirmation
        """
        path = payload.get("path")
        if not isinstance(path, str) or not path:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.VAULT,
                operation="vault.delete_secret",
                target_name="vault_handler",
                correlation_id=correlation_id,
                namespace=self._config.namespace if self._config else None,
            )
            raise RuntimeHostError(
                "Missing or invalid 'path' in payload",
                context=ctx,
            )

        mount_point = payload.get("mount_point", DEFAULT_MOUNT_POINT)
        if not isinstance(mount_point, str):
            mount_point = DEFAULT_MOUNT_POINT

        if self._client is None:
            raise RuntimeError("Client not initialized")

        def delete_func() -> None:
            if self._client is None:
                raise RuntimeError("Client not initialized")
            # Delete latest version
            self._client.secrets.kv.v2.delete_latest_version_of_secret(
                path=path,
                mount_point=mount_point,
            )

        await self._execute_with_retry(
            "vault.delete_secret",
            delete_func,
            correlation_id,
        )

        return ModelHandlerOutput.for_compute(
            input_envelope_id=input_envelope_id,
            correlation_id=correlation_id,
            handler_id=HANDLER_ID_VAULT,
            result={
                "status": "success",
                "payload": {"deleted": True},
                "correlation_id": str(correlation_id),
            },
        )

    async def _list_secrets(
        self,
        payload: dict[str, JsonValue],
        correlation_id: UUID,
        input_envelope_id: UUID,
    ) -> ModelHandlerOutput[dict[str, JsonValue]]:
        """List secrets at path in Vault KV v2 secrets engine.

        Args:
            payload: dict containing:
                - path: Secret path (required)
                - mount_point: KV mount point (default: "secret")
            correlation_id: Correlation ID for tracing
            input_envelope_id: Input envelope ID for causality tracking

        Returns:
            ModelHandlerOutput with list of secret keys
        """
        path = payload.get("path")
        if not isinstance(path, str) or not path:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.VAULT,
                operation="vault.list_secrets",
                target_name="vault_handler",
                correlation_id=correlation_id,
                namespace=self._config.namespace if self._config else None,
            )
            raise RuntimeHostError(
                "Missing or invalid 'path' in payload",
                context=ctx,
            )

        mount_point = payload.get("mount_point", DEFAULT_MOUNT_POINT)
        if not isinstance(mount_point, str):
            mount_point = DEFAULT_MOUNT_POINT

        if self._client is None:
            raise RuntimeError("Client not initialized")

        def list_func() -> dict[str, JsonValue]:
            if self._client is None:
                raise RuntimeError("Client not initialized")
            result: dict[str, JsonValue] = self._client.secrets.kv.v2.list_secrets(
                path=path,
                mount_point=mount_point,
            )
            return result

        result = await self._execute_with_retry(
            "vault.list_secrets",
            list_func,
            correlation_id,
        )

        # Extract nested data with type checking
        data_obj = result.get("data", {})
        data_dict = data_obj if isinstance(data_obj, dict) else {}
        keys = data_dict.get("keys", [])

        return ModelHandlerOutput.for_compute(
            input_envelope_id=input_envelope_id,
            correlation_id=correlation_id,
            handler_id=HANDLER_ID_VAULT,
            result={
                "status": "success",
                "payload": {"keys": keys if isinstance(keys, list) else []},
                "correlation_id": str(correlation_id),
            },
        )

    async def renew_token(
        self, correlation_id: UUID | None = None
    ) -> dict[str, JsonValue]:
        """Renew Vault authentication token.

        Token TTL Extraction Logic:
            1. Extract 'auth.lease_duration' from Vault renewal response
            2. If lease_duration is invalid or missing, use default_token_ttl
            3. Update _token_expires_at = current_time + extracted_ttl
            4. Log warning when falling back to default TTL

        Args:
            correlation_id: Optional correlation ID for tracing. When called via
                envelope dispatch, this preserves the request's correlation_id.
                When called directly (e.g., by monitoring), a new ID is generated.

        Returns:
            Token renewal information including new TTL

        Raises:
            InfraAuthenticationError: If token renewal fails
        """
        if correlation_id is None:
            correlation_id = uuid4()

        if self._client is None or self._config is None:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.VAULT,
                operation="vault.renew_token",
                target_name="vault_handler",
                correlation_id=correlation_id,
                namespace=self._config.namespace if self._config else None,
            )
            raise RuntimeHostError(
                "VaultHandler not initialized",
                context=ctx,
            )

        def renew_func() -> dict[str, JsonValue]:
            if self._client is None:
                raise RuntimeError("Client not initialized")
            result: dict[str, JsonValue] = self._client.auth.token.renew_self()
            return result

        try:
            result = await self._execute_with_retry(
                "vault.renew_token",
                renew_func,
                correlation_id,
            )

            # Update token expiration tracking
            auth_data = result.get("auth", {})
            token_ttl = None

            if isinstance(auth_data, dict):
                lease_duration = auth_data.get("lease_duration")
                if isinstance(lease_duration, int) and lease_duration > 0:
                    token_ttl = lease_duration

            if token_ttl is None:
                # Fallback to config or safe default
                token_ttl = self._config.default_token_ttl
                logger.warning(
                    "Token TTL not in renewal response, using fallback",
                    extra={
                        "ttl": token_ttl,
                        "correlation_id": str(correlation_id),
                    },
                )

            # After successful renewal, query actual TTL from Vault to handle
            # server-side TTL updates or policies that modify actual TTL
            loop = asyncio.get_running_loop()
            try:
                token_info = await loop.run_in_executor(
                    self._executor,
                    self._client.auth.token.lookup_self,
                )
                token_data = token_info.get("data", {})
                if isinstance(token_data, dict):
                    ttl_seconds = token_data.get("ttl")
                    if isinstance(ttl_seconds, int) and ttl_seconds > 0:
                        token_ttl = ttl_seconds
                        logger.info(
                            "Token TTL refreshed from Vault lookup",
                            extra={
                                "ttl_seconds": token_ttl,
                                "correlation_id": str(correlation_id),
                            },
                        )
            except Exception as e:
                # Fallback to lease_duration from renewal response (already set above)
                logger.debug(
                    "Token lookup after renewal failed, using lease_duration",
                    extra={
                        "error_type": type(e).__name__,
                        "correlation_id": str(correlation_id),
                    },
                )

            self._token_expires_at = time.time() + token_ttl

            logger.info(
                "Token renewed successfully",
                extra={
                    "new_ttl_seconds": token_ttl,
                    "correlation_id": str(correlation_id),
                },
            )

            return result

        except Exception as e:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.VAULT,
                operation="vault.renew_token",
                target_name="vault_handler",
                correlation_id=correlation_id,
                namespace=self._config.namespace if self._config else None,
            )
            raise InfraAuthenticationError(
                "Failed to renew Vault token",
                context=ctx,
            ) from e

    async def _renew_token_operation(
        self,
        correlation_id: UUID,
        input_envelope_id: UUID,
    ) -> ModelHandlerOutput[dict[str, JsonValue]]:
        """Execute token renewal operation from envelope.

        Args:
            correlation_id: Correlation ID for tracing
            input_envelope_id: Input envelope ID for causality tracking

        Returns:
            ModelHandlerOutput with renewal information
        """
        result = await self.renew_token(correlation_id=correlation_id)

        # Extract nested auth data with type checking
        auth_obj = result.get("auth", {})
        auth_data = auth_obj if isinstance(auth_obj, dict) else {}

        return ModelHandlerOutput.for_compute(
            input_envelope_id=input_envelope_id,
            correlation_id=correlation_id,
            handler_id=HANDLER_ID_VAULT,
            result={
                "status": "success",
                "payload": {
                    "renewable": auth_data.get("renewable", False),
                    "lease_duration": auth_data.get("lease_duration", 0),
                },
                "correlation_id": str(correlation_id),
            },
        )

    async def health_check(
        self, correlation_id: UUID | None = None
    ) -> dict[str, JsonValue]:
        """Return handler health status with operational metrics.

        Uses thread pool executor and retry logic for consistency with other operations.
        Includes circuit breaker protection and exponential backoff on transient failures.

        This is the standalone health check method intended for direct invocation by
        monitoring systems, health check endpoints, or diagnostic tools. When called
        directly, a new correlation_id is generated for tracing purposes.

        Envelope-Based vs Direct Invocation:
            - Direct: Call health_check() for monitoring/diagnostics. If no correlation_id
              is provided, a new one is generated internally for tracing.
            - Envelope: Use execute() with operation="vault.health_check" for dispatch.
              The envelope's correlation_id and envelope_id are preserved for causality.

        Note:
            This method does not accept envelope_id because it's designed for direct
            invocation outside the envelope dispatch context. For envelope-based health
            checks that preserve causality tracking, use _health_check_operation() via
            the execute() method.

        Args:
            correlation_id: Optional correlation ID for tracing. When called via
                envelope dispatch, this preserves the request's correlation_id.
                When called directly (e.g., by monitoring), a new ID is generated.

        Returns:
            Health status dict with handler state information including:
            - Basic health status (healthy, initialized, handler_type, timeout_seconds)
            - Token TTL remaining (sanitized - no actual token value)
            - Circuit breaker state and failure count
            - Thread pool utilization metrics

        Raises:
            RuntimeHostError: If health check fails (errors are propagated, not swallowed)
        """
        healthy = False
        if correlation_id is None:
            correlation_id = uuid4()

        # Calculate operational metrics (safe even if not initialized)
        token_ttl_remaining: int | None = None
        circuit_state: str | None = None
        circuit_failure_count: int = 0
        thread_pool_active: int = 0
        thread_pool_max: int = 0

        if self._initialized and self._config is not None:
            # Token TTL remaining (sanitized - never expose actual token)
            current_time = time.time()
            ttl_remaining = self._token_expires_at - current_time
            token_ttl_remaining = max(0, int(ttl_remaining))

            # Circuit breaker state (thread-safe access via mixin)
            if self._circuit_breaker_initialized:
                async with self._circuit_breaker_lock:
                    circuit_state = "open" if self._circuit_breaker_open else "closed"
                    circuit_failure_count = self._circuit_breaker_failures

            # Thread pool metrics
            thread_pool_max = self._max_workers
            if self._executor is not None:
                # Access _threads with getattr for safety - ThreadPoolExecutor
                # tracks active threads in _threads set but it's internal.
                # This is acceptable for observability metrics; if the attribute
                # is removed in future Python versions, we gracefully return 0.
                threads_set = getattr(self._executor, "_threads", None)
                if threads_set is not None:
                    thread_pool_active = len(threads_set)
                # Note: There's no public API for active thread count in
                # ThreadPoolExecutor. The _threads attribute exists in all
                # Python 3.x versions and is unlikely to change.

        if self._initialized and self._client is not None:

            def health_check_func() -> dict[str, JsonValue]:
                if self._client is None:
                    raise RuntimeError("Client not initialized")
                response = self._client.sys.read_health_status()
                # hvac returns Response object for some status codes, dict for others
                if hasattr(response, "json"):
                    # Response may have empty body (200 OK with no content)
                    if hasattr(response, "text") and response.text:
                        result: dict[str, JsonValue] = response.json()
                    else:
                        # Empty response means healthy (200 OK)
                        result = {"initialized": True, "sealed": False}
                else:
                    result = response
                return result

            # Use thread pool executor with retry logic for consistency
            # Errors are propagated (not caught) per PR #38 feedback
            health_result = await self._execute_with_retry(
                "vault.health_check",
                health_check_func,
                correlation_id,
            )
            # Type checking for healthy status extraction
            initialized_val = health_result.get("initialized", False)
            healthy = initialized_val if isinstance(initialized_val, bool) else False

        return {
            "healthy": healthy,
            "initialized": self._initialized,
            "handler_type": self.handler_type.value,
            "timeout_seconds": self._config.timeout_seconds if self._config else 30.0,
            # Operational metrics for visibility
            "token_ttl_remaining_seconds": token_ttl_remaining,
            "circuit_breaker_state": circuit_state,
            "circuit_breaker_failure_count": circuit_failure_count,
            "thread_pool_active_workers": thread_pool_active,
            "thread_pool_max_workers": thread_pool_max,
        }

    async def _health_check_operation(
        self,
        correlation_id: UUID,
        input_envelope_id: UUID,
    ) -> ModelHandlerOutput[dict[str, JsonValue]]:
        """Execute health check operation from envelope.

        This method wraps the core health_check() functionality in a ModelHandlerOutput
        for envelope-based operation dispatch. It differs from health_check() in that:

        1. It accepts pre-extracted IDs from the request envelope
        2. It returns ModelHandlerOutput (suitable for envelope dispatch)
        3. It preserves causality tracking via input_envelope_id

        ID Semantics:
            correlation_id: Groups related operations across distributed services.
                Used for filtering logs, tracing request flows, and debugging.
                Propagated from the request envelope or auto-generated if missing.

            input_envelope_id: Links this response to the originating request envelope.
                Enables request/response correlation in observability systems.
                When called via execute(), extracted from the request envelope.
                Auto-generated if not provided, ensuring all responses have valid
                causality tracking IDs.

        The standalone health_check() method generates its own correlation_id since
        it may be called directly (not via envelope dispatch) for monitoring purposes.

        Args:
            correlation_id: Correlation ID for distributed tracing across services.
            input_envelope_id: Envelope ID for causality tracking. Links this health
                check response to the original request envelope, enabling end-to-end
                request/response correlation in observability systems.

        Returns:
            ModelHandlerOutput with health check information
        """
        health_status = await self.health_check(correlation_id=correlation_id)

        return ModelHandlerOutput.for_compute(
            input_envelope_id=input_envelope_id,
            correlation_id=correlation_id,
            handler_id=HANDLER_ID_VAULT,
            result={
                "status": "success",
                "payload": health_status,
                "correlation_id": str(correlation_id),
            },
        )

    def describe(self) -> dict[str, JsonValue]:
        """Return handler metadata and capabilities.

        Returns:
            Handler description with supported operations and configuration
        """
        return {
            "handler_type": self.handler_type.value,
            "supported_operations": sorted(SUPPORTED_OPERATIONS),
            "timeout_seconds": self._config.timeout_seconds if self._config else 30.0,
            "initialized": self._initialized,
            "version": "0.1.0-mvp",
        }


__all__: list[str] = ["VaultHandler"]
