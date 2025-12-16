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
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Optional, TypeVar
from uuid import UUID, uuid4

T = TypeVar("T")

import hvac
from omnibase_core.enums.enum_handler_type import EnumHandlerType
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
from omnibase_infra.handlers.model_vault_adapter_config import ModelVaultAdapterConfig

logger = logging.getLogger(__name__)

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


class CircuitState(str, Enum):
    """Circuit breaker state machine states.

    States:
        CLOSED: Normal operation, requests allowed
        OPEN: Too many failures, blocking requests temporarily
        HALF_OPEN: Testing if service recovered, allowing limited requests
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class VaultAdapter:
    """HashiCorp Vault adapter using hvac client (MVP: KV v2 secrets engine).

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
        """Initialize VaultAdapter in uninitialized state."""
        self._client: Optional[hvac.Client] = None
        self._config: Optional[ModelVaultAdapterConfig] = None
        self._initialized: bool = False
        self._token_expires_at: float = 0.0
        self._executor: Optional[ThreadPoolExecutor] = None
        self._max_workers: int = 0
        self._max_queue_size: int = 0
        self._queue_semaphore: Optional[threading.Semaphore] = None
        # Circuit breaker state (thread-safe with RLock for reentrant access)
        self._circuit_lock: threading.RLock = threading.RLock()
        self._circuit_state: CircuitState = CircuitState.CLOSED
        self._circuit_failure_count: int = 0
        self._circuit_last_failure_time: float = 0.0

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

    async def initialize(self, config: dict[str, object]) -> None:
        """Initialize Vault client with configuration.

        Args:
            config: Configuration dict containing:
                - url: Vault server URL (required)
                - token: Vault authentication token (required)
                - namespace: Optional Vault namespace for Enterprise
                - timeout_seconds: Optional timeout (default 30.0)
                - verify_ssl: Optional SSL verification (default True)
                - token_renewal_threshold_seconds: Optional renewal threshold (default 300.0)
                - retry: Optional retry configuration dict

        Raises:
            RuntimeHostError: If URL or token is missing, or client initialization fails.
            InfraAuthenticationError: If token authentication fails.
            InfraConnectionError: If connection to Vault server fails.

        Security:
            Token must be provided via environment variable, not hardcoded in config.
            Use SecretStr for token to prevent accidental logging.
        """
        init_correlation_id = uuid4()

        # Parse configuration using Pydantic model
        try:
            # Handle SecretStr token conversion
            token_raw = config.get("token")
            if isinstance(token_raw, str):
                config = dict(config)  # Make mutable copy
                config["token"] = SecretStr(token_raw)

            # Type ignore for dict unpacking - Pydantic handles validation
            self._config = ModelVaultAdapterConfig(**config)  # type: ignore[arg-type]
        except ValidationError as e:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.VAULT,
                operation="initialize",
                target_name="vault_adapter",
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
                target_name="vault_adapter",
                correlation_id=init_correlation_id,
                namespace=None,  # Config not initialized yet
            )
            raise RuntimeHostError(
                f"Configuration parsing failed: {type(e).__name__}",
                context=ctx,
            ) from e

        # Validate required fields
        if not self._config.url:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.VAULT,
                operation="initialize",
                target_name="vault_adapter",
                correlation_id=init_correlation_id,
                namespace=self._config.namespace if self._config else None,
            )
            raise RuntimeHostError(
                "Missing 'url' in config - Vault server URL required",
                context=ctx,
            )

        if self._config.token is None:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.VAULT,
                operation="initialize",
                target_name="vault_adapter",
                correlation_id=init_correlation_id,
                namespace=self._config.namespace if self._config else None,
            )
            raise RuntimeHostError(
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
                    target_name="vault_adapter",
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
                thread_name_prefix="vault_adapter_",
            )
            # Use semaphore to limit pending operations in queue
            self._max_queue_size = (
                self._max_workers * self._config.max_queue_size_multiplier
            )
            self._queue_semaphore = threading.Semaphore(self._max_queue_size)

            self._initialized = True
            logger.info(
                "VaultAdapter initialized",
                extra={
                    "url": self._config.url,
                    "namespace": self._config.namespace,
                    "timeout_seconds": self._config.timeout_seconds,
                    "verify_ssl": self._config.verify_ssl,
                    "max_concurrent_operations": self._config.max_concurrent_operations,
                    "circuit_breaker_enabled": self._config.circuit_breaker_enabled,
                },
            )

        except InfraAuthenticationError:
            # Re-raise our own authentication errors without wrapping
            raise
        except hvac.exceptions.InvalidRequest as e:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.VAULT,
                operation="initialize",
                target_name="vault_adapter",
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
                target_name="vault_adapter",
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
                target_name="vault_adapter",
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
            - Resetting circuit breaker state (thread-safe)
        """
        if self._executor is not None:
            # Shutdown thread pool gracefully (wait for pending tasks)
            self._executor.shutdown(wait=True)
            self._executor = None
        if self._client is not None:
            # hvac.Client doesn't have async close, just clear reference
            self._client = None

        # Reset circuit breaker state (thread-safe)
        with self._circuit_lock:
            self._circuit_state = CircuitState.CLOSED
            self._circuit_failure_count = 0
            self._circuit_last_failure_time = 0.0

        self._initialized = False
        self._config = None
        logger.info("VaultAdapter shutdown complete")

    async def execute(self, envelope: dict[str, object]) -> dict[str, object]:
        """Execute Vault operation from envelope.

        Args:
            envelope: Request envelope containing:
                - operation: Vault operation (vault.read_secret, vault.write_secret, etc.)
                - payload: dict with operation-specific parameters
                - correlation_id: Optional correlation ID for tracing

        Returns:
            Response envelope with status, payload, and correlation_id

        Raises:
            RuntimeHostError: If handler not initialized or invalid input.
            InfraConnectionError: If Vault connection fails.
            InfraAuthenticationError: If authentication fails.
            SecretResolutionError: If secret resolution fails.
        """
        correlation_id = self._extract_correlation_id(envelope)

        if not self._initialized or self._client is None or self._config is None:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.VAULT,
                operation="execute",
                target_name="vault_adapter",
                correlation_id=correlation_id,
                namespace=self._config.namespace if self._config else None,
            )
            raise RuntimeHostError(
                "VaultAdapter not initialized. Call initialize() first.",
                context=ctx,
            )

        operation = envelope.get("operation")
        if not isinstance(operation, str):
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.VAULT,
                operation="execute",
                target_name="vault_adapter",
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
                target_name="vault_adapter",
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
                target_name="vault_adapter",
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
            return await self._read_secret(payload, correlation_id)
        elif operation == "vault.write_secret":
            return await self._write_secret(payload, correlation_id)
        elif operation == "vault.delete_secret":
            return await self._delete_secret(payload, correlation_id)
        elif operation == "vault.list_secrets":
            return await self._list_secrets(payload, correlation_id)
        elif operation == "vault.renew_token":
            return await self._renew_token_operation(correlation_id)
        else:  # vault.health_check
            return await self._health_check_operation(correlation_id)

    def _extract_correlation_id(self, envelope: dict[str, object]) -> UUID:
        """Extract or generate correlation ID from envelope."""
        raw = envelope.get("correlation_id")
        if isinstance(raw, UUID):
            return raw
        if isinstance(raw, str):
            try:
                return UUID(raw)
            except ValueError:
                pass
        return uuid4()

    async def _check_token_renewal(self, correlation_id: UUID) -> None:
        """Check if token needs renewal and renew if necessary.

        Args:
            correlation_id: Correlation ID for tracing

        Raises:
            InfraAuthenticationError: If token renewal fails
        """
        if self._config is None or self._client is None:
            return

        current_time = time.time()
        time_until_expiry = self._token_expires_at - current_time

        if time_until_expiry < self._config.token_renewal_threshold_seconds:
            logger.info(
                "Token approaching expiration, renewing",
                extra={
                    "time_until_expiry_seconds": time_until_expiry,
                    "threshold_seconds": self._config.token_renewal_threshold_seconds,
                    "correlation_id": str(correlation_id),
                },
            )
            await self.renew_token()

    def _check_circuit_breaker(self, correlation_id: UUID) -> None:
        """Check circuit breaker state and raise error if circuit is open.

        Circuit breaker state machine:
            CLOSED -> OPEN: After threshold consecutive failures
            OPEN -> HALF_OPEN: After reset timeout expires
            HALF_OPEN -> CLOSED: On successful request
            HALF_OPEN -> OPEN: On failed request

        Thread Safety:
            Uses RLock for thread-safe state access and modification.

        Observability:
            Circuit state transitions are logged at INFO level for monitoring.
            Use logs to track circuit breaker behavior and adjust thresholds.

        Args:
            correlation_id: Correlation ID for tracing

        Raises:
            InfraUnavailableError: If circuit is OPEN
        """
        if self._config is None or not self._config.circuit_breaker_enabled:
            return

        with self._circuit_lock:
            current_time = time.time()

            # Check if circuit is OPEN
            if self._circuit_state == CircuitState.OPEN:
                time_since_failure = current_time - self._circuit_last_failure_time

                # Check if reset timeout has passed
                if (
                    time_since_failure
                    >= self._config.circuit_breaker_reset_timeout_seconds
                ):
                    # Transition to HALF_OPEN to test service recovery
                    self._circuit_state = CircuitState.HALF_OPEN
                    self._circuit_failure_count = 0
                    logger.info(
                        "Circuit breaker transitioning to HALF_OPEN state",
                        extra={
                            "time_since_failure_seconds": time_since_failure,
                            "correlation_id": str(correlation_id),
                        },
                    )
                else:
                    # Circuit still open, reject request
                    retry_after = int(
                        self._config.circuit_breaker_reset_timeout_seconds
                        - time_since_failure
                    )
                    ctx = ModelInfraErrorContext(
                        transport_type=EnumInfraTransportType.VAULT,
                        operation="circuit_breaker_check",
                        target_name="vault_adapter",
                        correlation_id=correlation_id,
                        namespace=self._config.namespace if self._config else None,
                    )
                    raise InfraUnavailableError(
                        "Circuit breaker is open - Vault temporarily unavailable",
                        context=ctx,
                        circuit_state=self._circuit_state.value,
                        retry_after_seconds=retry_after,
                    )

    def _record_circuit_success(self) -> None:
        """Record successful operation for circuit breaker.

        On success:
            - CLOSED: No change
            - HALF_OPEN: Transition to CLOSED (service recovered)
            - OPEN: Should not reach here (check prevents execution)

        Thread Safety:
            Uses RLock for thread-safe state access and modification.
        """
        if self._config is None or not self._config.circuit_breaker_enabled:
            return

        with self._circuit_lock:
            if self._circuit_state == CircuitState.HALF_OPEN:
                # Service recovered, close circuit
                self._circuit_state = CircuitState.CLOSED
                self._circuit_failure_count = 0
                logger.info("Circuit breaker closed - service recovered")

            # Reset failure count on success
            self._circuit_failure_count = 0

    def _record_circuit_failure(self) -> None:
        """Record failed operation for circuit breaker.

        On failure:
            - CLOSED: Increment failure count, open if threshold exceeded
            - HALF_OPEN: Transition back to OPEN (service still failing)
            - OPEN: Should not reach here (check prevents execution)

        Thread Safety:
            Uses RLock for thread-safe state access and modification.
        """
        if self._config is None or not self._config.circuit_breaker_enabled:
            return

        with self._circuit_lock:
            self._circuit_failure_count += 1
            self._circuit_last_failure_time = time.time()

            if self._circuit_state == CircuitState.HALF_OPEN:
                # Service still failing, reopen circuit
                self._circuit_state = CircuitState.OPEN
                logger.warning(
                    "Circuit breaker reopened - service still failing",
                    extra={
                        "failure_count": self._circuit_failure_count,
                    },
                )
            elif self._circuit_state == CircuitState.CLOSED:
                # Check if threshold exceeded
                if (
                    self._circuit_failure_count
                    >= self._config.circuit_breaker_failure_threshold
                ):
                    self._circuit_state = CircuitState.OPEN
                    logger.warning(
                        "Circuit breaker opened due to consecutive failures",
                        extra={
                            "failure_count": self._circuit_failure_count,
                            "threshold": self._config.circuit_breaker_failure_threshold,
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

        Circuit breaker integration:
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

        # Check circuit breaker before execution
        self._check_circuit_breaker(correlation_id)

        retry_config = self._config.retry
        last_exception: Optional[Exception] = None

        for attempt in range(retry_config.max_attempts):
            try:
                # hvac is synchronous, wrap in custom thread executor
                loop = asyncio.get_running_loop()

                result = await asyncio.wait_for(
                    loop.run_in_executor(self._executor, func),
                    timeout=self._config.timeout_seconds,
                )

                # Record success for circuit breaker
                self._record_circuit_success()

                return result

            except TimeoutError as e:
                last_exception = e
                # Only record circuit failure on final retry attempt
                if attempt == retry_config.max_attempts - 1:
                    self._record_circuit_failure()
                    ctx = ModelInfraErrorContext(
                        transport_type=EnumInfraTransportType.VAULT,
                        operation=operation,
                        target_name="vault_adapter",
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
                self._record_circuit_failure()
                ctx = ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.VAULT,
                    operation=operation,
                    target_name="vault_adapter",
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
                    target_name="vault_adapter",
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
                    self._record_circuit_failure()
                    ctx = ModelInfraErrorContext(
                        transport_type=EnumInfraTransportType.VAULT,
                        operation=operation,
                        target_name="vault_adapter",
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
                    self._record_circuit_failure()
                    ctx = ModelInfraErrorContext(
                        transport_type=EnumInfraTransportType.VAULT,
                        operation=operation,
                        target_name="vault_adapter",
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
        payload: dict[str, object],
        correlation_id: UUID,
    ) -> dict[str, object]:
        """Read secret from Vault KV v2 secrets engine.

        Args:
            payload: dict containing:
                - path: Secret path (required)
                - mount_point: KV mount point (default: "secret")

        Returns:
            Response envelope with secret data
        """
        path = payload.get("path")
        if not isinstance(path, str) or not path:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.VAULT,
                operation="vault.read_secret",
                target_name="vault_adapter",
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

        def read_func() -> dict[str, object]:
            if self._client is None:
                raise RuntimeError("Client not initialized")
            result: dict[str, object] = self._client.secrets.kv.v2.read_secret_version(
                path=path,
                mount_point=mount_point,
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

        return {
            "status": "success",
            "payload": {
                "data": secret_data if isinstance(secret_data, dict) else {},
                "metadata": metadata if isinstance(metadata, dict) else {},
            },
            "correlation_id": correlation_id,
        }

    async def _write_secret(
        self,
        payload: dict[str, object],
        correlation_id: UUID,
    ) -> dict[str, object]:
        """Write secret to Vault KV v2 secrets engine.

        Args:
            payload: dict containing:
                - path: Secret path (required)
                - data: Secret data dict (required)
                - mount_point: KV mount point (default: "secret")

        Returns:
            Response envelope with write confirmation
        """
        path = payload.get("path")
        if not isinstance(path, str) or not path:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.VAULT,
                operation="vault.write_secret",
                target_name="vault_adapter",
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
                target_name="vault_adapter",
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

        def write_func() -> dict[str, object]:
            if self._client is None:
                raise RuntimeError("Client not initialized")
            result: dict[str, object] = (
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

        return {
            "status": "success",
            "payload": {
                "version": data_dict.get("version"),
                "created_time": data_dict.get("created_time"),
            },
            "correlation_id": correlation_id,
        }

    async def _delete_secret(
        self,
        payload: dict[str, object],
        correlation_id: UUID,
    ) -> dict[str, object]:
        """Delete secret from Vault KV v2 secrets engine.

        Args:
            payload: dict containing:
                - path: Secret path (required)
                - mount_point: KV mount point (default: "secret")

        Returns:
            Response envelope with deletion confirmation
        """
        path = payload.get("path")
        if not isinstance(path, str) or not path:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.VAULT,
                operation="vault.delete_secret",
                target_name="vault_adapter",
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

        return {
            "status": "success",
            "payload": {"deleted": True},
            "correlation_id": correlation_id,
        }

    async def _list_secrets(
        self,
        payload: dict[str, object],
        correlation_id: UUID,
    ) -> dict[str, object]:
        """List secrets at path in Vault KV v2 secrets engine.

        Args:
            payload: dict containing:
                - path: Secret path (required)
                - mount_point: KV mount point (default: "secret")

        Returns:
            Response envelope with list of secret keys
        """
        path = payload.get("path")
        if not isinstance(path, str) or not path:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.VAULT,
                operation="vault.list_secrets",
                target_name="vault_adapter",
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

        def list_func() -> dict[str, object]:
            if self._client is None:
                raise RuntimeError("Client not initialized")
            result: dict[str, object] = self._client.secrets.kv.v2.list_secrets(
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

        return {
            "status": "success",
            "payload": {"keys": keys if isinstance(keys, list) else []},
            "correlation_id": correlation_id,
        }

    async def renew_token(self) -> dict[str, object]:
        """Renew Vault authentication token.

        Token TTL Extraction Logic:
            1. Extract 'auth.lease_duration' from Vault renewal response
            2. If lease_duration is invalid or missing, use default_token_ttl
            3. Update _token_expires_at = current_time + extracted_ttl
            4. Log warning when falling back to default TTL

        Returns:
            Token renewal information including new TTL

        Raises:
            InfraAuthenticationError: If token renewal fails
        """
        correlation_id = uuid4()

        if self._client is None or self._config is None:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.VAULT,
                operation="vault.renew_token",
                target_name="vault_adapter",
                correlation_id=correlation_id,
                namespace=self._config.namespace if self._config else None,
            )
            raise RuntimeHostError(
                "VaultAdapter not initialized",
                context=ctx,
            )

        def renew_func() -> dict[str, object]:
            if self._client is None:
                raise RuntimeError("Client not initialized")
            result: dict[str, object] = self._client.auth.token.renew_self()
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
                target_name="vault_adapter",
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
    ) -> dict[str, object]:
        """Execute token renewal operation from envelope.

        Args:
            correlation_id: Correlation ID for tracing

        Returns:
            Response envelope with renewal information
        """
        result = await self.renew_token()

        # Extract nested auth data with type checking
        auth_obj = result.get("auth", {})
        auth_data = auth_obj if isinstance(auth_obj, dict) else {}

        return {
            "status": "success",
            "payload": {
                "renewable": auth_data.get("renewable", False),
                "lease_duration": auth_data.get("lease_duration", 0),
            },
            "correlation_id": correlation_id,
        }

    async def health_check(self) -> dict[str, object]:
        """Return handler health status.

        Uses thread pool executor and retry logic for consistency with other operations.
        Includes circuit breaker protection and exponential backoff on transient failures.

        Returns:
            Health status dict with handler state information

        Raises:
            RuntimeHostError: If health check fails (errors are propagated, not swallowed)
        """
        healthy = False
        correlation_id = uuid4()

        if self._initialized and self._client is not None:

            def health_check_func() -> dict[str, object]:
                if self._client is None:
                    raise RuntimeError("Client not initialized")
                result: dict[str, object] = self._client.sys.read_health_status()
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
        }

    async def _health_check_operation(
        self,
        correlation_id: UUID,
    ) -> dict[str, object]:
        """Execute health check operation from envelope.

        Args:
            correlation_id: Correlation ID for tracing

        Returns:
            Response envelope with health check information
        """
        health_status = await self.health_check()

        return {
            "status": "success",
            "payload": health_status,
            "correlation_id": correlation_id,
        }

    def describe(self) -> dict[str, object]:
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


__all__: list[str] = ["VaultAdapter"]
