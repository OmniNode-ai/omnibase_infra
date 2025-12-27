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
from omnibase_infra.handlers.models import ModelOperationContext, ModelRetryState
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
    }
)


class VaultHandler(MixinAsyncCircuitBreaker, MixinEnvelopeExtraction):
    """HashiCorp Vault handler using hvac client (MVP: KV v2 secrets engine).

    Security Policy - Token Handling:
        The Vault token contains sensitive credentials and is treated as a secret
        throughout this handler. The following security measures are enforced:

        1. Token is stored as SecretStr in config (never logged or exposed)
        2. All error messages use generic descriptions without exposing token
        3. Token renewal is automatic when TTL falls below threshold
        4. The describe() method returns capabilities without credentials

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

    # -------------------------------------------------------------------------
    # Helper methods for initialize (complexity reduction)
    # -------------------------------------------------------------------------

    def _create_init_error_context(
        self, correlation_id: UUID, namespace: str | None = None
    ) -> ModelInfraErrorContext:
        """Create error context for initialization operations.

        Args:
            correlation_id: Correlation ID for tracing
            namespace: Optional namespace (may not be set during early init)

        Returns:
            ModelInfraErrorContext configured for initialization
        """
        return ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.VAULT,
            operation="initialize",
            target_name="vault_handler",
            correlation_id=correlation_id,
            namespace=namespace,
        )

    def _parse_vault_config(
        self, config: dict[str, JsonValue], correlation_id: UUID
    ) -> ModelVaultHandlerConfig:
        """Parse and validate vault configuration.

        Args:
            config: Raw configuration dict
            correlation_id: Correlation ID for tracing

        Returns:
            Validated ModelVaultHandlerConfig

        Raises:
            ProtocolConfigurationError: If Pydantic validation fails
            RuntimeHostError: If unexpected error during parsing
        """
        try:
            # Handle SecretStr token conversion
            token_raw = config.get("token")
            if isinstance(token_raw, str):
                config = dict(config)  # Make mutable copy
                config["token"] = SecretStr(token_raw)

            # Use model_validate for type-safe dict parsing (Pydantic v2 pattern)
            return ModelVaultHandlerConfig.model_validate(config)
        except ValidationError as e:
            ctx = self._create_init_error_context(correlation_id, namespace=None)
            raise ProtocolConfigurationError(
                f"Invalid Vault configuration: {e}",
                context=ctx,
            ) from e
        except Exception as e:
            ctx = self._create_init_error_context(correlation_id, namespace=None)
            raise RuntimeHostError(
                f"Configuration parsing failed: {type(e).__name__}",
                context=ctx,
            ) from e

    def _validate_vault_config_defensive(
        self, config: ModelVaultHandlerConfig, correlation_id: UUID
    ) -> None:
        """Defensive validation for required config fields.

        Note: These checks are defensive programming since Pydantic validation
        should catch missing/empty URL and missing token. However, we keep them
        to ensure consistent error handling if the Pydantic model changes.

        Args:
            config: Validated config model
            correlation_id: Correlation ID for tracing

        Raises:
            ProtocolConfigurationError: If required fields are missing
        """
        if not config.url:
            ctx = self._create_init_error_context(correlation_id, config.namespace)
            raise ProtocolConfigurationError(
                "Missing 'url' in config - Vault server URL required",
                context=ctx,
            )

        if config.token is None:
            ctx = self._create_init_error_context(correlation_id, config.namespace)
            raise ProtocolConfigurationError(
                "Missing 'token' in config - Vault authentication token required",
                context=ctx,
            )

    def _create_hvac_client(self, config: ModelVaultHandlerConfig) -> hvac.Client:
        """Create and return hvac client instance.

        Args:
            config: Validated config model

        Returns:
            Configured hvac.Client instance
        """
        return hvac.Client(
            url=config.url,
            token=config.token.get_secret_value() if config.token else "",
            namespace=config.namespace,
            verify=config.verify_ssl,
            timeout=config.timeout_seconds,
        )

    def _verify_vault_auth(
        self, client: hvac.Client, correlation_id: UUID, namespace: str | None
    ) -> None:
        """Verify vault authentication.

        Args:
            client: hvac client instance
            correlation_id: Correlation ID for tracing
            namespace: Vault namespace

        Raises:
            InfraAuthenticationError: If authentication fails
        """
        if not client.is_authenticated():
            ctx = self._create_init_error_context(correlation_id, namespace)
            raise InfraAuthenticationError(
                "Vault authentication failed - check token validity",
                context=ctx,
            )

    def _initialize_token_ttl(
        self,
        client: hvac.Client,
        config: ModelVaultHandlerConfig,
        correlation_id: UUID,
    ) -> None:
        """Initialize token expiration tracking.

        Queries actual TTL from Vault, falls back to config default on failure.

        Args:
            client: hvac client instance
            config: Validated config model
            correlation_id: Correlation ID for tracing
        """
        try:
            token_ttl = self._extract_token_ttl_from_lookup(
                client, config.default_token_ttl, correlation_id
            )
            self._token_expires_at = time.time() + token_ttl
            logger.info(
                "Token TTL initialized",
                extra={
                    "ttl_seconds": token_ttl,
                    "correlation_id": str(correlation_id),
                },
            )
        except Exception as e:
            # Fallback to config default TTL if lookup fails
            token_ttl = config.default_token_ttl
            logger.warning(
                "Failed to query token TTL, using fallback",
                extra={
                    "error_type": type(e).__name__,
                    "default_ttl_seconds": token_ttl,
                    "correlation_id": str(correlation_id),
                },
            )
            self._token_expires_at = time.time() + token_ttl

    def _extract_token_ttl_from_lookup(
        self,
        client: hvac.Client,
        default_ttl: int,
        correlation_id: UUID,
    ) -> int:
        """Extract token TTL from vault lookup response.

        Args:
            client: hvac client instance
            default_ttl: Default TTL to use if extraction fails
            correlation_id: Correlation ID for tracing

        Returns:
            Token TTL in seconds
        """
        token_info = client.auth.token.lookup_self()
        token_data = token_info.get("data", {})

        if isinstance(token_data, dict):
            ttl_seconds = token_data.get("ttl")
            if isinstance(ttl_seconds, int) and ttl_seconds > 0:
                return ttl_seconds

        # Fallback to config or safe default
        logger.warning(
            "Token TTL not in Vault response, using fallback",
            extra={
                "ttl": default_ttl,
                "correlation_id": str(correlation_id),
            },
        )
        return default_ttl

    def _setup_thread_pool(self, config: ModelVaultHandlerConfig) -> None:
        """Setup bounded thread pool for vault operations.

        Args:
            config: Validated config model
        """
        self._max_workers = config.max_concurrent_operations
        self._executor = ThreadPoolExecutor(
            max_workers=self._max_workers,
            thread_name_prefix="vault_handler_",
        )
        self._max_queue_size = self._max_workers * config.max_queue_size_multiplier

    def _setup_circuit_breaker(self, config: ModelVaultHandlerConfig) -> None:
        """Setup circuit breaker if enabled.

        Args:
            config: Validated config model
        """
        if config.circuit_breaker_enabled:
            self._init_circuit_breaker(
                threshold=config.circuit_breaker_failure_threshold,
                reset_timeout=config.circuit_breaker_reset_timeout_seconds,
                service_name=f"vault.{config.namespace or 'default'}",
                transport_type=EnumInfraTransportType.VAULT,
            )
            self._circuit_breaker_initialized = True

    def _log_init_success(
        self, config: ModelVaultHandlerConfig, correlation_id: UUID
    ) -> None:
        """Log successful initialization.

        Args:
            config: Validated config model
            correlation_id: Correlation ID for tracing
        """
        logger.info(
            "%s initialized successfully",
            self.__class__.__name__,
            extra={
                "handler": self.__class__.__name__,
                "url": config.url,
                "namespace": config.namespace,
                "timeout_seconds": config.timeout_seconds,
                "verify_ssl": config.verify_ssl,
                "thread_pool_max_workers": self._max_workers,
                "thread_pool_max_queue_size": self._max_queue_size,
                "circuit_breaker_enabled": config.circuit_breaker_enabled,
                "correlation_id": str(correlation_id),
            },
        )

    def _handle_init_hvac_error(
        self,
        error: Exception,
        correlation_id: UUID,
        namespace: str | None,
    ) -> None:
        """Handle hvac-related errors during initialization.

        Args:
            error: The exception that occurred
            correlation_id: Correlation ID for tracing
            namespace: Vault namespace

        Raises:
            InfraAuthenticationError: For InvalidRequest errors
            InfraConnectionError: For VaultError errors
            RuntimeHostError: For other errors
        """
        ctx = self._create_init_error_context(correlation_id, namespace)

        if isinstance(error, hvac.exceptions.InvalidRequest):
            raise InfraAuthenticationError(
                "Vault authentication failed - invalid token or permissions",
                context=ctx,
            ) from error
        if isinstance(error, hvac.exceptions.VaultError):
            raise InfraConnectionError(
                f"Failed to connect to Vault: {type(error).__name__}",
                context=ctx,
            ) from error
        raise RuntimeHostError(
            f"Failed to initialize Vault client: {type(error).__name__}",
            context=ctx,
        ) from error

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
            ProtocolConfigurationError: If configuration validation fails.
            RuntimeHostError: If client initialization fails for non-auth/non-connection reasons.
            InfraAuthenticationError: If token authentication fails.
            InfraConnectionError: If connection to Vault server fails.
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

        # Phase 1: Parse and validate configuration
        self._config = self._parse_vault_config(config, init_correlation_id)
        self._validate_vault_config_defensive(self._config, init_correlation_id)

        # Phase 2: Create client and verify connection
        try:
            self._client = self._create_hvac_client(self._config)
            self._verify_vault_auth(
                self._client, init_correlation_id, self._config.namespace
            )

            # Phase 3: Initialize token TTL tracking
            self._initialize_token_ttl(self._client, self._config, init_correlation_id)

            # Phase 4: Setup thread pool and circuit breaker
            self._setup_thread_pool(self._config)
            self._setup_circuit_breaker(self._config)

            # Phase 5: Mark as initialized and log success
            self._initialized = True
            self._log_init_success(self._config, init_correlation_id)

        except InfraAuthenticationError:
            # Re-raise our own authentication errors without wrapping
            raise
        except Exception as e:
            self._handle_init_hvac_error(
                e, init_correlation_id, self._config.namespace if self._config else None
            )

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
        else:  # vault.renew_token
            return await self._renew_token_operation(correlation_id, input_envelope_id)

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

    # -------------------------------------------------------------------------
    # Helper methods for _execute_with_retry (complexity reduction)
    # -------------------------------------------------------------------------

    def _create_vault_error_context(
        self, operation: str, correlation_id: UUID
    ) -> ModelInfraErrorContext:
        """Create standard error context for Vault operations.

        Args:
            operation: Operation name
            correlation_id: Correlation ID for tracing

        Returns:
            ModelInfraErrorContext configured for Vault transport
        """
        return ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.VAULT,
            operation=operation,
            target_name="vault_handler",
            correlation_id=correlation_id,
            namespace=self._config.namespace if self._config else None,
        )

    async def _record_circuit_failure_if_final(
        self,
        retry_state: ModelRetryState,
        operation: str,
        correlation_id: UUID,
    ) -> None:
        """Record circuit breaker failure only on final retry attempt.

        Args:
            retry_state: Current retry state
            operation: Operation name
            correlation_id: Correlation ID for tracing
        """
        if not retry_state.is_retriable() and self._circuit_breaker_initialized:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(operation, correlation_id)

    async def _handle_vault_timeout(
        self,
        error: TimeoutError,
        retry_state: ModelRetryState,
        operation: str,
        correlation_id: UUID,
        timeout_seconds: float,
    ) -> None:
        """Handle timeout error - raise InfraTimeoutError if retries exhausted.

        Args:
            error: The timeout error
            retry_state: Current retry state (after next_attempt called)
            operation: Operation name
            correlation_id: Correlation ID for tracing
            timeout_seconds: Timeout value for error message

        Raises:
            InfraTimeoutError: If retries exhausted
        """
        await self._record_circuit_failure_if_final(
            retry_state, operation, correlation_id
        )
        if not retry_state.is_retriable():
            ctx = self._create_vault_error_context(operation, correlation_id)
            raise InfraTimeoutError(
                f"Vault operation timed out after {timeout_seconds}s",
                context=ctx,
                timeout_seconds=timeout_seconds,
            ) from error

    async def _handle_vault_forbidden(
        self,
        error: hvac.exceptions.Forbidden,
        operation: str,
        correlation_id: UUID,
    ) -> None:
        """Handle forbidden error - always raise, no retry.

        Args:
            error: The forbidden exception
            operation: Operation name
            correlation_id: Correlation ID for tracing

        Raises:
            InfraAuthenticationError: Always
        """
        if self._circuit_breaker_initialized:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(operation, correlation_id)
        ctx = self._create_vault_error_context(operation, correlation_id)
        raise InfraAuthenticationError(
            "Vault operation forbidden - check token permissions",
            context=ctx,
        ) from error

    def _handle_vault_invalid_path(
        self,
        error: hvac.exceptions.InvalidPath,
        operation: str,
        correlation_id: UUID,
    ) -> None:
        """Handle invalid path error - always raise, no retry, no circuit breaker.

        Args:
            error: The invalid path exception
            operation: Operation name
            correlation_id: Correlation ID for tracing

        Raises:
            SecretResolutionError: Always
        """
        ctx = self._create_vault_error_context(operation, correlation_id)
        raise SecretResolutionError(
            "Secret path not found or invalid",
            context=ctx,
        ) from error

    async def _handle_vault_down(
        self,
        error: hvac.exceptions.VaultDown,
        retry_state: ModelRetryState,
        operation: str,
        correlation_id: UUID,
    ) -> None:
        """Handle Vault down error - raise InfraUnavailableError if retries exhausted.

        Args:
            error: The VaultDown exception
            retry_state: Current retry state (after next_attempt called)
            operation: Operation name
            correlation_id: Correlation ID for tracing

        Raises:
            InfraUnavailableError: If retries exhausted
        """
        await self._record_circuit_failure_if_final(
            retry_state, operation, correlation_id
        )
        if not retry_state.is_retriable():
            ctx = self._create_vault_error_context(operation, correlation_id)
            raise InfraUnavailableError(
                "Vault server is unavailable",
                context=ctx,
            ) from error

    async def _handle_vault_general_error(
        self,
        error: Exception,
        retry_state: ModelRetryState,
        operation: str,
        correlation_id: UUID,
    ) -> None:
        """Handle general error - raise InfraConnectionError if retries exhausted.

        Args:
            error: The exception
            retry_state: Current retry state (after next_attempt called)
            operation: Operation name
            correlation_id: Correlation ID for tracing

        Raises:
            InfraConnectionError: If retries exhausted
        """
        await self._record_circuit_failure_if_final(
            retry_state, operation, correlation_id
        )
        if not retry_state.is_retriable():
            ctx = self._create_vault_error_context(operation, correlation_id)
            raise InfraConnectionError(
                f"Vault operation failed: {type(error).__name__}",
                context=ctx,
            ) from error

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

        # Initialize retry state with config values
        retry_config = self._config.retry
        retry_state = ModelRetryState(
            attempt=0,
            max_attempts=retry_config.max_attempts,
            delay_seconds=retry_config.initial_backoff_seconds,
            backoff_multiplier=retry_config.exponential_base,
        )

        # Create operation context for tracking
        op_context = ModelOperationContext.create(
            operation_name=operation,
            correlation_id=correlation_id,
            timeout_seconds=self._config.timeout_seconds,
            metadata={"namespace": self._config.namespace or "default"},
        )

        while retry_state.is_retriable():
            try:
                result = await self._execute_vault_operation(func, op_context)
                await self._record_circuit_success()
                return result
            except TimeoutError as e:
                retry_state = retry_state.next_attempt(
                    error_message=f"Timeout after {op_context.timeout_seconds}s",
                    max_delay_seconds=retry_config.max_backoff_seconds,
                )
                await self._handle_vault_timeout(
                    e,
                    retry_state,
                    operation,
                    correlation_id,
                    op_context.timeout_seconds,
                )
            except hvac.exceptions.Forbidden as e:
                await self._handle_vault_forbidden(e, operation, correlation_id)
            except hvac.exceptions.InvalidPath as e:
                self._handle_vault_invalid_path(e, operation, correlation_id)
            except hvac.exceptions.VaultDown as e:
                retry_state = retry_state.next_attempt(
                    error_message=f"Vault down: {type(e).__name__}",
                    max_delay_seconds=retry_config.max_backoff_seconds,
                )
                await self._handle_vault_down(e, retry_state, operation, correlation_id)
            except Exception as e:
                retry_state = retry_state.next_attempt(
                    error_message=f"Unexpected error: {type(e).__name__}",
                    max_delay_seconds=retry_config.max_backoff_seconds,
                )
                await self._handle_vault_general_error(
                    e, retry_state, operation, correlation_id
                )

            self._log_retry_attempt(retry_state, operation, correlation_id)
            await asyncio.sleep(retry_state.delay_seconds)

        # Should never reach here, but satisfy type checker
        if retry_state.last_error is not None:
            raise RuntimeError(f"Retry exhausted: {retry_state.last_error}")
        raise RuntimeError("Retry loop completed without result")

    async def _execute_vault_operation(
        self, func: Callable[[], T], op_context: ModelOperationContext
    ) -> T:
        """Execute a vault operation in thread pool with timeout.

        Args:
            func: Callable to execute (synchronous hvac method)
            op_context: Operation context with timeout

        Returns:
            Result from func()
        """
        loop = asyncio.get_running_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(self._executor, func),
            timeout=op_context.timeout_seconds,
        )

    async def _record_circuit_success(self) -> None:
        """Record success for circuit breaker if initialized."""
        if self._circuit_breaker_initialized:
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

    def _log_retry_attempt(
        self,
        retry_state: ModelRetryState,
        operation: str,
        correlation_id: UUID,
    ) -> None:
        """Log retry attempt details.

        Args:
            retry_state: Current retry state
            operation: Operation name
            correlation_id: Correlation ID for tracing
        """
        logger.debug(
            "Retrying Vault operation",
            extra={
                "operation": operation,
                "attempt": retry_state.attempt,
                "max_attempts": retry_state.max_attempts,
                "backoff_seconds": retry_state.delay_seconds,
                "last_error": retry_state.last_error,
                "correlation_id": str(correlation_id),
            },
        )

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

    # -------------------------------------------------------------------------
    # Helper methods for renew_token (complexity reduction)
    # -------------------------------------------------------------------------

    def _validate_renewal_preconditions(self, correlation_id: UUID) -> None:
        """Validate that handler is initialized before renewal.

        Args:
            correlation_id: Correlation ID for tracing

        Raises:
            RuntimeHostError: If handler not initialized
        """
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

    def _extract_ttl_from_renewal_response(
        self,
        result: dict[str, JsonValue],
        default_ttl: int,
        correlation_id: UUID,
    ) -> int:
        """Extract TTL from token renewal response.

        Args:
            result: Renewal response from Vault
            default_ttl: Default TTL to use if extraction fails
            correlation_id: Correlation ID for tracing

        Returns:
            Token TTL in seconds
        """
        auth_data = result.get("auth", {})

        if isinstance(auth_data, dict):
            lease_duration = auth_data.get("lease_duration")
            if isinstance(lease_duration, int) and lease_duration > 0:
                return lease_duration

        # Fallback to config or safe default
        logger.warning(
            "Token TTL not in renewal response, using fallback",
            extra={
                "ttl": default_ttl,
                "correlation_id": str(correlation_id),
            },
        )
        return default_ttl

    async def _refresh_ttl_from_vault_lookup(
        self,
        current_ttl: int,
        correlation_id: UUID,
    ) -> int:
        """Refresh TTL by querying actual value from Vault.

        Args:
            current_ttl: Current TTL value (used as fallback)
            correlation_id: Correlation ID for tracing

        Returns:
            Refreshed TTL in seconds
        """
        if self._client is None:
            return current_ttl

        try:
            loop = asyncio.get_running_loop()
            token_info = await loop.run_in_executor(
                self._executor,
                self._client.auth.token.lookup_self,
            )
            token_data = token_info.get("data", {})
            if isinstance(token_data, dict):
                ttl_seconds = token_data.get("ttl")
                if isinstance(ttl_seconds, int) and ttl_seconds > 0:
                    logger.info(
                        "Token TTL refreshed from Vault lookup",
                        extra={
                            "ttl_seconds": ttl_seconds,
                            "correlation_id": str(correlation_id),
                        },
                    )
                    return ttl_seconds
        except Exception as e:
            logger.debug(
                "Token lookup after renewal failed, using lease_duration",
                extra={
                    "error_type": type(e).__name__,
                    "correlation_id": str(correlation_id),
                },
            )

        return current_ttl

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

        self._validate_renewal_preconditions(correlation_id)

        # At this point, _client and _config are guaranteed non-None
        assert self._client is not None
        assert self._config is not None

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

            # Extract TTL from renewal response
            token_ttl = self._extract_ttl_from_renewal_response(
                result, self._config.default_token_ttl, correlation_id
            )

            # Refresh TTL from Vault lookup (may override renewal response)
            token_ttl = await self._refresh_ttl_from_vault_lookup(
                token_ttl, correlation_id
            )

            # Update token expiration tracking
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
