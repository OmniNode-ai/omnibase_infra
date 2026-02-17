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
    All operations return ModelHandlerOutput[dict[str, object]] per OMN-975.
    Uses ModelHandlerOutput.for_compute() since handlers return synchronous results
    rather than emitting events to the event bus.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import hvac

from omnibase_core.container import ModelONEXContainer
from omnibase_core.models.dispatch import ModelHandlerOutput
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
)
from omnibase_infra.errors import (
    InfraAuthenticationError,
    ModelInfraErrorContext,
    RuntimeHostError,
)
from omnibase_infra.handlers.mixins import (
    MixinVaultInitialization,
    MixinVaultRetry,
    MixinVaultSecrets,
    MixinVaultToken,
)
from omnibase_infra.handlers.models.vault import ModelVaultHandlerConfig
from omnibase_infra.mixins import MixinAsyncCircuitBreaker, MixinEnvelopeExtraction

logger = logging.getLogger(__name__)

# Handler ID for ModelHandlerOutput
HANDLER_ID_VAULT: str = "vault-handler"

SUPPORTED_OPERATIONS: frozenset[str] = frozenset(
    {
        "vault.read_secret",
        "vault.write_secret",
        "vault.delete_secret",
        "vault.list_secrets",
        "vault.renew_token",
    }
)


class HandlerVault(
    MixinAsyncCircuitBreaker,
    MixinEnvelopeExtraction,
    MixinVaultInitialization,
    MixinVaultRetry,
    MixinVaultSecrets,
    MixinVaultToken,
):
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

    Mixin Composition:
        - MixinAsyncCircuitBreaker: Circuit breaker pattern
        - MixinEnvelopeExtraction: Envelope parsing utilities
        - MixinVaultInitialization: Configuration and client setup
        - MixinVaultRetry: Retry logic with exponential backoff
        - MixinVaultSecrets: CRUD operations for secrets
        - MixinVaultToken: Token management and renewal
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize HandlerVault with ONEX container for dependency injection.

        Args:
            container: ONEX container for dependency injection.

        Note:
            Circuit breaker is initialized during initialize() call when
            configuration is available. The mixin's _init_circuit_breaker() method
            is called there with the actual config values.
        """
        self._container = container
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
            I/O operations (Vault secret management). EFFECT handlers are not
            deterministic and interact with external systems.

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

    async def initialize(self, config: dict[str, object]) -> None:
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

        # Phase 1: Parse and validate configuration (includes defensive validation)
        self._config = self._parse_vault_config(config, init_correlation_id)

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
        logger.info("HandlerVault shutdown complete")

    async def execute(
        self, envelope: dict[str, object]
    ) -> ModelHandlerOutput[dict[str, object]]:
        """Execute Vault operation from envelope.

        Args:
            envelope: Request envelope containing:
                - operation: Vault operation (vault.read_secret, vault.write_secret, etc.)
                - payload: dict with operation-specific parameters
                - correlation_id: Optional correlation ID for tracing
                - envelope_id: Optional envelope ID for causality tracking

        Returns:
            ModelHandlerOutput[dict[str, object]] with status, payload, and correlation_id
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
                "HandlerVault not initialized. Call initialize() first.",
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

            The transport type for this handler is VAULT (secret management).

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


__all__: list[str] = ["HandlerVault"]
