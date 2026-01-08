# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Vault initialization mixin for HandlerVault.

Provides initialization-related helper methods for parsing configuration,
creating the hvac client, and setting up infrastructure components.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from uuid import UUID

import hvac
from pydantic import SecretStr, ValidationError

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraAuthenticationError,
    InfraConnectionError,
    ModelInfraErrorContext,
    ProtocolConfigurationError,
    RuntimeHostError,
)
from omnibase_infra.handlers.models.vault import ModelVaultHandlerConfig

logger = logging.getLogger(__name__)


class MixinVaultInitialization:
    """Mixin providing initialization helpers for HandlerVault.

    Contains methods for:
    - Configuration parsing and validation
    - hvac client creation
    - Authentication verification
    - Token TTL initialization
    - Thread pool and circuit breaker setup
    - Error handling during initialization
    """

    # Instance attributes (declared for type checking)
    _client: hvac.Client | None
    _config: ModelVaultHandlerConfig | None
    _token_expires_at: float
    _executor: ThreadPoolExecutor | None
    _max_workers: int
    _max_queue_size: int
    _circuit_breaker_initialized: bool

    # Methods from MixinAsyncCircuitBreaker that will be available at runtime
    def _init_circuit_breaker(
        self,
        threshold: int,
        reset_timeout: float,
        service_name: str,
        transport_type: EnumInfraTransportType,
    ) -> None:
        """Initialize circuit breaker - provided by MixinAsyncCircuitBreaker."""

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
        self, config: dict[str, object], correlation_id: UUID
    ) -> ModelVaultHandlerConfig:
        """Parse and validate vault configuration.

        Performs both Pydantic validation and defensive checks for required fields.

        Args:
            config: Raw configuration dict
            correlation_id: Correlation ID for tracing

        Returns:
            Validated ModelVaultHandlerConfig

        Raises:
            ProtocolConfigurationError: If Pydantic validation fails or required
                fields are missing/empty
            RuntimeHostError: If unexpected error during parsing
        """
        try:
            # Handle SecretStr token conversion
            token_raw = config.get("token")
            if isinstance(token_raw, str):
                config = dict(config)  # Make mutable copy
                config["token"] = SecretStr(token_raw)

            # Use model_validate for type-safe dict parsing (Pydantic v2 pattern)
            parsed_config = ModelVaultHandlerConfig.model_validate(config)

            # Defensive validation: ensure required fields are present
            # These checks are defensive programming since Pydantic validation
            # should catch missing/empty URL and missing token. However, we keep
            # them to ensure consistent error handling if the Pydantic model changes.
            if not parsed_config.url:
                ctx = self._create_init_error_context(
                    correlation_id, parsed_config.namespace
                )
                raise ProtocolConfigurationError(
                    "Missing 'url' in config - Vault server URL required",
                    context=ctx,
                )

            if parsed_config.token is None:
                ctx = self._create_init_error_context(
                    correlation_id, parsed_config.namespace
                )
                raise ProtocolConfigurationError(
                    "Missing 'token' in config - Vault authentication token required",
                    context=ctx,
                )

            return parsed_config
        except ProtocolConfigurationError:
            # Re-raise our own configuration errors without wrapping
            raise
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

        Queries actual TTL from Vault via token lookup, falls back to config
        default on failure. Sets self._token_expires_at for token renewal tracking.

        Args:
            client: hvac client instance
            config: Validated config model
            correlation_id: Correlation ID for tracing
        """
        default_ttl = config.default_token_ttl
        token_ttl = default_ttl

        try:
            # Query token info from Vault
            token_info = client.auth.token.lookup_self()
            token_data = token_info.get("data", {})

            # Extract TTL from response
            if isinstance(token_data, dict):
                ttl_seconds = token_data.get("ttl")
                if isinstance(ttl_seconds, int) and ttl_seconds > 0:
                    token_ttl = ttl_seconds
                    self._token_expires_at = time.time() + token_ttl
                    logger.info(
                        "Token TTL initialized",
                        extra={
                            "ttl_seconds": token_ttl,
                            "correlation_id": str(correlation_id),
                        },
                    )
                    return

            # TTL not in response - use fallback
            logger.warning(
                "Token TTL not in Vault response, using fallback",
                extra={
                    "ttl": default_ttl,
                    "correlation_id": str(correlation_id),
                },
            )
            self._token_expires_at = time.time() + default_ttl

        except Exception as e:
            # Fallback to config default TTL if lookup fails
            logger.warning(
                "Failed to query token TTL, using fallback",
                extra={
                    "error_type": type(e).__name__,
                    "default_ttl_seconds": default_ttl,
                    "correlation_id": str(correlation_id),
                },
            )
            self._token_expires_at = time.time() + default_ttl

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


__all__: list[str] = ["MixinVaultInitialization"]
