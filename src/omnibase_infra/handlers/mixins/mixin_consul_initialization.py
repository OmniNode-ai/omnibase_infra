# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Consul initialization operations mixin.

This mixin provides initialization and setup helper methods for HandlerConsul,
extracted to reduce class complexity and improve maintainability.

Provides:
    - Configuration validation and parsing
    - Client setup and connection verification
    - Thread pool and circuit breaker initialization
    - Error handling helpers for initialization
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol
from uuid import UUID

import consul
from pydantic import SecretStr, ValidationError

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraConnectionError,
    ModelInfraErrorContext,
    ProtocolConfigurationError,
    RuntimeHostError,
)
from omnibase_infra.handlers.models.consul import ModelConsulHandlerConfig

if TYPE_CHECKING:
    from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class ProtocolConsulInitDependencies(Protocol):
    """Protocol defining required dependencies for initialization operations.

    HandlerConsul must provide these attributes for the mixin to work.
    """

    _executor: ThreadPoolExecutor | None
    _max_workers: int
    _max_queue_size: int
    _circuit_breaker_initialized: bool

    def _init_circuit_breaker(
        self,
        threshold: int,
        reset_timeout: float,
        service_name: str,
        transport_type: EnumInfraTransportType,
    ) -> None:
        """Initialize circuit breaker."""
        ...


class MixinConsulInitialization:
    """Mixin providing Consul initialization helper methods.

    This mixin extracts initialization operations from HandlerConsul to reduce
    class complexity while maintaining full functionality.

    Required Dependencies (from host class):
        - _executor: ThreadPoolExecutor instance
        - _max_workers: Max worker count
        - _max_queue_size: Max queue size
        - _circuit_breaker_initialized: Circuit breaker init flag
        - _init_circuit_breaker: Circuit breaker initialization method
    """

    # Instance attribute declarations for type checking (must match handler)
    _executor: ThreadPoolExecutor | None
    _max_workers: int
    _max_queue_size: int
    _circuit_breaker_initialized: bool

    # Methods from other mixins that will be available at runtime
    def _init_circuit_breaker(
        self,
        threshold: int,
        reset_timeout: float,
        service_name: str,
        transport_type: EnumInfraTransportType,
    ) -> None:
        """Initialize circuit breaker - provided by MixinAsyncCircuitBreaker."""
        raise NotImplementedError("Must be provided by implementing class")

    def _validate_consul_config(
        self,
        config: dict[str, object],
        correlation_id: UUID,
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
        self,
        consul_config: ModelConsulHandlerConfig,
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
        self,
        client: consul.Consul,
        correlation_id: UUID,
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
                    "Consul leader check failed: expected a leader, got empty response",
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

    def _setup_thread_pool(
        self,
        consul_config: ModelConsulHandlerConfig,
    ) -> None:
        """Set up the thread pool executor for async operations.

        Args:
            consul_config: Validated Consul configuration.
        """
        from concurrent.futures import ThreadPoolExecutor

        self._max_workers = consul_config.max_concurrent_operations
        self._executor = ThreadPoolExecutor(
            max_workers=self._max_workers,
            thread_name_prefix="consul_handler_",
        )
        self._max_queue_size = (
            self._max_workers * consul_config.max_queue_size_multiplier
        )

    def _setup_circuit_breaker(
        self,
        consul_config: ModelConsulHandlerConfig,
    ) -> None:
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

    def _log_initialization_success(
        self,
        config: ModelConsulHandlerConfig,
        correlation_id: UUID,
    ) -> None:
        """Log successful initialization with handler details.

        Args:
            config: The validated configuration.
            correlation_id: Correlation ID for logging.
        """
        logger.info(
            "%s initialized successfully",
            self.__class__.__name__,
            extra={
                "handler": self.__class__.__name__,
                "host": config.host,
                "port": config.port,
                "scheme": config.scheme,
                "datacenter": config.datacenter,
                "timeout_seconds": config.timeout_seconds,
                "thread_pool_max_workers": self._max_workers,
                "thread_pool_max_queue_size": self._max_queue_size,
                "circuit_breaker_enabled": self._circuit_breaker_initialized,
                "correlation_id": str(correlation_id),
            },
        )

    def _raise_auth_error(
        self,
        correlation_id: UUID,
        original_error: Exception,
    ) -> None:
        """Raise InfraAuthenticationError with context.

        Args:
            correlation_id: Correlation ID for error context.
            original_error: The original exception for chaining.

        Raises:
            InfraAuthenticationError: Always raises this error.
        """
        from omnibase_infra.errors import InfraAuthenticationError

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
        self,
        correlation_id: UUID,
        original_error: Exception,
    ) -> None:
        """Raise InfraConnectionError with context.

        Args:
            correlation_id: Correlation ID for error context.
            original_error: The original exception for chaining.

        Raises:
            InfraConnectionError: Always raises this error.
        """
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
        self,
        correlation_id: UUID,
        original_error: Exception,
    ) -> None:
        """Raise RuntimeHostError with context.

        Args:
            correlation_id: Correlation ID for error context.
            original_error: The original exception for chaining.

        Raises:
            RuntimeHostError: Always raises this error.
        """
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


__all__: list[str] = ["MixinConsulInitialization"]
