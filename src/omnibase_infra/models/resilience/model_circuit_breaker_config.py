# Copyright 2025 OmniNode Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration model for circuit breaker initialization.

Environment Variable Support:
    This model supports creation from environment variables via the from_env()
    class method. This enables runtime configuration of circuit breaker thresholds
    without code changes.

    Environment Variables:
        {prefix}_THRESHOLD: Maximum consecutive failures before opening circuit (default: 5)
        {prefix}_RESET_TIMEOUT: Seconds before auto-reset to half-open (default: 60.0)

    Default prefix is "ONEX_CB", so variables would be:
        ONEX_CB_THRESHOLD=5
        ONEX_CB_RESET_TIMEOUT=60.0

This module provides the configuration model used to initialize the
MixinAsyncCircuitBreaker mixin. Grouping parameters into a configuration
model follows ONEX patterns for reducing function parameter count and
union patterns.

Circuit Breaker Overview:
    The circuit breaker pattern prevents cascading failures in distributed systems
    by temporarily blocking requests to a failing service. It operates in three states:
    - CLOSED: Normal operation, requests allowed
    - OPEN: Circuit tripped after threshold failures, requests blocked
    - HALF_OPEN: Testing recovery after reset timeout, limited requests allowed

See Also:
    - MixinAsyncCircuitBreaker: The mixin that uses this configuration
    - docs/patterns/circuit_breaker_implementation.md: Implementation details
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.utils.util_env_parsing import parse_env_float, parse_env_int


class ModelCircuitBreakerConfig(BaseModel):
    """Configuration model for circuit breaker initialization.

    This model groups all parameters required by ``_init_circuit_breaker()``
    into a single configuration object, following ONEX conventions for functions
    with more than 5 parameters and reducing union type patterns.

    Attributes:
        threshold: Maximum number of consecutive failures before opening the
            circuit. Must be >= 1. Default: 5.
        reset_timeout_seconds: Time in seconds before the circuit automatically
            transitions from OPEN to HALF_OPEN for recovery testing.
            Must be >= 0. Default: 60.0.
        service_name: Identifier for the service being protected. Used in error
            context and logging. Default: "unknown".
        transport_type: Transport type for error context classification.
            Determines which error code is used when the circuit opens.
            Default: HTTP.

    Example:
        ```python
        from omnibase_infra.models.resilience import ModelCircuitBreakerConfig
        from omnibase_infra.mixins import MixinAsyncCircuitBreaker
        from omnibase_infra.enums import EnumInfraTransportType

        class EventBusKafka(MixinAsyncCircuitBreaker):
            def __init__(self, environment: str):
                config = ModelCircuitBreakerConfig(
                    threshold=5,
                    reset_timeout_seconds=60.0,
                    service_name=f"kafka.{environment}",
                    transport_type=EnumInfraTransportType.KAFKA,
                )
                self._init_circuit_breaker_from_config(config)

        # With defaults
        class HttpClient(MixinAsyncCircuitBreaker):
            def __init__(self, service_name: str):
                config = ModelCircuitBreakerConfig(service_name=service_name)
                self._init_circuit_breaker_from_config(config)
        ```

    Configuration Guidelines:
        - High-reliability services: Use lower threshold (3) and longer timeout (120s)
        - Best-effort services: Use higher threshold (10) and shorter timeout (30s)
        - Tune based on service SLAs and failure characteristics

    See Also:
        MixinAsyncCircuitBreaker: The mixin that uses this configuration.
    """

    threshold: int = Field(
        default=5,
        ge=1,
        description="Maximum consecutive failures before opening the circuit",
    )

    reset_timeout_seconds: float = Field(
        default=60.0,
        ge=0.0,
        description="Seconds before automatic transition from OPEN to HALF_OPEN",
    )

    service_name: str = Field(
        default="unknown",
        min_length=1,
        description="Service identifier for error context and logging",
    )

    transport_type: EnumInfraTransportType = Field(
        default=EnumInfraTransportType.HTTP,
        description="Transport type for error context classification",
    )

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "threshold": 5,
                    "reset_timeout_seconds": 60.0,
                    "service_name": "kafka.production",
                    "transport_type": "kafka",
                },
                {
                    "threshold": 3,
                    "reset_timeout_seconds": 120.0,
                    "service_name": "postgresql-primary",
                    "transport_type": "db",
                },
            ]
        },
    )

    @classmethod
    def from_env(
        cls,
        service_name: str = "unknown",
        transport_type: EnumInfraTransportType = EnumInfraTransportType.HTTP,
        prefix: str = "ONEX_CB",
    ) -> ModelCircuitBreakerConfig:
        """Create configuration from environment variables.

        Reads circuit breaker settings from environment variables, falling back
        to defaults if not set. This enables runtime configuration without
        code changes.

        Environment Variables:
            {prefix}_THRESHOLD: Maximum consecutive failures before opening circuit.
                Must be >= 1 and <= 1000. Default: 5.
            {prefix}_RESET_TIMEOUT: Seconds before auto-reset from OPEN to HALF_OPEN.
                Must be >= 0 and <= 3600 (1 hour). Default: 60.0.

        Args:
            service_name: Service identifier for error context and logging.
                This is NOT read from environment - must be provided by caller.
            transport_type: Transport type for error context classification.
                This is NOT read from environment - must be provided by caller.
            prefix: Environment variable prefix. Default: "ONEX_CB".
                Override to create service-specific configurations.

        Returns:
            ModelCircuitBreakerConfig with values from environment or defaults.

        Raises:
            ProtocolConfigurationError: If environment variable values cannot be
                parsed as int (threshold) or float (reset_timeout). The error
                includes ModelInfraErrorContext with transport_type and service_name.

        Example:
            ```python
            # With default prefix (reads ONEX_CB_THRESHOLD, ONEX_CB_RESET_TIMEOUT)
            config = ModelCircuitBreakerConfig.from_env(
                service_name="kafka.production",
                transport_type=EnumInfraTransportType.KAFKA,
            )

            # With custom prefix (reads KAFKA_CB_THRESHOLD, KAFKA_CB_RESET_TIMEOUT)
            config = ModelCircuitBreakerConfig.from_env(
                service_name="kafka.production",
                transport_type=EnumInfraTransportType.KAFKA,
                prefix="KAFKA_CB",
            )

            # Use with circuit breaker mixin
            self._init_circuit_breaker_from_config(config)
            ```

        Note:
            The service_name and transport_type are not read from environment
            because they are context-specific and should be provided by the
            calling code. Only the numeric thresholds are configurable via
            environment to allow tuning without identifying which service
            each variable affects.
        """
        threshold_var = f"{prefix}_THRESHOLD"
        reset_timeout_var = f"{prefix}_RESET_TIMEOUT"

        threshold = parse_env_int(
            threshold_var,
            5,
            min_value=1,
            max_value=1000,
            transport_type=transport_type,
            service_name=service_name,
        )
        reset_timeout = parse_env_float(
            reset_timeout_var,
            60.0,
            min_value=0.0,
            max_value=3600.0,
            transport_type=transport_type,
            service_name=service_name,
        )

        return cls(
            threshold=threshold,
            reset_timeout_seconds=reset_timeout,
            service_name=service_name,
            transport_type=transport_type,
        )


__all__ = ["ModelCircuitBreakerConfig"]
