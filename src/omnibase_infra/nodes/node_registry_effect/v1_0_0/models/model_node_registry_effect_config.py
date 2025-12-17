# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Configuration model for NodeRegistryEffect."""

from pydantic import BaseModel, ConfigDict, Field


class ModelNodeRegistryEffectConfig(BaseModel):
    """Configuration for NodeRegistryEffect circuit breaker and resilience settings.

    This model encapsulates configuration parameters for the registry effect node,
    following the ONEX pattern of using config models to reduce __init__ parameter count.

    Note:
        This is an MVP implementation. Additional configuration options (registry
        endpoints, connection pooling, timeouts, retry policies) will be added in
        future iterations as the node matures.

    Attributes:
        circuit_breaker_threshold: Number of consecutive failures before opening circuit.
            When this threshold is reached, subsequent requests will fail fast with
            InfraUnavailableError until the reset timeout expires.
        circuit_breaker_reset_timeout: Seconds before circuit breaker auto-resets.
            After this timeout, the circuit enters half-open state and allows
            a test request to determine if the service has recovered.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    circuit_breaker_threshold: int = Field(
        default=5,
        ge=1,
        description="Number of failures before opening circuit breaker",
    )
    circuit_breaker_reset_timeout: float = Field(
        default=60.0,
        ge=0.0,
        description="Seconds before circuit breaker auto-resets to half-open state",
    )
