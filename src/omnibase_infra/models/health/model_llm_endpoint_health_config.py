# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Configuration model for the LLM endpoint health checker.

.. versionadded:: 0.9.0
    Part of OMN-2255 LLM endpoint health checker.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelLlmEndpointHealthConfig(BaseModel):
    """Configuration for the LLM endpoint health checker.

    Attributes:
        endpoints: Mapping of logical endpoint name to base URL.
            Example: ``{"coder-14b": "http://192.168.86.201:8000"}``.
        probe_interval_seconds: Seconds between probe cycles. Default: 30.
        probe_timeout_seconds: HTTP timeout for individual probe requests.
            Default: 5.0.
        circuit_breaker_threshold: Consecutive failures before opening
            the circuit for an endpoint. Default: 3.
        circuit_breaker_reset_timeout: Seconds before a tripped circuit
            transitions to half-open. Default: 60.0.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    endpoints: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of logical name to base URL",
    )

    @field_validator("endpoints")
    @classmethod
    def _validate_endpoint_urls(cls, v: dict[str, str]) -> dict[str, str]:
        """Validate that every endpoint URL uses an HTTP(S) scheme.

        Raises:
            ValueError: If any URL does not start with ``http://`` or
                ``https://``.
        """
        for name, url in v.items():
            if not url.startswith(("http://", "https://")):
                msg = (
                    f"Endpoint '{name}' has invalid URL '{url}': "
                    "must start with 'http://' or 'https://'"
                )
                raise ValueError(msg)
        return v

    probe_interval_seconds: float = Field(
        default=30.0,
        ge=1.0,
        description="Seconds between probe cycles",
    )
    probe_timeout_seconds: float = Field(
        default=5.0,
        ge=0.5,
        le=30.0,
        description="HTTP timeout per probe request",
    )
    circuit_breaker_threshold: int = Field(
        default=3,
        ge=1,
        description="Consecutive failures before opening circuit per endpoint",
    )
    circuit_breaker_reset_timeout: float = Field(
        default=60.0,
        ge=0.0,
        description="Seconds before circuit transitions from OPEN to HALF_OPEN",
    )


__all__: list[str] = ["ModelLlmEndpointHealthConfig"]
