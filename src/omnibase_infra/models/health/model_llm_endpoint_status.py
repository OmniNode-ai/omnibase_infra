# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Status model for a single LLM endpoint health probe.

.. versionadded:: 0.9.0
    Part of OMN-2255 LLM endpoint health checker.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelLlmEndpointStatus(BaseModel):
    """Point-in-time health status of a single LLM endpoint.

    Attributes:
        url: Base URL of the endpoint.
        name: Logical name of the endpoint (e.g. ``coder-14b``).
        available: Whether the last probe succeeded.
        last_check: UTC timestamp of the most recent probe.
        latency_ms: Round-trip latency of the most recent probe in
            milliseconds.  ``-1.0`` if the probe failed.
        error: Human-readable error string from the last failed probe,
            or empty string if healthy.
        circuit_state: Current circuit breaker state for this endpoint
            (``closed``, ``open``, or ``half_open``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    url: str = Field(..., description="Endpoint base URL")

    @field_validator("url")
    @classmethod
    def _validate_url_scheme(cls, v: str) -> str:
        """Validate that the URL uses an HTTP(S) scheme.

        Raises:
            ValueError: If the URL does not start with ``http://`` or
                ``https://``.
        """
        if not v.startswith(("http://", "https://")):
            msg = f"Invalid URL '{v}': must start with 'http://' or 'https://'"
            raise ValueError(msg)
        return v

    name: str = Field(..., description="Logical endpoint name")
    available: bool = Field(..., description="Whether the endpoint is healthy")
    last_check: datetime = Field(..., description="UTC timestamp of last probe")
    latency_ms: float = Field(..., description="Probe latency in ms (-1.0 on failure)")
    error: str = Field(default="", description="Error message if probe failed")
    circuit_state: Literal["closed", "open", "half_open"] = Field(
        default="closed", description="Circuit breaker state"
    )


__all__: list[str] = ["ModelLlmEndpointStatus"]
