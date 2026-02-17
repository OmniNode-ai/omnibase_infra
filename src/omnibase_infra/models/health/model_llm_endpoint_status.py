# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Status model for a single LLM endpoint health probe.

Each instance represents a point-in-time snapshot produced by
``ServiceLlmEndpointHealth`` after probing a single endpoint. Instances
are frozen and stored in the service's in-memory status map and included
in ``ModelLlmEndpointHealthEvent`` payloads emitted to Kafka.

.. versionadded:: 0.9.0
    Part of OMN-2255 LLM endpoint health checker.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.utils.util_error_sanitization import sanitize_url


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
        """Validate that the URL uses an HTTP(S) scheme and has a hostname.

        Rejects non-HTTP schemes to prevent construction of status objects
        with invalid endpoint URLs.  Also rejects URLs with empty netloc
        (e.g. ``http://``) which would produce invalid probe requests.
        Error messages are sanitized via ``sanitize_url`` to avoid leaking
        credentials embedded in URLs.

        Raises:
            ValueError: If the URL does not start with ``http://`` or
                ``https://``, or has an empty netloc (no hostname).
        """
        if not v.startswith(("http://", "https://")):
            safe_url = sanitize_url(v)
            msg = f"Invalid URL '{safe_url}': must start with 'http://' or 'https://'"
            raise ValueError(msg)
        parsed = urlparse(v)
        if not parsed.netloc:
            safe_url = sanitize_url(v)
            msg = f"Invalid URL '{safe_url}': URL must have a hostname"
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
