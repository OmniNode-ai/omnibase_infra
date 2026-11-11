# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""HTTP client configuration.

Part of OMN-1976: Contract dependency materialization.
"""

from __future__ import annotations

import os

from pydantic import BaseModel, ConfigDict, Field

_TRUTHY_VALUES = frozenset({"true", "1", "yes", "on"})


class ModelHttpClientConfig(BaseModel):
    """HTTP client configuration for dependency materialization.

    Sources configuration from HTTP_CLIENT_* environment variables.
    Used by ProviderHttpClient to create shared httpx.AsyncClient instances.

    Attributes:
        timeout_seconds: Request timeout in seconds (positive, at most 300).
        follow_redirects: Whether to follow HTTP redirects (default True).
        trust_env: Whether environment proxy and TLS settings apply (default True).
        max_keepalive_connections: Idle connection limit (default 20; 0 disables reuse).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    timeout_seconds: float = Field(
        default=30.0,
        gt=0.0,
        le=300.0,
        description="Request timeout in seconds",
    )
    follow_redirects: bool = Field(default=True, description="Follow HTTP redirects")
    trust_env: bool = True
    max_keepalive_connections: int = Field(default=20, ge=0)

    @classmethod
    def from_env(cls) -> ModelHttpClientConfig:
        """Create config from HTTP_* environment variables.

        Raises:
            ValueError: If numeric env vars contain non-numeric values.
        """
        try:
            return cls(
                timeout_seconds=float(os.getenv("HTTP_CLIENT_TIMEOUT_SECONDS", "30.0")),
                follow_redirects=os.getenv(
                    "HTTP_CLIENT_FOLLOW_REDIRECTS", "true"
                ).lower()
                in _TRUTHY_VALUES,
            )
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid HTTP client configuration: {e}") from e


__all__ = ["ModelHttpClientConfig"]
