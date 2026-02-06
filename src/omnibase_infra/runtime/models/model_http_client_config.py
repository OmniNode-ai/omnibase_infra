# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""HTTP client configuration.

Part of OMN-1976: Contract dependency materialization.
"""

from __future__ import annotations

import os

from pydantic import BaseModel, ConfigDict, Field

_TRUTHY_VALUES = frozenset({"true", "1", "yes", "on"})


class ModelHttpClientConfig(BaseModel):
    """HTTP client configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Request timeout in seconds",
    )
    follow_redirects: bool = Field(
        default=True,
        description="Follow HTTP redirects",
    )

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
            msg = f"Invalid HTTP_CLIENT_* environment variable: {e}"
            raise ValueError(msg) from e


__all__ = ["ModelHttpClientConfig"]
