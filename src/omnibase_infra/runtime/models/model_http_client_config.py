# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""HTTP client configuration.

Part of OMN-1976: Contract dependency materialization.
"""

from __future__ import annotations

import os

from pydantic import BaseModel, ConfigDict, Field


class ModelHttpClientConfig(BaseModel):
    """HTTP client configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

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
        """Create config from HTTP_* environment variables."""
        return cls(
            timeout_seconds=float(os.getenv("HTTP_CLIENT_TIMEOUT_SECONDS", "30.0")),
            follow_redirects=os.getenv("HTTP_CLIENT_FOLLOW_REDIRECTS", "true").lower()
            == "true",
        )


__all__ = ["ModelHttpClientConfig"]
