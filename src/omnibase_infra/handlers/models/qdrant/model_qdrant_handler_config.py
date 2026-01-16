# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Configuration model for HandlerQdrant."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelQdrantHandlerConfig(BaseModel):
    """Configuration for Qdrant handler initialization.

    Attributes:
        url: Qdrant server URL (e.g., http://localhost:6333)
        api_key: Optional API key for authentication
        timeout_seconds: Request timeout in seconds
        prefer_grpc: Use gRPC instead of HTTP for better performance
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    url: str = Field(
        default="http://localhost:6333",
        description="Qdrant server URL",
    )
    api_key: str | None = Field(
        default=None,
        description="Optional API key for authentication",
    )
    timeout_seconds: float = Field(
        default=30.0,
        ge=0.1,
        le=3600.0,
        description="Request timeout in seconds",
    )
    prefer_grpc: bool = Field(
        default=False,
        description="Use gRPC instead of HTTP for better performance",
    )


__all__: list[str] = ["ModelQdrantHandlerConfig"]
