# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Configuration model for HandlerQdrant."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_vector_store_effect.contract_descriptor import (
    contract_qdrant_url,
)


class ModelQdrantHandlerConfig(BaseModel):
    """Configuration for Qdrant handler initialization.

    Attributes:
        url: Qdrant server URL (resolved via the vector-store node's
            overlay-declared ``descriptor.qdrant_url``; fails closed on unset
            ``QDRANT_URL`` rather than silently defaulting to localhost)
        api_key: Optional API key for authentication
        timeout_seconds: Request timeout in seconds
        prefer_grpc: Use gRPC instead of HTTP for better performance
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    url: str = Field(
        default_factory=contract_qdrant_url,
        description="Qdrant server URL (overlay-resolved via descriptor.qdrant_url)",
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
