# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed runtime health details for local runtime ingress."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelLocalRuntimeIngressHealth(BaseModel):
    """Operational health details for the local runtime ingress."""

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    enabled: bool = Field(..., description="Whether local ingress is configured.")
    running: bool = Field(..., description="Whether the socket server is serving.")
    socket_path: str | None = Field(
        default=None,
        description="Bound socket path when local ingress is configured.",
    )
    route_count: int = Field(
        default=0,
        ge=0,
        description="Number of local node routes exposed through the ingress.",
    )
    active_packages: tuple[str, ...] = Field(
        default=(),
        description="Resolved runtime packages scanned for ingress routes.",
    )
    request_model: str = Field(
        default="ModelLocalRuntimeIngressRequest",
        description="Stable request model name for client compatibility.",
    )
    response_model: str = Field(
        default="ModelLocalRuntimeIngressResponse",
        description="Stable response model name for client compatibility.",
    )


__all__ = ["ModelLocalRuntimeIngressHealth"]
