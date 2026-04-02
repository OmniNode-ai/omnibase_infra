# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Aggregated health snapshot across all probed endpoints."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_model_health_effect.models.model_endpoint_health import (
    ModelEndpointHealth,
)


class ModelHealthSnapshot(BaseModel):
    """Aggregated health snapshot across all probed endpoints."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Workflow correlation ID.")
    endpoints: tuple[ModelEndpointHealth, ...] = Field(
        default_factory=tuple, description="Per-endpoint health results."
    )
    success: bool = Field(
        default=True, description="Whether the overall probe succeeded."
    )
    error_message: str = Field(default="", description="Error if overall probe failed.")
