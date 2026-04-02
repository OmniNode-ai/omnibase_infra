# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Health probe request model."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_model_health_effect.models.model_health_probe_target import (
    ModelHealthProbeTarget,
)


class ModelHealthRequest(BaseModel):
    """Request to probe model endpoints for health/latency."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Workflow correlation ID.")
    targets: tuple[ModelHealthProbeTarget, ...] = Field(
        ..., description="Model endpoints to probe."
    )
    timeout_ms: int = Field(
        default=5000, description="Per-endpoint probe timeout in ms."
    )
