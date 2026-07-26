# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Recommended remediation action (OMN-13942) -- recorded/surfaced only.

Nothing in Increment 1 executes a ``ModelRecommendedAction``. This model is
the typed hand-off point for a future Increment 2 remediation effect.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_runner_fleet_health_compute.models.enum_recommended_action_type import (
    EnumRecommendedActionType,
)


class ModelRecommendedAction(BaseModel):
    """A single recommended (never executed) remediation action."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    action_type: EnumRecommendedActionType = Field(
        ..., description="What kind of remediation is recommended."
    )
    target_id: str = Field(
        ...,
        description="Runner name, run ID, or queue-entry identifier the action targets.",
    )
    reason: str = Field(
        ..., description="Human-readable justification for the recommendation."
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Classifier confidence in this recommendation."
    )


__all__ = ["ModelRecommendedAction"]
