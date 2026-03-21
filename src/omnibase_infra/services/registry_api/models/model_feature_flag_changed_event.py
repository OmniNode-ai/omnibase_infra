# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Feature flag changed event model.

OMN-5580: Kafka event emitted when a feature flag is toggled via the
control-plane API.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelFeatureFlagChangedEvent(BaseModel):
    """Kafka event payload for feature flag state changes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # ONEX_EXCLUDE: pattern_validator - flag_name is a feature flag identifier, not an entity reference
    flag_name: str = Field(description="Flag identifier")
    enabled: bool = Field(description="New flag value")
    env_var: str | None = Field(
        default=None,
        description="Env var associated with the flag",
    )
    previous_value: bool | None = Field(
        default=None,
        description="Previous flag value (None if unknown)",
    )
    value_source: str = Field(
        default="control_plane",
        description="Source of the new value",
    )
    correlation_id: UUID = Field(description="Correlation ID for tracing")
    timestamp: datetime = Field(description="When the change occurred")


__all__ = ["ModelFeatureFlagChangedEvent"]
