# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Feature flag toggle result model for Registry API responses.

OMN-5580: Result of a feature flag toggle operation.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelFeatureFlagToggleResult(BaseModel):
    """Result of a feature flag toggle operation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # ONEX_EXCLUDE: pattern_validator - flag_name is a feature flag identifier, not an entity reference
    flag_name: str = Field(description="Name of the toggled flag")
    requested_value: bool = Field(description="Value requested by the caller")
    last_resolved_value: bool = Field(description="Value before toggle")
    outcome: str = Field(
        description=(
            "Toggle outcome: 'persisted_and_emitted', 'persist_failed', "
            "'persisted_emit_failed', or 'emit_succeeded_persist_skipped'"
        ),
    )
    message: str = Field(description="Human-readable outcome description")
    persistence_key: str | None = Field(
        default=None,
        description="Infisical key where the value was persisted",
    )
    event_id: UUID | None = Field(
        default=None,
        description="Correlation ID of the emitted Kafka event",
    )


__all__ = ["ModelFeatureFlagToggleResult"]
