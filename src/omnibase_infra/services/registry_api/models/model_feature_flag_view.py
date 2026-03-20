# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Feature flag view model for Registry API responses.

OMN-5579: Aggregated feature flag view across all registered nodes.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.enums.enum_feature_flag_category import EnumFeatureFlagCategory


class ModelFeatureFlagView(BaseModel):
    """Aggregated view of a single feature flag across all declaring nodes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(description="Flag identifier")
    default_value: bool = Field(description="Contract-declared default")
    requested_value: bool | None = Field(
        default=None,
        description="Explicitly requested value (None if never toggled)",
    )
    process_value: bool | None = Field(
        default=None,
        description="Current value across nodes (None if mixed)",
    )
    effective_value: bool | None = Field(
        default=None,
        description="Always None in MVP — reserved for control-plane resolution",
    )
    effective_value_status: str = Field(
        default="deferred",
        description="Resolution status for effective_value",
    )
    state_alignment: str = Field(
        description="'aligned' or 'requested_differs_from_process'",
    )
    value_source: str = Field(
        default="default",
        description="How the current value was resolved",
    )
    description: str = Field(default="")
    category: EnumFeatureFlagCategory = Field(
        default=EnumFeatureFlagCategory.GENERAL,
    )
    env_var: str | None = Field(default=None)
    owner: str | None = Field(default=None)
    ownership_mode: str = Field(
        default="node_owned",
        description="'node_owned' or 'platform_housed'",
    )
    conflict_status: str = Field(
        default="clean",
        description="'clean', 'conflicted', or 'warning'",
    )
    conflict_details: list[str] | None = Field(default=None)
    declaring_nodes: list[str] = Field(default_factory=list)
    declaring_nodes_count: int = Field(default=0)
    writable: bool = Field(
        default=False,
        description="Whether Infisical is configured for writes",
    )
    last_changed_at: datetime | None = Field(default=None)


__all__ = ["ModelFeatureFlagView"]
