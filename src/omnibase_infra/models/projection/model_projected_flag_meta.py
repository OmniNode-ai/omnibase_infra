# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Projected feature flag metadata model.

OMN-5578: Flat metadata for feature flags stored in registration projections.
Each flag's metadata is stored alongside its value for efficient querying
without requiring re-extraction from the contract.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.enums.enum_feature_flag_category import EnumFeatureFlagCategory


class ModelProjectedFlagMeta(BaseModel):
    """Metadata for a single projected feature flag.

    Stored in ``ModelRegistrationProjection.feature_flag_metadata`` keyed by
    flag name. Provides enough context for omnidash display and API responses
    without round-tripping to the contract.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    description: str = Field(
        default="",
        description="Human-readable description of what the flag controls.",
    )
    category: EnumFeatureFlagCategory = Field(
        default=EnumFeatureFlagCategory.GENERAL,
        description="Functional domain grouping.",
    )
    env_var: str | None = Field(
        default=None,
        description="Env var name for runtime override, if declared in contract.",
    )
    owner: str | None = Field(
        default=None,
        description="Free-text owner identifier (team or service name).",
    )
    value_source: str = Field(
        default="default",
        description="How the current value was resolved: 'default', 'env', or 'infisical'.",
    )
    platform_housed: bool = Field(
        default=False,
        description="Whether the flag is managed by the platform control plane.",
    )
    declaring_node_id: str | None = Field(
        default=None,
        description="Node ID that declared this flag (for cross-node dedup).",
    )


__all__ = ["ModelProjectedFlagMeta"]
