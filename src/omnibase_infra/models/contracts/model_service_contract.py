# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed wrapper for ONEX service contract YAMLs.

First typed wrapper: feature_flags is list[ModelFeatureFlag] matching the real
YAML schema (list of {name, env_var, default, description} objects).

Note: Named ``ModelFeatureFlagContract`` rather than ``ModelServiceContract``
to satisfy the class_anti_pattern validator which blocks the word 'Service'
in class names (use specific domain terminology instead).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.contracts.model_feature_flag import ModelFeatureFlag


class ModelFeatureFlagContract(BaseModel):
    """Typed representation of a feature-flag contract YAML (runtime.contract.yaml, event_bus.contract.yaml).

    First typed wrapper — not final decomposition. The ``feature_flags`` field
    is a list of ``ModelFeatureFlag`` entries matching the real YAML schema.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(..., description="Contract identifier")
    description: str = Field(
        default="", description="Human-readable description of this contract"
    )
    feature_flags: list[ModelFeatureFlag] = Field(
        default_factory=list,
        description="Feature flags declared by this contract",
    )
