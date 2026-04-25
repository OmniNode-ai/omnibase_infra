# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed wrapper for ONEX service contract YAMLs.

First typed wrapper: feature_flags stays dict[str, bool] for now.
Tightening into per-flag submodels is tracked as a follow-up.

Note: Named ``ModelFeatureFlagContract`` rather than ``ModelServiceContract``
to satisfy the class_anti_pattern validator which blocks the word 'Service'
in class names (use specific domain terminology instead).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelFeatureFlagContract(BaseModel):
    """Typed representation of a feature-flag contract YAML (runtime.contract.yaml, event_bus.contract.yaml).

    First typed wrapper — not final decomposition. The ``feature_flags`` field
    remains ``dict[str, bool]`` for compatibility with existing call sites.
    Tightening into an explicit per-flag submodel with typed fields is a
    follow-up tracked separately.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(..., description="Service contract identifier")
    description: str = Field(
        default="", description="Human-readable description of this contract"
    )
    feature_flags: dict[str, bool] = Field(
        default_factory=dict,
        description="Named boolean feature flags declared by this service contract",
    )
