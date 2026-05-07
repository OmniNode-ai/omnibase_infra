# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Compatibility top-level scope contract model."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.scope.fallback_models.model_activation_scope import (
    ModelActivationScope,
)
from omnibase_infra.scope.fallback_models.model_applicability_scope import (
    ModelApplicabilityScope,
)
from omnibase_infra.scope.fallback_models.model_artifact_enforcement import (
    ModelArtifactEnforcement,
)
from omnibase_infra.scope.fallback_models.model_unavailable_behavior import (
    ModelUnavailableBehavior,
)


class ModelEnforcementScope(BaseModel):
    """Compatibility top-level scope contract model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    activation: ModelActivationScope = Field(default_factory=ModelActivationScope)
    applicability: ModelApplicabilityScope = Field(
        default_factory=ModelApplicabilityScope
    )
    enforcement: ModelArtifactEnforcement = Field(
        default_factory=ModelArtifactEnforcement
    )
    unavailable_behavior: ModelUnavailableBehavior = Field(
        default_factory=ModelUnavailableBehavior
    )


__all__ = ["ModelEnforcementScope"]
