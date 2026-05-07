# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Fallback scope contract models for packaged omnibase-core compatibility."""

from omnibase_infra.scope.fallback_models.model_artifact_enforcement import (
    ModelArtifactEnforcement,
)
from omnibase_infra.scope.fallback_models.model_enforcement_scope import (
    ModelEnforcementScope,
)

__all__ = ["ModelArtifactEnforcement", "ModelEnforcementScope"]
