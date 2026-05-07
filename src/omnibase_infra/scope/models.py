# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Scope contract model imports for the overlay loader.

``omnibase-core==0.40.1`` does not publish the ``models.scope`` package yet,
while editable core checkouts already contain it. Keep the loader pointed at
core when available, and provide narrowly compatible contract models until the
next core release is available in CI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_infra.scope.fallback_models import (
        ModelArtifactEnforcement,
        ModelEnforcementScope,
    )
else:
    try:  # pragma: no cover - exercised when the next omnibase-core release lands.
        from omnibase_core.models.scope.model_artifact_enforcement import (
            ModelArtifactEnforcement,
        )
        from omnibase_core.models.scope.model_enforcement_scope import (
            ModelEnforcementScope,
        )
    except ModuleNotFoundError:
        from omnibase_infra.scope.fallback_models import (
            ModelArtifactEnforcement,
            ModelEnforcementScope,
        )


__all__ = ["ModelArtifactEnforcement", "ModelEnforcementScope"]
