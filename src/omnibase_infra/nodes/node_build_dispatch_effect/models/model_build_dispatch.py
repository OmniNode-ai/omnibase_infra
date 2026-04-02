# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Re-export shim for backwards compatibility.

Models have been split into individual files per ONEX architecture rules.
Import directly from the individual model files instead.
"""

from __future__ import annotations

from omnibase_infra.nodes.node_build_dispatch_effect.models.model_build_dispatch_input import (
    ModelBuildDispatchInput,
)
from omnibase_infra.nodes.node_build_dispatch_effect.models.model_build_dispatch_outcome import (
    ModelBuildDispatchOutcome,
)
from omnibase_infra.nodes.node_build_dispatch_effect.models.model_build_dispatch_result import (
    ModelBuildDispatchResult,
)
from omnibase_infra.nodes.node_build_dispatch_effect.models.model_build_target import (
    ModelBuildTarget,
)

__all__: list[str] = [
    "ModelBuildDispatchInput",
    "ModelBuildDispatchOutcome",
    "ModelBuildDispatchResult",
    "ModelBuildTarget",
]
