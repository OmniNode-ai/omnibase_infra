# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Models for NodeMergeGateEffect."""

from __future__ import annotations

from omnibase_infra.nodes.node_merge_gate_effect.models.model_merge_gate_result import (
    ModelMergeGateResult,
)
from omnibase_infra.nodes.node_merge_gate_effect.models.model_merge_gate_violation import (
    ModelMergeGateViolation,
)

__all__: list[str] = ["ModelMergeGateResult", "ModelMergeGateViolation"]
