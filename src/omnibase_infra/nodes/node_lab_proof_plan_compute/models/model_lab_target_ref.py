# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The primary target and sibling composition refs."""

from typing import Literal

from omnibase_infra.nodes.node_lab_proof_plan_compute.models.model_lab_source_ref import (
    ModelLabSourceRef,
)


class ModelLabTargetRef(ModelLabSourceRef):
    """Release or last accepted merge, with every staged sibling ref."""

    kind: Literal["release", "merge"]
    composition: tuple[ModelLabSourceRef, ...]
