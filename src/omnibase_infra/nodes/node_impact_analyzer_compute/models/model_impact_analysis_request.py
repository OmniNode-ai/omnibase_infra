# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Typed request payload for the impact analyzer COMPUTE handler.

The canonical definition-B handler entrypoint takes a SINGLE typed request.
Impact scoring needs two inputs -- the change trigger and the artifact registry
to score against -- so both are bound into one frozen request model that the
shared runtime adapter passes to ``HandlerImpactAnalysis.handle``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from omnibase_infra.nodes.node_artifact_change_detector_effect.models.model_update_trigger import (
    ModelUpdateTrigger,
)
from omnibase_infra.registry.models.model_artifact_registry import ModelArtifactRegistry


class ModelImpactAnalysisRequest(BaseModel):
    """Single typed request for canonical def-B impact analysis.

    Attributes:
        trigger: The change trigger with file list and trigger type.
        registry: The artifact registry to score the trigger against.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    trigger: ModelUpdateTrigger
    registry: ModelArtifactRegistry


__all__: list[str] = ["ModelImpactAnalysisRequest"]
