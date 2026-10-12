# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Typed request payload for the impact analyzer COMPUTE handler.

The canonical definition-B handler entrypoint takes a SINGLE typed request.
Impact scoring needs two inputs -- the change trigger and the artifact registry
to score against -- so both are bound into one frozen request model that the
shared runtime adapter passes to ``HandlerImpactAnalysis.handle``.

On the bus the node's input is the ``ModelUpdateTrigger`` that the change detector
publishes to ``onex.evt.artifact.change-detected.v1`` (OMN-21053). A bare trigger
is therefore accepted as the request's ``trigger``, and the registry defaults to
the one shipped with the package; an in-process caller may still bind both.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.nodes.node_artifact_change_detector_effect.models.model_update_trigger import (
    ModelUpdateTrigger,
)
from omnibase_infra.registry.loader import load_packaged_artifact_registry
from omnibase_infra.registry.models.model_artifact_registry import ModelArtifactRegistry


class ModelImpactAnalysisRequest(BaseModel):
    """Single typed request for canonical def-B impact analysis.

    Attributes:
        trigger: The change trigger with file list and trigger type.
        registry: The artifact registry to score the trigger against.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    trigger: ModelUpdateTrigger
    registry: ModelArtifactRegistry = Field(
        default_factory=load_packaged_artifact_registry
    )

    @model_validator(mode="before")
    @classmethod
    def _bind_bare_trigger(cls, data: object) -> object:
        """Read a wire payload that is the trigger itself as ``trigger=<payload>``."""
        if isinstance(data, Mapping) and "trigger" not in data and "trigger_id" in data:
            return {"trigger": data}
        return data


__all__: list[str] = ["ModelImpactAnalysisRequest"]
