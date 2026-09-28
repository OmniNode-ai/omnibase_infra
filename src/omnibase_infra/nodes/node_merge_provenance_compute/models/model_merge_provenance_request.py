# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Input of the merge-provenance compute node (OMN-19927)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_merge_provenance_observe_effect.models.model_merge_provenance_observation import (
    ModelMergeProvenanceObservation,
)


class ModelMergeProvenanceRequest(BaseModel):
    """A commit, and the merge-group observation read for it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    repository: str = Field(description="owner/name of the repository pushed to.")
    sha: str = Field(description="The pushed commit.")
    observation: ModelMergeProvenanceObservation = Field(
        description="What node_merge_provenance_observe_effect read for the sha."
    )


__all__: list[str] = ["ModelMergeProvenanceRequest"]
