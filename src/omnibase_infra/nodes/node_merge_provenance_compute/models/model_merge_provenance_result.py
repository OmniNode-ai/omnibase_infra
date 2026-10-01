# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Output of the merge-provenance compute node (OMN-19927)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_merge_provenance_compute.models.enum_merge_provenance_reason import (
    EnumMergeProvenanceReason,
)
from omnibase_infra.nodes.node_merge_provenance_compute.models.enum_merge_provenance_verdict import (
    EnumMergeProvenanceVerdict,
)


class ModelMergeProvenanceResult(BaseModel):
    """The provenance verdict for one commit, with the run ids it was read from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    repository: str = Field(description="owner/name of the repository.")
    sha: str = Field(description="The commit graded.")
    verdict: EnumMergeProvenanceVerdict = Field(description="The verdict.")
    reason: EnumMergeProvenanceReason = Field(description="Why.")
    reason_detail: str = Field(description="Human-readable detail of the reason.")
    run_ids_read: tuple[int, ...] = Field(
        default=(), description="Every merge-group run id the verdict considered."
    )
    validating_run_id: int | None = Field(
        default=None, description="The run whose summary succeeded, when VALIDATED."
    )
    forces_full_suite: bool = Field(
        description="True for UNVALIDATED and UNDECIDABLE: the push runs the full suite."
    )


__all__: list[str] = ["ModelMergeProvenanceResult"]
