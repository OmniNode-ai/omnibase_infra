# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""What the observe effect saw for one commit (OMN-19927)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.nodes.node_merge_provenance_observe_effect.models.model_merge_group_run_fact import (
    ModelMergeGroupRunFact,
)


class ModelMergeProvenanceObservation(BaseModel):
    """The merge-group runs recorded for one sha, or the reason none could be read.

    ``read_ok=False`` carries an empty ``runs`` and a ``read_error``. An empty
    ``runs`` with ``read_ok=True`` means the read succeeded and found no
    merge-group run: the two must never be confused, because the first is
    UNDECIDABLE and the second is UNVALIDATED.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    repository: str = Field(description="owner/name of the repository.")
    sha: str = Field(description="The commit observed.")
    workflow_path: str = Field(description="Workflow whose runs were read.")
    summary_job: str = Field(description="Summary job looked up in each run.")
    read_ok: bool = Field(description="True only when every read completed.")
    read_error: str | None = Field(
        default=None, description="Why the read failed; required when read_ok is False."
    )
    runs: tuple[ModelMergeGroupRunFact, ...] = Field(
        default=(), description="Merge-group runs of the workflow for this sha."
    )
    observed_at: str = Field(description="ISO-8601 UTC time of the read.")

    @model_validator(mode="after")
    def _failed_read_has_a_reason_and_no_runs(self) -> ModelMergeProvenanceObservation:
        if not self.read_ok and (not self.read_error or self.runs):
            raise ValueError(
                "a failed read carries a read_error and no runs; an empty list "
                "must never stand in for an unread one"
            )
        return self


__all__: list[str] = ["ModelMergeProvenanceObservation"]
