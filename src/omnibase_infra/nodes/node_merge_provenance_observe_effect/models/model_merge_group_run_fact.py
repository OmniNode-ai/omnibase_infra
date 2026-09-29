# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A merge-group run of the watched workflow, with its summary job (OMN-19927)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_merge_provenance_observe_effect.models.enum_actions_execution_status import (
    EnumActionsExecutionStatus,
)


class ModelMergeGroupRunFact(BaseModel):
    """One run the observe effect read, joined to its summary job.

    ``summary_job_status`` and ``summary_job_conclusion`` are both None when
    the run's latest attempt has no job of the requested name. That is an
    observed absence (the job was renamed, or the run never reached it), which
    the compute node grades UNDECIDABLE rather than as a missing validation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: int = Field(description="Actions run id.")
    run_attempt: int = Field(ge=1, description="Latest attempt number of the run.")
    event: str = Field(description="Triggering event; merge_group for a queue run.")
    head_sha: str = Field(description="The commit the run executed against.")
    head_branch: str = Field(description="The queue ref, gh-readonly-queue/<base>/...")
    workflow_path: str = Field(description="Repository path of the workflow file.")
    run_status: EnumActionsExecutionStatus = Field(description="Run status.")
    run_conclusion: str | None = Field(default=None, description="Run conclusion.")
    summary_job_status: EnumActionsExecutionStatus | None = Field(
        default=None, description="Summary job status; None when the job is absent."
    )
    summary_job_conclusion: str | None = Field(
        default=None,
        description="Summary job conclusion; None when absent or not completed.",
    )


__all__: list[str] = ["ModelMergeGroupRunFact"]
