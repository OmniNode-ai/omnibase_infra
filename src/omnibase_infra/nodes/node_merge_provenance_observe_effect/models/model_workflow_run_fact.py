# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One workflow run as the reader returned it (OMN-19927)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_merge_provenance_observe_effect.models.enum_actions_execution_status import (
    EnumActionsExecutionStatus,
)


class ModelWorkflowRunFact(BaseModel):
    """A workflow run recorded for a sha, facts only."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: int = Field(description="Actions run id.")
    run_attempt: int = Field(ge=1, description="Latest attempt number of the run.")
    event: str = Field(description="Triggering event, e.g. merge_group.")
    head_sha: str = Field(description="The commit the run executed against.")
    head_branch: str = Field(description="Ref the run executed on.")
    workflow_path: str = Field(description="Repository path of the workflow file.")
    status: EnumActionsExecutionStatus = Field(description="Run status.")
    conclusion: str | None = Field(
        default=None, description="Run conclusion; None while not completed."
    )


__all__: list[str] = ["ModelWorkflowRunFact"]
