# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One job of a workflow run's latest attempt (OMN-19927)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_merge_provenance_observe_effect.models.enum_actions_execution_status import (
    EnumActionsExecutionStatus,
)


class ModelWorkflowJobFact(BaseModel):
    """A job's display name, status and conclusion, facts only."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(description="Job display name as the jobs API returns it.")
    status: EnumActionsExecutionStatus = Field(description="Job status.")
    conclusion: str | None = Field(
        default=None, description="Job conclusion; None while not completed."
    )


__all__: list[str] = ["ModelWorkflowJobFact"]
