# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Result model for completed scope-check workflow."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_scope_workflow_orchestrator.models.enum_scope_check_status import (
    EnumScopeCheckStatus,
)


class ModelScopeCheckResult(BaseModel):
    """Final result emitted when scope-check workflow completes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID = Field(
        ..., description="Correlation ID from the originating command."
    )
    manifest_path: str = Field(
        ..., description="Path where the scope manifest was written."
    )
    files_count: int = Field(default=0, description="Number of files in scope.")
    directories_count: int = Field(
        default=0, description="Number of directories in scope."
    )
    repos_count: int = Field(default=0, description="Number of repos in scope.")
    systems_count: int = Field(default=0, description="Number of systems in scope.")
    status: EnumScopeCheckStatus = Field(
        default=EnumScopeCheckStatus.COMPLETE,
        description="Workflow completion status.",
    )
