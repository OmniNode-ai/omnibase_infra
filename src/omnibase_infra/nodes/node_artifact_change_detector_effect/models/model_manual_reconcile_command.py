# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Domain model for manual artifact reconciliation commands."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelManualReconcileCommand(BaseModel):
    """Command to trigger a full manual artifact reconciliation.

    Published to ``onex.cmd.artifact.reconcile.v1`` by the CLI command
    (``omni-infra artifact-reconcile``) and consumed by HandlerManualTrigger.

    When ``changed_files`` is empty, HandlerManualTrigger builds a trigger
    with ``trigger_type="manual_plan_request"`` and empty ``changed_files``.
    The downstream COMPUTE node treats empty changed_files + manual trigger
    as a full-repo reconciliation (matches all artifacts).

    Related Tickets:
        - OMN-3940: Task 5 — Change Detector EFFECT Node
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    command_id: UUID = Field(description="Unique command identifier")
    source_repo: str = Field(
        description="Repository to reconcile, e.g. 'omnibase_infra'"
    )
    changed_files: list[str] = Field(
        default_factory=list,
        description=(
            "Optional list of file paths to restrict reconciliation scope. "
            "Empty means full-repo reconciliation."
        ),
    )
    actor: str | None = Field(
        default=None,
        description="User or agent that issued the command",
    )
    reason: str = Field(
        default="",
        description="Human-readable reason for the manual trigger",
    )


__all__ = ["ModelManualReconcileCommand"]
