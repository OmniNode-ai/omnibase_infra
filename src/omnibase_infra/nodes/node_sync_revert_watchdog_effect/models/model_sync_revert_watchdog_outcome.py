# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Per-ticket outcome record for the sync-revert watchdog sweep."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_sync_revert_watchdog_effect.models.enum_sync_revert_watchdog_decision import (
    EnumSyncRevertWatchdogDecision,
)


class ModelSyncRevertWatchdogOutcome(BaseModel):
    """One scanned ticket's terminal sweep decision."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    ticket_id: str = Field(
        default="", description="Linear ticket identifier, e.g. OMN-16536."
    )
    decision: EnumSyncRevertWatchdogDecision = Field(
        ..., description="Terminal classification for this ticket."
    )
    reason: str = Field(default="", description="Human-readable explanation.")
    reverted_at: str = Field(
        default="",
        description="ISO-8601 createdAt of the detected revert history entry, if any.",
    )
    from_state_name: str = Field(
        default="",
        description=(
            "Descriptive label for the workflow state the ticket reverted "
            "FROM (e.g. 'Done') — audit-trail display text, not a foreign "
            "key; the state itself is resolved and validated live by id "
            "internally at the point of mutation, never round-tripped "
            "through this outcome record."
        ),
    )
    to_state_name: str = Field(
        default="",
        description=(
            "Descriptive label for the workflow state the ticket reverted "
            "TO (e.g. 'In Progress') — audit-trail display text only."
        ),
    )
    bot_actor_type: str = Field(
        default="",
        description="Linear botActor.type on the detected revert entry (e.g. 'github').",
    )
    linear_comment_posted: bool = Field(
        default=False, description="Whether a diagnosis comment was posted."
    )
    applied: bool = Field(
        default=False,
        description="True only when a real Linear mutation was made (apply=True run).",
    )


__all__ = ["ModelSyncRevertWatchdogOutcome"]
