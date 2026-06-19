# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""FSM input event for the coding-agent workflow (OMN-13247, plan §5.2).

The reducer folds one of these events into the FSM state. The event kind
(``EnumCodingAgentEventKind``) carries the transition trigger.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.coding_agent.enum_cli_backend_status import (
    EnumCliBackendStatus,
)
from omnibase_infra.models.coding_agent.enum_coding_agent_event_kind import (
    EnumCodingAgentEventKind,
)


class ModelCodingAgentEvent(BaseModel):
    """One FSM input event folded by the reducer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID = Field(..., description="Workflow run correlation id.")
    kind: EnumCodingAgentEventKind = Field(..., description="Transition trigger.")
    error_class: EnumCliBackendStatus | None = Field(
        default=None,
        description="Failure class for INVOKE_FAILED events; None otherwise.",
    )
    error_message: str | None = Field(
        default=None, description="Human-readable reason for a failure/rejection."
    )
    is_replay: bool = Field(
        default=False,
        description="True when re-folding a historical event. Replay recomputes "
        "state + projection but the reducer issues NO live intent.",
    )


__all__: list[str] = ["ModelCodingAgentEvent"]
