# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""FSM intent for the coding-agent workflow (OMN-13247, plan §5.2).

The pure reducer returns ``(state, intents[])``. An intent is the reducer's
declaration of what should happen next — the orchestrator translates a live
intent into a bus command. On replay the reducer recomputes state but returns NO
live intent (``is_live=False``), so replay can never trigger a subprocess.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.coding_agent.enum_coding_agent_intent_kind import (
    EnumCodingAgentIntentKind,
)


class ModelCodingAgentIntent(BaseModel):
    """One next-step intent the reducer emits for the orchestrator to route."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID = Field(..., description="Workflow run correlation id.")
    kind: EnumCodingAgentIntentKind = Field(..., description="Next action requested.")
    is_live: bool = Field(
        default=True,
        description="False on replay; the orchestrator must not act on a "
        "non-live intent. A pure reducer cannot itself perform I/O.",
    )


__all__: list[str] = ["ModelCodingAgentIntent"]
