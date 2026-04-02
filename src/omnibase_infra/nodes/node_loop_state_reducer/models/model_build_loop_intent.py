# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Build loop intent model emitted by the reducer.

Related:
    - OMN-7311: ModelBuildLoopState foundation models
    - OMN-5113: Autonomous Build Loop epic
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums.enum_build_loop_intent_type import EnumBuildLoopIntentType
from omnibase_infra.enums.enum_build_loop_phase import EnumBuildLoopPhase


class ModelBuildLoopIntent(BaseModel):
    """Intent emitted by the build loop reducer to drive orchestrator actions.

    The orchestrator consumes these intents and routes them to the
    appropriate effect or compute node.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    intent_type: EnumBuildLoopIntentType = Field(
        ..., description="The intent type determining which node to invoke."
    )
    correlation_id: UUID = Field(
        ..., description="Cycle correlation ID for tracing."
    )
    cycle_number: int = Field(
        ..., ge=0, description="Current cycle number."
    )
    from_phase: EnumBuildLoopPhase = Field(
        ..., description="Phase that produced this intent."
    )
    payload: dict[str, str | int | float | bool | None] = Field(
        default_factory=dict,
        description="Optional key-value payload for the target node.",
    )


__all__: list[str] = ["ModelBuildLoopIntent"]
