# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One deploy agent reading for the wait-while-busy test (OMN-21866)."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.health.model_lab_lane_compose_busy_policy import (
    ModelLabLaneComposeBusyPolicy,
)


class ModelLabLaneComposeBusyRequest(BaseModel):
    """The agent fields the decision reads, and the contract's settle stages."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    state: str = Field(default="", description="The agent's /health state.")
    active_job: bool = Field(
        default=False, description="Whether /health names an active job."
    )
    settling_stage: str = Field(
        default="", description="last_result.settling_stage from /health."
    )
    shared_busy_reason: str = Field(
        default="",
        description="The action's shared busy test on the same reading; empty when not busy.",
    )
    policy: ModelLabLaneComposeBusyPolicy = Field(
        ..., description="The node contract's compose_busy_policy block."
    )

    @classmethod
    def from_agent_snapshot(
        cls, snapshot: Mapping[str, object], shared_busy_reason: str | None
    ) -> ModelLabLaneComposeBusyRequest:
        """Build from the resolve-lab-lane action's ``{health, queue}`` snapshot."""
        health = snapshot.get("health")
        health = health if isinstance(health, Mapping) else {}
        last = health.get("last_result")
        stage = last.get("settling_stage") if isinstance(last, Mapping) else None
        state = health.get("state")
        return cls(
            state=state if isinstance(state, str) else "",
            active_job=bool(health.get("active_job")),
            settling_stage=stage if isinstance(stage, str) else "",
            shared_busy_reason=shared_busy_reason or "",
            policy=ModelLabLaneComposeBusyPolicy.from_node_contract(),
        )


__all__ = ["ModelLabLaneComposeBusyRequest"]
