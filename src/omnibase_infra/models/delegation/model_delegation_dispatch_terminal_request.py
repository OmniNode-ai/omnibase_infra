# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""What the dispatch terminal node needs once the broker has answered a run.

OMN-19972. ``RuntimeDelegationDispatchPort`` hands the broker's terminal to
``node_delegation_dispatch_terminal_effect``, which turns it into the caller's
result and, for a run the caller waited on, publishes the run's numbered steps.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from omnibase_core.models.dispatch.model_dispatch_bus_terminal_result import (
    ModelDispatchBusTerminalResult,
)


class ModelDelegationDispatchTerminalRequest(BaseModel):
    """One dispatched run's terminal, with the facts its steps are numbered from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    terminal: ModelDispatchBusTerminalResult
    correlation_id: UUID
    task_type: str
    tenant_id: str | None = None
    started_at: AwareDatetime
    started_monotonic: float = Field(ge=0.0)
    engine_package: str = Field(min_length=1)
    publish_steps: bool


__all__ = ["ModelDelegationDispatchTerminalRequest"]
