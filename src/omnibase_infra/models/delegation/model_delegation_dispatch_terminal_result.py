# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dispatch terminal node's answer: the caller's result and the run's steps.

OMN-19972. ``result`` is the flattened delegate-skill shape the dispatch port
returns to its caller. ``run_steps`` are the run's numbered step events in step
order, for ``run_step_topic``; the port wraps and sends them, because a handler
holds no event bus and no envelope. Both are empty when no steps were asked for
or none can be sent.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.types import JsonType
from omnibase_infra.models.delegation.model_delegation_run_step_event import (
    ModelDelegationRunStepEvent,
)


class ModelDelegationDispatchTerminalResult(BaseModel):
    """The caller's flattened result and the run's numbered steps to send."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    result: dict[str, JsonType]
    run_step_topic: str | None = None
    run_steps: tuple[ModelDelegationRunStepEvent, ...] = Field(default_factory=tuple)


__all__ = ["ModelDelegationDispatchTerminalResult"]
