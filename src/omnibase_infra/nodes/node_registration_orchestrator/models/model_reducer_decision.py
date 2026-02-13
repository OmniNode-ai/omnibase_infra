# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Reducer Decision Model.

Frozen Pydantic model returned by all ``decide_*`` methods on
RegistrationReducerService. Encapsulates the pure outcome of a reducer
decision — events to publish, intents for the effect layer, and the
target FSM state — without performing any I/O.

This model is the single return type for all four reducer decision methods:
    - decide_introspection
    - decide_ack
    - decide_heartbeat
    - decide_timeout

Related Tickets:
    - OMN-888 (C1): Registration Orchestrator
    - OMN-889 (D1): Registration Reducer
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.models.reducer.model_intent import ModelIntent
from omnibase_infra.enums import EnumRegistrationState


class ModelReducerDecision(BaseModel):
    """Outcome of a reducer decision -- pure data, no side effects.

    Attributes:
        action: Whether to emit events/intents or do nothing.
        new_state: Target FSM state if action=emit, None if no state change.
        events: Events to publish to the event bus.
        intents: Intents for the effect layer to execute.
        reason: Human-readable explanation for logging and debugging.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    action: Literal["emit", "no_op"] = Field(
        ...,
        description="Whether to emit events/intents or do nothing",
    )
    new_state: EnumRegistrationState | None = Field(
        default=None,
        description="Target FSM state if action=emit, None if no state change",
    )
    events: tuple[BaseModel, ...] = Field(
        default_factory=tuple,
        description="Events to publish",
    )
    intents: tuple[ModelIntent, ...] = Field(
        default_factory=tuple,
        description="Intents for effect layer",
    )
    reason: str = Field(
        ...,
        description="Human-readable explanation for logging",
    )


__all__: list[str] = ["ModelReducerDecision"]
