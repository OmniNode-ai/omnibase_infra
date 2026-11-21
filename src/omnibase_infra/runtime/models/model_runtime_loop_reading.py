# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One window's event-loop reading for one runtime member (OMN-26249).

A stalled event loop is invisible to a health probe that runs on the same loop:
the probe is late, not wrong. This model is what a member publishes about its
own loop every window, so a stall reads as a number (the maximum gap, in
milliseconds) instead of as the absence of a number.

Every field is required. A reading that could leave out the tick lag or the
abandoned count would let a reader mistake "not measured" for "zero", so the
two ways a member may have no tick lag are stated, not defaulted:

- ``not_a_tick_subscriber``: the member does not consume the runtime tick, so
  tick lag does not apply to it.
- ``no_tick_consumed_yet``: the member subscribes to the tick and has not
  consumed one yet. This is NOT the not-applicable case; a verdict reading the
  topic must tell the two apart.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

TickLagAbsentReason = Literal["not_a_tick_subscriber", "no_tick_consumed_yet"]


class ModelRuntimeLoopReading(BaseModel):
    """Event-loop lag, tick lag and abandoned-dispatch count for one window."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    member_id: str = Field(
        ...,
        min_length=1,
        description="Identity of the runtime member that took the reading.",
    )
    host: str = Field(
        ...,
        min_length=1,
        description="Host the member runs on.",
    )
    window_start: datetime = Field(
        ...,
        description="Start of the window this reading covers (UTC, timezone-aware).",
    )
    window_end: datetime = Field(
        ...,
        description="End of the window this reading covers (UTC, timezone-aware).",
    )
    max_gap_ms: float = Field(
        ...,
        ge=0.0,
        description=(
            "Largest amount, in milliseconds, by which the sampler's short sleep "
            "returned late within the window. A loop blocked for 0.6 s reads at "
            "least 480."
        ),
    )
    tick_lag_ms: float | None = Field(
        ...,
        ge=0.0,
        description=(
            "Consume time minus the newest tick's scheduled_at, in milliseconds. "
            "None only together with tick_lag_absent_reason."
        ),
    )
    tick_lag_absent_reason: TickLagAbsentReason | None = Field(
        ...,
        description=(
            "Why tick_lag_ms is None. None only when tick_lag_ms is a number."
        ),
    )
    abandoned_count: int = Field(
        ...,
        ge=0,
        description=(
            "Dispatches the event bus abandoned at their deadline that have not "
            "returned yet (dispatch_deadline_status().orphaned_dispatches, "
            "OMN-19355)."
        ),
    )

    @model_validator(mode="after")
    def _window_and_tick_lag_are_coherent(self) -> Self:
        if self.window_start.tzinfo is None or self.window_end.tzinfo is None:
            raise ValueError("window_start and window_end must be timezone-aware")
        if self.window_end <= self.window_start:
            raise ValueError("window_end must be after window_start")
        if (self.tick_lag_ms is None) == (self.tick_lag_absent_reason is None):
            raise ValueError(
                "exactly one of tick_lag_ms and tick_lag_absent_reason must be set"
            )
        return self


__all__: list[str] = ["ModelRuntimeLoopReading", "TickLagAbsentReason"]
