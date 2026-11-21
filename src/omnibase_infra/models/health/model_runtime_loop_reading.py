# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One window's event-loop health reading of one runtime member (OMN-26249).

A runtime member whose event loop is stalled cannot report that it is stalled
through any path that runs on that loop, so the stall has to be read as a
number after the fact. Each window the member publishes the longest time its
loop was late to a short scheduled wake-up, how late the newest runtime tick
was when the member consumed it, and how many dispatches it abandoned at their
deadline (OMN-19355).

Every field is required. A reading that omits one is not a smaller reading, it
is a reading nobody can tell apart from a member that did not report. The tick
lag is the one field that may be absent in value: a member that does not
subscribe to the tick has no tick lag to report, and it says so with a reason,
which a verdict can tell apart from a missing reading of an expected
subscriber.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelRuntimeLoopReading(BaseModel):
    """Event-loop lag, tick lag and abandoned dispatches of one member in one window."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Identifies this reading.")
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
        ..., description="Start of the window the reading covers (UTC)."
    )
    window_end: datetime = Field(
        ..., description="End of the window the reading covers (UTC)."
    )
    max_loop_gap_ms: float = Field(
        ...,
        ge=0.0,
        description=(
            "Longest time in the window the event loop was late to a short "
            "scheduled wake-up, in milliseconds."
        ),
    )
    tick_subscriber: bool = Field(
        ...,
        description="Whether this member subscribes to the runtime tick topic.",
    )
    tick_lag_ms: float | None = Field(
        ...,
        description=(
            "Consume time minus the tick's scheduled_at for the newest tick "
            "consumed in the window, in milliseconds. None when "
            "tick_lag_not_applicable_reason is set."
        ),
    )
    tick_lag_not_applicable_reason: str | None = Field(
        ...,
        min_length=1,
        description=(
            "Why there is no tick lag: the member does not subscribe to the "
            "tick, or it subscribes and consumed none in the window. None "
            "when tick_lag_ms is set."
        ),
    )
    abandoned_dispatches: int = Field(
        ...,
        ge=0,
        description=(
            "Dispatches this member abandoned at their deadline that have not "
            "returned yet, from the dispatch-deadline status (OMN-19355)."
        ),
    )

    @model_validator(mode="after")
    def _validate_reading(self) -> ModelRuntimeLoopReading:
        if (self.tick_lag_ms is None) == (self.tick_lag_not_applicable_reason is None):
            raise ValueError(
                "exactly one of tick_lag_ms and tick_lag_not_applicable_reason "
                "must be set"
            )
        if not self.tick_subscriber and self.tick_lag_ms is not None:
            raise ValueError("a member that does not subscribe to the tick has no lag")
        if self.window_end <= self.window_start:
            raise ValueError("window_end must be after window_start")
        return self


__all__: list[str] = ["ModelRuntimeLoopReading"]
