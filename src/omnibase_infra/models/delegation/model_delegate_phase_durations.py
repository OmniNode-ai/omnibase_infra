# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Completed phase durations for a delegation's receipt (OMN-19452).

A total duration hides whether startup, a broker operation, or terminal
waiting accounted for the latency. Each phase therefore carries its own
measurement, with ``None`` preserving the absence of a completed span and
``0.0`` recording a phase that completed without measurable elapsed time.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ModelDelegatePhaseDurations"]


class ModelDelegatePhaseDurations(BaseModel):
    """Nonnegative seconds per phase, or absence of a completed measurement."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    startup_seconds: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Seconds from entry of run_delegate to the start of the locus probe, "
            "including the drift guard, lane and credential resolution, and payload write."
        ),
    )
    locus_probe_seconds: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Seconds spent in the pre-dispatch probe that proves a deployed "
            "orchestrator is consuming the command topic."
        ),
    )
    bus_connect_seconds: float | None = Field(
        default=None,
        ge=0.0,
        description="Seconds spent in bus.start().",
    )
    reply_subscribe_seconds: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Seconds summed across every bus.subscribe the runtime made before "
            "publishing, including the reply/terminal subscription and handler "
            "subscriptions when the run hosts handlers in-process."
        ),
    )
    publish_seconds: float | None = Field(
        default=None,
        ge=0.0,
        description="Seconds spent in bus.publish of the command.",
    )
    terminal_wait_seconds: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Seconds from the publish returning to the runtime closing the bus, "
            "after a terminal is received or the wait times out."
        ),
    )
