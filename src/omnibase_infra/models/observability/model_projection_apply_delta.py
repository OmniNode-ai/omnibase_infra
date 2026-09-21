# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Per-projection apply accounting for one closed window (OMN-18910).

What this measures that nothing else does
-----------------------------------------
Every pre-existing dimension answers whether a consumer is attached and
MOVING. ``ModelConsumerFlowDelta`` (OMN-16777) counts envelopes in, out, DLQ'd
and raised; none of those four is "a row landed". A consumer that refuses a
message still commits its offset. A projection that returns without writing
still commits its offset. A cache that drops a delta still consumed it. Lag was
zero throughout OMN-18880, OMN-18905 and OMN-18769, and all three ran for hours
behind green surfaces.

So this is the pair the runtime never held in one place: what a projection
CONSUMED, and what it WROTE, over the same window, taken at the one seam that
knows both — the projection dispatch arm in ``handler_wiring``.

Why it is not on the wire
-------------------------
Deliberately process-local. It is read by ``ServiceRuntimeHealthMonitor`` in
the same process that fills it, so it needs no envelope, no topic and no
consumer, and adding a field to the heartbeat's ``ModelConsumerFlowDelta``
would have made a cross-repo wire change out of an in-process reading. The
verdict this feeds IS published, on the health event that already exists.

Three counts and one gauge
--------------------------
``consumed``, ``upserted`` and ``refused_by_guard`` are per-window counts, reset
every drain. ``deltas_dropped_total`` is a cumulative GAUGE carried forward, not
a delta, because the fact being graded is accumulation across windows: a high
flat value is legitimate idempotence and a rising one is loss. Collapsing the
gauge to a per-window delta would erase exactly that distinction.

Related Tickets:
    - OMN-18910: this model (epic OMN-18906 AC-4)
    - OMN-16777: the sibling wire delta this deliberately does not extend
    - OMN-18992: the guard-refusal classification this counts the accumulation of
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelProjectionApplyDelta(BaseModel):
    """One projection's consume-versus-write accounting over one window."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    projection: str = Field(
        ...,
        min_length=1,
        description="The projection handler or contract node this accounts for",
    )
    topic: str = Field(
        ...,
        min_length=1,
        description="The source topic the events were consumed from",
    )
    consumed: int = Field(
        ...,
        ge=0,
        description="Envelopes dispatched to this projection's handler in the window",
    )
    upserted: int = Field(
        ...,
        ge=0,
        description="Rows the handler reported persisting in the window",
    )
    refused_by_guard: int = Field(
        default=0,
        ge=0,
        description=(
            "Zero-row writes the handler attributed to its ordering guard "
            "(OMN-18992). Correct behaviour, counted separately so it cannot be "
            "read as a write and cannot be read as a defect."
        ),
    )
    deltas_dropped_total: int = Field(
        default=0,
        ge=0,
        description=(
            "Cumulative deltas discarded for this projection since process "
            "start. A GAUGE, not a per-window count: the graded fact is whether "
            "it rises, never how large it is."
        ),
    )


__all__ = ["ModelProjectionApplyDelta"]
