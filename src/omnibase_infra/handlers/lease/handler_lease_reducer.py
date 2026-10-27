# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pure lease projection reducer (OMN-21424).

``delta(state, event) -> (new_state, intents)`` with no I/O. Each row per
``(kind, resource)`` is a deterministic commutative join: a higher fence wins
outright using exactly that event's data, without inheriting a lower fence's
holder; a lower fence is dropped. At equal fences, state is the maximum of the
total order ``held < released < expired`` and holder is the non-empty value,
or the lexicographically greater value if both are non-empty and differ.
The lease ID comes from the higher-ranked state; equal-ranked conflicting IDs
use the greater UUID, so even inconsistent replay data converges. Lower-ranked
events can still supply a missing holder without changing
state. An unchanged merged row emits no intent, so replaying events in any order
reaches the same rows.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.lease import (
    ModelLeaseProjectionRow,
    ModelLeaseReply,
    ModelPayloadLeaseProjectionUpsert,
)

LeaseProjectionState = Mapping[tuple[str, str], ModelLeaseProjectionRow]

_STATE: dict[str, Literal["held", "released", "expired"]] = {
    "lease-granted": "held",
    "lease-renewed": "held",
    "lease-released": "released",
    "lease-expired": "expired",
}
# Higher fences use exactly the event's data; lower fences are dropped.
# Equal fences join by the total order held < released < expired, retain the
# non-empty or lexicographically greater holder even from a lower-ranked event,
# and take the lease ID from the higher-ranked state, then the greater UUID
# when equal-ranked events disagree about the grant.
# An unchanged merged row emits no intent.
_RANK = {"held": 0, "released": 1, "expired": 2}


def reduce_lease_event(
    state: LeaseProjectionState, event: ModelLeaseReply
) -> tuple[
    dict[tuple[str, str], ModelLeaseProjectionRow],
    tuple[ModelPayloadLeaseProjectionUpsert, ...],
]:
    """Fold one lease event; returns the new state and its upsert intents."""
    new_state = dict(state)
    row_state = _STATE.get(event.outcome)
    if row_state is None or event.fence < 1 or event.lease_id is None:
        return new_state, ()
    key = (event.kind, event.resource)
    current = state.get(key)
    if current is not None and event.fence < current.fence:
        return new_state, ()
    holder = event.holder
    lease_id = event.lease_id
    if current is not None and event.fence == current.fence:
        holder = max(holder, current.holder)
        if _RANK[row_state] < _RANK[current.state]:
            row_state = current.state
            lease_id = current.lease_id
        elif _RANK[row_state] == _RANK[current.state]:
            lease_id = max(lease_id, current.lease_id)
    row = ModelLeaseProjectionRow(
        kind=event.kind,
        resource=event.resource,
        lease_id=lease_id,
        fence=event.fence,
        holder=holder,
        state=row_state,
    )
    if row == current:
        return new_state, ()
    new_state[key] = row
    return new_state, (ModelPayloadLeaseProjectionUpsert(row=row),)


class HandlerLeaseReducer:
    """Holds the reducer state between events; the fold itself is ``reduce_lease_event``."""

    def __init__(self) -> None:
        self._state: dict[tuple[str, str], ModelLeaseProjectionRow] = {}

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    def handle(
        self, event: ModelLeaseReply
    ) -> tuple[ModelPayloadLeaseProjectionUpsert, ...]:
        self._state, intents = reduce_lease_event(self._state, event)
        return intents


__all__ = ["HandlerLeaseReducer", "LeaseProjectionState", "reduce_lease_event"]
