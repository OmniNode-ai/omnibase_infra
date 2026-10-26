# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pure lease projection reducer (OMN-21424).

``delta(state, event) -> (new_state, intents)`` with no I/O. The state holds one
row per ``(kind, resource)``: the grant with the highest fence seen. An event
with a lower fence than the row's is dropped, so replaying events in any order
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
# At one fence a release or expiry ends the grant; a late renewal does not revive it.
_RANK = {"held": 0, "released": 1, "expired": 1}


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
    if current is not None and (
        event.fence < current.fence
        or (event.fence == current.fence and _RANK[row_state] < _RANK[current.state])
    ):
        return new_state, ()
    row = ModelLeaseProjectionRow(
        kind=event.kind,
        resource=event.resource,
        lease_id=event.lease_id,
        fence=event.fence,
        holder=event.holder or (current.holder if current is not None else ""),
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
