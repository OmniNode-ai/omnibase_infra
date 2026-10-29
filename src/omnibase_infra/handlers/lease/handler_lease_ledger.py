# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Ledger CLAIM and TERMINAL rows as a projection of ``lane-work`` leases (OMN-21428).

``node_projection_lease`` folds lease events of the ``lane-work`` kind into
rolling-ledger rows: a grant writes the lane's CLAIM, a release writes its
TERMINAL with the outcome the lane released with, and an expiry writes a
TERMINAL with ``outcome=unknown cause=lease-expired``. The lane runner writes
neither row. ``project_lane_work`` is the pure fold; the handler holds its
state and appends the rows.

The fold keeps the GenericLease invariants (``formal/generic_lease``): per
resource it keeps the highest fence seen, so an event of a lower fence writes
nothing and CLAIMs appear in rising fence order (``FenceMonotonicPerResource``);
a CLAIM is closed exactly once; and a grant above an open CLAIM's fence means
the earlier grant ended (``ExpiredLeaseFreesResource``), so that CLAIM is
closed first with ``cause=lease-superseded``.

Expiry is performed by etcd with no reaper. ``expired_claims`` compares the
open CLAIMs with the live grants read from the store and answers the
``lease-expired`` events for the ones whose grant is gone; ``sweep_expired``
reads the store and writes their TERMINALs.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.handlers.handler_ledger_write_guard import check_file
from omnibase_infra.handlers.lease.handler_lease_store import HandlerLeaseStore
from omnibase_infra.models.lease import (
    ModelLaneWorkClaim,
    ModelLaneWorkHolder,
    ModelLeaseReply,
)

LANE_WORK_KIND = "lane-work"
PROJECTION_ACTOR = "script:node_projection_lease"

LaneWorkLedgerState = Mapping[str, ModelLaneWorkClaim]


# Appends one rolling-ledger row.
LedgerRowAppend = Callable[[str], Awaitable[None]]


def _ts(at: datetime) -> str:
    return at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def claim_row(resource: str, claim: ModelLaneWorkClaim) -> str:
    holder = claim.holder
    cells = [
        claim.claimed_at,
        "CLAIM",
        f"lane={holder.lane}",
        f"principal={holder.principal}",
        f"ticket={holder.ticket}",
        f"actor={PROJECTION_ACTOR}",
        "model=none",
        f"host={holder.host}",
        f"run={holder.run}",
    ]
    if holder.parent:
        cells.append(f"parent={holder.parent}")
    cells += [
        f"kind={LANE_WORK_KIND}",
        f"resource={resource}",
        f"lease={claim.lease_id}",
        f"fence={claim.fence}",
    ]
    return " | ".join(cells)


def terminal_row(
    claim: ModelLaneWorkClaim, *, at: datetime, outcome: str, cause: str
) -> str:
    holder = claim.holder
    return " | ".join(
        [
            _ts(at),
            "TERMINAL",
            f"lane={holder.lane}",
            f"principal={holder.principal}",
            f"ticket={holder.ticket}",
            f"actor={PROJECTION_ACTOR}",
            "model=none",
            "friction=none",
            f"outcome={outcome}",
            f"cause={cause}",
            f"closes-CLAIM={claim.claimed_at}",
            f"lease={claim.lease_id}",
            f"fence={claim.fence}",
            "delegated=0 delegation_reason=no-text-or-code",
        ]
    )


def project_lane_work(
    state: LaneWorkLedgerState, event: ModelLeaseReply, *, at: datetime
) -> tuple[dict[str, ModelLaneWorkClaim], tuple[str, ...]]:
    """Fold one lease event; returns the new state and the ledger rows it writes."""
    new_state = dict(state)
    if event.kind != LANE_WORK_KIND or event.fence < 1 or event.lease_id is None:
        return new_state, ()
    current = state.get(event.resource)
    if event.outcome == "lease-granted":
        if current is not None and event.fence <= current.fence:
            return new_state, ()
        rows: list[str] = []
        if current is not None and current.open:
            rows.append(
                terminal_row(
                    current, at=at, outcome="unknown", cause="lease-superseded"
                )
            )
        claim = ModelLaneWorkClaim(
            lease_id=event.lease_id,
            fence=event.fence,
            holder=ModelLaneWorkHolder.parse(event.holder),
            claimed_at=_ts(at),
            open=True,
        )
        new_state[event.resource] = claim
        rows.append(claim_row(event.resource, claim))
        return new_state, tuple(rows)
    if event.outcome in ("lease-released", "lease-expired"):
        if (
            current is None
            or not current.open
            or current.lease_id != event.lease_id
            or current.fence != event.fence
        ):
            return new_state, ()
        new_state[event.resource] = current.model_copy(update={"open": False})
        released = event.outcome == "lease-released"
        outcome = (event.reason or "unknown") if released else "unknown"
        return new_state, (
            terminal_row(current, at=at, outcome=outcome, cause=event.outcome),
        )
    return new_state, ()


def expired_claims(
    state: LaneWorkLedgerState, live: Mapping[str, set[UUID]]
) -> tuple[ModelLeaseReply, ...]:
    """``lease-expired`` for every open CLAIM whose grant is not among the live ones."""
    return tuple(
        ModelLeaseReply(
            outcome="lease-expired",
            kind=LANE_WORK_KIND,
            resource=resource,
            lease_id=claim.lease_id,
            fence=claim.fence,
        )
        for resource, claim in sorted(state.items())
        if claim.open and claim.lease_id not in live.get(resource, set())
    )


class LedgerFileAppender:
    """Appends rows to a ledger file; a test's write to a canonical ledger is refused.

    ``env`` is the process environment the write guard reads (``OMNI_HOME`` and
    the test markers); the caller passes ``os.environ``.
    """

    def __init__(self, path: Path, env: Mapping[str, str]) -> None:
        self._path = path
        self._env = env

    async def append(self, row: str) -> None:
        check_file(self._path, self._env)
        await asyncio.to_thread(self._write, row)

    def _write(self, row: str) -> None:
        with self._path.open("a", encoding="utf-8") as ledger:
            ledger.write(row + "\n")


class HandlerLeaseLedger:
    """Writes the ``lane-work`` CLAIM and TERMINAL rows; the fold is ``project_lane_work``."""

    def __init__(
        self,
        append: LedgerRowAppend,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._append = append
        self._clock = clock
        self._state: dict[str, ModelLaneWorkClaim] = {}

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def handle(self, event: ModelLeaseReply) -> tuple[str, ...]:
        self._state, rows = project_lane_work(self._state, event, at=self._clock())
        for row in rows:
            await self._append(row)
        return rows

    def open_claims(self) -> dict[str, ModelLaneWorkClaim]:
        return {resource: c for resource, c in self._state.items() if c.open}

    async def sweep_expired(self, store: HandlerLeaseStore) -> tuple[str, ...]:
        """Read the live grants of every open CLAIM and close the expired ones."""
        live: dict[str, set[UUID]] = {}
        for resource in self.open_claims():
            grants = await store.show(LANE_WORK_KIND, resource)
            live[resource] = {g.lease_id for g in grants if g.lease_id is not None}
        rows: list[str] = []
        for event in expired_claims(self._state, live):
            rows.extend(await self.handle(event))
        return tuple(rows)


__all__ = [
    "LANE_WORK_KIND",
    "PROJECTION_ACTOR",
    "HandlerLeaseLedger",
    "LaneWorkLedgerState",
    "LedgerFileAppender",
    "LedgerRowAppend",
    "claim_row",
    "expired_claims",
    "project_lane_work",
    "terminal_row",
]
