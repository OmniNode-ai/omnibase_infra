# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21428: ledger CLAIM and TERMINAL rows projected from ``lane-work`` leases.

AC1: a ``lane-work`` grant writes the CLAIM row, and its actor is the
projection. AC2: an expired grant whose holder wrote no TERMINAL gets a
TERMINAL row that closes the CLAIM. The GenericLease invariants carry over:
every event order writes CLAIMs in rising fence order per resource, never a
TERMINAL without its CLAIM, and never two TERMINALs for one CLAIM.
"""

from __future__ import annotations

import itertools
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from omnibase_infra.handlers.handler_ledger_write_guard import (
    LedgerTestWriteRefusedError,
)
from omnibase_infra.handlers.lease.handler_lane_work_lease import (
    HandlerLaneWorkLease,
    LeaseCommands,
    renew_interval_s,
)
from omnibase_infra.handlers.lease.handler_lease_ledger import (
    LANE_WORK_KIND,
    PROJECTION_ACTOR,
    HandlerLeaseLedger,
    LedgerFileAppender,
    expired_claims,
    project_lane_work,
)
from omnibase_infra.models.lease import (
    LeaseOutcome,
    ModelLaneWorkClaim,
    ModelLaneWorkHolder,
    ModelLeaseAcquireRequest,
    ModelLeaseReleaseRequest,
    ModelLeaseRenewRequest,
    ModelLeaseReply,
)

pytestmark = [pytest.mark.unit]

AT = datetime(2026, 10, 26, 4, 30, tzinfo=UTC)
HOLDER = ModelLaneWorkHolder(
    host="h201",
    lane="claims-test",
    run="rlane-claims-test",
    principal="dev-cli-test",
    ticket="OMN-21428",
    parent="orchestrator-test",
)
RESOURCE = "lane:claims-test"


def _cells(row: str) -> dict[str, str]:
    parts = [part.strip() for part in row.split(" | ")]
    cells = {"ts": parts[0], "verb": parts[1]}
    for part in parts[2:]:
        name, _, value = part.partition("=")
        cells.setdefault(name, value)
    return cells


def _reply(
    outcome: LeaseOutcome,
    lease_id: UUID,
    fence: int,
    *,
    kind: str = LANE_WORK_KIND,
    holder: str = "",
    reason: str = "",
) -> ModelLeaseReply:
    return ModelLeaseReply(
        outcome=outcome,
        kind=kind,
        resource=RESOURCE,
        holder=holder,
        lease_id=lease_id,
        fence=fence,
        reason=reason,
    )


class _MemoryAppender:
    def __init__(self) -> None:
        self.rows: list[str] = []

    async def append(self, row: str) -> None:
        self.rows.append(row)


def test_ac1_grant_writes_claim_row_with_projection_actor() -> None:
    lease_id = uuid4()
    _, rows = project_lane_work(
        {}, _reply("lease-granted", lease_id, 7, holder=HOLDER.holder()), at=AT
    )
    assert len(rows) == 1
    cells = _cells(rows[0])
    assert cells["verb"] == "CLAIM"
    assert cells["actor"] == PROJECTION_ACTOR == "script:node_projection_lease"
    assert cells["lane"] == "claims-test"
    assert cells["ticket"] == "OMN-21428"
    assert cells["principal"] == "dev-cli-test"
    assert cells["host"] == "h201"
    assert cells["run"] == "rlane-claims-test"
    assert cells["parent"] == "orchestrator-test"
    assert cells["lease"] == str(lease_id)
    assert cells["fence"] == "7"
    assert cells["ts"] == "2026-10-26T04:30:00Z"


def test_ac1_renew_replay_and_other_kinds_write_no_row() -> None:
    lease_id = uuid4()
    granted = _reply("lease-granted", lease_id, 7, holder=HOLDER.holder())
    state, rows = project_lane_work({}, granted, at=AT)
    assert len(rows) == 1
    for event in (
        granted,
        granted.model_copy(update={"reason": "replayed request_id"}),
        _reply("lease-renewed", lease_id, 7),
        _reply("lease-granted", uuid4(), 3, kind="slot", holder=HOLDER.holder()),
    ):
        state, again = project_lane_work(state, event, at=AT)
        assert again == ()


def test_ac2_expiry_writes_terminal_closing_the_claim() -> None:
    lease_id = uuid4()
    state, (claim,) = project_lane_work(
        {}, _reply("lease-granted", lease_id, 7, holder=HOLDER.holder()), at=AT
    )
    later = AT + timedelta(seconds=90)
    state, rows = project_lane_work(
        state, _reply("lease-expired", lease_id, 7), at=later
    )
    assert len(rows) == 1
    cells = _cells(rows[0])
    assert cells["verb"] == "TERMINAL"
    assert cells["actor"] == PROJECTION_ACTOR
    assert cells["lane"] == "claims-test"
    assert cells["outcome"] == "unknown"
    assert cells["cause"] == "lease-expired"
    assert cells["closes-CLAIM"] == _cells(claim)["ts"]
    assert cells["lease"] == str(lease_id)
    assert cells["ts"] == "2026-10-26T04:31:30Z"
    _, again = project_lane_work(state, _reply("lease-expired", lease_id, 7), at=later)
    assert again == ()


def test_ac2_release_carries_the_lane_outcome() -> None:
    lease_id = uuid4()
    state, _ = project_lane_work(
        {}, _reply("lease-granted", lease_id, 7, holder=HOLDER.holder()), at=AT
    )
    state, (row,) = project_lane_work(
        state, _reply("lease-released", lease_id, 7, reason="handed-off"), at=AT
    )
    assert _cells(row)["outcome"] == "handed-off"
    assert _cells(row)["cause"] == "lease-released"
    _, again = project_lane_work(state, _reply("lease-expired", lease_id, 7), at=AT)
    assert again == ()


def test_ac2_expired_claims_lists_open_claims_with_no_live_grant() -> None:
    live, gone = uuid4(), uuid4()
    state, _ = project_lane_work(
        {}, _reply("lease-granted", live, 4, holder=HOLDER.holder()), at=AT
    )
    other = ModelLeaseReply(
        outcome="lease-granted",
        kind=LANE_WORK_KIND,
        resource="lane:other",
        holder=HOLDER.model_copy(update={"lane": "other"}).holder(),
        lease_id=gone,
        fence=5,
    )
    state, _ = project_lane_work(state, other, at=AT)
    expired = expired_claims(state, {RESOURCE: {live}, "lane:other": set()})
    assert [(e.outcome, e.resource, e.lease_id, e.fence) for e in expired] == [
        ("lease-expired", "lane:other", gone, 5)
    ]


def test_tla_invariants_hold_for_every_event_order() -> None:
    """FenceMonotonicPerResource and one TERMINAL per CLAIM, over all 120 orders."""
    first, second = uuid4(), uuid4()
    events = [
        _reply("lease-granted", first, 5, holder=HOLDER.holder()),
        _reply("lease-renewed", first, 5),
        _reply("lease-expired", first, 5),
        _reply("lease-granted", second, 9, holder=HOLDER.holder()),
        _reply("lease-released", second, 9, reason="done"),
    ]
    for order in itertools.permutations(events):
        state: dict[str, ModelLaneWorkClaim] = {}
        written: list[dict[str, str]] = []
        for event in order:
            state, rows = project_lane_work(state, event, at=AT)
            written.extend(_cells(row) for row in rows)
        claims = [int(c["fence"]) for c in written if c["verb"] == "CLAIM"]
        assert claims == sorted(set(claims)), order
        open_leases: set[str] = set()
        closed: set[str] = set()
        for cells in written:
            if cells["verb"] == "CLAIM":
                open_leases.add(cells["lease"])
            else:
                assert cells["lease"] in open_leases, order
                assert cells["lease"] not in closed, order
                closed.add(cells["lease"])
        # A later grant implies the earlier one ended: no CLAIM stays open under a newer one.
        assert len(open_leases - closed) <= 1, order


async def test_handler_appends_rows_through_its_appender() -> None:
    appender = _MemoryAppender()
    ledger = HandlerLeaseLedger(appender.append, clock=lambda: AT)
    lease_id = uuid4()
    await ledger.handle(_reply("lease-granted", lease_id, 7, holder=HOLDER.holder()))
    await ledger.handle(_reply("lease-renewed", lease_id, 7))
    await ledger.handle(_reply("lease-expired", lease_id, 7))
    assert [_cells(row)["verb"] for row in appender.rows] == ["CLAIM", "TERMINAL"]
    assert ledger.open_claims() == {}


async def test_file_appender_writes_rows_and_refuses_a_canonical_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger_file = tmp_path / "ROLLING_WORK_LEDGER.md"
    await LedgerFileAppender(ledger_file, os.environ).append("row one")
    await LedgerFileAppender(ledger_file, os.environ).append("row two")
    assert ledger_file.read_text() == "row one\nrow two\n"
    home = Path.home() / "omni_home_not_tmp"
    monkeypatch.setenv("OMNI_HOME", str(home))
    with pytest.raises(LedgerTestWriteRefusedError):
        await LedgerFileAppender(home / "ROLLING_WORK_LEDGER.md", os.environ).append(
            "row"
        )


class _FakeCommands(LeaseCommands):
    def __init__(self) -> None:
        self.acquired: list[ModelLeaseAcquireRequest] = []
        self.released: list[ModelLeaseReleaseRequest] = []
        self.lease_id = uuid4()

    async def acquire(self, request: ModelLeaseAcquireRequest) -> ModelLeaseReply:
        self.acquired.append(request)
        return ModelLeaseReply(
            outcome="lease-granted",
            kind=request.kind,
            resource=request.resource,
            holder=request.holder,
            lease_id=self.lease_id,
            fence=11,
            ttl_s=request.ttl_s or 0,
        )

    async def renew(self, request: ModelLeaseRenewRequest) -> ModelLeaseReply:
        return ModelLeaseReply(
            outcome="lease-renewed",
            kind=request.kind,
            resource=request.resource,
            lease_id=request.lease_id,
            fence=11,
        )

    async def release(self, request: ModelLeaseReleaseRequest) -> ModelLeaseReply:
        self.released.append(request)
        return ModelLeaseReply(
            outcome="lease-released",
            kind=request.kind,
            resource=request.resource,
            lease_id=request.lease_id,
            fence=11,
            reason=request.outcome,
        )


async def test_runner_acquires_lane_work_and_releases_with_its_outcome() -> None:
    commands = _FakeCommands()
    runner = HandlerLaneWorkLease(commands, HOLDER, ttl_s=30)
    granted = await runner.acquire()
    assert granted.outcome == "lease-granted"
    (request,) = commands.acquired
    assert request.kind == LANE_WORK_KIND
    assert request.resource == RESOURCE
    assert ModelLaneWorkHolder.parse(request.holder) == HOLDER
    assert (await runner.renew()).outcome == "lease-renewed"
    released = await runner.release("handed-off")
    assert released.reason == "handed-off"
    assert commands.released[0].outcome == "handed-off"
    # The runner holds no ledger writer: the CLAIM and TERMINAL are the projection's.
    assert not [name for name in vars(runner) if "ledger" in name or "append" in name]


def test_renew_interval_is_a_third_of_the_ttl() -> None:
    assert renew_interval_s(30) == 10.0
    assert renew_interval_s(3) == 1.0


class _LostReplyCommands(_FakeCommands):
    """Commits the first grant but loses its reply, as a dropped connection would."""

    def __init__(self) -> None:
        super().__init__()
        self.lost = False

    async def acquire(self, request: ModelLeaseAcquireRequest) -> ModelLeaseReply:
        reply = await super().acquire(request)
        if not self.lost:
            self.lost = True
            raise ConnectionError("reply lost after the grant committed")
        return reply


async def test_acquire_retry_after_a_lost_reply_reuses_the_request_id() -> None:
    commands = _LostReplyCommands()
    runner = HandlerLaneWorkLease(commands, HOLDER, ttl_s=30)
    with pytest.raises(ConnectionError):
        await runner.acquire()
    assert (await runner.acquire()).outcome == "lease-granted"
    first, retry = commands.acquired
    # The store's request-key idempotency reads back the committed grant.
    assert retry.request_id == first.request_id
    await runner.release("done")
    await runner.acquire()
    assert commands.acquired[2].request_id != first.request_id
