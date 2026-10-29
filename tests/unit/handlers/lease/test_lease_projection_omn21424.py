# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21424 AC4: the lease projection upserts on (kind, resource), never to a lower fence."""

from __future__ import annotations

import itertools
import re
import sqlite3
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest

from omnibase_infra.handlers.lease import (
    UPSERT_LEASE_PROJECTION_SQL,
    lease_projection_params,
    reduce_lease_event,
)
from omnibase_infra.models.lease import (
    ModelLeaseProjectionRow,
    ModelLeaseReply,
    ModelPayloadLeaseProjectionUpsert,
)

pytestmark = pytest.mark.unit

_L1, _L2, _L3, _L4 = (UUID(int=i) for i in range(1, 5))

MIGRATION = (
    Path(__file__).parents[4]
    / "docker/migrations/forward/nodes/node_projection_lease"
    / "0000_create_lease_projection.sql"
)


def _events() -> list[ModelLeaseReply]:
    """Three grants of one resource, a renewal and a release, plus another resource."""
    return [
        ModelLeaseReply(
            outcome="lease-granted",
            kind="k",
            resource="r",
            holder="a",
            lease_id=_L1,
            fence=10,
        ),
        ModelLeaseReply(
            outcome="lease-renewed",
            kind="k",
            resource="r",
            holder="a",
            lease_id=_L1,
            fence=10,
        ),
        ModelLeaseReply(
            outcome="lease-granted",
            kind="k",
            resource="r",
            holder="b",
            lease_id=_L2,
            fence=17,
        ),
        ModelLeaseReply(
            outcome="lease-released",
            kind="k",
            resource="r",
            holder="b",
            lease_id=_L2,
            fence=17,
        ),
        ModelLeaseReply(
            outcome="lease-granted",
            kind="k",
            resource="r",
            holder="c",
            lease_id=_L3,
            fence=23,
        ),
        ModelLeaseReply(
            outcome="lease-granted",
            kind="k",
            resource="s",
            holder="a",
            lease_id=_L4,
            fence=12,
        ),
        ModelLeaseReply(
            outcome="lease-refused", kind="k", resource="r", holder="d", reason="held"
        ),
    ]


def test_ac4_reducer_replay_in_any_order_keeps_the_highest_fence() -> None:
    for order in itertools.permutations(_events()):
        state: dict[tuple[str, str], ModelLeaseProjectionRow] = {}
        for event in order:
            state, _ = reduce_lease_event(state, event)
        assert state[("k", "r")].fence == 23, order
        assert state[("k", "r")].holder == "c"
        assert state[("k", "s")].fence == 12
        assert len(state) == 2


def test_ac4_reducer_intents_end_at_the_final_state_for_every_key() -> None:
    for order in itertools.permutations(_events()):
        state: dict[tuple[str, str], ModelLeaseProjectionRow] = {}
        last: dict[tuple[str, str], ModelLeaseProjectionRow] = {}
        for event in order:
            state, intents = reduce_lease_event(state, event)
            for intent in intents:
                last[(intent.row.kind, intent.row.resource)] = intent.row
        assert last == state, order


def test_ac4_reducer_is_pure_and_emits_no_intent_for_a_lower_fence() -> None:
    high = ModelLeaseReply(
        outcome="lease-granted",
        kind="k",
        resource="r",
        holder="b",
        lease_id=_L2,
        fence=17,
    )
    low = ModelLeaseReply(
        outcome="lease-granted",
        kind="k",
        resource="r",
        holder="a",
        lease_id=_L1,
        fence=10,
    )
    state, intents = reduce_lease_event({}, high)
    assert [i.row.fence for i in intents] == [17]
    assert isinstance(intents[0], ModelPayloadLeaseProjectionUpsert)
    assert intents[0].target == "postgres://lease_projection/k/r"
    frozen = dict(state)
    after, intents = reduce_lease_event(state, low)
    assert intents == ()
    assert after == frozen and state == frozen


def _sqlite_with_migration() -> sqlite3.Connection:
    db = sqlite3.connect(":memory:")
    ddl = MIGRATION.read_text()
    table = re.search(
        r"CREATE TABLE IF NOT EXISTS public\.lease_projection \((.*?)\n\);", ddl, re.S
    )
    assert table is not None, "the migration creates lease_projection"
    body = re.sub(r"TIMESTAMPTZ NOT NULL DEFAULT now\(\)", "TEXT", table.group(1))
    db.execute(f"CREATE TABLE lease_projection ({body}\n)")
    db.create_function("now", 0, lambda: "2026-10-27T00:00:00Z")
    return db


def test_ac4_sql_upsert_replayed_out_of_order_never_lowers_the_fence() -> None:
    """The migration's table and the handler's upsert statement, run in SQLite."""
    sql = re.sub(r"\$(\d+)", r"?\1", UPSERT_LEASE_PROJECTION_SQL)
    rows = [
        ModelLeaseProjectionRow(
            kind="k",
            resource="r",
            lease_id=UUID(int=f),
            fence=f,
            holder=f"h{f}",
            state="held",
        )
        for f in (10, 17, 23)
    ]
    for order in itertools.permutations(rows):
        db = _sqlite_with_migration()
        for row in order:
            db.execute(sql, lease_projection_params(row))
        found = db.execute(
            "SELECT kind, resource, fence, holder FROM lease_projection"
        ).fetchall()
        assert found == [("k", "r", 23, "h23")], order


def _equal_fence_events() -> list[ModelLeaseReply]:
    """One grant at fence 10 seen as grant, renewal, release and expiry, plus fence 7."""

    def event(outcome: str, holder: str, fence: int, lease: UUID) -> ModelLeaseReply:
        return ModelLeaseReply(
            outcome=cast("Any", outcome),
            kind="k",
            resource="r",
            holder=holder,
            lease_id=lease,
            fence=fence,
        )

    return [
        event("lease-granted", "a", 10, _L1),
        event("lease-released", "", 10, _L1),
        event("lease-expired", "", 10, _L1),
        event("lease-granted", "z", 7, _L2),
    ]


def test_ac4_sql_upsert_joins_equal_fence_rows_like_the_reducer_in_any_order() -> None:
    """Replayed rows reach the table in any order, as after a reducer restart."""
    sql = re.sub(r"\$(\d+)", r"?\1", UPSERT_LEASE_PROJECTION_SQL)
    events = _equal_fence_events()
    expected: dict[tuple[str, str], ModelLeaseProjectionRow] = {}
    for event in events:
        expected, _ = reduce_lease_event(expected, event)
    want = expected[("k", "r")]
    assert (want.state, want.holder, want.fence) == ("expired", "a", 10)
    rows = [
        intent.row for event in events for intent in reduce_lease_event({}, event)[1]
    ]
    for order in itertools.permutations(rows):
        db = _sqlite_with_migration()
        for row in order:
            db.execute(sql, lease_projection_params(row))
        found = db.execute(
            "SELECT lease_id, fence, holder, state FROM lease_projection"
        ).fetchall()
        assert found == [(str(want.lease_id), 10, "a", "expired")], order


def test_ac4_sql_upsert_refreshes_updated_at_on_every_conflict() -> None:
    sql = re.sub(r"\$(\d+)", r"?\1", UPSERT_LEASE_PROJECTION_SQL)
    db = _sqlite_with_migration()
    first = ModelLeaseProjectionRow(
        kind="k", resource="r", lease_id=_L1, fence=10, holder="a", state="held"
    )
    db.execute(sql, lease_projection_params(first))
    db.execute("UPDATE lease_projection SET updated_at = 'stale'")
    db.execute(
        sql,
        lease_projection_params(first.model_copy(update={"state": "released"})),
    )
    (stamp,) = db.execute("SELECT updated_at FROM lease_projection").fetchone()
    assert stamp == "2026-10-27T00:00:00Z"


def test_ac4_equal_rank_conflicting_lease_ids_converge_in_sql_and_reducer() -> None:
    """Even conflicting replay data has one result, independent of delivery order."""
    events = [
        ModelLeaseReply(
            outcome=outcome,
            kind="k",
            resource="r",
            lease_id=lease_id,
            fence=10,
            holder=holder,
        )
        for outcome, lease_id, holder in (
            ("lease-expired", _L1, ""),
            ("lease-expired", _L2, ""),
            ("lease-released", _L3, "a"),
        )
    ]
    expected = ModelLeaseProjectionRow(
        kind="k", resource="r", lease_id=_L2, fence=10, holder="a", state="expired"
    )
    sql = re.sub(r"\$(\d+)", r"?\1", UPSERT_LEASE_PROJECTION_SQL)
    for order in itertools.permutations(events):
        state: dict[tuple[str, str], ModelLeaseProjectionRow] = {}
        with _sqlite_with_migration() as db:
            for event in order:
                state, _ = reduce_lease_event(state, event)
                # Independent reducers can emit these rows after a restart;
                # SQL must converge without relying on a previous reducer fold.
                for intent in reduce_lease_event({}, event)[1]:
                    db.execute(sql, lease_projection_params(intent.row))
            assert state[("k", "r")] == expected, order
            assert db.execute(
                "SELECT lease_id, fence, holder, state FROM lease_projection"
            ).fetchall() == [(str(_L2), 10, "a", "expired")], order


def test_ac4_sql_upsert_matches_the_reducer_for_every_pair_of_rows() -> None:
    """Every ordered pair over fence, state, lease ID and holder lands the reducer's row."""
    sql = re.sub(r"\$(\d+)", r"?\1", UPSERT_LEASE_PROJECTION_SQL)
    rows = [
        ModelLeaseProjectionRow(
            kind="k",
            resource="r",
            lease_id=lease_id,
            fence=fence,
            holder=holder,
            state=cast("Any", state),
        )
        for fence in (10, 11)
        for state in ("held", "released", "expired")
        for lease_id in (_L1, _L2)
        for holder in ("", "a", "b")
    ]
    outcome = {"held": "lease-granted", "released": "lease-released"}
    for first, second in itertools.product(rows, repeat=2):
        state: dict[tuple[str, str], ModelLeaseProjectionRow] = {("k", "r"): first}
        event = ModelLeaseReply(
            outcome=cast("Any", outcome.get(second.state, "lease-expired")),
            kind="k",
            resource="r",
            holder=second.holder,
            lease_id=second.lease_id,
            fence=second.fence,
        )
        state, _ = reduce_lease_event(state, event)
        want = state[("k", "r")]
        with _sqlite_with_migration() as db:
            db.execute(sql, lease_projection_params(first))
            db.execute(sql, lease_projection_params(second))
            found = db.execute(
                "SELECT lease_id, fence, holder, state FROM lease_projection"
            ).fetchall()
        assert found == [(str(want.lease_id), want.fence, want.holder, want.state)], (
            first,
            second,
        )
