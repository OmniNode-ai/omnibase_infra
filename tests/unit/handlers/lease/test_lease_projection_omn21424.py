# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21424 AC4: the lease projection upserts on (kind, resource), never to a lower fence."""

from __future__ import annotations

import itertools
import re
import sqlite3
from pathlib import Path
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
    / "docker/migrations/forward/nodes/node_projection_lease/0000_create_lease_projection.sql"
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


def test_ac4_sql_upsert_at_an_equal_fence_never_regresses_the_state() -> None:
    """A replayed ``held`` at the same fence keeps ``released``/``expired``."""
    sql = re.sub(r"\$(\d+)", r"?\1", UPSERT_LEASE_PROJECTION_SQL)
    rows = [
        ModelLeaseProjectionRow(
            kind="k",
            resource="r",
            lease_id=UUID(int=7),
            fence=7,
            holder="h7",
            state=state,
        )
        for state in ("held", "released", "expired")
    ]
    for order in itertools.permutations(rows):
        db = _sqlite_with_migration()
        for row in order:
            db.execute(sql, lease_projection_params(row))
        found = db.execute("SELECT fence, state FROM lease_projection").fetchall()
        assert found == [(7, "expired")], order


@pytest.mark.parametrize(
    ("grant_holder", "terminal_holder"), [("a", ""), ("a", "z"), ("z", "a")]
)
def test_equal_fence_sql_join_matches_reducer_in_every_order(
    grant_holder: str, terminal_holder: str
) -> None:
    events = [
        ModelLeaseReply(
            outcome=outcome,
            kind="k",
            resource="r",
            lease_id=UUID(int=7),
            fence=7,
            holder=holder,
        )
        for outcome, holder in (
            ("lease-granted", grant_holder),
            ("lease-renewed", ""),
            ("lease-released", terminal_holder),
            ("lease-expired", ""),
        )
    ]
    sql = re.sub(r"\$(\d+)", r"?\1", UPSERT_LEASE_PROJECTION_SQL)
    for order in itertools.permutations(events):
        expected: dict[tuple[str, str], ModelLeaseProjectionRow] = {}
        with _sqlite_with_migration() as db:
            for event in order:
                expected, _ = reduce_lease_event(expected, event)
                _, intents = reduce_lease_event({}, event)
                db.execute(sql, lease_projection_params(intents[0].row))
            found = db.execute(
                "SELECT kind, resource, lease_id, fence, holder, state FROM lease_projection"
            ).fetchall()
            assert found == [lease_projection_params(expected[("k", "r")])], order
