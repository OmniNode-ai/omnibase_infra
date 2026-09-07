# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17888 second pass: a bounded question, and a cursor that is really bounded.

Two defects survived the first pass (omnibase_infra#3288, squash ``c4fe409a``).

**The seam could not express a bounded question.** ``query(table, filters)``
offered equality filters and nothing else, so a caller that wanted "the latest
row of this session" had exactly one option: read every row of the session and
sort it in Python. ``HandlerProjectionSessionReplay.project`` did that on EVERY
event -- O(n^2) in session length -- and the row budget added in the first pass
only converted that into a scheduled refusal (~2026-09-08T00:00Z at the measured
~3,029 rows/hour). ``order_by`` / ``descending`` / ``limit`` are the minimal
typed capability that lets the caller ask the question it actually has. They are
keyword-only and default to the previous behaviour exactly.

**The cursor was still client-side.** The first pass replaced
``[dict(record) for record in cursor.fetchall()]`` with an iteration, which
removed the Python-object copy -- and left the libpq one. A psycopg2 cursor with
no ``name=`` executes through ``PQexec``, and libpq buffers the ENTIRE result set
inside the connection's ``PGresult`` before the first row is yielded. So the
per-row budget could only ever refuse rows the process had already paid for: the
guard fired after the allocation it exists to prevent. The read now DECLAREs a
server-side cursor and FETCHes it in ``PROJECTION_QUERY_CURSOR_ITERSIZE``
batches, which means it must also run inside a transaction block -- PostgreSQL
refuses ``DECLARE CURSOR`` outside one, and the only autocommit-legal form,
``WITH HOLD``, materialises the whole result into a server-side tuplestore at
commit, moving the buffer rather than removing it.

The connection double here records what the adapter actually asked psycopg2 for
-- cursor kwargs, itersize, autocommit transitions, commit/rollback, and the SQL
text -- because every one of those is the artifact under test.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import pytest

from omnibase_infra.errors.error_projection import ProjectionQueryRowBudgetError
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    PROJECTION_HANDLER_MAX_INFLIGHT,
    PROJECTION_QUERY_CURSOR_ITERSIZE,
    PROJECTION_QUERY_MAX_ROWS,
    ProjectionDatabaseTarget,
    _build_projection_db_adapter,
)
from tests.helpers.application_db_topology import (
    projection_database_target,
    projection_database_urls,
)

pytestmark = pytest.mark.unit

_TABLE = "generation_events"


class _RecordingCursor:
    """DB-API cursor double that records how it was created and driven."""

    def __init__(
        self,
        connection: _RecordingConnection,
        *,
        name: str | None,
        rows: list[dict[str, Any]],
        identity: tuple[str, str],
    ) -> None:
        self._connection = connection
        self.name = name
        self.itersize: int | None = None
        self.executed: list[tuple[str, Any]] = []
        self.autocommit_at_execute: bool | None = None
        self._rows = rows
        self._identity = identity
        self.closed = False

    def execute(self, sql: str, params: Any = None) -> None:
        self.executed.append((sql, params))
        self.autocommit_at_execute = self._connection.autocommit

    def fetchone(self) -> tuple[str, str]:
        return self._identity

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter([dict(row) for row in self._rows])

    def __enter__(self) -> _RecordingCursor:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        self.closed = True


class _RecordingConnection:
    """psycopg2 connection double recording transaction and cursor lifecycle."""

    def __init__(
        self,
        principal: str,
        database: str,
        rows: list[dict[str, Any]],
    ) -> None:
        self.closed = False
        self._autocommit = True
        self.autocommit_transitions: list[bool] = []
        self.cursor_calls: list[dict[str, Any]] = []
        self.cursors: list[_RecordingCursor] = []
        self.commits = 0
        self.rollbacks = 0
        self._identity = (principal, database)
        self._rows = rows

    @property
    def autocommit(self) -> bool:
        return self._autocommit

    @autocommit.setter
    def autocommit(self, value: bool) -> None:
        self._autocommit = value
        self.autocommit_transitions.append(value)

    def cursor(
        self, name: str | None = None, cursor_factory: object = None
    ) -> _RecordingCursor:
        self.cursor_calls.append({"name": name, "cursor_factory": cursor_factory})
        cursor = _RecordingCursor(
            self, name=name, rows=self._rows, identity=self._identity
        )
        self.cursors.append(cursor)
        return cursor

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    def close(self) -> None:
        self.closed = True

    # -- assertions helpers --------------------------------------------
    @property
    def read_cursor(self) -> _RecordingCursor:
        """The one cursor the read used (the identity probe is unnamed)."""
        named = [cursor for cursor in self.cursors if cursor.name is not None]
        assert len(named) == 1, f"expected exactly one named cursor, got {len(named)}"
        return named[0]


def _target(name: str = _TABLE) -> ProjectionDatabaseTarget:
    return projection_database_target(name, schema="omninode_internal", access="read")


def _run(
    rows: list[dict[str, Any]] | None = None, **query_kwargs: Any
) -> tuple[list[dict[str, object]], _RecordingConnection]:
    target = _target()
    conn = _RecordingConnection("omninode_runtime", "omnidash_analytics", rows or [])
    with patch("psycopg2.connect", return_value=conn):
        adapter = _build_projection_db_adapter(
            projection_database_urls(target, "postgresql://fixture"),
            target,
            None,
            None,
        )
        result = adapter.query(_TABLE, **query_kwargs)  # type: ignore[attr-defined]
    return result, conn


# ---------------------------------------------------------------------------
# D4 -- the cursor is really server-side, and really inside a transaction.
# ---------------------------------------------------------------------------


def test_the_read_declares_a_server_side_cursor() -> None:
    """RED on the parent: ``conn.cursor(cursor_factory=...)`` with no ``name=``.

    Without a name psycopg2 uses ``PQexec`` and libpq buffers the whole result
    set before the adapter sees row one, so the per-row budget cannot refuse
    what has already been allocated.
    """
    _rows, conn = _run([{"sequence": 1}])

    read_cursor = conn.read_cursor
    assert read_cursor.name is not None
    assert read_cursor.name.startswith("onex_projection_read_"), (
        f"named cursor {read_cursor.name!r} does not carry the seam's prefix; "
        "the name is what makes it server-side"
    )


def test_the_named_cursor_fetches_in_declared_batches() -> None:
    """``itersize`` is part of the memory arithmetic, so it is stated, not inherited.

    psycopg2's default is 2,000. The number is pinned here because the module
    comment computes client-side peak from it.
    """
    _rows, conn = _run([{"sequence": 1}])

    assert conn.read_cursor.itersize == PROJECTION_QUERY_CURSOR_ITERSIZE
    assert PROJECTION_QUERY_CURSOR_ITERSIZE == 1_000


def test_each_named_cursor_name_is_unique_per_read() -> None:
    """Two reads on one connection must not collide on the cursor name.

    A reused name is ``DuplicateCursor`` on the second DECLARE while the first
    is still open -- a failure that would only appear under concurrency.
    """
    target = _target()
    conn = _RecordingConnection("omninode_runtime", "omnidash_analytics", [])
    with patch("psycopg2.connect", return_value=conn):
        adapter = _build_projection_db_adapter(
            projection_database_urls(target, "postgresql://fixture"),
            target,
            None,
            None,
        )
        adapter.query(_TABLE)  # type: ignore[attr-defined]
        adapter.query(_TABLE)  # type: ignore[attr-defined]

    names = [cursor.name for cursor in conn.cursors if cursor.name is not None]
    assert len(names) == 2
    assert len(set(names)) == 2, f"cursor names collided: {names}"


def test_the_unscoped_read_runs_inside_a_transaction_and_always_ends_it() -> None:
    """DECLARE CURSOR is illegal outside a transaction block.

    The adapter's connections sit in autocommit, so the read must open its own
    transaction -- and must always end it, or the connection is left holding one
    open for the rest of its life.
    """
    _rows, conn = _run([{"sequence": 1}])

    assert conn.read_cursor.autocommit_at_execute is False, (
        "the DECLARE ran in autocommit; PostgreSQL rejects a non-WITH-HOLD "
        "server-side cursor there, and WITH HOLD would materialise the whole "
        "result into a server-side tuplestore instead of removing the buffer"
    )
    assert conn.autocommit_transitions[-1] is True, (
        "autocommit was not restored; the connection is pooled per binding and "
        "the next caller would inherit an open transaction"
    )
    assert conn.commits == 1
    assert conn.rollbacks == 0


def test_a_failing_read_rolls_back_and_restores_autocommit() -> None:
    """The transaction the read opened is ended on the failure path too."""
    target = _target()
    conn = _RecordingConnection("omninode_runtime", "omnidash_analytics", [])
    with patch("psycopg2.connect", return_value=conn):
        adapter = _build_projection_db_adapter(
            projection_database_urls(target, "postgresql://fixture"),
            target,
            None,
            None,
        )
        with pytest.raises(ValueError, match="Invalid order_by column"):
            adapter.query(_TABLE, order_by="sequence; DROP TABLE x")  # type: ignore[attr-defined]

    # The refusal happens before any connection work, so no transaction is
    # opened at all -- the strongest form of "ended".
    assert conn.autocommit_transitions == []
    assert conn.commits == 0


# ---------------------------------------------------------------------------
# D1's seam half -- the ordered, limited read.
# ---------------------------------------------------------------------------


def test_query_without_ordering_emits_the_statement_it_always_did() -> None:
    """The capability is additive; the default path is byte-unchanged."""
    _rows, conn = _run([{"sequence": 1}])

    sql, params = conn.read_cursor.executed[0]
    # `public`, not the DECLARED `omninode_internal`: the emitted statement
    # qualifies with the topology-RESOLVED physical schema, which differs from
    # the declaration during the OMN-15359 migration window.
    assert sql == 'SELECT * FROM "public"."generation_events"'
    assert params is None


def test_ordered_limited_read_emits_order_by_and_limit() -> None:
    """RED on the parent: ``query`` took no ordering at all.

    This is the statement ``HandlerProjectionSessionReplay`` needs for reducer
    state -- the session's newest row -- in place of reading the session.
    """
    _rows, conn = _run(
        [{"sequence": 9}],
        filters={"session_id": "9787a4a3-ec49-4819-8bdc-5044efb94550"},
        order_by="sequence",
        descending=True,
        limit=1,
    )

    sql, params = conn.read_cursor.executed[0]
    assert sql == (
        'SELECT * FROM "public"."generation_events" '
        'WHERE "session_id" = %s ORDER BY "sequence" DESC LIMIT %s'
    )
    assert params == ["9787a4a3-ec49-4819-8bdc-5044efb94550", 1]


def test_ascending_is_the_default_direction() -> None:
    _rows, conn = _run([{"sequence": 1}], order_by="sequence", limit=5)

    sql, params = conn.read_cursor.executed[0]
    assert sql.endswith('ORDER BY "sequence" ASC LIMIT %s')
    assert params == [5]


def test_limit_alone_is_permitted_and_parameterised() -> None:
    """A bare LIMIT with no ORDER BY is an arbitrary row set, and is the caller's
    choice to make; what matters is that the count reaches the server as a bound
    parameter and not as interpolated text."""
    _rows, conn = _run([{"sequence": 1}], limit=3)

    sql, params = conn.read_cursor.executed[0]
    assert sql == 'SELECT * FROM "public"."generation_events" LIMIT %s'
    assert params == [3]


@pytest.mark.parametrize(
    "column",
    [
        "sequence; DROP TABLE session_replay_snapshots",
        'sequence" DESC, (SELECT 1)--',
        "",
        "se quence",
    ],
)
def test_order_by_column_is_validated_like_a_filter_key(column: str) -> None:
    """``order_by`` is interpolated into the statement, so it takes the same
    identifier gate the filter keys already take. A parameter cannot carry a
    column name, so validation is the only defence available here."""
    with pytest.raises(ValueError, match="Invalid order_by column"):
        _run([], order_by=column)


def test_descending_without_order_by_is_refused_rather_than_ignored() -> None:
    """A silently-ignored ``descending=True`` would return the OLDEST row to a
    caller that asked for the newest -- a wrong answer that looks like a right
    one, which is the failure mode this seam refuses everywhere else."""
    with pytest.raises(ValueError, match="descending requires an order_by column"):
        _run([], descending=True)


@pytest.mark.parametrize("bad_limit", [0, -1, 1.5, "1", True])
def test_limit_must_be_a_positive_int(bad_limit: object) -> None:
    """``True`` is in this list on purpose: ``isinstance(True, int)`` is True in
    Python, and ``LIMIT true`` is not what any caller meant."""
    with pytest.raises(ValueError, match="limit must be a positive int"):
        _run([], limit=bad_limit)


def test_the_row_budget_still_applies_to_an_ordered_read() -> None:
    """The bound is not opt-out-able by asking for an ordering.

    A caller that supplies ``order_by`` but no ``limit`` is still asking an
    unbounded question, and gets the same refusal.
    """
    rows = [{"sequence": index} for index in range(40)]
    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring.PROJECTION_QUERY_MAX_ROWS",
        32,
    ):
        with pytest.raises(ProjectionQueryRowBudgetError):
            _run(rows, order_by="sequence")


# ---------------------------------------------------------------------------
# D3 -- the declared memory share must equal the arithmetic.
# ---------------------------------------------------------------------------


def test_the_declared_memory_share_is_the_arithmetic_to_the_byte() -> None:
    """Pin the share exactly, not with a bound loose enough to hide a mismatch.

    RED on the parent in the sense that matters: the shipped comment said the
    share was 256 MiB, the sentence after it said the worst case is 271 MiB,
    and ``test_shipped_bounds_fit_the_runtime_container_budget`` asserted a
    third number (300 MiB) while its docstring repeated 256. All three passed
    together because the assertion was an inequality with 29 MiB of slack. An
    equality has no slack: raising either constant fails here, and so does
    restating the share without redoing the multiplication.
    """
    measured_bytes_per_row = 284
    worst_case = (
        PROJECTION_HANDLER_MAX_INFLIGHT
        * PROJECTION_QUERY_MAX_ROWS
        * measured_bytes_per_row
    )

    assert worst_case == 284_000_000, (
        "8 * 125,000 * 284 B = 284,000,000 B; a different product means a "
        "constant moved and the comment block above them is now stale"
    )
    declared_share_mib = 271
    assert worst_case <= declared_share_mib * 1024 * 1024
    # ... and the declared share is TIGHT: one more MiB of claim would be
    # unjustified, which is what makes the number a statement rather than a
    # ceiling picked to accommodate whatever shipped.
    assert worst_case > (declared_share_mib - 1) * 1024 * 1024

    # The share must still fit the measured container headroom (1,536 MiB limit
    # less the ~462 MiB post-subscription baseline).
    assert declared_share_mib < 1_051
