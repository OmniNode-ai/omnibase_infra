# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17888: the projection read seam must materialise ONE bounded copy.

``ProjectionDatabaseOperations._execute_query`` emitted
``SELECT * FROM "<schema>"."<table>"`` with no row bound for any caller, and
then materialised the whole result set **twice, both copies alive at once**::

    return [dict(record) for record in cursor.fetchall()]

``fetchall()`` builds the full list of driver row objects; the comprehension
then builds a second full list of plain ``dict`` s while the first is still
referenced by the temporary. That is the allocation site of the ~1,062 MB RSS
step that memcg-OOM-killed ``onex-runtime`` on the ``.201`` DEV lane roughly
every three minutes.

Measured in the deployed container (image ``sha256:21dd9d6a7401``, revision
``743881e38f4c``) against the live ``omnidash_analytics`` database, for the one
``session_id`` that holds 91,633 of ``public.session_replay_snapshots``' 94,571
rows::

    after cursor.fetchall()          +205,512 KB  (200.7 MiB, 91,571 driver rows)
    after the dict comprehension      +25,344 KB  ( 24.8 MiB, both lists alive)
    ONE CALL                          230,856 KB  (225.4 MiB)

Controls run in the same container: a 453-row session cost +540 KB, and an
absent ``session_id`` cost +0 KB -- so the cost is the rows, not the connect or
the statement.

``asyncio.to_thread`` runs the blocking handler on the event loop's DEFAULT
executor, whose worker count is ``min(32, os.cpu_count() + 4)`` = 32 in this
container. py-spy confirmed 32 live ``asyncio_N`` threads and put 80 of 93
sampled stacks (86%) inside this one query. Four to five concurrent calls
reproduce the step exactly: 4.5 x 225.4 MiB = 1,014 MiB.

These tests drive the REAL adapter through ``_build_projection_db_adapter``
against a DB-API double whose cursor synthesises rows on demand -- the way a
server does -- so the allocation shape under test is the deployed one and not
a mock of it.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Final
from unittest.mock import MagicMock, patch

import pytest

from omnibase_infra.errors.error_projection import ProjectionQueryRowBudgetError
from omnibase_infra.runtime.auto_wiring import handler_wiring
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    PROJECTION_HANDLER_MAX_INFLIGHT,
    PROJECTION_QUERY_MAX_ROWS,
    ProjectionDatabaseTarget,
    ProjectionDispatchSinks,
    _build_projection_db_adapter,
    _make_projection_dispatch_callback,
)
from tests.helpers.application_db_topology import (
    configure_projection_dsns,
    projection_database_target,
    projection_database_urls,
)

pytestmark = pytest.mark.unit

# Deliberately wide-ish rows so the two-copy shape is unambiguous in
# tracemalloc, and small enough that the whole module stays a unit test.
_COLUMNS: Final[tuple[str, ...]] = (
    "snapshot_id",
    "session_id",
    "sequence",
    "event_type",
    "payload",
    "recorded_at",
)


class _DriverRow(dict):  # type: ignore[type-arg]
    """Stand-in for ``psycopg2.extras.RealDictRow``, counting live instances.

    A ``dict`` subclass exactly as ``RealDictRow`` is, so ``dict(record)``
    copies it the way the deployed code does rather than aliasing it.

    ``live``/``peak_live`` are the measurement. Counting live driver rows is
    the allocation shape itself, and under CPython refcounting it is exact: a
    row the seam has finished copying is released at that statement, so a
    streaming read never has more than one or two alive while a ``fetchall()``
    read has the entire result set alive at once. A byte measurement would say
    the same thing far more weakly -- the copied values are shared between the
    two dicts, so bytes understate a shape that object counts state exactly.
    """

    live: int = 0
    peak_live: int = 0

    def __init__(self, values: dict[str, str]) -> None:
        super().__init__(values)
        type(self).live += 1
        type(self).peak_live = max(type(self).peak_live, type(self).live)

    def __del__(self) -> None:
        type(self).live -= 1

    @classmethod
    def reset(cls) -> None:
        cls.live = 0
        cls.peak_live = 0


class _StreamingCursor:
    """DB-API cursor double that synthesises rows on demand.

    ``fetchall()`` materialises the entire result set at once, the way psycopg2
    does; iteration yields one row at a time. Which of the two the adapter
    reaches for is the whole subject of this module, so both are implemented
    honestly and both are counted.
    """

    def __init__(self, principal: str, row_count: int) -> None:
        self._principal = principal
        self._row_count = row_count
        self.fetchall_calls = 0
        self.rows_yielded = 0
        self.executed: list[str] = []

    # -- DB-API surface ------------------------------------------------
    def execute(self, sql: str, params: object = None) -> None:
        self.executed.append(sql)

    def fetchone(self) -> tuple[str, str]:
        return (self._principal, "omnidash_analytics")

    def _generate(self) -> Iterator[_DriverRow]:
        for index in range(self._row_count):
            self.rows_yielded += 1
            yield _DriverRow(
                {column: f"{column}-value-{index:08d}" for column in _COLUMNS}
            )

    def fetchall(self) -> list[_DriverRow]:
        self.fetchall_calls += 1
        return list(self._generate())

    def __iter__(self) -> Iterator[_DriverRow]:
        return self._generate()

    def __enter__(self) -> _StreamingCursor:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def close(self) -> None:
        return None


def _connection(principal: str, row_count: int) -> tuple[MagicMock, _StreamingCursor]:
    cursor = _StreamingCursor(principal, row_count)
    conn = MagicMock()
    conn.closed = False
    conn.autocommit = True
    conn.cursor.return_value = cursor
    return conn, cursor


def _read_target(name: str = "generation_events") -> ProjectionDatabaseTarget:
    return projection_database_target(name, schema="omninode_internal", access="read")


def _adapter(target: ProjectionDatabaseTarget) -> object:
    return _build_projection_db_adapter(
        projection_database_urls(target, "postgresql://fixture"),
        target,
        None,
        None,
    )


def _query(row_count: int, table: str = "generation_events") -> object:
    target = _read_target(table)
    conn, cursor = _connection("omninode_runtime", row_count)
    with patch("psycopg2.connect", return_value=conn):
        adapter = _adapter(target)
        rows = adapter.query(table)  # type: ignore[attr-defined]
    return rows, cursor


def test_query_never_materialises_the_result_set_twice() -> None:
    """The seam must stream the cursor, not call ``fetchall()``.

    RED on the parent: the shipped line is
    ``[dict(record) for record in cursor.fetchall()]``, so ``fetchall_calls``
    is 1 and both full lists are alive at the comprehension's peak.
    """
    rows, cursor = _query(2_000)

    assert len(rows) == 2_000  # type: ignore[arg-type]
    assert cursor.fetchall_calls == 0, (
        "the read seam called fetchall(), which holds the entire driver-row "
        "list alive while a second full list of dicts is built -- the 225.4 MiB "
        "per-call allocation measured on the .201 DEV lane"
    )
    assert cursor.rows_yielded == 2_000


def test_query_holds_one_driver_row_at_a_time_not_the_whole_result_set() -> None:
    """The peak number of live driver rows must not scale with the result set.

    This is the 225.4 MiB directly: 91,571 ``RealDictCursor`` rows alive at
    once were 200.7 MiB of it, and they were alive only because ``fetchall()``
    had to build the whole list before the comprehension could start copying
    it. Streaming releases each row at the copy, so the peak is a constant.
    """
    _DriverRow.reset()
    rows, cursor = _query(20_000)

    assert len(rows) == 20_000  # type: ignore[arg-type]
    assert cursor.rows_yielded == 20_000
    assert _DriverRow.peak_live <= 2, (
        f"{_DriverRow.peak_live} driver rows were alive at once for a "
        "20,000-row read: the seam materialises the whole result set before "
        "copying it, which is the shape that cost 200.7 MiB on one call"
    )


def test_query_refuses_a_result_set_above_the_row_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Above the budget the seam REFUSES; it never truncates.

    A ``LIMIT`` appended to the statement would return a short answer that
    reads exactly like a complete one -- the silent-fallback shape this
    codebase refuses everywhere else. The budget raises instead, naming the
    relation and the bound so the caller is repairable.
    """
    monkeypatch.setattr(handler_wiring, "PROJECTION_QUERY_MAX_ROWS", 32)

    with pytest.raises(ProjectionQueryRowBudgetError) as excinfo:
        _query(33)

    message = str(excinfo.value)
    assert "generation_events" in message
    assert "32" in message


def test_the_refusal_itself_stays_inside_the_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refusing must not first allocate the thing it is refusing.

    Counting after materialising would make the guard useless: the OOM happens
    while the list is being built. The bound is therefore checked per row, so
    at most ``PROJECTION_QUERY_MAX_ROWS`` rows are ever converted.
    """
    monkeypatch.setattr(handler_wiring, "PROJECTION_QUERY_MAX_ROWS", 32)

    target = _read_target()
    conn, cursor = _connection("omninode_runtime", 10_000)
    with patch("psycopg2.connect", return_value=conn):
        adapter = _adapter(target)
        with pytest.raises(ProjectionQueryRowBudgetError):
            adapter.query("generation_events")  # type: ignore[attr-defined]

    assert cursor.rows_yielded <= 33, (
        f"the seam pulled {cursor.rows_yielded} rows before refusing a 32-row "
        "budget; the refusal allocated the result set it exists to prevent"
    )


def test_query_at_the_budget_returns_every_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive control: the budget is a ceiling, not an off-by-one truncation."""
    monkeypatch.setattr(handler_wiring, "PROJECTION_QUERY_MAX_ROWS", 32)

    rows, cursor = _query(32)

    assert len(rows) == 32  # type: ignore[arg-type]
    assert cursor.rows_yielded == 32
    assert rows[0]["snapshot_id"] == "snapshot_id-value-00000000"  # type: ignore[index]


def test_shipped_bounds_fit_the_runtime_container_budget() -> None:
    """Pin the two shipped numbers to the arithmetic that chose them.

    The runtime container limit is 1536 MiB and its post-subscription baseline
    on the DEV lane measured ~462 MiB, leaving ~1,051 MiB of headroom. The
    per-row cost of the retained copy measured 284 B (24.8 MiB / 91,571 rows).
    The seam's worst case is therefore
    ``PROJECTION_HANDLER_MAX_INFLIGHT * PROJECTION_QUERY_MAX_ROWS * 284 B``.

    THIS ASSERTION WAS WRONG AS SHIPPED, in the way a loose bound always is: the
    docstring said the share was 256 MiB, the module comment said 256 MiB, the
    real product is 270.84 MiB, and the assertion admitted anything up to 300
    MiB -- so three mutually inconsistent numbers all passed together. The share
    is now stated as the product itself, 271 MiB, which is 25.8% of the measured
    headroom. ``test_the_declared_memory_share_is_the_arithmetic_to_the_byte``
    in ``test_projection_ordered_read_omn17888.py`` pins it as an equality;
    this test keeps the containment statement it was written to make.
    """
    measured_bytes_per_row = 284
    worst_case = (
        PROJECTION_HANDLER_MAX_INFLIGHT
        * PROJECTION_QUERY_MAX_ROWS
        * measured_bytes_per_row
    )
    assert worst_case <= 271 * 1024 * 1024, (
        f"worst-case projection read = {worst_case / 1024 / 1024:.2f} MiB "
        "exceeds the declared 271 MiB share of the 1,051 MiB container headroom"
    )
    # The measured baseline plus the whole share must still leave the container
    # room to work in, which is the statement the share is FOR.
    assert 462 + (worst_case / 1024 / 1024) < 1_536


def test_blocking_projection_invocations_are_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The REAL dispatch callback must cap its own in-flight blocking calls.

    ``asyncio.to_thread`` runs on the loop's DEFAULT executor, whose worker
    count is ``min(32, os.cpu_count() + 4)`` -- 32 in the runtime container,
    and py-spy counted exactly 32 live ``asyncio_N`` threads there with 86% of
    sampled stacks inside one projection query. The multiplier on the per-call
    read budget was therefore a property of the host's core count.

    The default executor is set to 32 workers explicitly here rather than left
    to ``os.cpu_count()``: the deployed multiplier is what is under test, and a
    4-core CI runner would otherwise cap the pre-fix shape at 8 by accident and
    report this green without the gate.
    """
    configure_projection_dsns(monkeypatch, url="postgresql://fixture")

    live = 0
    peak = 0
    lock = threading.Lock()

    class _BlockingProjectionHandler:
        def handle(self, input_data: dict[str, object]) -> dict[str, object]:
            nonlocal live, peak
            with lock:
                live += 1
                peak = max(peak, live)
            time.sleep(0.05)
            with lock:
                live -= 1
            return {"rows_upserted": 1}

    callback = _make_projection_dispatch_callback(
        _BlockingProjectionHandler(),
        projection_database_target("pr_merged_events", schema="omninode_internal"),
        ("onex.evt.github.pr-merged.v1",),
        sinks=ProjectionDispatchSinks(),
    )

    def _envelope() -> MagicMock:
        envelope = MagicMock()
        envelope.topic = "onex.evt.github.pr-merged.v1"
        envelope.payload = {"pr_number": 1}
        return envelope

    async def _drive() -> None:
        loop = asyncio.get_running_loop()
        executor = ThreadPoolExecutor(max_workers=32)
        loop.set_default_executor(executor)
        try:
            await asyncio.gather(*(callback(_envelope()) for _ in range(24)))
        finally:
            executor.shutdown(wait=True)

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring."
        "_build_projection_db_adapter",
        return_value=MagicMock(),
    ):
        asyncio.run(_drive())

    assert peak > 0, "no dispatch ran; the measurement is vacuous"
    assert peak <= PROJECTION_HANDLER_MAX_INFLIGHT, (
        f"{peak} projection handler invocations were blocking at once against a "
        f"declared ceiling of {PROJECTION_HANDLER_MAX_INFLIGHT}; the read budget "
        "is multiplied by whatever the loop's default executor admits"
    )
