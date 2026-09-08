# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-16770 durable close: idempotency is INTERNAL-domain, so the seam passes.

``test_savings_estimates_rls_guc_seam_omn16770.py`` pins the OMN-16770 seam
itself — that an idempotency read which cannot answer truthfully is refused
rather than silently inverted. That seam is correct and stays. This file pins
the other half: that the read the seam guards is one this node can actually
answer, so the refusal is not the permanent steady state.

Before this change ``_find_ready_sessions`` anti-joined ``savings_estimates``,
a TENANT relation under ``ENABLE`` + ``FORCE ROW LEVEL SECURITY`` with the
policy ``tenant_id = current_setting('app.tenant_id', true)``, which this node
neither owns nor writes. Every compose lane runs the correlation pool as
``omninode_runtime`` — NOSUPERUSER / NOBYPASSRLS / non-owner (OMN-16843) — and
this node carries no tenant attribution to bind (neither signal table has a
``tenant_id`` column; inventing one is what the OMN-16831 ruling forbids). So
the seam refused on every 60s tick, forever: measured on the ``.201`` dev lane
as 480 ``SavingsCorrelationUnscopedReadError`` refusals in four hours, and the
batch has never produced an estimate on any lane.

The close, named on OMN-16770 itself as its remaining acceptance criterion, is
to stop reading a TENANT relation for INTERNAL idempotency at all. The node
now records its own publications in ``omninode_internal.
savings_correlation_finalizations`` — a relation it owns, writes, and can read
truthfully — and the anti-join reads that. The seam is unchanged and still
runs on the same connection immediately before the candidate query; it now
passes *by construction* rather than being satisfied by a grant or a bound
scope, which is the difference between a guard that is met and a guard that is
routed around.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

pytestmark = pytest.mark.unit

from omnibase_infra.nodes.node_savings_estimation_compute.handlers import (
    handler_savings_correlation as module,
)
from omnibase_infra.nodes.node_savings_estimation_compute.handlers.handler_savings_correlation import (
    FINALIZATION_RELATION,
    IDEMPOTENCY_RELATION,
    HandlerSavingsCorrelation,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
NODE_MIGRATIONS = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "nodes"
    / "node_savings_estimation_compute"
)
FINALIZATION_MIGRATION = (
    NODE_MIGRATIONS / "0002_create_savings_correlation_finalizations.sql"
)
MIGRATION_LEDGER = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "_ledger"
    / "application-migrations.tsv"
)

#: The TENANT relation the anti-join used to read. Spelled here so that a
#: reintroduction is caught by name rather than by symptom.
TENANT_RELATION = "savings_estimates"


def _executable_sql(path: Path) -> str:
    """The migration's statements with its `--` commentary removed.

    The header explains at length why `savings_estimates` is TENANT-domain and
    what a `tenant_id` column would do here. Asserting over the raw file would
    match that prose and prove nothing about the DDL.
    """
    return "\n".join(line.split("--", 1)[0] for line in path.read_text().splitlines())


# ---------------------------------------------------------------------------
# A connection fake that answers the seam's probe and records what ran.
# ---------------------------------------------------------------------------


class _FakeConnection:
    def __init__(
        self,
        *,
        rls_enforced: bool,
        tenant_scope: str | None = None,
        candidate_session_ids: tuple[str, ...] = (),
    ) -> None:
        self._rls_enforced = rls_enforced
        self._tenant_scope = tenant_scope
        self._candidate_session_ids = candidate_session_ids
        self.executed: list[tuple[str, tuple[object, ...]]] = []
        self.fetched: list[str] = []

    async def execute(self, sql: str, *args: object) -> str:
        self.executed.append((sql, args))
        return "INSERT 0 1"

    async def fetchrow(self, sql: str, *args: object) -> dict[str, Any] | None:
        if "row_security_active" in sql:
            return {
                "rls_enforced": self._rls_enforced,
                "tenant_scope": self._tenant_scope,
            }
        if "session_outcomes" in sql:
            return {"outcome": "success"}
        return None

    @property
    def finalization_inserts(self) -> list[tuple[str, tuple[object, ...]]]:
        return [
            (sql, args)
            for sql, args in self.executed
            if "savings_correlation_finalizations" in sql
        ]

    async def fetch(self, sql: str, *args: object) -> list[dict[str, Any]]:
        self.fetched.append(sql)
        if "candidate_sessions" in sql:
            return [{"session_id": s} for s in self._candidate_session_ids]
        if "savings_injection_signals" in sql:
            return [{"tokens_injected": 1200, "patterns_count": 3}]
        if "llm_call_metrics" in sql:
            return [
                {
                    "model_id": "qwen2.5-coder-32b",
                    "prompt_tokens": 900,
                    "completion_tokens": 300,
                }
            ]
        return []

    @property
    def candidate_query_ran(self) -> bool:
        return any("candidate_sessions" in sql for sql in self.fetched)


class _FakeAcquire:
    def __init__(self, conn: _FakeConnection) -> None:
        self._conn = conn

    async def __aenter__(self) -> _FakeConnection:
        return self._conn

    async def __aexit__(self, *exc: object) -> bool:
        return False


class _FakePool:
    def __init__(self, conn: _FakeConnection) -> None:
        self.conn = conn

    def acquire(self) -> _FakeAcquire:
        return _FakeAcquire(self.conn)


def _handler(conn: _FakeConnection) -> HandlerSavingsCorrelation:
    return HandlerSavingsCorrelation(
        pool=_FakePool(conn),  # type: ignore[arg-type]
        publisher=None,
    )


# ---------------------------------------------------------------------------
# The read is INTERNAL-domain now, and the anti-join proves it.
# ---------------------------------------------------------------------------


def test_the_idempotency_read_is_the_nodes_own_internal_relation() -> None:
    """The relation the seam probes is the one this node owns and writes."""
    assert IDEMPOTENCY_RELATION == FINALIZATION_RELATION
    assert FINALIZATION_RELATION.startswith("omninode_internal.")
    assert FINALIZATION_RELATION.endswith("savings_correlation_finalizations")


def test_the_candidate_query_no_longer_reads_the_tenant_relation() -> None:
    """The whole defect was reading a TENANT relation over an INTERNAL binding.

    Asserted on the SQL text rather than on behaviour, because a behavioural
    test cannot distinguish "does not read it" from "reads it and the fake
    happened to answer". If someone re-adds the anti-join against
    ``savings_estimates``, this fails by name.
    """
    source = inspect.getsource(HandlerSavingsCorrelation._find_ready_sessions)
    assert TENANT_RELATION not in source
    assert f"FROM {FINALIZATION_RELATION} se" in source


@pytest.mark.asyncio
async def test_the_batch_runs_on_the_live_dev_lane_connection_state() -> None:
    """The exact connection state that refused 480 times in four hours.

    ``omninode_runtime`` on the ``.201`` dev lane: no ``app.tenant_id`` bound,
    and row-level security is NOT active on the relation the anti-join reads,
    because that relation is now the node's own internal one. Before the close
    this same state raised ``SavingsCorrelationUnscopedReadError`` and the
    candidate query never ran.
    """
    conn = _FakeConnection(
        rls_enforced=False,
        tenant_scope=None,
        candidate_session_ids=("s1", "s2"),
    )
    assert await _handler(conn)._find_ready_sessions() == ["s1", "s2"]
    assert conn.candidate_query_ran is True


@pytest.mark.asyncio
async def test_a_published_session_is_recorded_so_it_is_not_republished() -> None:
    """Publishing without recording would re-publish the session every tick.

    The marker is what makes the anti-join mean anything, so the WRITE is
    driven end-to-end through ``_finalize_session`` rather than grepped for:
    a test that only reads the source cannot tell a statement that runs from
    one that is unreachable.
    """
    published: list[str] = []

    async def _publisher(**kwargs: Any) -> None:
        published.append(str(kwargs["payload"]["session_id"]))

    conn = _FakeConnection(rls_enforced=False)
    handler = HandlerSavingsCorrelation(
        pool=_FakePool(conn),  # type: ignore[arg-type]
        publisher=_publisher,
    )
    correlation_id = uuid4()
    assert await handler._finalize_session("sess-a", correlation_id) is True
    assert published == ["sess-a"]

    inserts = conn.finalization_inserts
    assert len(inserts) == 1, "exactly one marker row per published estimate"
    sql, args = inserts[0]
    assert "INSERT INTO" in sql
    assert "ON CONFLICT" in sql, (
        "a retried or racing tick must not raise on the marker insert"
    )
    assert args == ("sess-a", correlation_id)


@pytest.mark.asyncio
async def test_no_marker_is_written_when_nothing_is_published() -> None:
    """A session that produced no estimate is not recorded as finalized.

    Recording it would silently drop the session forever the moment its
    signals did arrive.
    """
    conn = _FakeConnection(rls_enforced=False)
    handler = HandlerSavingsCorrelation(
        pool=_FakePool(conn),  # type: ignore[arg-type]
        publisher=None,
    )
    assert await handler._finalize_session("sess-b", uuid4()) is False
    assert conn.finalization_inserts == []


def test_the_marker_write_names_the_declared_relation() -> None:
    """The INSERT's literal relation is the one the constant declares.

    ``_record_finalization`` spells the relation out rather than interpolating
    it, so this is what keeps the write and the anti-join from drifting apart
    — a marker written somewhere else is a marker the anti-join never sees.
    """
    source = inspect.getsource(HandlerSavingsCorrelation._record_finalization)
    assert f"INSERT INTO {FINALIZATION_RELATION}" in source


def test_the_marker_is_written_only_after_a_successful_publish() -> None:
    """At-least-once, deliberately: mark AFTER the publish, never before.

    The behavioural test above proves both happen; it cannot prove the order,
    and the order is the whole failure-mode choice. Marking first and
    publishing second loses the estimate permanently when the publish raises.
    Pinned against a well-meaning reorder.
    """
    source = inspect.getsource(HandlerSavingsCorrelation._finalize_session)
    assert source.index("await self._publisher(") < source.index(
        "_record_finalization("
    )


# ---------------------------------------------------------------------------
# The relation is INTERNAL in the migration too, not only in the Python.
# ---------------------------------------------------------------------------


def test_the_migration_creates_the_relation_in_the_internal_schema() -> None:
    sql = _executable_sql(FINALIZATION_MIGRATION)
    assert re.search(
        r"CREATE TABLE IF NOT EXISTS\s+omninode_internal\.savings_correlation_finalizations",
        sql,
    )


def test_the_migration_puts_no_row_level_security_on_the_relation() -> None:
    """RLS here would rebuild the defect: the seam would refuse again.

    The node has no tenant scope to bind, so a GUC-predicated policy on the
    relation it reads for idempotency puts it straight back into the permanent
    refusal this close removes.
    """
    sql = _executable_sql(FINALIZATION_MIGRATION).upper()
    assert "ROW LEVEL SECURITY" not in sql
    assert "CREATE POLICY" not in sql


def test_the_migration_declares_no_tenant_column() -> None:
    """A `tenant_id` here would be invented attribution (OMN-16831)."""
    assert "tenant_id" not in _executable_sql(FINALIZATION_MIGRATION)


def test_the_migration_grants_the_pool_principal() -> None:
    """`omninode_runtime` is the principal OMNINODE_INTERNAL_DB_URL connects as.

    Without both the read and the write grant the close swaps one runtime
    failure for another.
    """
    sql = _executable_sql(FINALIZATION_MIGRATION)
    grant = re.search(
        r"GRANT\s+([A-Z, ]+)\s+ON\s+omninode_internal\.savings_correlation_finalizations\s+TO\s+omninode_runtime",
        sql,
    )
    assert grant is not None
    granted = {word.strip() for word in grant.group(1).split(",")}
    assert {"SELECT", "INSERT"} <= granted


def test_the_migration_is_registered_in_the_application_ledger() -> None:
    """An unregistered node migration is never applied by the runner."""
    rows = [
        line.split("\t")
        for line in MIGRATION_LEDGER.read_text().splitlines()
        if line.strip()
    ]
    matching = [
        row
        for row in rows
        if row[0]
        == "nodes/node_savings_estimation_compute/0002_create_savings_correlation_finalizations.sql"
    ]
    assert len(matching) == 1, "the 0002 migration is not registered exactly once"
    assert matching[0][3] == "omninode_internal"


def test_the_ledger_sha_matches_the_file_on_disk() -> None:
    """The ledger pins content, so a silent in-place rewrite fails here."""
    import hashlib

    rows = [
        line.split("\t")
        for line in MIGRATION_LEDGER.read_text().splitlines()
        if line.strip()
    ]
    row = next(
        r
        for r in rows
        if r[0]
        == "nodes/node_savings_estimation_compute/0002_create_savings_correlation_finalizations.sql"
    )
    digest = hashlib.sha256(FINALIZATION_MIGRATION.read_bytes()).hexdigest()
    assert row[5].strip() == digest


def test_the_seam_module_still_exports_its_refusal() -> None:
    """The close does not delete the guard — it satisfies it.

    ``SavingsCorrelationUnscopedReadError`` still exists and still fires if the
    relation the anti-join reads ever becomes unanswerable again.
    """
    assert hasattr(module, "SavingsCorrelationUnscopedReadError")
    assert hasattr(module, "_assert_idempotency_read_is_scoped")
    source = inspect.getsource(HandlerSavingsCorrelation._find_ready_sessions)
    assert "_assert_idempotency_read_is_scoped(conn)" in source
