# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-16770: the savings_estimates idempotency read must never run unscoped.

The defect this pins
--------------------
``HandlerSavingsCorrelation._find_ready_sessions`` ends its candidate query
with an anti-join::

    AND NOT EXISTS (
        SELECT 1 FROM savings_estimates se WHERE se.session_id = cs.session_id
    )

``savings_estimates`` is a TENANT relation. ``node_projection_savings/
081_savings_estimates_rls_tenant_isolation.sql`` puts it under ``ENABLE`` **and**
``FORCE ROW LEVEL SECURITY`` with the policy
``tenant_id = current_setting('app.tenant_id', true)``. The pool that runs this
query is built from ``OMNINODE_INTERNAL_DB_URL``, whose principal
``omninode_runtime`` is pinned NOSUPERUSER / NOBYPASSRLS / non-owner by
``docker/docker-compose.infra.yml``.

That leaves exactly two reachable states, and BOTH are wrong:

* **No grant (today).** The read raises ``InsufficientPrivilegeError`` deep
  inside the candidate query. Loud, but no estimate is ever produced.
* **A bare ``GRANT SELECT`` (the obvious "fix").** The policy now evaluates
  ``tenant_id = NULL`` for every row, which is NULL, which is not TRUE — so the
  subquery matches nothing. ``NOT EXISTS`` becomes universally true, every
  candidate session reads as "never finalized", and the batch re-publishes an
  estimate for **every** session on **every** 60s tick, forever. Silent,
  unbounded, and strictly worse than the error it replaced.

So the grant is not the seam, and adding one must not be able to invert the
anti-join. These tests pin that: an RLS-enforced connection with no tenant
scope must **REFUSE**, and must refuse *before* the candidate query is ever
issued — a query that never runs cannot return the zero rows that invert it.

The complement is pinned too: a connection for which row-level security is not
active (the owner / BYPASSRLS / superuser case ``081``'s own header calls out
for compose lanes) reads legitimately and is NOT refused. "Proper tenant scope
or the right role" — anything else refuses.

Ticket: OMN-16770
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

pytestmark = pytest.mark.unit

from omnibase_infra.nodes.node_savings_estimation_compute.handlers.handler_savings_correlation import (
    IDEMPOTENCY_RELATION,
    TENANT_GUC,
    HandlerSavingsCorrelation,
    SavingsCorrelationUnscopedReadError,
)
from omnibase_infra.nodes.node_savings_estimation_compute.models.model_savings_correlation_batch_command import (
    ModelSavingsCorrelationBatchCommand,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
RLS_MIGRATION = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "nodes"
    / "node_projection_savings"
    / "081_savings_estimates_rls_tenant_isolation.sql"
)


# ---------------------------------------------------------------------------
# A connection fake that models the two premises the seam reads, and records
# every statement so "the candidate query never ran" is assertable.
# ---------------------------------------------------------------------------


class _FakeConnection:
    def __init__(
        self,
        *,
        rls_enforced: bool | None,
        tenant_scope: str | None = None,
        answer_probe: bool = True,
        candidate_session_ids: tuple[str, ...] = (),
    ) -> None:
        self._rls_enforced = rls_enforced
        self._tenant_scope = tenant_scope
        self._answer_probe = answer_probe
        self._candidate_session_ids = candidate_session_ids
        self.executed: list[str] = []
        self.fetched: list[str] = []

    # -- asyncpg surface ---------------------------------------------------
    async def execute(self, sql: str, *args: object) -> str:
        self.executed.append(sql)
        return "SET"

    async def fetchrow(self, sql: str, *args: object) -> dict[str, Any] | None:
        if "row_security_active" in sql:
            if not self._answer_probe:
                return None
            return {
                "rls_enforced": self._rls_enforced,
                "tenant_scope": self._tenant_scope,
            }
        return None

    async def fetch(self, sql: str, *args: object) -> list[dict[str, Any]]:
        self.fetched.append(sql)
        if "candidate_sessions" in sql:
            return [{"session_id": s} for s in self._candidate_session_ids]
        return []

    # -- helpers -----------------------------------------------------------
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


def _handler(conn: _FakeConnection, publisher: Any = None) -> HandlerSavingsCorrelation:
    return HandlerSavingsCorrelation(
        pool=_FakePool(conn),  # type: ignore[arg-type]
        publisher=publisher,
    )


# ---------------------------------------------------------------------------
# The inversion itself: granted-but-unscoped must refuse, not return 0 rows.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rls_enforced_without_tenant_scope_refuses() -> None:
    """The exact granted-but-unscoped state a bare GRANT would produce."""
    conn = _FakeConnection(
        rls_enforced=True,
        tenant_scope=None,
        candidate_session_ids=("s1", "s2", "s3"),
    )

    with pytest.raises(SavingsCorrelationUnscopedReadError) as excinfo:
        await _handler(conn)._find_ready_sessions()

    message = str(excinfo.value)
    assert IDEMPOTENCY_RELATION in message
    assert TENANT_GUC in message


@pytest.mark.asyncio
async def test_the_candidate_query_never_runs_when_the_read_is_unscoped() -> None:
    """The anti-join inversion is structurally unreachable, not merely caught.

    A refusal raised *after* the candidate query would still have executed the
    inverted anti-join. Asserting the statement was never issued is what makes
    "the inversion CANNOT happen" a proof rather than a hope.
    """
    conn = _FakeConnection(
        rls_enforced=True,
        tenant_scope=None,
        candidate_session_ids=("s1", "s2", "s3"),
    )

    with pytest.raises(SavingsCorrelationUnscopedReadError):
        await _handler(conn)._find_ready_sessions()

    assert conn.candidate_query_ran is False


@pytest.mark.asyncio
async def test_blank_tenant_scope_is_not_a_scope() -> None:
    """An empty-string GUC compares against '' and matches no tenant row.

    ``nullif(..., '')`` normalizes it in SQL; this pins that the Python side
    refuses the same way rather than treating a blank string as scoped.
    """
    conn = _FakeConnection(rls_enforced=True, tenant_scope="   ")

    with pytest.raises(SavingsCorrelationUnscopedReadError):
        await _handler(conn)._find_ready_sessions()

    assert conn.candidate_query_ran is False


@pytest.mark.asyncio
async def test_an_unreadable_premise_fails_closed() -> None:
    """A probe that answers nothing must refuse, not read as "no RLS here".

    Otherwise a driver/permission change that silences the probe would
    re-open the inversion with no signal at all.
    """
    conn = _FakeConnection(rls_enforced=True, answer_probe=False)

    with pytest.raises(SavingsCorrelationUnscopedReadError):
        await _handler(conn)._find_ready_sessions()

    assert conn.candidate_query_ran is False


@pytest.mark.asyncio
async def test_a_non_boolean_rls_answer_fails_closed() -> None:
    """``row_security_active`` returns a boolean; anything else is not an answer."""
    conn = _FakeConnection(rls_enforced=None)

    with pytest.raises(SavingsCorrelationUnscopedReadError):
        await _handler(conn)._find_ready_sessions()

    assert conn.candidate_query_ran is False


# ---------------------------------------------------------------------------
# The batch-level consequence: refuse loudly, publish nothing.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_batch_publishes_nothing_when_the_read_is_unscoped() -> None:
    """The inverted anti-join's whole harm is re-publishing every session.

    Under refusal the tick raises out of ``run_correlation_batch`` — which
    ``service_kernel`` already logs and retries in 60s — and the publisher is
    never reached even once.
    """
    published: list[object] = []

    async def _publisher(**kwargs: object) -> bool:
        published.append(kwargs)
        return True

    conn = _FakeConnection(
        rls_enforced=True,
        tenant_scope=None,
        candidate_session_ids=("s1", "s2", "s3"),
    )
    handler = _handler(conn, publisher=_publisher)

    with pytest.raises(SavingsCorrelationUnscopedReadError):
        await handler.run_correlation_batch(
            ModelSavingsCorrelationBatchCommand(correlation_id=uuid4())
        )

    assert published == []


# ---------------------------------------------------------------------------
# The complement: the legitimate reads are NOT refused.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_role_rls_does_not_apply_to_reads_without_a_guc() -> None:
    """The "right role" branch.

    ``row_security_active`` is false for a table owner without FORCE, for a
    BYPASSRLS role, and for a superuser — the compose-lane case ``081``'s own
    header names. Such a connection sees every tenant's rows, so the anti-join
    is answered truthfully and there is nothing to refuse.
    """
    conn = _FakeConnection(
        rls_enforced=False,
        tenant_scope=None,
        candidate_session_ids=("s1", "s2"),
    )

    assert await _handler(conn)._find_ready_sessions() == ["s1", "s2"]
    assert conn.candidate_query_ran is True


@pytest.mark.asyncio
async def test_an_rls_enforced_connection_with_a_tenant_scope_reads() -> None:
    """The "proper tenant scope" branch: RLS applies, and a scope is bound."""
    conn = _FakeConnection(
        rls_enforced=True,
        tenant_scope="omninode",
        candidate_session_ids=("s1",),
    )

    assert await _handler(conn)._find_ready_sessions() == ["s1"]
    assert conn.candidate_query_ran is True


# ---------------------------------------------------------------------------
# The seam's premise is DERIVED from the migration, not asserted from memory.
# ---------------------------------------------------------------------------


def test_the_migration_still_declares_the_premise_the_seam_reads() -> None:
    """If 081 stops forcing RLS or changes its predicate, this seam's
    reasoning changed and the guard must be re-derived rather than trusted."""
    sql = RLS_MIGRATION.read_text()

    assert re.search(
        r"ALTER TABLE savings_estimates\s+FORCE ROW LEVEL SECURITY", sql
    ), "081 no longer FORCEs RLS — the owner is no longer constrained"
    assert f"current_setting('{TENANT_GUC}', true)" in sql, (
        "081's policy no longer compares against the GUC this seam probes"
    )


def test_the_probed_relation_is_the_one_the_anti_join_reads() -> None:
    """A guard aimed at a different relation than the query reads is a no-op."""
    import inspect

    from omnibase_infra.nodes.node_savings_estimation_compute.handlers import (
        handler_savings_correlation as module,
    )

    source = inspect.getsource(module.HandlerSavingsCorrelation._find_ready_sessions)
    assert f"FROM {IDEMPOTENCY_RELATION} se" in source
