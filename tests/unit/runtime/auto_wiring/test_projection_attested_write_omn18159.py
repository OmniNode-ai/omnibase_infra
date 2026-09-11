# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18159: the runtime kernel's projection adapter can perform an attested write.

WHAT THIS CLOSES

``delegation_events`` carries a durable writer attestation -- ``writer_identity``
stamped by Postgres as ``CURRENT_USER`` and ``written_at`` as ``NOW()``, both
restated on the ``DO UPDATE`` arm -- and the staging green bar reads it to answer
"which database principal wrote this row".

Four implementations of that write exist across this workspace. Three of them
(the delegation projection runner's asyncpg path and ``omnimarket``'s three sync
adapters) can express it. ``ProjectionDatabaseOperations`` -- the adapter the
runtime kernel injects into every in-process projection handler -- could not:
every column was a bound parameter, only the conflict keys were held off the
update arm, there was no ``RETURNING`` clause, and the method returned ``bool``.

So an in-process handler running under the kernel could not stamp the
attestation at all. The omnimarket handler therefore REFUSES a store without the
capability rather than falling back, because the fallback persists the row with
``writer_identity`` NULL on every update arm -- a column DEFAULT is consulted
only on INSERT -- and a NULL there reads exactly like "nobody wrote this",
which is indistinguishable from "an unscoped principal wrote this". This module
removes that refusal by giving the kernel the capability.

WHY THE DECISIONS COME FROM ``omnibase_core`` AND THE SQL DOES NOT

``ModelUpsertPlan`` (``omnibase_core.models.projection``) owns which column goes
on which arm, which expression is admissible, and which identifier is safe. It
is shared precisely so a security decision -- the closed set of expressions that
reach the statement uncast -- cannot exist in four divergent copies.

The rendered SQL is NOT taken from it, deliberately. This adapter quotes every
identifier and schema-qualifies the table (``"schema"."table"``), which no other
consumer does. Rendering here from the shared plan keeps the statement this
adapter has always emitted byte-for-byte while the decisions stay in one place,
and ``test_a_plain_upsert_statement_is_unchanged`` pins that byte-for-byte claim
rather than leaving it asserted.
"""

from __future__ import annotations

from collections.abc import Iterator
from types import MappingProxyType
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from omnibase_core.models.projection import WRITE_ATTESTATION_COLUMNS
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    ProjectionDatabaseTarget,
    _build_projection_db_adapter,
)
from tests.helpers.application_db_topology import (
    projection_database_target,
    projection_database_urls,
)

pytestmark = pytest.mark.unit

TABLE = "generation_events"


class _RecordingCursor:
    """DB-API cursor double that records SQL and serves a RETURNING row."""

    def __init__(self, returned: list[dict[str, Any]] | None = None) -> None:
        self.executed: list[tuple[str, object]] = []
        self._returned = returned or []
        self.description: object | None = None

    def execute(self, sql: str, params: object = None) -> None:
        self.executed.append((sql, params))
        # A real driver sets `description` only for a row-returning statement.
        self.description = [("col",)] if " RETURNING " in sql else None

    def fetchall(self) -> list[dict[str, Any]]:
        return list(self._returned)

    def fetchone(self) -> tuple[str, str]:
        return ("omninode_runtime", "omnidash_analytics")

    def __enter__(self) -> _RecordingCursor:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


@pytest.fixture(autouse=True)
def _patched_driver() -> Iterator[None]:
    """Keep ``psycopg2.connect`` patched for the whole test.

    This adapter opens its connection LAZILY, at the first statement, so a
    patch that covered only construction would let the write escape to a real
    driver. Patching for the test duration is what makes the statement under
    test the one the adapter actually composed.
    """
    with patch("psycopg2.connect", new=lambda *a, **k: _CONNECTIONS[-1]):
        yield


_CONNECTIONS: list[Any] = []


def _adapter(
    cursor: _RecordingCursor, *, access: str = "write"
) -> tuple[Any, ProjectionDatabaseTarget]:
    """Build the REAL adapter over a DB-API double."""
    target = projection_database_target(
        TABLE, schema="omninode_internal", access=access
    )
    conn = MagicMock()
    conn.closed = False
    conn.autocommit = True
    conn.cursor.return_value = cursor
    _CONNECTIONS.append(conn)
    adapter = _build_projection_db_adapter(
        projection_database_urls(target, "postgresql://fixture"),
        target,
        None,
        None,
    )
    return adapter, target


def _sql(cursor: _RecordingCursor) -> str:
    """The last non-probe statement the adapter issued."""
    statements = [
        sql for sql, _ in cursor.executed if sql.lstrip().upper().startswith("INSERT")
    ]
    assert statements, cursor.executed
    return statements[-1]


class TestTheCapabilityExists:
    def test_the_operations_router_exposes_upsert_returning(self) -> None:
        adapter, _ = _adapter(_RecordingCursor())
        assert hasattr(adapter, "upsert_returning")

    def test_plain_upsert_still_returns_a_bool(self) -> None:
        """Its many callers ask "did it write", not "what is stored"."""
        cursor = _RecordingCursor()
        adapter, _ = _adapter(cursor)
        assert adapter.upsert(TABLE, "correlation_id", {"correlation_id": "c1"}) is True


class TestTheAttestationReachesBothArms:
    def test_the_expressions_are_uncast_on_insert_and_restated_on_update(self) -> None:
        """A bound parameter would let this process choose what the row says.

        And the update arm has to RESTATE the expression rather than copy it
        from ``EXCLUDED``: a column DEFAULT is consulted only on INSERT, so an
        existing row would otherwise wear its first writer's stamp forever.
        """
        cursor = _RecordingCursor([{"correlation_id": "c1"}])
        adapter, _ = _adapter(cursor)
        adapter.upsert_returning(
            TABLE,
            "correlation_id",
            {"correlation_id": "c1", "task_type": "t"},
            sql_expression_columns=WRITE_ATTESTATION_COLUMNS,
        )
        sql = _sql(cursor)
        # Present on the INSERT arm, unquoted and unparameterised.
        assert "CURRENT_USER, NOW()" in sql
        assert "%(writer_identity)s" not in sql
        assert "%(written_at)s" not in sql
        # Restated on the UPDATE arm, not assigned from EXCLUDED.
        assert '"writer_identity" = CURRENT_USER' in sql
        assert '"written_at" = NOW()' in sql
        assert 'EXCLUDED."writer_identity"' not in sql

    def test_an_expression_that_is_also_a_row_value_is_refused(self) -> None:
        cursor = _RecordingCursor()
        adapter, _ = _adapter(cursor)
        with pytest.raises(ValueError, match="must not also be supplied as row values"):
            adapter.upsert_returning(
                TABLE,
                "correlation_id",
                {"correlation_id": "c1", "writer_identity": "i_said_so"},
                sql_expression_columns=WRITE_ATTESTATION_COLUMNS,
            )

    def test_an_expression_outside_the_closed_set_is_refused(self) -> None:
        """The set is closed because these strings reach the statement uncast."""
        cursor = _RecordingCursor()
        adapter, _ = _adapter(cursor)
        with pytest.raises(ValueError, match="not in the allowed write-attestation"):
            adapter.upsert_returning(
                TABLE,
                "correlation_id",
                {"correlation_id": "c1"},
                sql_expression_columns=MappingProxyType(
                    {"writer_identity": "(SELECT 1)"}
                ),
            )


class TestInsertOnlyColumns:
    def test_they_are_on_the_insert_arm_and_off_the_update_arm(self) -> None:
        """Both halves are load-bearing under row-level security.

        Postgres evaluates the policy's WITH CHECK against the PROPOSED insert
        row before the conflict is resolved, so dropping ``tenant_id`` from the
        column list entirely is refused outright -- even for a statement that
        was only ever going to take the update arm.
        """
        cursor = _RecordingCursor([{"correlation_id": "c1"}])
        adapter, _ = _adapter(cursor)
        adapter.upsert_returning(
            TABLE,
            "correlation_id",
            {"correlation_id": "c1", "created_at": "t", "task_type": "x"},
            insert_only_columns=frozenset({"created_at"}),
        )
        sql = _sql(cursor)
        assert '"created_at"' in sql.split("ON CONFLICT")[0]
        assert 'EXCLUDED."created_at"' not in sql
        assert 'EXCLUDED."task_type"' in sql


class TestReturning:
    def test_the_clause_is_appended_and_the_rows_come_back(self) -> None:
        cursor = _RecordingCursor([{"correlation_id": "c1", "writer_identity": "who"}])
        adapter, _ = _adapter(cursor)
        rows = adapter.upsert_returning(
            TABLE,
            "correlation_id",
            {"correlation_id": "c1"},
            returning=("correlation_id", "writer_identity"),
        )
        assert ' RETURNING "correlation_id", "writer_identity"' in _sql(cursor)
        assert rows == [{"correlation_id": "c1", "writer_identity": "who"}]

    def test_no_returning_clause_returns_no_rows(self) -> None:
        """Not an empty-row guess: the statement was never asked for rows."""
        cursor = _RecordingCursor([{"correlation_id": "c1"}])
        adapter, _ = _adapter(cursor)
        assert (
            adapter.upsert_returning(TABLE, "correlation_id", {"correlation_id": "c1"})
            == []
        )
        assert " RETURNING " not in _sql(cursor)


class TestTheExistingStatementIsUnchanged:
    """The non-regression claim, pinned rather than asserted.

    This adapter quotes every identifier and schema-qualifies the table, which
    no other consumer of the shared plan does. Driving the rendering from the
    plan is only safe if the statement it emits for an ordinary upsert is the
    one it has always emitted.
    """

    def test_a_plain_upsert_statement_is_unchanged(self) -> None:
        cursor = _RecordingCursor()
        adapter, target = _adapter(cursor)
        adapter.upsert(
            TABLE, "correlation_id", {"correlation_id": "c1", "task_type": "t"}
        )
        schema = target.table_targets[0].physical_schema
        # S608 reads any f-string containing SQL as an injection vector. This
        # one is the EXPECTED value of an assertion -- it is compared against,
        # never executed -- and interpolating the schema is what makes the
        # comparison byte-exact against whatever the topology resolved.
        assert _sql(cursor) == (
            f'INSERT INTO "{schema}"."{TABLE}" ("correlation_id", "task_type") '  # noqa: S608
            "VALUES (%(correlation_id)s, %(task_type)s) "
            'ON CONFLICT ("correlation_id") '
            'DO UPDATE SET "task_type" = EXCLUDED."task_type"'
        )

    def test_a_row_of_only_conflict_keys_still_does_nothing(self) -> None:
        cursor = _RecordingCursor()
        adapter, _ = _adapter(cursor)
        adapter.upsert(TABLE, "correlation_id", {"correlation_id": "c1"})
        assert _sql(cursor).endswith('ON CONFLICT ("correlation_id") DO NOTHING')


class TestWriteAccessIsStillEnforced:
    def test_a_read_only_relation_refuses_the_attested_write_too(self) -> None:
        """The new entry point must not become a way around the access check."""
        cursor = _RecordingCursor()
        adapter, _ = _adapter(cursor, access="read")
        with pytest.raises(PermissionError, match="write refused"):
            adapter.upsert_returning(TABLE, "correlation_id", {"correlation_id": "c1"})


class TestDomainGuardsReachTheNewEntryPoint:
    """The bug this nearly shipped with, now pinned.

    Each domain subclass guards writes -- the internal one rejects a canonical
    ``tenant_id``, the tenant one resolves attribution and scope, the catalog
    one enforces its declared access. Those guards hung off ``upsert``, so an
    ``upsert_returning`` added beside it inherited the BASE implementation and
    silently skipped every one of them. Both entry points now route through
    one ``_prepare_write`` hook, and this asserts it rather than trusting it.
    """

    def test_the_internal_domain_still_rejects_a_canonical_tenant_id(self) -> None:
        cursor = _RecordingCursor()
        adapter, _ = _adapter(cursor)
        with pytest.raises(ValueError, match="rejects canonical tenant_id"):
            adapter.upsert_returning(
                TABLE,
                "correlation_id",
                {"correlation_id": "c1", "tenant_id": "t"},
            )

    def test_a_caller_supplied_tenant_override_is_refused(self) -> None:
        """Scope is derived, never asserted by the caller.

        Accepting an override here would reintroduce the attribution-from-the-
        caller path the tenant operation's guards exist to close.
        """
        cursor = _RecordingCursor()
        adapter, _ = _adapter(cursor)
        with pytest.raises(ValueError, match="not a caller-supplied value"):
            adapter.upsert_returning(
                TABLE, "correlation_id", {"correlation_id": "c1"}, tenant="someone"
            )
