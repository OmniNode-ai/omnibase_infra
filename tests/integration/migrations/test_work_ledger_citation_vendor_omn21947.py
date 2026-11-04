# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Apply the citation prerequisite through the production psql migration path."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.integration.migrations.conftest import EphemeralPostgres

pytestmark = [pytest.mark.integration, pytest.mark.postgres, pytest.mark.ephemeral_pg]

_CHAIN = (
    Path(__file__).resolve().parents[3]
    / "docker/migrations/forward/nodes/node_projection_work_ledger"
)
_HISTORY = "0007_work_ledger_history_import.sql"
_GRANT = "0008_grant_omninode_runtime_ledger_line_citation.sql"


def _sql(pg: EphemeralPostgres, sql: str) -> str:
    result = pg.psql("-v", "ON_ERROR_STOP=1", "-At", "-c", sql)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _apply(pg: EphemeralPostgres, name: str) -> None:
    result = pg.psql("-v", "ON_ERROR_STOP=1", "-f", str(_CHAIN / name))
    assert result.returncode == 0, result.stderr


@pytest.fixture
def ledger(ephemeral_postgres: EphemeralPostgres) -> EphemeralPostgres:
    _sql(
        ephemeral_postgres,
        "CREATE SCHEMA omninode_internal; "
        "CREATE ROLE omninode_runtime NOLOGIN; "
        "CREATE TABLE omninode_internal.work_ledger_rows "
        "(row_id text PRIMARY KEY, ledger_id text, ledger_seq bigint); "
        "INSERT INTO omninode_internal.work_ledger_rows VALUES "
        "('assigned', 'ledger', 99042), ('legacy', 'ledger', NULL)",
    )
    return ephemeral_postgres


@pytest.mark.parametrize("existing", [False, True], ids=["fresh", "existing-index"])
def test_citation_resolution_has_a_primary_key_on_both_paths(
    ledger: EphemeralPostgres, existing: bool
) -> None:
    if existing:
        _sql(
            ledger,
            "CREATE TABLE omninode_internal.ledger_line_citation "
            "(ledger_id text, file_key text, generation int, line_no int, row_id text); "
            "CREATE UNIQUE INDEX ledger_line_citation_pkey ON "
            "omninode_internal.ledger_line_citation "
            "(ledger_id, file_key, generation, line_no); "
            "INSERT INTO omninode_internal.ledger_line_citation "
            "VALUES ('ledger', 'live', 0, 1, 'legacy')",
        )
    _apply(ledger, _HISTORY)
    _apply(ledger, _HISTORY)
    assert (
        _sql(
            ledger,
            "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
            "WHERE conrelid='omninode_internal.ledger_line_citation'::regclass "
            "AND contype='p'",
        )
        == "PRIMARY KEY (ledger_id, file_key, generation, line_no)"
    )
    assert (
        _sql(
            ledger,
            "SELECT row_id || ':' || coalesce(ledger_seq::text, 'null') "
            "FROM omninode_internal.work_ledger_rows ORDER BY row_id",
        )
        == "assigned:99042\nlegacy:null"
    )
    _sql(
        ledger,
        "UPDATE omninode_internal.work_ledger_rows SET legacy_ordinal=1 "
        "WHERE row_id='legacy'",
    )
    duplicate = ledger.psql(
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        "INSERT INTO omninode_internal.work_ledger_rows "
        "VALUES ('duplicate', 'ledger', NULL, 1)",
    )
    assert duplicate.returncode != 0
    assert "idx_work_ledger_rows_legacy_ordinal" in duplicate.stderr


def test_citation_resolution_generation_and_immutable_runtime_grants(
    ledger: EphemeralPostgres,
) -> None:
    _apply(ledger, _HISTORY)
    _apply(ledger, _GRANT)
    _apply(ledger, _GRANT)
    assert (
        _sql(
            ledger,
            "SELECT privilege_type FROM information_schema.role_table_grants "
            "WHERE grantee='omninode_runtime' AND table_schema='omninode_internal' "
            "AND table_name='ledger_line_citation' ORDER BY privilege_type",
        )
        == "INSERT\nSELECT\nUPDATE"
    )
    _sql(
        ledger,
        "SET ROLE omninode_runtime; "
        "INSERT INTO omninode_internal.ledger_line_citation VALUES "
        "('ledger', 'live', 0, 1, 'legacy'), ('ledger', 'live', 1, 1, 'assigned')",
    )
    assert _sql(
        ledger,
        "SET ROLE omninode_runtime; SELECT row_id FROM "
        "omninode_internal.ledger_line_citation ORDER BY generation",
    ).splitlines()[-2:] == ["legacy", "assigned"]
    for operation in (
        "UPDATE omninode_internal.ledger_line_citation SET row_id='changed'",
        "DELETE FROM omninode_internal.ledger_line_citation",
    ):
        refused = ledger.psql("-v", "ON_ERROR_STOP=1", "-c", operation)
        assert refused.returncode != 0
        assert "ledger_line_citation is immutable" in refused.stderr
    duplicate = ledger.psql(
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        "INSERT INTO omninode_internal.ledger_line_citation "
        "VALUES ('ledger', 'live', 0, 1, 'different')",
    )
    assert duplicate.returncode != 0
    assert "duplicate key" in duplicate.stderr


def test_history_import_refuses_to_invent_missing_citation_identities(
    ledger: EphemeralPostgres,
) -> None:
    _sql(
        ledger,
        "CREATE TABLE omninode_internal.ledger_line_citation (row_id text); "
        "INSERT INTO omninode_internal.ledger_line_citation VALUES ('legacy')",
    )
    refused = ledger.psql("-v", "ON_ERROR_STOP=1", "-f", str(_CHAIN / _HISTORY))
    assert refused.returncode != 0
    assert 'column "ledger_id"' in refused.stderr
    assert "contains null values" in refused.stderr
    assert (
        _sql(ledger, "SELECT row_id FROM omninode_internal.ledger_line_citation")
        == "legacy"
    )


def test_runtime_grant_refuses_inherited_delete(ledger: EphemeralPostgres) -> None:
    _apply(ledger, _HISTORY)
    _sql(
        ledger,
        "CREATE ROLE inherited_writer NOLOGIN; "
        "GRANT DELETE ON omninode_internal.ledger_line_citation TO inherited_writer; "
        "GRANT inherited_writer TO omninode_runtime",
    )
    refused = ledger.psql("-v", "ON_ERROR_STOP=1", "-f", str(_CHAIN / _GRANT))
    assert refused.returncode != 0
    assert "division by zero" in refused.stderr


def test_history_import_rolls_back_incompatible_citation_schema(
    ledger: EphemeralPostgres,
) -> None:
    _sql(
        ledger,
        "CREATE TABLE omninode_internal.ledger_line_citation (row_id text); "
        "INSERT INTO omninode_internal.ledger_line_citation VALUES ('legacy')",
    )
    refused = ledger.psql("-v", "ON_ERROR_STOP=1", "-f", str(_CHAIN / _HISTORY))
    assert refused.returncode != 0
    assert "contains null values" in refused.stderr
    assert (
        _sql(
            ledger,
            "SELECT count(*) FROM information_schema.columns "
            "WHERE table_schema='omninode_internal' AND table_name='work_ledger_rows' "
            "AND column_name='legacy_ordinal'",
        )
        == "0"
    )
    assert (
        _sql(
            ledger,
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema='omninode_internal' AND table_name='ledger_line_citation'",
        )
        == "row_id"
    )
    assert (
        _sql(ledger, "SELECT row_id FROM omninode_internal.ledger_line_citation")
        == "legacy"
    )


def test_history_import_refuses_an_incompatible_existing_primary_key(
    ledger: EphemeralPostgres,
) -> None:
    _sql(
        ledger,
        "CREATE TABLE omninode_internal.ledger_line_citation "
        "(ledger_id text, file_key text, generation int, line_no int, "
        "row_id text PRIMARY KEY); "
        "INSERT INTO omninode_internal.ledger_line_citation "
        "VALUES ('ledger', 'live', 0, 1, 'legacy')",
    )
    refused = ledger.psql("-v", "ON_ERROR_STOP=1", "-f", str(_CHAIN / _HISTORY))
    assert refused.returncode != 0
    assert "ledger_line_citation has an incompatible primary key" in refused.stderr
    assert (
        _sql(
            ledger,
            "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
            "WHERE conrelid='omninode_internal.ledger_line_citation'::regclass "
            "AND contype='p'",
        )
        == "PRIMARY KEY (row_id)"
    )
    assert (
        _sql(ledger, "SELECT row_id FROM omninode_internal.ledger_line_citation")
        == "legacy"
    )


def test_history_import_and_runtime_grant_follow_ingestion_time() -> None:
    order = sorted(path.name for path in _CHAIN.glob("*.sql"))
    assert order.index("0006_work_ledger_ingested_at.sql") < order.index(_HISTORY)
    assert order.index(_HISTORY) < order.index(_GRANT)
