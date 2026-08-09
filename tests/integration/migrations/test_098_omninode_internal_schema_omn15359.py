# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-15359 — the `omninode_internal` schema must exist, additively, in
`omnidash_analytics`.

THE GAP (live evidence, OMN-15426 readback, rolling ledger 2026-08-03T19:2xZ)

    `handler_wiring.py` auto-wiring issues schema-qualified SQL against
    contract-declared `db_io.schema=omninode_internal` for the 41-table
    `omninode_runtime` domain. The schema itself has never been created
    anywhere in the migration corpus -- every table lands, unqualified, in
    `public`. A grant was withheld because the target relation would not
    resolve: "relation does not exist", not a permission error.

This proof applies `098_create_omninode_internal_schema.sql` through the same
`psql -v ON_ERROR_STOP=1 -f` invocation `scripts/run-forward-migrations.sh`
uses (matching the pattern of `test_097_app_dashboard_connect_omn15297.py`),
against a real ephemeral cluster, and asserts real catalog state -- not string
matching the SQL text.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.integration.migrations.conftest import EphemeralPostgres

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]
FORWARD_DIR = REPO_ROOT / "docker" / "migrations" / "forward"
ROLLBACK_DIR = REPO_ROOT / "docker" / "migrations" / "rollback"
MIGRATION = FORWARD_DIR / "098_create_omninode_internal_schema.sql"
ROLLBACK = ROLLBACK_DIR / "rollback_098_create_omninode_internal_schema.sql"
ANALYTICS_DB = "omnidash_analytics"
TARGET_SCHEMA = "omninode_internal"


def _apply(
    pg: EphemeralPostgres, path: Path, *, dbname: str = "postgres"
) -> subprocess.CompletedProcess[str]:
    result = pg.psql("-v", "ON_ERROR_STOP=1", "-f", str(path), dbname=dbname)
    assert result.returncode == 0, (
        f"{path.name} failed to apply against {dbname}:\n{result.stderr}"
    )
    return result


def _seed_analytics_database(pg: EphemeralPostgres) -> None:
    bootstrap = pg.connect()
    bootstrap.autocommit = True
    with bootstrap.cursor() as cur:
        cur.execute(f'CREATE DATABASE "{ANALYTICS_DB}"')
    bootstrap.close()


def _schema_exists(pg: EphemeralPostgres) -> bool:
    conn = pg.connect(dbname=ANALYTICS_DB)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM pg_catalog.pg_namespace WHERE nspname = %s",
            (TARGET_SCHEMA,),
        )
        exists = cur.fetchone() is not None
    conn.close()
    return exists


def _table_count_in_schema(pg: EphemeralPostgres) -> int:
    conn = pg.connect(dbname=ANALYTICS_DB)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_schema = %s",
            (TARGET_SCHEMA,),
        )
        (count,) = cur.fetchone()
    conn.close()
    return int(count)


def test_098_creates_the_schema_in_omnidash_analytics(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    """The migration's only mutation: `omninode_internal` starts absent, ends present."""
    _seed_analytics_database(ephemeral_postgres)
    assert not _schema_exists(ephemeral_postgres), (
        "test setup invariant: the schema must not pre-exist"
    )

    _apply(ephemeral_postgres, MIGRATION)

    assert _schema_exists(ephemeral_postgres)


def test_098_is_additive_only_zero_tables_created(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    """No table lands in the new schema -- this migration only builds the target."""
    _seed_analytics_database(ephemeral_postgres)
    _apply(ephemeral_postgres, MIGRATION)

    assert _table_count_in_schema(ephemeral_postgres) == 0


def test_098_is_idempotent_on_reapply(ephemeral_postgres: EphemeralPostgres) -> None:
    """Safe to re-run: `CREATE SCHEMA IF NOT EXISTS` on an already-applied lane."""
    _seed_analytics_database(ephemeral_postgres)
    _apply(ephemeral_postgres, MIGRATION)
    _apply(ephemeral_postgres, MIGRATION)

    assert _schema_exists(ephemeral_postgres)


def test_098_rollback_drops_the_empty_schema(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    """Rollback undoes exactly what 098 did, on a lane where it is still empty."""
    _seed_analytics_database(ephemeral_postgres)
    _apply(ephemeral_postgres, MIGRATION)
    assert _schema_exists(ephemeral_postgres)

    _apply(ephemeral_postgres, ROLLBACK)

    assert not _schema_exists(ephemeral_postgres)


def test_098_rollback_refuses_once_a_table_has_landed(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    """RESTRICT, not CASCADE: rollback must fail closed once data has moved in.

    Proves the rollback file's own safety claim rather than trusting its
    comment: this is the exact scenario -- a later migration has copied a
    table into the schema -- the RESTRICT clause exists to protect against.
    """
    _seed_analytics_database(ephemeral_postgres)
    _apply(ephemeral_postgres, MIGRATION)

    conn = ephemeral_postgres.connect(dbname=ANALYTICS_DB)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(f"CREATE TABLE {TARGET_SCHEMA}.copied_family (id TEXT PRIMARY KEY)")
    conn.close()

    result = ephemeral_postgres.psql(
        "-v", "ON_ERROR_STOP=1", "-f", str(ROLLBACK), dbname="postgres"
    )
    assert result.returncode != 0
    assert "depends on" in result.stderr or "cannot drop" in result.stderr
    assert _schema_exists(ephemeral_postgres), (
        "RESTRICT must have refused the drop; the schema must still be present"
    )
