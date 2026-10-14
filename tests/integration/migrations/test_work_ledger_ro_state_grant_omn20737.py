# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Live apply proof for the work_ledger_ro grant on work_ledger_state (OMN-20737).

``work_ledger_ro`` is provisioned out of band on the lab dev database, so the
migration is applied here in the two situations it meets: the role exists and
must come away able to read the state table (and nothing more), and the role is
absent, as on a fresh build, where the file must apply cleanly and create none.
The file is driven through ``psql -v ON_ERROR_STOP=1 -f`` as the forward runner
does, by a non-superuser that owns the table.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.integration.migrations.conftest import EphemeralPostgres

REPO_ROOT = Path(__file__).resolve().parents[3]
MIGRATION_FILE = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "nodes"
    / "node_projection_work_ledger"
    / "0004_grant_work_ledger_ro_work_ledger_state.sql"
)
_ROLE = "work_ledger_ro"
_TABLE = "omninode_internal.work_ledger_state"
_MIGRATOR = "omn20737_migrator"
_OTHER_OWNER = "omn20737_other_owner"
_READ_AS_RO = (
    "SET ROLE work_ledger_ro; SELECT count(*) FROM omninode_internal.work_ledger_state;"
)
_DELETE_AS_RO = (
    "SET ROLE work_ledger_ro; DELETE FROM omninode_internal.work_ledger_state;"
)
_RO_PRIVILEGES = (
    "SELECT a.privilege_type FROM pg_catalog.pg_class c"
    " JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace,"
    " aclexplode(c.relacl) a"
    " WHERE n.nspname = 'omninode_internal' AND c.relname = 'work_ledger_state'"
    " AND a.grantee = 'work_ledger_ro'::regrole"
)
_RO_ROLE_COUNT = "SELECT count(*) FROM pg_roles WHERE rolname = 'work_ledger_ro'"


def _run(
    pg: EphemeralPostgres, *args: str, user: str = "postgres"
) -> subprocess.CompletedProcess[str]:
    return pg.psql("-X", "-v", "ON_ERROR_STOP=1", *args, user=user)


def _bootstrap(pg: EphemeralPostgres, *, table_owner: str, ro_role: bool) -> None:
    statements = [
        f"CREATE ROLE {_MIGRATOR} LOGIN NOSUPERUSER NOBYPASSRLS",
        f"CREATE ROLE {_OTHER_OWNER} NOLOGIN",
        f"CREATE SCHEMA omninode_internal AUTHORIZATION {_MIGRATOR}",
        f"GRANT USAGE ON SCHEMA omninode_internal TO {_OTHER_OWNER}",
        f"CREATE TABLE {_TABLE} (entity_id text PRIMARY KEY)",
        f"ALTER TABLE {_TABLE} OWNER TO {table_owner}",
    ]
    if ro_role:
        statements.insert(0, f"CREATE ROLE {_ROLE} NOLOGIN NOSUPERUSER NOBYPASSRLS")
        statements.append(f"GRANT USAGE ON SCHEMA omninode_internal TO {_ROLE}")
    result = _run(pg, "-c", ";\n".join(statements) + ";")
    assert result.returncode == 0, result.stderr


def _apply(pg: EphemeralPostgres) -> subprocess.CompletedProcess[str]:
    return _run(pg, "-f", str(MIGRATION_FILE), user=_MIGRATOR)


def _privileges(pg: EphemeralPostgres) -> set[str]:
    result = _run(pg, "-At", "-c", _RO_PRIVILEGES)
    assert result.returncode == 0, result.stderr
    return set(result.stdout.split())


@pytest.mark.integration
@pytest.mark.ephemeral_pg
def test_the_read_only_role_gains_select_and_nothing_more(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    pg = ephemeral_postgres
    _bootstrap(pg, table_owner=_MIGRATOR, ro_role=True)
    # Positive control: before the file the role is refused the read.
    before = _run(pg, "-c", _READ_AS_RO)
    assert before.returncode != 0
    assert "permission denied" in before.stderr
    assert _privileges(pg) == set()

    first = _apply(pg)
    assert first.returncode == 0, first.stderr
    assert _privileges(pg) == {"SELECT"}
    read = _run(pg, "-At", "-c", _READ_AS_RO)
    assert read.returncode == 0, read.stderr
    write = _run(pg, "-c", _DELETE_AS_RO)
    assert write.returncode != 0
    assert "permission denied" in write.stderr

    again = _apply(pg)
    assert again.returncode == 0, again.stderr
    assert _privileges(pg) == {"SELECT"}


@pytest.mark.integration
@pytest.mark.ephemeral_pg
def test_a_build_without_the_role_applies_cleanly_and_creates_none(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    pg = ephemeral_postgres
    _bootstrap(pg, table_owner=_MIGRATOR, ro_role=False)

    result = _apply(pg)

    assert result.returncode == 0, result.stderr
    roles = _run(pg, "-At", "-c", _RO_ROLE_COUNT)
    assert roles.stdout.strip() == "0"


@pytest.mark.integration
@pytest.mark.ephemeral_pg
def test_a_grant_the_migrator_cannot_make_stops_the_file(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    pg = ephemeral_postgres
    _bootstrap(pg, table_owner=_OTHER_OWNER, ro_role=True)
    _run(pg, "-c", f"GRANT SELECT ON {_TABLE} TO {_MIGRATOR}")

    result = _apply(pg)

    assert result.returncode != 0
    assert "cannot read" in result.stderr
    assert _privileges(pg) == set()
