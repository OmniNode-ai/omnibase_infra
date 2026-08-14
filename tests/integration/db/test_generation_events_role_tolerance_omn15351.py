# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Real-Postgres execution proof for node migration 0027 role tolerance (OMN-15351).

``docker/migrations/forward/nodes/node_projection_delegation/0027_generation_events_tenant_rls.sql``
opened with a fail-closed guard that raised on a missing ``role_omnidash``::

    psql:/migrations/forward/nodes/node_projection_delegation/0027_generation_events_tenant_rls.sql:26:
      ERROR:  role_omnidash role missing — generation_events writer access cannot be granted

``role_omnidash`` is ENVIRONMENT-provisioned (out-of-band on cloud RDS, or from
``ROLE_OMNIDASH_PASSWORD`` at first-startup init in
``docker/migrations/forward/000_create_multiple_databases.sh``); no forward
migration anywhere creates it. The ``.201`` dev-lane cluster carries
``pg_roles = {app_dashboard, postgres, role_omniweb}``, so that guard made EVERY
dev-lane deploy fatal at this file (OMN-15348 AC4 redeploy, workflow
``wf_55998f90``).

This module executes the REAL vendored SQL file with the SAME psql invocation the
deploy-time runner uses (``scripts/run-forward-migrations.sh``:
``psql -v ON_ERROR_STOP=1 -f <migration_file>``) against a REAL ephemeral
Postgres, in three role states:

* ``role_omnidash`` absent  -> exit 0, WARNING naming BOTH skipped grants, and
  zero ``role_omnidash`` privileges anywhere in the resulting ACLs.
* ``role_omnidash`` present -> the exact pre-OMN-15351 grant set is applied
  (schema USAGE + SELECT/INSERT/UPDATE on ``generation_events``).
* ``app_dashboard`` absent  -> STILL FATAL. Forward migration 094 (OMN-14899)
  creates ``app_dashboard`` in-repo, so its absence is a real ordering bug, not
  an environment difference. Nothing but the ``role_omnidash`` guard was softened.

Hermetic by construction: each test initdb's its own cluster into a temp
directory listening on a unix socket only, so role state (which is CLUSTER-wide,
not database-wide) cannot leak between tests and nothing can collide with a lane
database or a TCP port. Skips when Postgres binaries are unavailable.

Run: uv run pytest tests/integration/db/test_generation_events_role_tolerance_omn15351.py -v

Ticket: OMN-15351
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.postgres, pytest.mark.slow]

REPO_ROOT = Path(__file__).resolve().parents[3]
NODE_MIGRATIONS = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "nodes"
    / "node_projection_delegation"
)
BASE_TABLE_SQL = NODE_MIGRATIONS / "0008_generation_events.sql"
MIGRATION_SQL = NODE_MIGRATIONS / "0027_generation_events_tenant_rls.sql"

OWNER_ROLE = "postgres"
DB = "omn15351"
# Non-owner/NOSUPERUSER/NOBYPASSRLS mirrors the live posture of both roles.
_ROLE_ATTRS = "NOSUPERUSER NOBYPASSRLS LOGIN PASSWORD 'omn15351_proof_only'"  # pragma: allowlist secret

# The two grants the migration must skip (and name) when role_omnidash is absent,
# and must apply verbatim when it is present.
SKIPPED_GRANT_SCHEMA = "GRANT USAGE ON SCHEMA public TO role_omnidash"
SKIPPED_GRANT_TABLE = (
    "GRANT SELECT, INSERT, UPDATE ON generation_events TO role_omnidash"
)
# Pre-OMN-15351 role_omnidash privileges on generation_events, proven by running
# the unmodified file on the same fixture (see the PR body's differential run).
EXPECTED_OMNIDASH_TABLE_PRIVILEGES = {"SELECT", "INSERT", "UPDATE"}


def _pg_bin(name: str) -> str | None:
    """Resolve a Postgres binary from PATH or a brew keg-only prefix."""
    found = shutil.which(name)
    if found:
        return found
    for prefix in sorted(Path("/opt/homebrew/opt").glob("postgresql@*"), reverse=True):
        candidate = prefix / "bin" / name
        if candidate.exists():
            return str(candidate)
    return None


_INITDB = _pg_bin("initdb")
_PG_CTL = _pg_bin("pg_ctl")
_PSQL = _pg_bin("psql")

if not _INITDB or not _PG_CTL or not _PSQL:  # pragma: no cover - environment dependent
    pytest.skip(
        "initdb/pg_ctl/psql not available — cannot bring up an ephemeral Postgres",
        allow_module_level=True,
    )


@pytest.fixture
def cluster() -> Iterator[str]:
    """Bring up a fresh ephemeral, unix-socket-only Postgres cluster per test.

    Per-test (not per-module) because ``CREATE ROLE`` is cluster-wide: a shared
    cluster would let one test's role leak into another's role-absent premise and
    silently make it vacuous.
    """
    root = Path(tempfile.mkdtemp(prefix="omn15351-pg-"))
    data_dir = root / "data"
    sock_dir = root / "sock"
    sock_dir.mkdir()

    subprocess.run(
        [str(_INITDB), "-D", str(data_dir), "-U", OWNER_ROLE, "-A", "trust"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            str(_PG_CTL),
            "-D",
            str(data_dir),
            "-l",
            str(root / "postgres.log"),
            "-o",
            f"-k {sock_dir} -h '' -c listen_addresses=''",
            "-w",
            "start",
        ],
        check=True,
        capture_output=True,
    )
    try:
        yield str(sock_dir)
    finally:
        subprocess.run(
            [str(_PG_CTL), "-D", str(data_dir), "-m", "immediate", "stop"],
            check=False,
            capture_output=True,
        )
        shutil.rmtree(root, ignore_errors=True)


def _psql(sock: str, database: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(_PSQL), "-X", "-q", "-h", sock, "-U", OWNER_ROLE, "-d", database, *args],
        capture_output=True,
        text=True,
        check=False,  # the exit code IS the assertion in the RED/fatal-guard cases
    )


def _query(sock: str, sql: str) -> str:
    result = _psql(sock, DB, "-t", "-A", "-v", "ON_ERROR_STOP=1", "-c", sql)
    assert result.returncode == 0, f"probe query failed: {result.stderr}"
    return result.stdout.strip()


def _prepare(sock: str, *, app_dashboard: bool, role_omnidash: bool) -> None:
    """Create the database, the requested roles, and the 0008 base table."""
    assert _psql(sock, "postgres", "-c", f"CREATE DATABASE {DB}").returncode == 0
    for role, wanted in (
        ("app_dashboard", app_dashboard),
        ("role_omnidash", role_omnidash),
    ):
        if wanted:
            created = _psql(sock, "postgres", "-c", f"CREATE ROLE {role} {_ROLE_ATTRS}")
            assert created.returncode == 0, created.stderr

    base = _psql(sock, DB, "-v", "ON_ERROR_STOP=1", "-f", str(BASE_TABLE_SQL))
    assert base.returncode == 0, base.stderr

    # Anti-vacuity: assert the premise this test rests on actually holds.
    roles = _query(sock, "SELECT rolname FROM pg_roles WHERE rolname !~ '^pg_'").split()
    assert ("app_dashboard" in roles) is app_dashboard, roles
    assert ("role_omnidash" in roles) is role_omnidash, roles


def _apply_migration(sock: str) -> subprocess.CompletedProcess[str]:
    """Apply 0027 exactly as scripts/run-forward-migrations.sh does."""
    return _psql(sock, DB, "-v", "ON_ERROR_STOP=1", "-f", str(MIGRATION_SQL))


def test_role_omnidash_absent_completes_and_names_every_skipped_grant(
    cluster: str,
) -> None:
    """Role absent: migration succeeds, warns by name, and grants nothing to it."""
    _prepare(cluster, app_dashboard=True, role_omnidash=False)

    result = _apply_migration(cluster)
    output = result.stdout + result.stderr

    assert result.returncode == 0, output
    assert "WARNING:" in output, output
    assert "role_omnidash role missing" in output, output
    # No silent skip: the warning must enumerate what it skipped.
    assert SKIPPED_GRANT_SCHEMA in output, output
    assert SKIPPED_GRANT_TABLE in output, output

    # Readback: zero role_omnidash privileges anywhere.
    assert (
        _query(
            cluster,
            "SELECT count(*) FROM information_schema.role_table_grants "
            "WHERE table_name = 'generation_events' AND grantee = 'role_omnidash'",
        )
        == "0"
    )
    assert (
        _query(
            cluster,
            "SELECT coalesce(nspacl::text, '') LIKE '%role_omnidash%' "
            "FROM pg_namespace WHERE nspname = 'public'",
        )
        == "f"
    )

    # Everything the migration does NOT gate on role_omnidash still applied.
    assert (
        _query(
            cluster,
            "SELECT relrowsecurity FROM pg_class WHERE relname = 'generation_events'",
        )
        == "t"
    )
    assert (
        _query(
            cluster,
            "SELECT count(*) FROM pg_policies WHERE tablename = 'generation_events' "
            "AND policyname = 'tenant_isolation'",
        )
        == "1"
    )
    assert (
        _query(
            cluster,
            "SELECT count(*) FROM information_schema.columns "
            "WHERE table_name = 'generation_events' AND column_name = 'tenant_id'",
        )
        == "1"
    )
    assert (
        _query(
            cluster,
            "SELECT privilege_type FROM information_schema.role_table_grants "
            "WHERE table_name = 'generation_events' AND grantee = 'app_dashboard'",
        )
        == "SELECT"
    )


def test_role_omnidash_absent_migration_is_idempotent(cluster: str) -> None:
    """Re-applying on a role-less lane stays green (the deploy retry path)."""
    _prepare(cluster, app_dashboard=True, role_omnidash=False)

    assert _apply_migration(cluster).returncode == 0
    second = _apply_migration(cluster)
    assert second.returncode == 0, second.stdout + second.stderr
    assert "role_omnidash role missing" in second.stdout + second.stderr


def test_role_omnidash_present_applies_the_unchanged_grant_set(cluster: str) -> None:
    """Role present: identical grants to the pre-OMN-15351 file, no warning."""
    _prepare(cluster, app_dashboard=True, role_omnidash=True)

    result = _apply_migration(cluster)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "role_omnidash role missing" not in output, output

    table_privileges = set(
        _query(
            cluster,
            "SELECT privilege_type FROM information_schema.role_table_grants "
            "WHERE table_name = 'generation_events' AND grantee = 'role_omnidash'",
        ).split()
    )
    assert table_privileges == EXPECTED_OMNIDASH_TABLE_PRIVILEGES

    assert (
        _query(
            cluster,
            "SELECT has_schema_privilege('role_omnidash', 'public', 'USAGE')",
        )
        == "t"
    )
    # arw = INSERT/SELECT/UPDATE, and nothing more (no DELETE 'd', no owner bits).
    assert "role_omnidash=arw/postgres" in _query(
        cluster,
        "SELECT relacl::text FROM pg_class WHERE relname = 'generation_events'",
    )


def test_app_dashboard_guard_is_still_fatal(cluster: str) -> None:
    """Only the role_omnidash guard was relaxed; 094's role stays fail-closed."""
    _prepare(cluster, app_dashboard=False, role_omnidash=True)

    result = _apply_migration(cluster)
    output = result.stdout + result.stderr

    assert result.returncode != 0, output
    assert "app_dashboard role missing" in output, output
    # Fail-closed means the table must NOT be left half-migrated with RLS on.
    assert (
        _query(
            cluster,
            "SELECT relrowsecurity FROM pg_class WHERE relname = 'generation_events'",
        )
        == "f"
    )
