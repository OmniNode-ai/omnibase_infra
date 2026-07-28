# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Real-Postgres RLS regression for the projection writer's tenant context (OMN-15301).

The auto-wiring DB-injection callback
(``handler_wiring._make_projection_dispatch_callback``) builds a
``SyncPsycopg2Adapter`` (``_build_sync_db_adapter``) and injects it as
``input_data["_db"]`` for EVERY projection handler. That adapter opens a
psycopg2 connection with ``autocommit = True`` and never sets the
``app.tenant_id`` GUC.

Against staging RDS, ``omnidash_analytics.delegation_events`` carries RLS
ENABLED with policy ``tenant_isolation`` FOR ALL:

    USING       (tenant_id = current_setting('app.tenant_id', true))
    WITH CHECK  (tenant_id = current_setting('app.tenant_id', true))

and the writer connects as ``role_omnidash`` -- non-owner, NOSUPERUSER,
NOBYPASSRLS. With the GUC unset ``current_setting(..., true)`` returns NULL,
the WITH CHECK predicate is NULL, and every INSERT is rejected:

    handler=HandlerProjectionDelegation
    topic=onex.evt.omnibase-infra.delegation-completed.v1
    error_type=InsufficientPrivilege
    error=new row violates row-level security policy for table "delegation_events"

This module proves that failure and its fix against a REAL Postgres cluster
with the REAL policy text and a REAL non-owner/NOBYPASSRLS role, driven
through the actual adapter the runtime builds -- not a mock, and not a
psql-as-superuser probe (a superuser bypasses RLS unconditionally and would
make every assertion here vacuously green).

Hermetic by construction: an ephemeral cluster is initdb'd into a temp
directory and listens on a unix socket only, so the test shares no state with
any lane database and cannot collide on a TCP port. Skips when Postgres
binaries are unavailable.

Run: uv run pytest tests/integration/db/test_projection_writer_tenant_rls_omn15301.py -v
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

psycopg2 = pytest.importorskip("psycopg2", reason="psycopg2 required for RLS proof")

from omnibase_infra.errors.error_projection import (
    ProjectionTenantContextError,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _build_sync_db_adapter,
)

pytestmark = [pytest.mark.integration, pytest.mark.postgres, pytest.mark.slow]

TABLE = "delegation_events"
CONFLICT_KEY = "correlation_id"
WRITER_ROLE = "role_omnidash_test"
WRITER_PASSWORD = "rls_proof_only_pw"  # pragma: allowlist secret
OWNER_ROLE = "postgres"
TENANT_A = "tenant-a"
TENANT_B = "tenant-b"

# Verbatim from omnimarket migration 0023_delegation_rls_tenant_isolation.sql.
# The policy compares TEXT (no ::uuid cast) by deliberate seam decision, and the
# live staging table has ENABLE (not FORCE) -- FORCE is unnecessary here because
# the writer is a non-owner, which RLS binds regardless.
_POLICY_SQL = """
ALTER TABLE delegation_events ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS tenant_isolation ON delegation_events;
CREATE POLICY tenant_isolation ON delegation_events
  FOR ALL
  USING (tenant_id = current_setting('app.tenant_id', true))
  WITH CHECK (tenant_id = current_setting('app.tenant_id', true));
"""

# Minimal shape of the live table: the conflict key plus the tenant column with
# migration 0022's exact DEFAULT, which the interim single-tenant fallback and
# the policy must agree on.
_TABLE_SQL = """
CREATE TABLE delegation_events (
    correlation_id TEXT PRIMARY KEY,
    task_type      TEXT NOT NULL DEFAULT '',
    delegated_to   TEXT NOT NULL DEFAULT '',
    tenant_id      TEXT NOT NULL DEFAULT 'omninode'
);
"""

# TABLE is a module constant, never user input — S608 is a false positive.
_DELETE_ALL_SQL = f"DELETE FROM {TABLE}"  # noqa: S608
_SELECT_ROWS_SQL = f"SELECT correlation_id, tenant_id FROM {TABLE} ORDER BY 1"  # noqa: S608
_INSERT_SQL = f"INSERT INTO {TABLE} (correlation_id, tenant_id) VALUES (%s, %s)"  # noqa: S608


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

if not _INITDB or not _PG_CTL:  # pragma: no cover - environment dependent
    pytest.skip(
        "initdb/pg_ctl not available — cannot bring up an ephemeral Postgres",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def pg_socket_dir() -> Iterator[str]:
    """Bring up an ephemeral, unix-socket-only Postgres cluster."""
    root = Path(tempfile.mkdtemp(prefix="omn15301-pg-"))
    data_dir = root / "data"
    sock_dir = root / "sock"
    sock_dir.mkdir()
    log_file = root / "postgres.log"

    subprocess.run(
        [
            str(_INITDB),
            "-D",
            str(data_dir),
            "-U",
            OWNER_ROLE,
            "--auth-local=trust",
            "--auth-host=trust",
            "-E",
            "UTF8",
        ],
        check=True,
        capture_output=True,
    )
    # -h '' disables TCP entirely: no port to collide with a lane database.
    subprocess.run(
        [
            str(_PG_CTL),
            "-D",
            str(data_dir),
            "-l",
            str(log_file),
            "-o",
            f"-k {sock_dir} -h ''",
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
            [str(_PG_CTL), "-D", str(data_dir), "-m", "immediate", "-w", "stop"],
            check=False,
            capture_output=True,
        )
        shutil.rmtree(root, ignore_errors=True)


def _dsn(sock_dir: str, user: str, password: str | None = None) -> str:
    parts = [f"host={sock_dir}", "dbname=rlsproof", f"user={user}"]
    if password:
        parts.append(f"password={password}")
    return " ".join(parts)


@pytest.fixture(scope="module")
def owner_dsn(pg_socket_dir: str) -> str:
    """Create the proof database, the RLS'd table, the policy and the writer role."""
    bootstrap = psycopg2.connect(
        f"host={pg_socket_dir} dbname=postgres user={OWNER_ROLE}"
    )
    bootstrap.autocommit = True
    with bootstrap.cursor() as cur:
        cur.execute("CREATE DATABASE rlsproof")
    bootstrap.close()

    dsn = _dsn(pg_socket_dir, OWNER_ROLE)
    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(_TABLE_SQL)
        cur.execute(_POLICY_SQL)
        # Non-owner, NOSUPERUSER, NOBYPASSRLS — the live role_omnidash posture.
        cur.execute(
            f"CREATE ROLE {WRITER_ROLE} LOGIN PASSWORD %s NOSUPERUSER NOBYPASSRLS",
            (WRITER_PASSWORD,),
        )
        cur.execute(f"GRANT USAGE ON SCHEMA public TO {WRITER_ROLE}")
        cur.execute(
            f"GRANT SELECT, INSERT, UPDATE ON {TABLE} TO {WRITER_ROLE}",
        )
    conn.close()
    return dsn


@pytest.fixture
def writer_dsn(pg_socket_dir: str, owner_dsn: str) -> str:
    return _dsn(pg_socket_dir, WRITER_ROLE, WRITER_PASSWORD)


@pytest.fixture(autouse=True)
def _clean_table(owner_dsn: str) -> Iterator[None]:
    yield
    conn = psycopg2.connect(owner_dsn)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(_DELETE_ALL_SQL)
    conn.close()


@pytest.fixture(autouse=True)
def _clear_tenant_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutral tenant env by default; individual tests opt in."""
    monkeypatch.delenv("ONEX_TENANT_ID", raising=False)
    monkeypatch.delenv("ENFORCE_TENANT_ISOLATION", raising=False)


def _rows(owner_dsn: str) -> list[dict[str, Any]]:
    """Read every row as the OWNER, bypassing the policy, to assert ground truth."""
    conn = psycopg2.connect(owner_dsn)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(_SELECT_ROWS_SQL)
            return [{"correlation_id": r[0], "tenant_id": r[1]} for r in cur.fetchall()]
    finally:
        conn.close()


def _row(correlation_id: str, tenant_id: str | None = None) -> dict[str, object]:
    row: dict[str, object] = {
        "correlation_id": correlation_id,
        "task_type": "code-review",
        "delegated_to": "local",
    }
    if tenant_id is not None:
        row["tenant_id"] = tenant_id
    return row


class TestEnvironmentReproducesLiveFailure:
    """The harness must reproduce the live error class before any fix is claimed."""

    def test_raw_insert_without_guc_is_rejected(self, writer_dsn: str) -> None:
        """No GUC → predicate NULL → INSERT rejected. This is the live blocker."""
        conn = psycopg2.connect(writer_dsn)
        conn.autocommit = True
        try:
            with pytest.raises(psycopg2.errors.InsufficientPrivilege) as exc_info:
                with conn.cursor() as cur:
                    cur.execute(
                        _INSERT_SQL,
                        ("live-repro", TENANT_A),
                    )
            assert "violates row-level security policy" in str(exc_info.value)
        finally:
            conn.close()

    def test_writer_role_is_not_superuser_or_bypassrls(self, writer_dsn: str) -> None:
        """A superuser/BYPASSRLS writer would make every other assertion vacuous."""
        conn = psycopg2.connect(writer_dsn)
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT rolsuper, rolbypassrls FROM pg_roles "
                    "WHERE rolname = current_user"
                )
                rolsuper, rolbypassrls = cur.fetchone()
            assert rolsuper is False
            assert rolbypassrls is False
        finally:
            conn.close()

    def test_set_local_alone_does_not_survive_autocommit(self, writer_dsn: str) -> None:
        """Pins the crux: under autocommit, SET LOCAL evaporates before the INSERT.

        This is why the fix cannot be "just issue SET LOCAL first" — each
        statement is its own implicit transaction, so the GUC is already gone by
        the time the INSERT runs.
        """
        conn = psycopg2.connect(writer_dsn)
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                cur.execute("SET LOCAL app.tenant_id = %s", (TENANT_A,))
            with pytest.raises(psycopg2.errors.InsufficientPrivilege):
                with conn.cursor() as cur:
                    cur.execute(
                        _INSERT_SQL,
                        ("set-local-lost", TENANT_A),
                    )
        finally:
            conn.close()


class TestAdapterWritesUnderTenantContext:
    """The adapter the runtime actually builds must land rows under RLS."""

    def test_upsert_lands_row_with_event_tenant(
        self, writer_dsn: str, owner_dsn: str
    ) -> None:
        adapter = _build_sync_db_adapter(writer_dsn)
        assert adapter.upsert(TABLE, CONFLICT_KEY, _row("cid-a", TENANT_A)) is True
        assert _rows(owner_dsn) == [{"correlation_id": "cid-a", "tenant_id": TENANT_A}]

    def test_upsert_updates_existing_row_of_same_tenant(
        self, writer_dsn: str, owner_dsn: str
    ) -> None:
        adapter = _build_sync_db_adapter(writer_dsn)
        adapter.upsert(TABLE, CONFLICT_KEY, _row("cid-a", TENANT_A))
        row = _row("cid-a", TENANT_A)
        row["delegated_to"] = "remote"
        assert adapter.upsert(TABLE, CONFLICT_KEY, row) is True
        assert len(_rows(owner_dsn)) == 1

    def test_query_runs_under_tenant_context(self, writer_dsn: str) -> None:
        """Without GUC on the read path, _preserve_existing_evidence sees nothing.

        A silently-empty read is worse than a failed one: the projection would
        conclude no prior row exists and clobber previously-written evidence.
        """
        adapter = _build_sync_db_adapter(writer_dsn)
        adapter.upsert(TABLE, CONFLICT_KEY, _row("cid-a", TENANT_A))
        os.environ["ONEX_TENANT_ID"] = TENANT_A
        try:
            found = adapter.query(TABLE, {CONFLICT_KEY: "cid-a"})
        finally:
            os.environ.pop("ONEX_TENANT_ID", None)
        assert [r["correlation_id"] for r in found] == ["cid-a"]

    def test_interim_default_tenant_matches_column_default(
        self, writer_dsn: str, owner_dsn: str
    ) -> None:
        """An event with no tenant lands under the 0022 DEFAULT, not rejected.

        ENFORCE_TENANT_ISOLATION is off (fleet default), so the OMN-14058
        operator-accepted single-tenant interim still applies: the row takes the
        column DEFAULT 'omninode' and the GUC must agree with it, or the write
        fails for no isolation benefit.
        """
        adapter = _build_sync_db_adapter(writer_dsn)
        assert adapter.upsert(TABLE, CONFLICT_KEY, _row("cid-default")) is True
        assert _rows(owner_dsn) == [
            {"correlation_id": "cid-default", "tenant_id": "omninode"}
        ]

    def test_lane_tenant_env_does_not_override_the_column_default(
        self, writer_dsn: str, owner_dsn: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The write GUC must follow the ROW, never ONEX_TENANT_ID.

        Regression pin. When the row omits tenant_id, Postgres applies the
        column DEFAULT — so resolving the session tenant from ONEX_TENANT_ID
        would set the GUC to a tenant the stored row does not carry, and the
        policy's WITH CHECK rejects the write. Tenant resolution from the
        environment is the handler's job (it stamps row["tenant_id"]); this
        boundary only makes the session agree with the row.
        """
        monkeypatch.setenv("ONEX_TENANT_ID", TENANT_B)
        adapter = _build_sync_db_adapter(writer_dsn)
        assert adapter.upsert(TABLE, CONFLICT_KEY, _row("cid-lane")) is True
        assert _rows(owner_dsn) == [
            {"correlation_id": "cid-lane", "tenant_id": "omninode"}
        ]

    def test_handler_stamped_tenant_wins_over_lane_env(
        self, writer_dsn: str, owner_dsn: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ONEX_TENANT_ID", TENANT_B)
        adapter = _build_sync_db_adapter(writer_dsn)
        assert adapter.upsert(TABLE, CONFLICT_KEY, _row("cid-a", TENANT_A)) is True
        assert _rows(owner_dsn) == [{"correlation_id": "cid-a", "tenant_id": TENANT_A}]

    def test_read_probe_follows_the_lane_tenant(
        self, writer_dsn: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reads mirror the HANDLER's resolution order so probes find real rows.

        On a lane with ONEX_TENANT_ID set, the handler stamps that tenant onto
        the rows, so the existing-row probe must look there — otherwise
        _preserve_existing_evidence sees nothing and clobbers real evidence.
        """
        adapter = _build_sync_db_adapter(writer_dsn)
        adapter.upsert(TABLE, CONFLICT_KEY, _row("cid-lane", TENANT_B))
        monkeypatch.setenv("ONEX_TENANT_ID", TENANT_B)
        found = adapter.query(TABLE, {CONFLICT_KEY: "cid-lane"})
        assert [r["correlation_id"] for r in found] == ["cid-lane"]


class TestCrossTenantIsolationHolds:
    """The fix must not become a way around the policy it is satisfying."""

    def test_with_check_rejects_row_stamped_for_another_tenant(
        self, writer_dsn: str
    ) -> None:
        """Tenant B's context cannot write a row carrying tenant A's id."""
        conn = psycopg2.connect(writer_dsn)
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT set_config('app.tenant_id', %s, true)", (TENANT_B,))
                with pytest.raises(psycopg2.errors.InsufficientPrivilege):
                    cur.execute(
                        _INSERT_SQL,
                        ("cross-tenant", TENANT_A),
                    )
            conn.rollback()
        finally:
            conn.close()

    def test_tenant_b_cannot_read_tenant_a_rows(self, writer_dsn: str) -> None:
        adapter = _build_sync_db_adapter(writer_dsn)
        adapter.upsert(TABLE, CONFLICT_KEY, _row("cid-a", TENANT_A))
        os.environ["ONEX_TENANT_ID"] = TENANT_B
        try:
            found = adapter.query(TABLE, {CONFLICT_KEY: "cid-a"})
        finally:
            os.environ.pop("ONEX_TENANT_ID", None)
        assert found == []

    def test_tenant_context_does_not_leak_between_writes(
        self, writer_dsn: str, owner_dsn: str
    ) -> None:
        """Transaction-scoped context must not persist onto the next write.

        The adapter caches its connection, so a session-level SET would leak
        tenant A's context into a later write that resolved a different tenant.
        """
        adapter = _build_sync_db_adapter(writer_dsn)
        # Three consecutive writes over the SAME cached connection, each under a
        # different tenant. A session-scoped SET would carry tenant A's context
        # into the second write and the policy's WITH CHECK would reject it.
        adapter.upsert(TABLE, CONFLICT_KEY, _row("cid-a", TENANT_A))
        adapter.upsert(TABLE, CONFLICT_KEY, _row("cid-b", TENANT_B))
        adapter.upsert(TABLE, CONFLICT_KEY, _row("cid-default"))
        assert _rows(owner_dsn) == [
            {"correlation_id": "cid-a", "tenant_id": TENANT_A},
            {"correlation_id": "cid-b", "tenant_id": TENANT_B},
            {"correlation_id": "cid-default", "tenant_id": "omninode"},
        ]


class TestFailClosedOnMissingTenant:
    """With enforcement on, a tenant-less write refuses loudly and writes nothing."""

    def test_enforced_missing_tenant_raises_typed_error(
        self, writer_dsn: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ENFORCE_TENANT_ISOLATION", "true")
        adapter = _build_sync_db_adapter(writer_dsn)
        with pytest.raises(ProjectionTenantContextError):
            adapter.upsert(TABLE, CONFLICT_KEY, _row("cid-refused"))

    def test_enforced_missing_tenant_writes_zero_rows(
        self, writer_dsn: str, owner_dsn: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ENFORCE_TENANT_ISOLATION", "true")
        adapter = _build_sync_db_adapter(writer_dsn)
        with pytest.raises(ProjectionTenantContextError):
            adapter.upsert(TABLE, CONFLICT_KEY, _row("cid-refused"))
        assert _rows(owner_dsn) == []

    def test_enforced_blank_tenant_is_refused(
        self, writer_dsn: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ENFORCE_TENANT_ISOLATION", "true")
        adapter = _build_sync_db_adapter(writer_dsn)
        with pytest.raises(ProjectionTenantContextError):
            adapter.upsert(TABLE, CONFLICT_KEY, _row("cid-blank", "   "))

    def test_enforced_write_with_real_tenant_still_lands(
        self, writer_dsn: str, owner_dsn: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ENFORCE_TENANT_ISOLATION", "true")
        adapter = _build_sync_db_adapter(writer_dsn)
        assert adapter.upsert(TABLE, CONFLICT_KEY, _row("cid-ok", TENANT_A)) is True
        assert _rows(owner_dsn) == [{"correlation_id": "cid-ok", "tenant_id": TENANT_A}]
