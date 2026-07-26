# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Migration contract checks for the app_dashboard role (OMN-14899).

The connecting role — not the RLS policy — is the real isolation control:
Postgres silently bypasses row-level security for SUPERUSER / BYPASSRLS
roles and for table owners. These checks pin the migration text to the
security-critical properties so a later edit cannot silently weaken them.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Iterator
from pathlib import Path

import psycopg2
import psycopg2.errors
import pytest

REPO_ROOT = Path(__file__).parent.parent.parent.parent
MIGRATION_FILE = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "094_create_app_dashboard_role.sql"
)
ROLLBACK_FILE = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "rollback"
    / "rollback_094_create_app_dashboard_role.sql"
)


@pytest.mark.integration
def test_094_creates_role_with_bypass_flags_off() -> None:
    sql = MIGRATION_FILE.read_text()

    assert "CREATE ROLE app_dashboard" in sql
    assert "NOSUPERUSER" in sql
    assert "NOBYPASSRLS" in sql
    assert "NOLOGIN" in sql
    assert "NOCREATEDB" in sql
    assert "NOCREATEROLE" in sql


@pytest.mark.integration
def test_094_enforces_flags_on_preexisting_role() -> None:
    """ALTER ROLE must re-assert the flags — presence is not the property."""
    sql = MIGRATION_FILE.read_text()

    assert "ALTER ROLE app_dashboard" in sql
    # The ALTER block (after the guarded CREATE) must carry both
    # security-critical negations.
    alter_block = sql.split("ALTER ROLE app_dashboard", 1)[1]
    assert "NOSUPERUSER" in alter_block
    assert "NOBYPASSRLS" in alter_block


@pytest.mark.integration
def test_094_contains_no_credential_material() -> None:
    sql = MIGRATION_FILE.read_text()

    assert (
        "PASSWORD '" not in sql.upper() and "ENCRYPTED PASSWORD" not in sql.upper()
    ), (
        "credential material must never live in a migration; the LOGIN + "
        "password attach is a deployment-owned gated step"
    )


@pytest.mark.integration
def test_094_is_role_only_no_grants_no_connect() -> None:
    """Grants ride with the RLS migrations (OMN-14894), never with the role.

    Keeping the file free of \\connect and GRANT means it is valid in any
    database context, and a table can never become readable by app_dashboard
    before its tenant_isolation policy exists.
    """
    sql = MIGRATION_FILE.read_text()
    executable = "\n".join(
        line for line in sql.splitlines() if not line.lstrip().startswith("--")
    )

    assert "GRANT" not in executable.upper(), "role migration must not carry grants"
    assert "\\connect" not in executable
    assert "ALTER DEFAULT PRIVILEGES" not in executable.upper()


@pytest.mark.integration
def test_094_rollback_drops_role_and_grants() -> None:
    sql = ROLLBACK_FILE.read_text()

    assert "REVOKE ALL ON ALL TABLES IN SCHEMA public FROM app_dashboard" in sql
    assert "DROP ROLE IF EXISTS app_dashboard" in sql


# =============================================================================
# Live-connection proof (OMN-14899 follow-up)
#
# The five checks above pin the migration's SQL *text*. None of them apply
# the migration or open a database connection, so none of them can prove the
# role actually behaves the way the text claims. These tests do: they spin up
# a real, throwaway Postgres 16 cluster, apply the real migration file, and
# drive every assertion through an actual client connection — not psql-as-
# superuser and not a string match.
# =============================================================================

_PG_TOOLS_MISSING = any(
    shutil.which(tool) is None for tool in ("initdb", "pg_ctl", "psql")
)


class _EphemeralPostgres:
    """A throwaway, superuser-owned Postgres 16 cluster for one test.

    Spun up via initdb/pg_ctl into a scratch directory — never the shared
    local docker Postgres and never any cloud/RDS/staging database — so a
    test can freely create, reshape, and drop roles without risking a
    shared-state collision or requiring cloud credentials.
    """

    def __init__(self, socket_dir: str, port: int) -> None:
        self.socket_dir = socket_dir
        self.port = port

    def connect(
        self,
        *,
        user: str = "postgres",
        password: str | None = None,
        dbname: str = "postgres",
    ) -> psycopg2.extensions.connection:
        return psycopg2.connect(
            host=self.socket_dir,
            port=self.port,
            user=user,
            password=password,
            dbname=dbname,
        )

    def psql(
        self, *args: str, user: str = "postgres"
    ) -> subprocess.CompletedProcess[str]:
        """Apply SQL the same way the real migration runner does.

        run-forward-migrations.sh invokes each file as
        ``psql -v ON_ERROR_STOP=1 -f <file>`` — matching that invocation
        (rather than executing the SQL text through a driver call) is what
        makes this an honest reproduction of the production apply path.
        """
        return subprocess.run(
            [
                "psql",
                "-h",
                self.socket_dir,
                "-p",
                str(self.port),
                "-U",
                user,
                "-d",
                "postgres",
                *args,
            ],
            capture_output=True,
            text=True,
            check=False,
        )


@pytest.fixture
def ephemeral_postgres() -> Iterator[_EphemeralPostgres]:
    if _PG_TOOLS_MISSING:
        pytest.skip(
            "initdb/pg_ctl/psql not on PATH — cannot spin up an ephemeral "
            "Postgres cluster for the live-connection proof"
        )

    scratch = tempfile.mkdtemp(prefix="pg094_")
    data_dir = Path(scratch) / "data"
    log_file = Path(scratch) / "server.log"
    # Arbitrary; listen_addresses='' below means no TCP bind, so this port
    # is only ever used to name the unix socket file and never contended.
    port = 55491

    init = subprocess.run(
        ["initdb", "-D", str(data_dir), "-U", "postgres", "--auth=trust", "--no-sync"],
        capture_output=True,
        text=True,
        check=False,
    )
    if init.returncode != 0:
        shutil.rmtree(scratch, ignore_errors=True)
        pytest.fail(f"initdb failed for the ephemeral test cluster: {init.stderr}")

    start = subprocess.run(
        [
            "pg_ctl",
            "-D",
            str(data_dir),
            "-o",
            f"-k {scratch} -p {port} -c listen_addresses=",
            "-l",
            str(log_file),
            "-w",
            "-t",
            "30",
            "start",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if start.returncode != 0:
        log_text = log_file.read_text() if log_file.exists() else ""
        shutil.rmtree(scratch, ignore_errors=True)
        pytest.fail(
            f"pg_ctl start failed for the ephemeral test cluster: "
            f"{start.stderr}\n{log_text}"
        )

    try:
        yield _EphemeralPostgres(socket_dir=scratch, port=port)
    finally:
        subprocess.run(
            ["pg_ctl", "-D", str(data_dir), "-m", "fast", "stop"],
            capture_output=True,
            text=True,
            check=False,
        )
        shutil.rmtree(scratch, ignore_errors=True)


def _apply_094_as_rds_shaped_role(
    pg: _EphemeralPostgres,
) -> subprocess.CompletedProcess[str]:
    """Create an RDS-master-shaped role and apply 094 through it.

    RDS never grants true SUPERUSER — the master account is CREATEROLE +
    CREATEDB but explicitly NOSUPERUSER / NOREPLICATION / NOBYPASSRLS.
    Applying through ``psql -v ON_ERROR_STOP=1 -f`` (the same invocation
    ``run-forward-migrations.sh`` uses in production) as this role is an
    honest reproduction of the real RDS apply path, not just a shape-alike
    approximation.
    """
    bootstrap = pg.connect()
    bootstrap.autocommit = True
    with bootstrap.cursor() as cur:
        cur.execute(
            "CREATE ROLE rds_master_shaped WITH LOGIN CREATEROLE CREATEDB "
            "NOSUPERUSER NOREPLICATION NOBYPASSRLS"
        )
    bootstrap.close()

    return pg.psql(
        "-v", "ON_ERROR_STOP=1", "-f", str(MIGRATION_FILE), user="rds_master_shaped"
    )


@pytest.mark.integration
@pytest.mark.postgres
def test_094_alter_role_succeeds_under_rds_shaped_master_role(
    ephemeral_postgres: _EphemeralPostgres,
) -> None:
    """094 must apply cleanly through an RDS-shaped (non-superuser) role.

    Postgres only allows ``ALTER ROLE ... NOSUPERUSER/NOBYPASSRLS/
    NOREPLICATION`` when the *executing* role already holds that attribute
    — even just to reassert an already-correct ``false``. A migration that
    unconditionally ALTERs those three flags therefore fails on every real
    RDS apply, not only on some pre-existing-role edge case.
    """
    result = _apply_094_as_rds_shaped_role(ephemeral_postgres)
    assert result.returncode == 0, (
        "094_create_app_dashboard_role.sql must apply cleanly through an "
        "RDS-shaped (CREATEROLE, NOSUPERUSER) connection — it must never "
        "require the executing role to already be a true Postgres "
        f"superuser.\npsql stderr:\n{result.stderr}"
    )

    verify = ephemeral_postgres.connect()
    verify.autocommit = True
    with verify.cursor() as cur:
        cur.execute(
            "SELECT rolsuper, rolbypassrls, rolreplication, rolcreatedb, "
            "rolcreaterole, rolcanlogin FROM pg_roles WHERE rolname = 'app_dashboard'"
        )
        row = cur.fetchone()
    verify.close()

    assert row is not None, "app_dashboard role was not created"
    rolsuper, rolbypassrls, rolreplication, rolcreatedb, rolcreaterole, rolcanlogin = (
        row
    )
    assert rolsuper is False
    assert rolbypassrls is False
    assert rolreplication is False
    assert rolcreatedb is False
    assert rolcreaterole is False
    assert rolcanlogin is False


@pytest.mark.integration
@pytest.mark.postgres
def test_094_denies_read_without_tenant_context_through_real_connection(
    ephemeral_postgres: _EphemeralPostgres,
) -> None:
    """The connecting role, not the RLS policy, must be the enforced boundary.

    Applies the real migration through the same RDS-shaped, non-superuser
    path as the RDS-compat test above (not a superuser shortcut), attaches
    a fixture tenant-scoped table with RLS enforced (ENABLE + FORCE), and
    drives every read/escape assertion through an actual client connection
    authenticated AS app_dashboard. A role assertion against pg_roles is
    necessary but not sufficient; this is the denied/allowed-read proof.
    """
    apply_result = _apply_094_as_rds_shaped_role(ephemeral_postgres)
    assert apply_result.returncode == 0, (
        "094_create_app_dashboard_role.sql must apply through the RDS-shaped "
        f"role before the RLS proof can run.\npsql stderr:\n{apply_result.stderr}"
    )

    setup = ephemeral_postgres.connect()
    setup.autocommit = True
    with setup.cursor() as cur:
        cur.execute(
            "CREATE TABLE test_tenant_rows ("
            "  tenant_id text NOT NULL,"
            "  payload text NOT NULL"
            ")"
        )
        cur.execute(
            "INSERT INTO test_tenant_rows (tenant_id, payload) VALUES "
            "('tenant-a', 'a-row-1'), ('tenant-a', 'a-row-2'), "
            "('tenant-b', 'b-row-1')"
        )
        cur.execute("ALTER TABLE test_tenant_rows ENABLE ROW LEVEL SECURITY")
        cur.execute("ALTER TABLE test_tenant_rows FORCE ROW LEVEL SECURITY")
        cur.execute(
            "CREATE POLICY tenant_isolation ON test_tenant_rows "
            "USING (tenant_id = current_setting('app.tenant_id', true)) "
            "WITH CHECK (tenant_id = current_setting('app.tenant_id', true))"
        )
        cur.execute("GRANT USAGE ON SCHEMA public TO app_dashboard")
        cur.execute("GRANT SELECT ON test_tenant_rows TO app_dashboard")

        # Test-local, per-run credential — never committed, never a real
        # secret. Real deployments attach LOGIN via AWS Secrets Manager
        # (OMN-14899), an operator-gated step that stays out of scope here.
        test_password = uuid.uuid4().hex
        cur.execute("ALTER ROLE app_dashboard LOGIN PASSWORD %s", (test_password,))

        cur.execute(
            "SELECT tableowner FROM pg_tables WHERE tablename = 'test_tenant_rows'"
        )
        owner_row = cur.fetchone()
    setup.close()

    assert owner_row is not None
    assert owner_row[0] != "app_dashboard", (
        "app_dashboard must never own a table it reads through — ownership "
        "silently bypasses ENABLE ROW LEVEL SECURITY"
    )

    app_conn = ephemeral_postgres.connect(user="app_dashboard", password=test_password)
    app_conn.autocommit = True
    try:
        with app_conn.cursor() as cur:
            cur.execute("SELECT current_user, session_user")
            current_user, session_user = cur.fetchone()
            assert current_user == "app_dashboard"
            assert session_user == "app_dashboard"

            # No app.tenant_id GUC set -> current_setting(..., true) is NULL,
            # so `tenant_id = NULL` is never true -> zero rows, fail-closed.
            cur.execute("SELECT count(*) FROM test_tenant_rows")
            (count_without_context,) = cur.fetchone()
            assert count_without_context == 0, (
                "app_dashboard must see zero rows with no tenant context set "
                "(fail-closed), not the whole table"
            )

            cur.execute("SET app.tenant_id = 'tenant-a'")
            cur.execute("SELECT tenant_id FROM test_tenant_rows ORDER BY payload")
            tenant_ids = [r[0] for r in cur.fetchall()]
            assert tenant_ids == ["tenant-a", "tenant-a"], (
                f"expected only tenant-a rows, got {tenant_ids} — cross-tenant leak"
            )

            with pytest.raises(psycopg2.errors.InsufficientPrivilege):
                cur.execute("ALTER ROLE app_dashboard BYPASSRLS")
            app_conn.rollback()

            with pytest.raises(psycopg2.errors.InsufficientPrivilege):
                cur.execute("SET ROLE postgres")
            app_conn.rollback()
    finally:
        app_conn.close()
