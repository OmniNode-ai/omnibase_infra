# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Migration contract checks for the app_dashboard role (OMN-14899).

The connecting role — not the RLS policy — is the real isolation control:
Postgres silently bypasses row-level security for SUPERUSER / BYPASSRLS
roles and for table owners. These checks pin the migration text to the
security-critical properties so a later edit cannot silently weaken them.
"""

from __future__ import annotations

import re
import subprocess
import uuid
from pathlib import Path

import psycopg2
import psycopg2.errors
import pytest

from tests.integration.migrations.conftest import EphemeralPostgres

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

    assert "ALTER ROLE app_dashboard NOSUPERUSER NOBYPASSRLS NOREPLICATION" in sql, (
        "the security-critical negations must still be issued when pg_roles "
        "shows an actual escalation"
    )


@pytest.mark.integration
def test_094_never_revokes_the_deployment_owned_login_attach() -> None:
    """OMN-15343: LOGIN is create-time only, never re-asserted.

    The LOGIN + password attach is a deployment-owned, operator-gated step
    (AWS Secrets Manager, OMN-14899). On the cloud instance app_dashboard
    already carries LOGIN (live readback 2026-07-29: rolcanlogin = t), so a
    blanket ``ALTER ROLE app_dashboard NOLOGIN`` would break the dashboard's
    runtime connection as a side effect of recording a migration.
    """
    sql = MIGRATION_FILE.read_text()
    executable = "\n".join(
        line for line in sql.splitlines() if not line.lstrip().startswith("--")
    )
    for statement in re.findall(r"ALTER ROLE app_dashboard[^';]*", executable):
        assert "NOLOGIN" not in statement, (
            "NOLOGIN belongs to CREATE ROLE only; re-asserting it on a "
            f"pre-existing role revokes a deployment-owned attach: {statement!r}"
        )


@pytest.mark.integration
def test_094_gates_every_privileged_statement_on_an_observed_divergence() -> None:
    """No unconditional ALTER ROLE may survive.

    An unconditional ALTER is a privilege demand made on every apply, which is
    why 094 could not run under the RDS-shaped master (OMN-14899 follow-up) and
    then could not run at all under the ordinary service role the k8s Job uses
    on the managed instance (OMN-15343). Every ALTER must live inside a DO
    block that first read pg_roles.
    """
    sql = MIGRATION_FILE.read_text()
    executable = [
        line for line in sql.splitlines() if not line.lstrip().startswith("--")
    ]
    for line in executable:
        stripped = line.strip()
        if stripped.startswith("ALTER ROLE"):
            msg = (
                "top-level (ungated) ALTER ROLE found; it must be inside a DO "
                f"block gated on a pg_roles read: {stripped!r}"
            )
            raise AssertionError(msg)
    assert executable  # the file is not empty


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

# The throwaway-cluster harness moved to tests/integration/migrations/conftest.py
# under OMN-15297, which needed the same cluster to prove the app_dashboard
# GRANT chain rather than the role migration alone. One copy, not two.


def _apply_094_as_rds_shaped_role(
    pg: EphemeralPostgres,
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
    ephemeral_postgres: EphemeralPostgres,
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


# -----------------------------------------------------------------------------
# OMN-15343 — the ordinary-service-role apply path (the live cloud case)
#
# The k8s migration Job (omninode_infra k8s/migrations/omnibase-infra-migrate
# .yaml) escalated role DDL to `-U postgres`. The managed RDS instance has no
# such role, so deploy run 30406741279 died at connect time on THIS file, at the
# last flat migration, before the node loop ran. The runner now resolves the
# execution identity per POSTGRES_TARGET and on RDS uses the ordinary
# per-database role — which holds no CREATEROLE. This file must therefore be a
# true no-op under that role when app_dashboard is already in the required
# state, or it can never be recorded and every migration behind it stays
# blocked.
#
# `app_dashboard` on the live instance carries LOGIN (readback 2026-07-29:
# rolcanlogin = t) with every other flag already correct, so that is the exact
# state reproduced below.
# -----------------------------------------------------------------------------

ORDINARY_ROLE = "omn15343_ordinary"


def _seed_live_cloud_role_state(pg: EphemeralPostgres) -> None:
    """app_dashboard as it exists on the managed instance, plus a service role
    shaped like the one the Job connects as (LOGIN, no CREATEROLE)."""
    bootstrap = pg.connect()
    bootstrap.autocommit = True
    with bootstrap.cursor() as cur:
        cur.execute(
            "CREATE ROLE app_dashboard WITH LOGIN NOSUPERUSER NOBYPASSRLS "
            "NOCREATEDB NOCREATEROLE NOREPLICATION"
        )
        cur.execute(
            f"CREATE ROLE {ORDINARY_ROLE} WITH LOGIN NOSUPERUSER NOBYPASSRLS "
            "NOCREATEDB NOCREATEROLE NOREPLICATION"
        )
    bootstrap.close()


def _login_flag(pg: EphemeralPostgres) -> bool:
    conn = pg.connect()
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("SELECT rolcanlogin FROM pg_roles WHERE rolname = 'app_dashboard'")
        row = cur.fetchone()
    conn.close()
    assert row is not None
    return bool(row[0])


@pytest.mark.integration
@pytest.mark.postgres
def test_094_applies_under_an_ordinary_role_when_the_role_already_exists(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    """GREEN: the live cloud state, applied by the identity the Job actually
    connects as. Must exit 0 so the runner records it legitimately."""
    _seed_live_cloud_role_state(ephemeral_postgres)

    result = ephemeral_postgres.psql(
        "-v", "ON_ERROR_STOP=1", "-f", str(MIGRATION_FILE), user=ORDINARY_ROLE
    )

    assert result.returncode == 0, (
        "094 must be a true no-op under a non-CREATEROLE role when "
        "app_dashboard is already in the required state, or the k8s Job can "
        f"never record it on RDS (OMN-15343).\npsql stderr:\n{result.stderr}"
    )
    assert _login_flag(ephemeral_postgres) is True, (
        "the deployment-owned LOGIN attach must survive the apply"
    )


@pytest.mark.integration
@pytest.mark.postgres
def test_094_pre_fix_shape_fails_under_an_ordinary_role(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    """RED baseline: the pre-OMN-15343 shape of this file — an unconditional
    CREATE plus an unconditional ALTER — is refused by the same role in the same
    state, so the GREEN above is not vacuous.

    Derived as the minimal pre-fix statements rather than a copy of the old
    file: what is under test is that UNGATED role DDL cannot run here at all.
    """
    _seed_live_cloud_role_state(ephemeral_postgres)

    for statement in (
        "CREATE ROLE app_dashboard WITH NOLOGIN NOSUPERUSER NOBYPASSRLS",
        "ALTER ROLE app_dashboard NOLOGIN NOCREATEDB NOCREATEROLE",
    ):
        refused = ephemeral_postgres.psql(
            "-v", "ON_ERROR_STOP=1", "-c", statement, user=ORDINARY_ROLE
        )
        assert refused.returncode != 0, (
            f"expected {statement!r} to be refused for a non-CREATEROLE role"
        )
        assert "permission denied" in refused.stderr.lower(), refused.stderr

    assert _login_flag(ephemeral_postgres) is True


@pytest.mark.integration
@pytest.mark.postgres
def test_094_still_refuses_when_the_role_is_absent_and_cannot_be_created(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    """Fail-closed: "no-op when already correct" must not become "succeed when
    the role is missing". A migration that did not achieve its effect must not
    exit 0, because the runner records anything that exits 0.
    """
    bootstrap = ephemeral_postgres.connect()
    bootstrap.autocommit = True
    with bootstrap.cursor() as cur:
        cur.execute(
            f"CREATE ROLE {ORDINARY_ROLE} WITH LOGIN NOSUPERUSER NOBYPASSRLS "
            "NOCREATEDB NOCREATEROLE NOREPLICATION"
        )
    bootstrap.close()

    result = ephemeral_postgres.psql(
        "-v", "ON_ERROR_STOP=1", "-f", str(MIGRATION_FILE), user=ORDINARY_ROLE
    )

    assert result.returncode != 0, (
        "with app_dashboard absent and no CREATEROLE, 094 must fail loudly:\n"
        + result.stdout
    )
    assert "permission denied to create role" in result.stderr, result.stderr

    verify = ephemeral_postgres.connect()
    verify.autocommit = True
    with verify.cursor() as cur:
        cur.execute("SELECT count(*) FROM pg_roles WHERE rolname = 'app_dashboard'")
        (count,) = cur.fetchone()
    verify.close()
    assert count == 0


@pytest.mark.integration
@pytest.mark.postgres
def test_094_still_corrects_an_escalated_preexisting_role(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    """Gating on divergence must not weaken the security invariant: a role that
    actually carries BYPASSRLS is still corrected when the executing identity
    can do it."""
    bootstrap = ephemeral_postgres.connect()
    bootstrap.autocommit = True
    with bootstrap.cursor() as cur:
        cur.execute("CREATE ROLE app_dashboard WITH LOGIN BYPASSRLS")
    bootstrap.close()

    result = ephemeral_postgres.psql("-v", "ON_ERROR_STOP=1", "-f", str(MIGRATION_FILE))
    assert result.returncode == 0, result.stderr

    verify = ephemeral_postgres.connect()
    verify.autocommit = True
    with verify.cursor() as cur:
        cur.execute(
            "SELECT rolbypassrls, rolsuper, rolreplication FROM pg_roles "
            "WHERE rolname = 'app_dashboard'"
        )
        row = cur.fetchone()
    verify.close()
    assert row == (False, False, False), row


@pytest.mark.integration
@pytest.mark.postgres
def test_094_denies_read_without_tenant_context_through_real_connection(
    ephemeral_postgres: EphemeralPostgres,
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
