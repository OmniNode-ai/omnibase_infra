# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-15297 — the app_dashboard GRANT CHAIN must yield a usable connection.

THE DEFECT (live readback, 2026-07-28, `.201` dev lane, db `omnidash_analytics`)

    FATAL:  permission denied for database "omnidash_analytics"
    DETAIL:  User does not have CONNECT privilege.

094 creates the role. 0023 grants ``USAGE ON SCHEMA public`` and ``SELECT`` on
the two RLS-covered delegation tables. Nothing in the chain ever grants
``CONNECT ON DATABASE``. On a stock Postgres ``CONNECT`` is held by ``PUBLIC``,
so the gap is LATENT — every existing test passes and the role appears to work.
On a database where ``PUBLIC``'s CONNECT has been revoked (the dev lane today,
and the explicit target state of OMN-15355) the role cannot open a session at
all, and every grant behind it is unreachable.

WHY THIS FILE IS SHAPED THE WAY IT IS
-------------------------------------
The chain is DISCOVERED, not enumerated (``_app_dashboard_grant_chain``). That
is deliberate and it is what makes the RED honest: with the fix absent this
module still collects, still applies a complete and self-consistent migration
chain, and fails on the OBSERVED PRIVILEGE — not on a missing file. A test that
red-fails with ``FileNotFoundError: 097_...sql`` proves only that a file is
absent; it cannot distinguish "not written yet" from "written and wrong", which
is the distinction the RED is supposed to establish.

Every assertion here is driven through a real client connection authenticated
AS ``app_dashboard`` against a real cluster, applied through the same
``psql -v ON_ERROR_STOP=1 -f`` invocation ``scripts/run-forward-migrations.sh``
uses. String-matching the SQL text would restate the migration rather than test
it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import psycopg2
import pytest

from tests.integration.migrations.conftest import EphemeralPostgres

REPO_ROOT = Path(__file__).resolve().parents[3]
FORWARD_DIR = REPO_ROOT / "docker" / "migrations" / "forward"
NODE_DELEGATION_DIR = FORWARD_DIR / "nodes" / "node_projection_delegation"
RLS_MIGRATION = NODE_DELEGATION_DIR / "0023_delegation_rls_tenant_isolation.sql"

# The database the dashboard read path connects to. Named here rather than
# derived so the test states the seam value it is asserting against, matching
# `APPLICATION_DATABASE_PHYSICAL_NAME` in omnibase_infra.topology.
ANALYTICS_DB = "omnidash_analytics"
READ_ROLE = "app_dashboard"


def _executable_sql(path: Path) -> str:
    """The migration text with ``--`` comment lines stripped.

    Comments mention roles constantly (096 names app_dashboard only to point at
    this very ticket). Discovery has to read what the file DOES.
    """
    return "\n".join(
        line
        for line in path.read_text().splitlines()
        if not line.lstrip().startswith("--")
    )


def _app_dashboard_grant_chain() -> list[Path]:
    """Every top-level forward migration that acts on ``app_dashboard``.

    Sorted by filename, which is the order the runner applies them in.
    """
    return sorted(
        path for path in FORWARD_DIR.glob("*.sql") if READ_ROLE in _executable_sql(path)
    )


def _apply(
    pg: EphemeralPostgres, path: Path, *, dbname: str
) -> subprocess.CompletedProcess[str]:
    result = pg.psql("-v", "ON_ERROR_STOP=1", "-f", str(path), dbname=dbname)
    assert result.returncode == 0, (
        f"{path.name} failed to apply against {dbname}:\n{result.stderr}"
    )
    return result


def _seed_analytics_database(pg: EphemeralPostgres) -> None:
    """Reproduce the lane condition the defect needs, and only that.

    Two facts matter and both are reproduced from live state rather than
    invented:

    * ``omnidash_analytics`` exists and ``PUBLIC`` has NO ``CONNECT`` on it.
      This is the dev-lane state recorded on OMN-15297 and the declared target
      state of OMN-15355 (revoke PUBLIC CONNECT). It is what turns the missing
      grant from latent into fatal.
    * The two tables 0023 covers exist, owned by the migration role (never by
      ``app_dashboard`` — an owner is exempt from RLS and the proof would be
      vacuous).
    """
    bootstrap = pg.connect()
    bootstrap.autocommit = True
    with bootstrap.cursor() as cur:
        cur.execute(f'CREATE DATABASE "{ANALYTICS_DB}"')
        cur.execute(f'REVOKE CONNECT ON DATABASE "{ANALYTICS_DB}" FROM PUBLIC')
    bootstrap.close()

    analytics = pg.connect(dbname=ANALYTICS_DB)
    analytics.autocommit = True
    with analytics.cursor() as cur:
        cur.execute(
            "CREATE TABLE delegation_events ("
            "  event_id TEXT PRIMARY KEY,"
            "  tenant_id TEXT NOT NULL"
            ")"
        )
        cur.execute(
            "CREATE TABLE delegation_budget_state ("
            "  budget_key TEXT PRIMARY KEY,"
            "  tenant_id TEXT NOT NULL"
            ")"
        )
        # A tenant-stamped table with NO tenant_isolation policy. Nothing in the
        # chain may grant SELECT on it: readable-before-policy is the exact
        # ordering hazard migration 094's header calls out.
        cur.execute(
            "CREATE TABLE uncovered_projection ("
            "  row_id TEXT PRIMARY KEY,"
            "  tenant_id TEXT NOT NULL"
            ")"
        )
    analytics.close()


def _attach_deployment_owned_login(pg: EphemeralPostgres) -> None:
    """The LOGIN + password attach the migrations deliberately do not carry.

    094 states this explicitly: credential material never lives in a migration,
    the attach is deployment-owned (AWS Secrets Manager on the cloud path). The
    test performs it here because a role that cannot log in cannot demonstrate
    whether CONNECT is the thing that is missing.
    """
    bootstrap = pg.connect()
    bootstrap.autocommit = True
    with bootstrap.cursor() as cur:
        cur.execute(f"ALTER ROLE {READ_ROLE} WITH LOGIN PASSWORD 'omn15297-ephemeral'")
    bootstrap.close()


def _apply_full_chain(pg: EphemeralPostgres) -> None:
    """Apply the chain exactly as the runner does.

    Top-level forward migrations run against ``POSTGRES_DB``; node migrations
    run against ``NODE_POSTGRES_DB`` (compose sets ``omnidash_analytics``). Any
    file needing the analytics context switches with its own ``\\connect``
    directive, which is why these go through psql rather than a driver.
    """
    for migration in _app_dashboard_grant_chain():
        _apply(pg, migration, dbname="postgres")
    _apply(pg, RLS_MIGRATION, dbname=ANALYTICS_DB)


def _database_privilege(pg: EphemeralPostgres, privilege: str) -> bool:
    conn = pg.connect()
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(
            "SELECT has_database_privilege(%s, %s, %s)",
            (READ_ROLE, ANALYTICS_DB, privilege),
        )
        row = cur.fetchone()
    conn.close()
    assert row is not None
    return bool(row[0])


# =============================================================================
# The seam: the chain must leave app_dashboard able to open a session.
# =============================================================================


@pytest.mark.integration
@pytest.mark.postgres
def test_grant_chain_leaves_app_dashboard_with_connect_on_the_analytics_database(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    """``has_database_privilege(app_dashboard, omnidash_analytics, CONNECT)``.

    This is OMN-15297's first acceptance test, stated against the catalog
    rather than against the migration text.
    """
    pg = ephemeral_postgres
    _seed_analytics_database(pg)
    _apply_full_chain(pg)

    assert _database_privilege(pg, "CONNECT"), (
        "the app_dashboard grant chain applied cleanly and still leaves the "
        f"role without CONNECT on {ANALYTICS_DB}. PUBLIC's CONNECT is revoked "
        "on this database (the .201 dev-lane state, and the declared target "
        "state of OMN-15355), so the role cannot open a session at all and "
        "every USAGE/SELECT grant behind it is unreachable. Chain applied: "
        f"{[p.name for p in _app_dashboard_grant_chain()]} + {RLS_MIGRATION.name}"
    )


@pytest.mark.integration
@pytest.mark.postgres
def test_app_dashboard_can_actually_open_a_session_and_read_its_own_tenant(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    """The catalog check above, driven through a real client connection.

    ``has_database_privilege`` is the catalog's opinion; this is the observed
    behaviour of the connection path the dashboard actually uses. Both are kept
    because they fail differently: a catalog-only assertion cannot see a
    ``pg_hba``/session-level refusal, and a connect-only assertion cannot say
    which privilege was missing.
    """
    pg = ephemeral_postgres
    _seed_analytics_database(pg)
    _apply_full_chain(pg)
    _attach_deployment_owned_login(pg)

    seed = pg.connect(dbname=ANALYTICS_DB)
    seed.autocommit = True
    with seed.cursor() as cur:
        cur.execute(
            "INSERT INTO delegation_events (event_id, tenant_id) VALUES (%s, %s)",
            ("omn15297-a", "tenant-a"),
        )
    seed.close()

    try:
        reader = pg.connect(
            user=READ_ROLE, password="omn15297-ephemeral", dbname=ANALYTICS_DB
        )
    except psycopg2.OperationalError as exc:  # pragma: no cover - the RED path
        pytest.fail(
            f"{READ_ROLE} could not open a session on {ANALYTICS_DB} after the "
            f"full grant chain applied cleanly: {exc}"
        )

    reader.autocommit = True
    with reader.cursor() as cur:
        cur.execute("SELECT current_user, session_user")
        identity = cur.fetchone()
        cur.execute("SET app.tenant_id = 'tenant-a'")
        cur.execute("SELECT count(*) FROM delegation_events")
        visible = cur.fetchone()
    reader.close()

    assert identity == (READ_ROLE, READ_ROLE)
    assert visible is not None and visible[0] == 1, (
        "app_dashboard connected but could not read its own tenant's row — "
        "CONNECT is granted and the SELECT/USAGE half of the chain is not"
    )


@pytest.mark.integration
@pytest.mark.postgres
def test_grant_chain_is_idempotent_on_reapply(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    """OMN-15297 acceptance 2. Re-running the runner must not error.

    ``_apply`` asserts returncode 0 on every file, so a second full pass is the
    assertion. Applied twice rather than once because the migration ledger does
    not protect a file that has been edited and re-run by hand on a lane.
    """
    pg = ephemeral_postgres
    _seed_analytics_database(pg)
    _apply_full_chain(pg)
    _apply_full_chain(pg)

    assert _database_privilege(pg, "CONNECT")


@pytest.mark.integration
@pytest.mark.postgres
def test_chain_grants_exactly_connect_on_the_database_and_nothing_more(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    """The database-level ACL entry for app_dashboard must be exactly CONNECT.

    ``CREATE ON DATABASE`` lets a role make schemas, and a role that creates
    objects OWNS them — and an owner is exempt from row-level security, FORCE
    included. Widening this grant would reopen the bypass the whole epic exists
    to close, so the ceiling is pinned rather than assumed.

    Read from ``pg_database.datacl`` filtered to this grantee, NOT from
    ``has_database_privilege``: the effective-privilege function also reports
    privileges inherited from ``PUBLIC`` (``TEMPORARY`` is a stock Postgres
    PUBLIC default), so it cannot distinguish what this chain granted from what
    the cluster already gave everyone. Revoking PUBLIC's remaining defaults is
    OMN-15355's blast radius, deliberately not this ticket's.
    """
    pg = ephemeral_postgres
    _seed_analytics_database(pg)
    _apply_full_chain(pg)

    conn = pg.connect()
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(
            "SELECT coalesce(array_agg(acl.privilege_type ORDER BY acl.privilege_type), "
            "ARRAY[]::text[]) "
            "FROM pg_database d, aclexplode(d.datacl) acl "
            "WHERE d.datname = %s AND acl.grantee = %s::regrole::oid",
            (ANALYTICS_DB, READ_ROLE),
        )
        row = cur.fetchone()
    conn.close()

    assert row is not None
    assert list(row[0]) == ["CONNECT"], (
        f"database-level grants to {READ_ROLE} on {ANALYTICS_DB} are {row[0]}; "
        "the read path needs CONNECT and nothing else — CREATE would let the "
        "role own objects, and an owner is exempt from row-level security"
    )


@pytest.mark.integration
@pytest.mark.postgres
def test_chain_never_grants_select_on_a_table_without_a_tenant_policy(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    """Readable-before-policy is the ordering hazard, and it is fail-closed.

    ``uncovered_projection`` carries ``tenant_id`` but has no RLS and no
    ``tenant_isolation`` policy. If any migration in the chain reaches for a
    blanket ``GRANT SELECT ON ALL TABLES``, this table becomes readable with no
    tenant predicate at all — a cross-tenant read that every RLS test in the
    repo would still report green, because none of them look at this table.
    """
    pg = ephemeral_postgres
    _seed_analytics_database(pg)
    _apply_full_chain(pg)

    conn = pg.connect(dbname=ANALYTICS_DB)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(
            "SELECT has_table_privilege(%s, 'public.uncovered_projection', 'SELECT')",
            (READ_ROLE,),
        )
        uncovered = cur.fetchone()
        cur.execute(
            "SELECT has_table_privilege(%s, 'public.delegation_events', 'SELECT')",
            (READ_ROLE,),
        )
        covered = cur.fetchone()
    conn.close()

    assert uncovered is not None and uncovered[0] is False, (
        "app_dashboard can SELECT a table that has no tenant_isolation policy "
        "— the grant chain must never make a table readable before its policy "
        "exists"
    )
    assert covered is not None and covered[0] is True


@pytest.mark.integration
@pytest.mark.postgres
def test_chain_never_grants_write_on_the_rls_covered_read_tables(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    """app_dashboard is the READ role. INSERT/UPDATE/DELETE are not its shape.

    096 had to re-narrow role_omnidash after a blanket grant silently re-added
    DELETE to three FORCE-RLS tables. Same class, pinned here before it can
    happen to this role.
    """
    pg = ephemeral_postgres
    _seed_analytics_database(pg)
    _apply_full_chain(pg)

    conn = pg.connect(dbname=ANALYTICS_DB)
    conn.autocommit = True
    granted: dict[tuple[str, str], bool] = {}
    with conn.cursor() as cur:
        for table in ("delegation_events", "delegation_budget_state"):
            for privilege in ("INSERT", "UPDATE", "DELETE", "TRUNCATE"):
                cur.execute(
                    "SELECT has_table_privilege(%s, %s, %s)",
                    (READ_ROLE, f"public.{table}", privilege),
                )
                row = cur.fetchone()
                assert row is not None
                granted[(table, privilege)] = bool(row[0])
    conn.close()

    offenders = [key for key, held in granted.items() if held]
    assert not offenders, (
        f"app_dashboard holds write privileges it must not: {offenders}"
    )


@pytest.mark.integration
@pytest.mark.postgres
def test_chain_leaves_the_read_role_non_owner_and_non_bypassing(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    """The role flags and ownership, re-read after the grant chain.

    094 asserts these at role-creation time. They are re-asserted AFTER the
    grants because the grant chain is the thing this ticket edits, and a grant
    file that reached for ownership or an ALTER ROLE would make every policy in
    OMN-14894 inert without failing any existing test.
    """
    pg = ephemeral_postgres
    _seed_analytics_database(pg)
    _apply_full_chain(pg)

    cluster = pg.connect()
    cluster.autocommit = True
    with cluster.cursor() as cur:
        cur.execute(
            "SELECT rolsuper, rolbypassrls, rolcreatedb, rolcreaterole, rolreplication "
            "FROM pg_roles WHERE rolname = %s",
            (READ_ROLE,),
        )
        flags = cur.fetchone()
    cluster.close()

    assert flags is not None, f"{READ_ROLE} was not created by the chain"
    assert flags == (False, False, False, False, False), (
        f"{READ_ROLE} carries an escalated flag after the grant chain: {flags}"
    )

    analytics = pg.connect(dbname=ANALYTICS_DB)
    analytics.autocommit = True
    with analytics.cursor() as cur:
        cur.execute(
            "SELECT tablename FROM pg_tables "
            "WHERE schemaname = 'public' AND tableowner = %s",
            (READ_ROLE,),
        )
        owned = [row[0] for row in cur.fetchall()]
    analytics.close()

    assert not owned, (
        f"{READ_ROLE} owns table(s) {owned} — an owner is exempt from row-level "
        "security (FORCE included), so every tenant_isolation policy on them is "
        "inert and any 'clean under RLS' reading is a false clean"
    )
