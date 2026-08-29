# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Provisioning proofs for the tenant_projection_writer identity (OMN-15425).

Enforcement shipped ahead of provisioning here, which is why these tests are
written the way they are. ``application_database.py`` has required
``tenant_projection -> tenant_projection_writer`` for the tenant projection
binding since OMN-15421, and OMN-16911's ``ProjectionBindingConnections.get()``
attests ``(current_user, current_database())`` against that declaration on every
connection — but nothing ever created the role. On the .201 dev lane
(2026-08-29) that produced 143/143 DLQ'd messages on
``node_projection_delegation_inference_response`` with

    PermissionError: Projection binding 'tenant_projection' connected as
      ('role_omnidash', 'omnidash_analytics'),
      expected ('tenant_projection_writer', 'omnidash_analytics')

Two halves close it and both are pinned here:

* ``102_create_tenant_projection_writer_role.sql`` — the cluster-wide role, with
  the RLS-relevant attributes off, no credential material, and no ``\\connect``
  (a new cross-database flat migration is a hard reject, OMN-15819).
* ``nodes/node_projection_delegation_inference_response/0004_grant_tenant_projection_writer.sql``
  — the schema/table authorization, delivered through the node-owned loop, which
  is the only sanctioned path that connects to ``omnidash_analytics``.

The text checks pin the security-critical properties so a later edit cannot
weaken them silently. The live-apply proofs then drive every claim through a
real throwaway cluster and a real client connection, because a role's behaviour
is not a property of the migration's prose.
"""

from __future__ import annotations

import re
from pathlib import Path

import psycopg2
import pytest
import yaml

from tests.integration.migrations.conftest import EphemeralPostgres

REPO_ROOT = Path(__file__).parent.parent.parent.parent
MIGRATION_FILE = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "102_create_tenant_projection_writer_role.sql"
)
ROLLBACK_FILE = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "rollback"
    / "rollback_102_create_tenant_projection_writer_role.sql"
)
GRANT_FILE = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "nodes"
    / "node_projection_delegation_inference_response"
    / "0004_grant_tenant_projection_writer.sql"
)
BOOTSTRAP_SCRIPT = (
    REPO_ROOT / "docker" / "migrations" / "forward" / "000_create_multiple_databases.sh"
)
LOCAL_TOPOLOGY = (
    REPO_ROOT / "src" / "omnibase_infra" / "topology" / "instances" / "local.yaml"
)

PRINCIPAL = "tenant_projection_writer"


def _executable_lines(path: Path) -> list[str]:
    """Drop SQL comment lines so a rationale cannot satisfy a code assertion."""
    return [
        line
        for line in path.read_text().splitlines()
        if not line.lstrip().startswith("--")
    ]


def _executable_text(path: Path) -> str:
    return "\n".join(_executable_lines(path))


# =============================================================================
# 102 — the role migration's text contract
# =============================================================================


@pytest.mark.integration
def test_102_creates_the_role_with_every_bypass_flag_off() -> None:
    sql = MIGRATION_FILE.read_text()

    assert f"CREATE ROLE {PRINCIPAL} WITH" in sql
    for attribute in (
        "NOLOGIN",
        "NOSUPERUSER",
        "NOBYPASSRLS",
        "NOCREATEDB",
        "NOCREATEROLE",
        "NOREPLICATION",
    ):
        assert attribute in sql, f"{attribute} must be pinned at create time"


@pytest.mark.integration
def test_102_enforces_the_escalation_negations_on_a_preexisting_role() -> None:
    """Presence of the role is not the property — the flags are.

    rolsuper or rolbypassrls on this principal exempts every tenant projection
    write from the OMN-14894 tenant_isolation policies, which turns the whole P5
    isolation proof into a false clean rather than evidence.
    """
    sql = MIGRATION_FILE.read_text()

    assert f"ALTER ROLE {PRINCIPAL} NOSUPERUSER NOBYPASSRLS NOREPLICATION" in sql


@pytest.mark.integration
def test_102_never_revokes_the_deployment_owned_login_attach() -> None:
    """NOLOGIN is create-time only (094's invariant, OMN-15343).

    The LOGIN + password attach is deployment-owned — minted by
    000_create_multiple_databases.sh's LOGIN_ONLY_ROLE_MAP on compose lanes and
    by the secret store on managed ones. Re-asserting NOLOGIN would revoke that
    attach as a side effect of recording a migration.
    """
    for statement in re.findall(
        rf"ALTER ROLE {PRINCIPAL}[^';]*", _executable_text(MIGRATION_FILE)
    ):
        assert "NOLOGIN" not in statement, (
            "NOLOGIN belongs to CREATE ROLE only; re-asserting it on a "
            f"pre-existing role revokes a deployment-owned attach: {statement!r}"
        )


@pytest.mark.integration
def test_102_gates_every_privileged_statement_on_an_observed_divergence() -> None:
    """No unconditional ALTER ROLE may survive.

    An unconditional ALTER is not idempotent, it is a privilege demand made on
    every apply — which is what made 094 undeliverable under the ordinary
    non-CREATEROLE service role the managed lane's migration Job runs as
    (OMN-15343). Every ALTER must sit inside a DO block that read pg_roles first.
    """
    for line in _executable_lines(MIGRATION_FILE):
        assert not line.strip().startswith("ALTER ROLE"), (
            "top-level (ungated) ALTER ROLE found; it must be inside a DO block "
            f"gated on a pg_roles read: {line.strip()!r}"
        )


@pytest.mark.integration
def test_102_contains_no_credential_material() -> None:
    sql = MIGRATION_FILE.read_text().upper()

    assert "PASSWORD '" not in sql and "ENCRYPTED PASSWORD" not in sql, (
        "credential material must never live in a migration; the LOGIN + "
        "password attach is a deployment-owned step, referenced by variable name"
    )


@pytest.mark.integration
def test_102_carries_no_foreign_connect_and_no_schema_or_table_grant() -> None:
    """The flat corpus's one-database rule (OMN-15819) is structural here.

    A NEW flat migration whose ``\\connect`` names a database other than
    ``omnibase_infra`` is a hard reject: the k8s Job that applies this corpus
    gates its ``psql -f`` loop on ``directive_db == "$DB_NAME"`` and can never
    deliver such a file. The schema/table half therefore rides the node-owned
    loop, and this file stays valid in any database context.
    """
    executable = _executable_text(MIGRATION_FILE)

    assert "\\connect" not in executable
    assert "GRANT USAGE ON SCHEMA" not in executable.upper()
    assert "ALTER DEFAULT PRIVILEGES" not in executable.upper()
    assert "GRANT CONNECT ON DATABASE omnidash_analytics" in executable, (
        "the topology declares DATABASE/CONNECT for this principal, and that "
        "grant is cluster-wide so it needs no \\connect"
    )


@pytest.mark.integration
def test_102_reads_back_the_connect_grant_instead_of_trusting_it() -> None:
    """A privilege-less GRANT warns and returns success — it does not raise.

    Proven in the OMN-15425 scratch replay (postgres:16-alpine, ordinary
    NOCREATEROLE login role): 102 completed with exit 0 and
    ``WARNING: no privileges were granted for "omnidash_analytics"``. Without the
    read-back the file would record itself as applied on a lane where the
    principal cannot connect at all — the same silent-success class OMN-15819
    unmasked for the ledger.
    """
    executable = _executable_text(MIGRATION_FILE)

    assert (
        f"has_database_privilege('{PRINCIPAL}', 'omnidash_analytics', 'CONNECT')"
        in executable
    )


@pytest.mark.integration
def test_102_rollback_revokes_before_dropping() -> None:
    sql = ROLLBACK_FILE.read_text()

    assert f"REVOKE ALL ON ALL TABLES IN SCHEMA public FROM {PRINCIPAL}" in sql
    assert f"REVOKE CONNECT ON DATABASE omnidash_analytics FROM {PRINCIPAL}" in sql
    assert f"DROP ROLE IF EXISTS {PRINCIPAL}" in sql


# =============================================================================
# The provisioning seam — credential by reference, never by value
# =============================================================================


@pytest.mark.integration
def test_login_credential_is_minted_by_reference_at_the_bootstrap_seam() -> None:
    """OMN-16843's LOGIN_ONLY_ROLE_MAP, not SERVICE_DB_MAP.

    SERVICE_DB_MAP's path issues ``GRANT USAGE, CREATE ON SCHEMA public`` plus
    blanket default privileges. CREATE on the schema lets the role OWN tables,
    and Postgres exempts a table's owner from RLS unconditionally — FORCE
    included. For this principal specifically that would silently undo the
    isolation the whole P5 cut exists to establish.
    """
    script = BOOTSTRAP_SCRIPT.read_text()
    login_only_block = script.split("LOGIN_ONLY_ROLE_MAP=(", 1)[1].split(")", 1)[0]

    assert f'"{PRINCIPAL}:TENANT_PROJECTION_WRITER_PASSWORD"' in login_only_block
    assert f"{PRINCIPAL}:" not in script.split("LOGIN_ONLY_ROLE_MAP", 1)[0], (
        "the principal must not also appear in SERVICE_DB_MAP"
    )


@pytest.mark.integration
def test_tenant_projection_binds_its_own_dsn_env_not_the_analytics_one() -> None:
    """One binding, one login role, one DSN env.

    ``app_dashboard`` and ``tenant_projection`` share the physical database but
    connect as different principals, and OMN-16911 attests ``current_user`` per
    binding — so a single DSN env cannot satisfy both. While it did, the dev
    lane's tenant projections resolved the analytics DSN (``role_omnidash`` on
    that lane) and DLQ'd 100% of their input at the attestation.
    """
    topology = yaml.safe_load(LOCAL_TOPOLOGY.read_text())
    bindings = topology["databases"]["application"]["bindings"]

    tenant_binding = bindings["tenant_projection"]
    assert tenant_binding["principal"] == PRINCIPAL
    assert tenant_binding["dsn_env"] == "ONEX_TENANT_DB_URL"

    dsn_envs = [binding["dsn_env"] for binding in bindings.values()]
    assert dsn_envs.count("ONEX_TENANT_DB_URL") == 1, (
        "the tenant projection DSN env must be exclusive to its own binding"
    )


# =============================================================================
# 0004 — the grant migration's text contract
# =============================================================================


@pytest.mark.integration
def test_grant_migration_matches_the_topology_declared_table_set() -> None:
    """The grant list is a transcription of the topology, not a hand-picked set.

    The topology's TABLE grants are themselves generated from node contract
    ``db_io.db_tables`` declarations by
    ``scripts/generate_application_database_table_grants.py --write``. If a
    contract adds a tenant-classified relation and this file is not updated in
    the same change, the new table's writes are denied at runtime — this test is
    what turns that into a red build instead of a silent zero-row projection.
    """
    topology = yaml.safe_load(LOCAL_TOPOLOGY.read_text())
    principal = topology["databases"]["application"]["principals"][PRINCIPAL]
    declared = {
        name
        for grant in principal["grants"]
        if grant["object_type"] == "TABLE"
        for name in grant["objects"]
    }

    granted = set(re.findall(r"^\s*'([a-z0-9_]+)',?$", GRANT_FILE.read_text(), re.M))

    assert granted == declared, (
        "grant migration drifted from the topology declaration: "
        f"missing={sorted(declared - granted)!r} extra={sorted(granted - declared)!r}"
    )


@pytest.mark.integration
def test_grant_migration_never_grants_delete_or_default_privileges() -> None:
    """A projection writer upserts; it does not reshape the table.

    And ``ALTER DEFAULT PRIVILEGES IN SCHEMA public`` is specifically forbidden:
    ``public`` still physically hosts ~40 OMNINODE_INTERNAL-domain families
    pending their own cutovers, so a default-privileges grant there would hand
    this principal access to every future internal-domain table as it appears —
    violating this ticket's own "cannot access internal relations" criterion.
    """
    executable = _executable_text(GRANT_FILE).upper()

    assert "DELETE" not in executable
    assert "ALTER DEFAULT PRIVILEGES" not in executable
    assert "GRANT SELECT, INSERT, UPDATE ON PUBLIC.%I" in executable


@pytest.mark.integration
def test_grant_migration_fails_loud_when_the_role_is_missing() -> None:
    sql = GRANT_FILE.read_text()

    assert "RAISE EXCEPTION" in sql
    assert "102_create_tenant_projection_writer_role.sql" in sql, (
        "the exception must name the migration that provisions the role"
    )


# =============================================================================
# Live-apply proofs
#
# Everything above pins text. None of it applies a migration or opens a
# connection, so none of it can prove the role behaves the way the prose claims.
# These do.
# =============================================================================


def _apply_102(pg: EphemeralPostgres, *, user: str = "postgres") -> None:
    result = pg.psql("-v", "ON_ERROR_STOP=1", "-f", str(MIGRATION_FILE), user=user)
    assert result.returncode == 0, result.stderr


@pytest.mark.integration
@pytest.mark.postgres
def test_102_applies_idempotently_with_the_flags_off(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    _apply_102(ephemeral_postgres)
    _apply_102(ephemeral_postgres)

    conn = ephemeral_postgres.connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT rolcanlogin, rolsuper, rolbypassrls, rolcreatedb, "
                "rolcreaterole, rolreplication FROM pg_roles WHERE rolname = %s",
                (PRINCIPAL,),
            )
            row = cur.fetchone()
    finally:
        conn.close()

    assert row is not None, "the role must exist after the migration applies"
    assert row == (False, False, False, False, False, False)


@pytest.mark.integration
@pytest.mark.postgres
def test_102_corrects_an_observed_escalation(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    """A pre-existing escalated role is corrected, never trusted."""
    _apply_102(ephemeral_postgres)

    conn = ephemeral_postgres.connect()
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(f"ALTER ROLE {PRINCIPAL} BYPASSRLS CREATEDB CREATEROLE")
    finally:
        conn.close()

    _apply_102(ephemeral_postgres)

    conn = ephemeral_postgres.connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT rolbypassrls, rolcreatedb, rolcreaterole FROM pg_roles "
                "WHERE rolname = %s",
                (PRINCIPAL,),
            )
            row = cur.fetchone()
    finally:
        conn.close()

    assert row == (False, False, False)


@pytest.mark.integration
@pytest.mark.postgres
def test_102_applies_cleanly_under_an_ordinary_non_createrole_role(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    """The live managed-lane case (OMN-15343).

    The k8s migration Job holds no cluster-admin credential by design, and the
    managed instance has no ``postgres`` role at all. A migration that cannot be
    recorded there blocks every migration behind it, so this file has to be a
    true no-op once the role is already in its required state.
    """
    _apply_102(ephemeral_postgres)

    conn = ephemeral_postgres.connect()
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(
                "CREATE ROLE plain_service_role WITH LOGIN NOSUPERUSER NOCREATEDB "
                "NOCREATEROLE NOBYPASSRLS NOREPLICATION"
            )
    finally:
        conn.close()

    _apply_102(ephemeral_postgres, user="plain_service_role")


@pytest.mark.integration
@pytest.mark.postgres
def test_the_provisioned_role_is_denied_internal_and_catalog_schemas(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    """AC: tenant projection writes cannot reach internal relations.

    102 grants CONNECT and nothing else, and 0004 grants USAGE on ``public`` and
    ``tenant`` only. No path in either file widens that, so the internal and
    catalog schemas are refused at the schema boundary — before any table ACL or
    RLS policy is consulted.
    """
    _apply_102(ephemeral_postgres)

    admin = ephemeral_postgres.connect()
    admin.autocommit = True
    try:
        with admin.cursor() as cur:
            cur.execute(
                f"ALTER ROLE {PRINCIPAL} WITH LOGIN PASSWORD 'ephemeral-proof-only'"  # pragma: allowlist secret
            )
            cur.execute("CREATE SCHEMA omninode_internal")
            cur.execute("CREATE TABLE omninode_internal.live_events (id int)")
    finally:
        admin.close()

    writer = ephemeral_postgres.connect(user=PRINCIPAL)
    try:
        with writer.cursor() as cur:
            with pytest.raises(psycopg2.errors.InsufficientPrivilege):
                cur.execute("SELECT count(*) FROM omninode_internal.live_events")
    finally:
        writer.close()


@pytest.mark.integration
@pytest.mark.postgres
def test_the_provisioned_role_reads_back_the_attested_identity_tuple(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    """The exact comparison ``ProjectionBindingConnections.get()`` performs.

    It executes ``SELECT current_user, current_database()`` and refuses unless
    the tuple equals ``(binding.principal, binding.physical_database)``. This
    proves the provisioned identity satisfies that check on the principal half —
    the half that was ``role_omnidash`` on the dev lane and produced 143/143 DLQ.
    """
    _apply_102(ephemeral_postgres)

    admin = ephemeral_postgres.connect()
    admin.autocommit = True
    try:
        with admin.cursor() as cur:
            cur.execute(
                f"ALTER ROLE {PRINCIPAL} WITH LOGIN PASSWORD 'ephemeral-proof-only'"  # pragma: allowlist secret
            )
    finally:
        admin.close()

    writer = ephemeral_postgres.connect(user=PRINCIPAL)
    try:
        with writer.cursor() as cur:
            cur.execute("SELECT current_user, current_database()")
            identity = cur.fetchone()
    finally:
        writer.close()

    assert identity is not None
    assert identity[0] == PRINCIPAL
