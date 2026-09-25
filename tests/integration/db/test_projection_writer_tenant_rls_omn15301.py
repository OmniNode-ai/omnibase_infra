# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Real PostgreSQL proof for split domain projection adapters (OMN-15421)."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest

psycopg2 = pytest.importorskip("psycopg2", reason="psycopg2 required for RLS proof")
psycopg2_extras = pytest.importorskip(
    "psycopg2.extras", reason="psycopg2 extras required for UUID proof"
)
psycopg2_extras.register_uuid()

from omnibase_infra.errors.error_projection import ProjectionTenantContextError
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _build_projection_db_adapter,
)
from omnibase_infra.topology.physical_schema_mapping import (
    physical_grant_schema_for_table,
)
from tests.helpers.application_db_topology import (
    projection_database_target,
    projection_database_urls,
)
from tests.helpers.projection_tenant_authority import verified_tenant_dispatch

pytestmark = [pytest.mark.integration, pytest.mark.postgres, pytest.mark.slow]

DATABASE = "omnidash_analytics"
TENANT_TABLE = "delegation_events"
INTERNAL_TABLE = "generation_events"
CATALOG_TABLE = "plan_tiers"
# OMN-17422 amendment 2 (K3): each relation is built where the runtime's own
# physical mapping says its SQL lands, never in a hard-coded schema. The suite
# was stranded twice by hard-coding: once by the tenant bridge, once by the
# internal bridge (`generation_events` is logically omninode_internal but
# physically in `public` until OMN-15359 moves it), and nothing noticed because
# the module is skipped in CI. Deriving the schema makes the next remap move the
# fixture with it.
TENANT_SCHEMA = physical_grant_schema_for_table("public", TENANT_TABLE)
INTERNAL_SCHEMA = physical_grant_schema_for_table("omninode_internal", INTERNAL_TABLE)
CATALOG_SCHEMA = physical_grant_schema_for_table("platform_catalog", CATALOG_TABLE)
TENANT_ROLE = "tenant_projection_writer"
INTERNAL_ROLE = "omninode_runtime"
CATALOG_READER_ROLE = "app_dashboard"
ROLE_PASSWORD = "domain_adapter_proof_only"  # pragma: allowlist secret
OWNER_ROLE = "postgres"
TENANT_A = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
TENANT_B = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")

# OMN-17887 (operator ruling 2026-09-24): the `tenant` schema is RETIRED and the
# TENANT domain's schema is `public` for good, so the tenant-domain relation is
# built in `public` -- exactly where the runtime's emitted SQL names it.
_SCHEMA_SQL = f"""
CREATE SCHEMA omninode_internal;
CREATE SCHEMA platform_catalog;
CREATE TABLE {TENANT_SCHEMA}.{TENANT_TABLE} (
    correlation_id UUID PRIMARY KEY,
    task_type TEXT NOT NULL,
    tenant_id UUID NOT NULL
);
ALTER TABLE {TENANT_SCHEMA}.{TENANT_TABLE} ENABLE ROW LEVEL SECURITY;
ALTER TABLE {TENANT_SCHEMA}.{TENANT_TABLE} FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON {TENANT_SCHEMA}.{TENANT_TABLE}
  FOR ALL
  USING (tenant_id = current_setting('app.tenant_id', true)::uuid)
  WITH CHECK (tenant_id = current_setting('app.tenant_id', true)::uuid);
CREATE TABLE {INTERNAL_SCHEMA}.{INTERNAL_TABLE} (
    correlation_id UUID PRIMARY KEY,
    source_tenant_id UUID NULL,
    status TEXT NOT NULL
);
CREATE TABLE {CATALOG_SCHEMA}.{CATALOG_TABLE} (
    tier_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL
);
"""


def _pg_bin(name: str) -> str | None:
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
        "initdb/pg_ctl not available — cannot bring up ephemeral PostgreSQL",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def pg_socket_dir() -> Iterator[str]:
    root = Path(tempfile.mkdtemp(prefix="omn15421-pg-"))
    data_dir = root / "data"
    socket_dir = root / "socket"
    socket_dir.mkdir()
    postgres_env = {**os.environ, "LANG": "C", "LC_ALL": "C", "LC_CTYPE": "C"}
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
        env=postgres_env,
    )
    subprocess.run(
        [
            str(_PG_CTL),
            "-D",
            str(data_dir),
            "-l",
            str(root / "postgres.log"),
            "-o",
            f"-k {socket_dir} -h ''",
            "-w",
            "start",
        ],
        check=True,
        capture_output=True,
        env=postgres_env,
    )
    try:
        yield str(socket_dir)
    finally:
        subprocess.run(
            [str(_PG_CTL), "-D", str(data_dir), "-m", "immediate", "-w", "stop"],
            check=False,
            capture_output=True,
            env=postgres_env,
        )
        shutil.rmtree(root, ignore_errors=True)


def _dsn(socket_dir: str, user: str, password: str | None = None) -> str:
    value = f"host={socket_dir} dbname={DATABASE} user={user}"
    return f"{value} password={password}" if password else value


@pytest.fixture(scope="module")
def domain_dsns(pg_socket_dir: str) -> dict[str, str]:
    bootstrap = psycopg2.connect(
        f"host={pg_socket_dir} dbname=postgres user={OWNER_ROLE}"
    )
    bootstrap.autocommit = True
    with bootstrap.cursor() as cursor:
        cursor.execute(f"CREATE DATABASE {DATABASE}")
    bootstrap.close()

    owner_dsn = _dsn(pg_socket_dir, OWNER_ROLE)
    conn = psycopg2.connect(owner_dsn)
    conn.autocommit = True
    with conn.cursor() as cursor:
        cursor.execute(_SCHEMA_SQL)
        for role in (TENANT_ROLE, INTERNAL_ROLE, CATALOG_READER_ROLE):
            cursor.execute(
                f"CREATE ROLE {role} LOGIN PASSWORD %s NOSUPERUSER NOBYPASSRLS",
                (ROLE_PASSWORD,),
            )
            cursor.execute(f"GRANT CONNECT ON DATABASE {DATABASE} TO {role}")
        cursor.execute(f"GRANT USAGE ON SCHEMA {TENANT_SCHEMA} TO {TENANT_ROLE}")
        cursor.execute(
            f"GRANT SELECT, INSERT, UPDATE ON {TENANT_SCHEMA}.{TENANT_TABLE} TO {TENANT_ROLE}"
        )
        cursor.execute(f"GRANT USAGE ON SCHEMA {INTERNAL_SCHEMA} TO {INTERNAL_ROLE}")
        cursor.execute(
            "GRANT SELECT, INSERT, UPDATE ON "
            f"{INTERNAL_SCHEMA}.{INTERNAL_TABLE} TO {INTERNAL_ROLE}"
        )
        cursor.execute(
            f"GRANT USAGE ON SCHEMA {CATALOG_SCHEMA} TO {CATALOG_READER_ROLE}"
        )
        cursor.execute(
            f"GRANT SELECT ON {CATALOG_SCHEMA}.{CATALOG_TABLE} TO {CATALOG_READER_ROLE}"
        )
    conn.close()
    return {
        "owner": owner_dsn,
        "tenant": _dsn(pg_socket_dir, TENANT_ROLE, ROLE_PASSWORD),
        "internal": _dsn(pg_socket_dir, INTERNAL_ROLE, ROLE_PASSWORD),
        "catalog": _dsn(pg_socket_dir, CATALOG_READER_ROLE, ROLE_PASSWORD),
    }


@pytest.fixture(autouse=True)
def _clean_tables(domain_dsns: dict[str, str]) -> Iterator[None]:
    yield
    conn = psycopg2.connect(domain_dsns["owner"])
    conn.autocommit = True
    with conn.cursor() as cursor:
        cursor.execute(f"TRUNCATE {TENANT_SCHEMA}.{TENANT_TABLE}")
        cursor.execute(f"TRUNCATE {INTERNAL_SCHEMA}.{INTERNAL_TABLE}")
        cursor.execute(f"TRUNCATE {CATALOG_SCHEMA}.{CATALOG_TABLE}")
    conn.close()


def _tenant_adapter(dsn: str, tenant_id: UUID | None) -> object:
    target = projection_database_target(TENANT_TABLE, schema="public")
    authority = None
    event = None
    if tenant_id is not None:
        authority, event = verified_tenant_dispatch(tenant_id)
    return _build_projection_db_adapter(
        projection_database_urls(target, dsn), target, authority, event
    )


def _internal_adapter(dsn: str) -> object:
    target = projection_database_target(INTERNAL_TABLE, schema="omninode_internal")
    return _build_projection_db_adapter(
        projection_database_urls(target, dsn), target, None, None
    )


def _catalog_reader_adapter(dsn: str) -> object:
    target = projection_database_target(
        CATALOG_TABLE,
        schema="platform_catalog",
        access="read",
        catalog_read_binding="app_dashboard",
        unshipped_grant_principal="app_dashboard",
        unshipped_grant_reason=(
            "PLATFORM_CATALOG grants are not derivable from node contracts: no "
            "db_io.db_tables block declares a catalog relation, so the shipped "
            "topology carries none (OMN-15355/OMN-15424 own that grant set). "
            "This asserts catalog read isolation, not catalog coverage."
        ),
    )
    return _build_projection_db_adapter(
        projection_database_urls(target, dsn), target, None, None
    )


def _tenant_rows(owner_dsn: str) -> list[tuple[UUID, UUID]]:
    conn = psycopg2.connect(owner_dsn)
    conn.autocommit = True
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                f"SELECT correlation_id, tenant_id FROM {TENANT_SCHEMA}.{TENANT_TABLE} "  # noqa: S608 - module constant
                "ORDER BY tenant_id"
            )
            return list(cursor.fetchall())
    finally:
        conn.close()


def test_fixture_relations_live_where_the_runtime_writes(
    domain_dsns: dict[str, str],
) -> None:
    """K3: the fixture is built from the runtime's own physical mapping.

    Every relation this suite writes must exist in the schema that
    ``physical_grant_schema_for_table`` resolves, which is the schema the
    adapter's SQL names. When a family is remapped, this fails first and names
    the relation, instead of every write test failing with ``UndefinedTable``.
    """
    expected = {
        (TENANT_TABLE, physical_grant_schema_for_table("public", TENANT_TABLE)),
        (
            INTERNAL_TABLE,
            physical_grant_schema_for_table("omninode_internal", INTERNAL_TABLE),
        ),
        (
            CATALOG_TABLE,
            physical_grant_schema_for_table("platform_catalog", CATALOG_TABLE),
        ),
    }
    conn = psycopg2.connect(domain_dsns["owner"])
    conn.autocommit = True
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT c.relname, n.nspname FROM pg_class c "
                "JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE c.relkind = 'r' AND c.relname = ANY(%s)",
                ([name for name, _ in expected],),
            )
            built = set(cursor.fetchall())
    finally:
        conn.close()
    assert built == expected, (
        f"fixture relations {sorted(built)} are not where the runtime writes "
        f"{sorted(expected)}"
    )


def test_verified_tenant_write_read_and_uuid_preservation(
    domain_dsns: dict[str, str],
) -> None:
    correlation_id = uuid4()
    adapter = _tenant_adapter(domain_dsns["tenant"], TENANT_A)
    try:
        assert adapter.upsert(
            TENANT_TABLE,
            "correlation_id",
            {"correlation_id": correlation_id, "task_type": "proof"},
        )
        found = adapter.query(TENANT_TABLE, {"correlation_id": correlation_id})
        assert found[0]["correlation_id"] == correlation_id
        assert isinstance(found[0]["correlation_id"], UUID)
    finally:
        adapter.close()
    assert _tenant_rows(domain_dsns["owner"]) == [(correlation_id, TENANT_A)]


def test_untrusted_row_tenant_cannot_choose_context(
    domain_dsns: dict[str, str],
) -> None:
    adapter = _tenant_adapter(domain_dsns["tenant"], TENANT_A)
    with pytest.raises(ProjectionTenantContextError, match="does not match"):
        adapter.upsert(
            TENANT_TABLE,
            "correlation_id",
            {"correlation_id": uuid4(), "task_type": "proof", "tenant_id": TENANT_B},
        )
    assert _tenant_rows(domain_dsns["owner"]) == []


def test_equal_tenant_string_is_replaced_with_uuid(domain_dsns: dict[str, str]) -> None:
    adapter = _tenant_adapter(domain_dsns["tenant"], TENANT_A)
    try:
        assert adapter.upsert(
            TENANT_TABLE,
            "correlation_id",
            {
                "correlation_id": uuid4(),
                "task_type": "proof",
                "tenant_id": str(TENANT_A),
            },
        )
    finally:
        adapter.close()


def test_unbound_write_is_scoped_to_its_recorded_tenant(
    domain_dsns: dict[str, str],
) -> None:
    """With no authority bound, the row lands under the tenant it records.

    OMN-17422, option (a) ruling, shipped in omnibase_infra#3088. This test used
    to assert the OMN-16831 interim shape, where Postgres refused every unbound
    write because no ``app.tenant_id`` GUC was set. That refused every tenant
    row for every tenant, which is the defect OMN-17422 was filed for. Under the
    ruling, the writer opens a transaction-local scope from the attribution it
    is about to write, so the FORCE-RLS ``WITH CHECK`` is satisfied by the row's
    own tenant, and isolation still holds on every other path:

    * the row lands, and lands as the tenant it records;
    * a session scoped to a different tenant cannot read it;
    * the recording tenant's own scoped session can;
    * a row that records no tenant opens no scope, and Postgres refuses it
      (nothing invents or defaults a tenant: OMN-16804 AC3, OMN-16831 AC2).
    """
    # The no-tenant refusal is asserted on a FRESH connection, where the GUC has
    # never been set and reads NULL, so the policy refuses with
    # InsufficientPrivilege. On a connection that already ran a scoped write,
    # Postgres reads a custom GUC back as '' rather than NULL, and the policy's
    # ``::uuid`` cast raises InvalidTextRepresentation instead: still a refusal,
    # but a different error class, so it is not the shape pinned here.
    no_tenant = _tenant_adapter(domain_dsns["tenant"], None)
    try:
        with pytest.raises(psycopg2.errors.InsufficientPrivilege):
            no_tenant.upsert(
                TENANT_TABLE,
                "correlation_id",
                {"correlation_id": uuid4(), "task_type": "no-tenant"},
            )
    finally:
        no_tenant.close()
    assert _tenant_rows(domain_dsns["owner"]) == []

    correlation_id = uuid4()
    unbound = _tenant_adapter(domain_dsns["tenant"], None)
    try:
        assert unbound.upsert(
            TENANT_TABLE,
            "correlation_id",
            {
                "correlation_id": correlation_id,
                "task_type": "proof",
                "tenant_id": TENANT_A,
            },
        )
    finally:
        unbound.close()
    assert _tenant_rows(domain_dsns["owner"]) == [(correlation_id, TENANT_A)]

    tenant_a = _tenant_adapter(domain_dsns["tenant"], TENANT_A)
    tenant_b = _tenant_adapter(domain_dsns["tenant"], TENANT_B)
    try:
        assert tenant_b.query(TENANT_TABLE, {"correlation_id": correlation_id}) == []
        found = tenant_a.query(TENANT_TABLE, {"correlation_id": correlation_id})
        assert [row["correlation_id"] for row in found] == [correlation_id]
    finally:
        tenant_a.close()
        tenant_b.close()


def test_tenant_b_cannot_read_tenant_a(domain_dsns: dict[str, str]) -> None:
    correlation_id = uuid4()
    tenant_a = _tenant_adapter(domain_dsns["tenant"], TENANT_A)
    tenant_b = _tenant_adapter(domain_dsns["tenant"], TENANT_B)
    try:
        tenant_a.upsert(
            TENANT_TABLE,
            "correlation_id",
            {"correlation_id": correlation_id, "task_type": "proof"},
        )
        assert tenant_b.query(TENANT_TABLE, {"correlation_id": correlation_id}) == []
    finally:
        tenant_a.close()
        tenant_b.close()


def test_reused_tenant_connection_does_not_leak_context(
    domain_dsns: dict[str, str],
) -> None:
    shared_connection = psycopg2.connect(domain_dsns["tenant"])
    try:
        with patch("psycopg2.connect", return_value=shared_connection):
            for tenant_id in (TENANT_A, TENANT_B):
                _tenant_adapter(domain_dsns["tenant"], tenant_id).upsert(
                    TENANT_TABLE,
                    "correlation_id",
                    {"correlation_id": uuid4(), "task_type": "pooled"},
                )
        with shared_connection.cursor() as cursor:
            cursor.execute("SELECT current_setting('app.tenant_id', true)")
            assert cursor.fetchone()[0] in (None, "")
    finally:
        shared_connection.close()
    assert [row[1] for row in _tenant_rows(domain_dsns["owner"])] == [
        TENANT_A,
        TENANT_B,
    ]


def test_real_rls_with_check_rejects_tenant_b_insert_and_update_under_a(
    domain_dsns: dict[str, str],
) -> None:
    conn = psycopg2.connect(domain_dsns["tenant"])
    try:
        with conn.cursor() as cursor:
            with pytest.raises(psycopg2.errors.InsufficientPrivilege):
                cursor.execute(
                    f"INSERT INTO {TENANT_SCHEMA}.{TENANT_TABLE} VALUES (%s, %s, %s)",  # noqa: S608 - module constant
                    (uuid4(), "unset-context", TENANT_A),
                )
        conn.rollback()

        with conn.cursor() as cursor:
            cursor.execute("SET LOCAL app.tenant_id = %s", (str(TENANT_A),))
            with pytest.raises(psycopg2.errors.InsufficientPrivilege):
                cursor.execute(
                    f"INSERT INTO {TENANT_SCHEMA}.{TENANT_TABLE} VALUES (%s, %s, %s)",  # noqa: S608 - module constant
                    (uuid4(), "wrong-insert", TENANT_B),
                )
        conn.rollback()

        correlation_id = uuid4()
        with conn.cursor() as cursor:
            cursor.execute("SET LOCAL app.tenant_id = %s", (str(TENANT_A),))
            cursor.execute(
                f"INSERT INTO {TENANT_SCHEMA}.{TENANT_TABLE} VALUES (%s, %s, %s)",  # noqa: S608 - module constant
                (correlation_id, "valid-a", TENANT_A),
            )
        conn.commit()
        with conn.cursor() as cursor:
            cursor.execute("SET LOCAL app.tenant_id = %s", (str(TENANT_A),))
            with pytest.raises(psycopg2.errors.InsufficientPrivilege):
                cursor.execute(
                    f"UPDATE {TENANT_SCHEMA}.{TENANT_TABLE} SET tenant_id = %s "  # noqa: S608 - module constant
                    "WHERE correlation_id = %s",
                    (TENANT_B, correlation_id),
                )
        conn.rollback()
    finally:
        conn.close()


def test_rollback_clears_guc_before_reusing_connection_for_tenant_b(
    domain_dsns: dict[str, str],
) -> None:
    shared_connection = psycopg2.connect(domain_dsns["tenant"])
    try:
        with patch("psycopg2.connect", return_value=shared_connection):
            tenant_a = _tenant_adapter(domain_dsns["tenant"], TENANT_A)
            with pytest.raises(psycopg2.errors.NotNullViolation):
                tenant_a.upsert(
                    TENANT_TABLE,
                    "correlation_id",
                    {"correlation_id": uuid4()},
                )
            with shared_connection.cursor() as cursor:
                cursor.execute("SELECT current_setting('app.tenant_id', true)")
                assert cursor.fetchone()[0] in (None, "")

            tenant_b = _tenant_adapter(domain_dsns["tenant"], TENANT_B)
            tenant_b.upsert(
                TENANT_TABLE,
                "correlation_id",
                {"correlation_id": uuid4(), "task_type": "after-rollback"},
            )
            with shared_connection.cursor() as cursor:
                cursor.execute("SELECT current_setting('app.tenant_id', true)")
                assert cursor.fetchone()[0] in (None, "")
    finally:
        shared_connection.close()

    assert _tenant_rows(domain_dsns["owner"])[0][1] == TENANT_B


def test_internal_write_read_has_no_tenant_guc(domain_dsns: dict[str, str]) -> None:
    correlation_id = uuid4()
    adapter = _internal_adapter(domain_dsns["internal"])
    try:
        adapter.upsert(
            INTERNAL_TABLE,
            "correlation_id",
            {
                "correlation_id": correlation_id,
                "source_tenant_id": TENANT_A,
                "status": "complete",
            },
        )
        assert (
            adapter.query(INTERNAL_TABLE, {"correlation_id": correlation_id})[0][
                "status"
            ]
            == "complete"
        )
        conn = next(iter(adapter._connections.values()))
        with conn.cursor() as cursor:
            cursor.execute("SELECT current_setting('app.tenant_id', true)")
            assert cursor.fetchone()[0] in (None, "")
    finally:
        adapter.close()


def test_catalog_reader_can_read_and_has_no_writer_operation(
    domain_dsns: dict[str, str],
) -> None:
    owner = psycopg2.connect(domain_dsns["owner"])
    owner.autocommit = True
    with owner.cursor() as cursor:
        cursor.execute(
            f"INSERT INTO {CATALOG_SCHEMA}.{CATALOG_TABLE} VALUES (%s, %s)",  # noqa: S608 - module constant
            ("beta", "Beta"),
        )
    owner.close()
    adapter = _catalog_reader_adapter(domain_dsns["catalog"])
    try:
        assert adapter.query(CATALOG_TABLE)[0]["tier_id"] == "beta"
        with pytest.raises(PermissionError, match="write refused"):
            adapter.upsert(
                CATALOG_TABLE,
                "tier_id",
                {"tier_id": "pro", "display_name": "Pro"},
            )
    finally:
        adapter.close()


@pytest.mark.parametrize(
    ("dsn_key", "sql"),
    [
        ("tenant", f"SELECT * FROM {INTERNAL_SCHEMA}.{INTERNAL_TABLE}"),  # noqa: S608 - module constant
        ("internal", f"SELECT * FROM {TENANT_SCHEMA}.{TENANT_TABLE}"),  # noqa: S608 - module constant
        ("catalog", f"INSERT INTO {CATALOG_SCHEMA}.{CATALOG_TABLE} VALUES ('x', 'X')"),  # noqa: S608 - module constant
    ],
)
def test_cross_domain_roles_are_denied(
    domain_dsns: dict[str, str], dsn_key: str, sql: str
) -> None:
    conn = psycopg2.connect(domain_dsns[dsn_key])
    conn.autocommit = True
    try:
        with pytest.raises(psycopg2.errors.InsufficientPrivilege):
            with conn.cursor() as cursor:
                cursor.execute(sql)
    finally:
        conn.close()


def test_miswired_dsn_fails_identity_attestation(domain_dsns: dict[str, str]) -> None:
    target = projection_database_target(TENANT_TABLE, schema="public")
    authority, event = verified_tenant_dispatch(TENANT_A)
    adapter = _build_projection_db_adapter(
        projection_database_urls(target, domain_dsns["internal"]),
        target,
        authority,
        event,
    )
    with pytest.raises(PermissionError, match="expected"):
        adapter.upsert(
            TENANT_TABLE,
            "correlation_id",
            {"correlation_id": uuid4(), "task_type": "proof"},
        )


def test_environment_tenant_is_not_authority(
    domain_dsns: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``ONEX_TENANT_ID`` never becomes the isolation context (OMN-16831).

    A process-wide deployment env var is not a request-scoped entitlement, and
    the option (a) scope (OMN-17422) must come from the row, never from the
    environment. The environment and the row therefore name DIFFERENT tenants
    here: were the environment promoted into the GUC, the row's own
    ``WITH CHECK`` would refuse it. Instead it lands as the tenant it records,
    and the environment's tenant cannot read it.
    """
    monkeypatch.setenv("ONEX_TENANT_ID", str(TENANT_A))
    monkeypatch.setenv("ENFORCE_TENANT_ISOLATION", "false")
    correlation_id = uuid4()
    adapter = _tenant_adapter(domain_dsns["tenant"], None)
    try:
        assert adapter.upsert(
            TENANT_TABLE,
            "correlation_id",
            {
                "correlation_id": correlation_id,
                "task_type": "proof",
                "tenant_id": TENANT_B,
            },
        )
    finally:
        adapter.close()
    assert _tenant_rows(domain_dsns["owner"]) == [(correlation_id, TENANT_B)]

    tenant_a = _tenant_adapter(domain_dsns["tenant"], TENANT_A)
    try:
        assert tenant_a.query(TENANT_TABLE, {"correlation_id": correlation_id}) == []
    finally:
        tenant_a.close()
