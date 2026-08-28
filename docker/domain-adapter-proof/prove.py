# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Rebuilt-image PostgreSQL 16 domain-adapter proof for OMN-15421."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from unittest.mock import patch
from uuid import UUID, uuid4

import psycopg2
import psycopg2.extras

from omnibase_core.crypto.crypto_ed25519_signer import generate_keypair
from omnibase_core.models.contracts.subcontracts.model_db_table_declaration import (
    ModelDbTableDeclaration,
)
from omnibase_core.models.core.model_deployment_topology import ModelDeploymentTopology
from omnibase_core.models.envelope.model_message_envelope import ModelMessageEnvelope
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.errors.error_projection import ProjectionTenantContextError
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    ProjectionDatabaseTarget,
    _build_projection_db_adapter,
    _resolve_projection_database_target,
)
from omnibase_infra.runtime.projection_tenant_authority import (
    VerifiedProjectionTenantAuthority,
    verify_signed_projection_tenant_authority,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--admin-dsn", required=True)
    parser.add_argument("--tenant-dsn", required=True)
    parser.add_argument("--internal-dsn", required=True)
    parser.add_argument("--catalog-dsn", required=True)
    return parser.parse_args()


_ARGS = _parse_args()
ADMIN_DSN = _ARGS.admin_dsn
TENANT_DSN = _ARGS.tenant_dsn
INTERNAL_DSN = _ARGS.internal_dsn
CATALOG_DSN = _ARGS.catalog_dsn
DATABASE = "omnidash_analytics"
TENANT_ROLE = "tenant_projection_writer"
INTERNAL_ROLE = "omninode_runtime"
CATALOG_ROLE = "app_dashboard"
ROLE_PASSWORD = "domain-adapter-proof-only"  # pragma: allowlist secret
TENANT_A = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
TENANT_B = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
TENANT_TABLE = "future_tenant_projection"
INTERNAL_TABLE = "future_internal_projection"
CATALOG_TABLE = "plan_tiers"

psycopg2.extras.register_uuid()

TOPOLOGY = ModelDeploymentTopology.from_yaml(Path(__file__).with_name("topology.yaml"))


class _KeyProvider:
    def __init__(self, runtime_id: str, public_key: bytes) -> None:
        self._keys = {runtime_id: public_key}

    def get_public_key(self, runtime_id: str) -> bytes | None:
        return self._keys.get(runtime_id)

    def register_key(self, runtime_id: str, public_key: bytes) -> None:
        self._keys[runtime_id] = public_key

    def has_key(self, runtime_id: str) -> bool:
        return runtime_id in self._keys

    def list_runtime_ids(self) -> list[str]:
        return sorted(self._keys)


@dataclass(frozen=True)
class _TenantBindingResolver:
    runtime_id: str
    realm: str
    bus_id: str
    tenant_id: UUID

    def resolve_tenant_id(
        self,
        *,
        runtime_id: str,
        realm: str,
        bus_id: str,
    ) -> UUID | None:
        if (runtime_id, realm, bus_id) != (
            self.runtime_id,
            self.realm,
            self.bus_id,
        ):
            return None
        return self.tenant_id


@dataclass(frozen=True)
class _SignedFixture:
    envelope: ModelMessageEnvelope[ModelEventEnvelope[dict[str, object]]]
    key_provider: _KeyProvider
    resolver: _TenantBindingResolver

    def verify(self) -> VerifiedProjectionTenantAuthority:
        return verify_signed_projection_tenant_authority(
            self.envelope,
            self.key_provider,
            self.resolver,
        )


def _signed_fixture(
    tenant_value: str,
    *,
    bound_tenant: UUID,
) -> _SignedFixture:
    runtime_id = "tenant-gateway-proof"
    realm = "docker-proof"
    bus_id = "domain-adapter-proof"
    event = ModelEventEnvelope[dict[str, object]](
        payload={"proof": True},
        correlation_id=uuid4(),
    )
    keypair = generate_keypair()
    envelope = ModelMessageEnvelope[
        ModelEventEnvelope[dict[str, object]]
    ].create_signed(
        realm=realm,
        runtime_id=runtime_id,
        bus_id=bus_id,
        tenant_id=tenant_value,
        payload=event,
        trace_id=event.correlation_id,
        private_key=keypair.private_key_bytes,
        emitted_at=datetime.now(UTC),
    )
    return _SignedFixture(
        envelope=envelope,
        key_provider=_KeyProvider(runtime_id, keypair.public_key_bytes),
        resolver=_TenantBindingResolver(runtime_id, realm, bus_id, bound_tenant),
    )


def _verified_dispatch(
    tenant_id: UUID,
) -> tuple[VerifiedProjectionTenantAuthority, ModelEventEnvelope[dict[str, object]]]:
    fixture = _signed_fixture(str(tenant_id), bound_tenant=tenant_id)
    return fixture.verify(), fixture.envelope.payload


def _target(
    table: str,
    schema: str,
    *,
    access: Literal["read", "write", "read_write"] = "read_write",
    catalog_read_binding: str | None = None,
    catalog_write_binding: str | None = None,
) -> ProjectionDatabaseTarget:
    declaration = ModelDbTableDeclaration(
        name=table,
        database_ref="application",
        schema=schema,
        migration=f"proof/{schema}/{table}.sql",
        access=access,
        role=f"{table}_proof",
    )
    return _resolve_projection_database_target(
        (declaration,),
        TOPOLOGY,
        catalog_read_binding=catalog_read_binding,
        catalog_write_binding=catalog_write_binding,
    )


def _adapter(
    target: ProjectionDatabaseTarget,
    authority: VerifiedProjectionTenantAuthority | None = None,
    event: ModelEventEnvelope[dict[str, object]] | None = None,
) -> object:
    urls = {
        "tenant_projection": TENANT_DSN,
        "omninode_runtime_service": INTERNAL_DSN,
        "app_dashboard": CATALOG_DSN,
    }
    return _build_projection_db_adapter(
        {binding.binding_ref: urls[binding.binding_ref] for binding in target.bindings},
        target,
        authority,
        event,
    )


def _raises(
    error_type: type[BaseException],
    action: Callable[[], object],
) -> BaseException:
    try:
        action()
    except error_type as exc:
        return exc
    raise AssertionError(f"Expected {error_type.__name__}")


def _admin_rows(
    sql: str,
    params: tuple[object, ...] | None = None,
) -> list[tuple[object, ...]]:
    conn = psycopg2.connect(ADMIN_DSN)
    conn.autocommit = True
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql, params)
            return list(cursor.fetchall())
    finally:
        conn.close()


def _initialize_database() -> None:
    conn = psycopg2.connect(ADMIN_DSN)
    conn.autocommit = True
    with conn.cursor() as cursor:
        for role in (TENANT_ROLE, INTERNAL_ROLE, CATALOG_ROLE):
            cursor.execute(
                f"CREATE ROLE {role} LOGIN PASSWORD %s NOSUPERUSER NOBYPASSRLS",
                (ROLE_PASSWORD,),
            )
            cursor.execute(f"GRANT CONNECT ON DATABASE {DATABASE} TO {role}")
        cursor.execute("CREATE SCHEMA tenant")
        cursor.execute("CREATE SCHEMA omninode_internal")
        cursor.execute("CREATE SCHEMA platform_catalog")
        cursor.execute(
            f"""
            CREATE TABLE tenant.{TENANT_TABLE} (
                correlation_id UUID PRIMARY KEY,
                task_type TEXT NOT NULL,
                tenant_id UUID NOT NULL
            );
            ALTER TABLE tenant.{TENANT_TABLE} ENABLE ROW LEVEL SECURITY;
            ALTER TABLE tenant.{TENANT_TABLE} FORCE ROW LEVEL SECURITY;
            CREATE POLICY tenant_isolation ON tenant.{TENANT_TABLE}
              FOR ALL
              USING (tenant_id = current_setting('app.tenant_id', true)::uuid)
              WITH CHECK (tenant_id = current_setting('app.tenant_id', true)::uuid);
            CREATE TABLE omninode_internal.{INTERNAL_TABLE} (
                correlation_id UUID PRIMARY KEY,
                source_tenant_id UUID NULL,
                status TEXT NOT NULL
            );
            CREATE TABLE platform_catalog.{CATALOG_TABLE} (
                tier_id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL
            );
            """
        )
        cursor.execute(f"GRANT USAGE ON SCHEMA tenant TO {TENANT_ROLE}")
        cursor.execute(
            f"GRANT SELECT, INSERT, UPDATE ON tenant.{TENANT_TABLE} TO {TENANT_ROLE}"
        )
        cursor.execute(f"GRANT USAGE ON SCHEMA omninode_internal TO {INTERNAL_ROLE}")
        cursor.execute(
            "GRANT SELECT, INSERT, UPDATE ON "
            f"omninode_internal.{INTERNAL_TABLE} TO {INTERNAL_ROLE}"
        )
        cursor.execute(f"GRANT USAGE ON SCHEMA platform_catalog TO {CATALOG_ROLE}")
        cursor.execute(
            f"GRANT SELECT ON platform_catalog.{CATALOG_TABLE} TO {CATALOG_ROLE}"
        )
        cursor.execute(
            "INSERT INTO platform_catalog.plan_tiers VALUES ('beta', 'Beta')"
        )
    conn.close()


def _assert_database_identities() -> None:
    for dsn, principal in (
        (TENANT_DSN, TENANT_ROLE),
        (INTERNAL_DSN, INTERNAL_ROLE),
        (CATALOG_DSN, CATALOG_ROLE),
    ):
        conn = psycopg2.connect(dsn)
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT current_user, current_database()")
                assert cursor.fetchone() == (principal, DATABASE)
        finally:
            conn.close()


def _prove_real_rls_with_check() -> None:
    """Prove the database itself rejects tenant-B rows under tenant-A GUC."""
    conn = psycopg2.connect(TENANT_DSN)
    try:
        with conn.cursor() as cursor:
            _raises(
                psycopg2.errors.InsufficientPrivilege,
                lambda: cursor.execute(
                    "INSERT INTO tenant.future_tenant_projection VALUES (%s, %s, %s)",
                    (uuid4(), "unset-context", TENANT_A),
                ),
            )
        conn.rollback()

        with conn.cursor() as cursor:
            cursor.execute("SET LOCAL app.tenant_id = %s", (str(TENANT_A),))
            _raises(
                psycopg2.errors.InsufficientPrivilege,
                lambda: cursor.execute(
                    "INSERT INTO tenant.future_tenant_projection VALUES (%s, %s, %s)",
                    (uuid4(), "wrong-insert", TENANT_B),
                ),
            )
        conn.rollback()

        correlation_id = uuid4()
        with conn.cursor() as cursor:
            cursor.execute("SET LOCAL app.tenant_id = %s", (str(TENANT_A),))
            cursor.execute(
                "INSERT INTO tenant.future_tenant_projection VALUES (%s, %s, %s)",
                (correlation_id, "valid-a", TENANT_A),
            )
        conn.commit()
        with conn.cursor() as cursor:
            cursor.execute("SET LOCAL app.tenant_id = %s", (str(TENANT_A),))
            _raises(
                psycopg2.errors.InsufficientPrivilege,
                lambda: cursor.execute(
                    "UPDATE tenant.future_tenant_projection SET tenant_id = %s "
                    "WHERE correlation_id = %s",
                    (TENANT_B, correlation_id),
                ),
            )
        conn.rollback()
    finally:
        conn.close()


def _prove_no_authority_records_the_claim(
    tenant_target: ProjectionDatabaseTarget,
) -> None:
    """Prove the no-authority path stopped destroying the tenant dimension.

    OMN-16831 (operator ruling 2026-08-28, option D). Before the ruling this
    seam raised :class:`ProjectionTenantContextError` whenever no verified
    capability was bound -- which is every real dispatch on every lane, because
    ``bind_projection_tenant_authority`` has zero non-test call sites. Every
    event on all 15 TENANT-classified relations was therefore quarantined
    before a single statement was issued, and the immutable log kept a DLQ
    record instead of a tenant-attributed fact.

    Two things are proven here, and they are the whole ruling:

    1. **The runtime no longer refuses.** The absence of an authorization
       artifact is not an attribution failure, so the write is not stopped at
       the runtime precondition -- the statement is actually issued. Falsified
       by any ``ProjectionTenantContextError`` escaping this call.
    2. **Isolation did not weaken -- it moved to its real enforcement point.**
       The write is still refused, by the DATABASE's own RLS policy
       (``InsufficientPrivilege``), exactly as :func:`_prove_real_rls_with_check`
       proves for a raw unattributed ``INSERT``. Nothing is invented, defaulted
       or substituted on the way there (OMN-16804 AC3).

    The discriminator between the pre- and post-ruling worlds is therefore the
    *class of the error*: the runtime's ``ProjectionTenantContextError`` before,
    the database's ``InsufficientPrivilege`` after. Asserting the database error
    is what makes this control falsifiable in both directions -- it fails if the
    refusal moves back into the runtime, and it fails if isolation is dropped.
    """
    missing = _adapter(tenant_target)
    try:
        error = _raises(
            psycopg2.errors.InsufficientPrivilege,
            lambda: missing.upsert(
                TENANT_TABLE,
                "correlation_id",
                {
                    "correlation_id": uuid4(),
                    "task_type": "no-authority-claim",
                    "tenant_id": TENANT_A,
                },
            ),
        )
        assert not isinstance(error, ProjectionTenantContextError)
    finally:
        missing.close()


def _prove_rollback_clears_reused_connection(
    tenant_target: ProjectionDatabaseTarget,
) -> None:
    """Force SQL failure after SET LOCAL, then reuse the connection for B."""
    shared = psycopg2.connect(TENANT_DSN)
    authority_a, event_a = _verified_dispatch(TENANT_A)
    authority_b, event_b = _verified_dispatch(TENANT_B)
    try:
        with patch("psycopg2.connect", return_value=shared):
            adapter_a = _adapter(tenant_target, authority_a, event_a)
            _raises(
                psycopg2.errors.NotNullViolation,
                lambda: adapter_a.upsert(
                    TENANT_TABLE,
                    "correlation_id",
                    {"correlation_id": uuid4()},
                ),
            )
            with shared.cursor() as cursor:
                cursor.execute("SELECT current_setting('app.tenant_id', true)")
                assert cursor.fetchone()[0] in (None, "")
            adapter_b = _adapter(tenant_target, authority_b, event_b)
            assert adapter_b.upsert(
                TENANT_TABLE,
                "correlation_id",
                {"correlation_id": uuid4(), "task_type": "after-rollback"},
            )
            with shared.cursor() as cursor:
                cursor.execute("SELECT current_setting('app.tenant_id', true)")
                assert cursor.fetchone()[0] in (None, "")
    finally:
        shared.close()


def _prove_signature_failures() -> None:
    malformed = _signed_fixture("not-a-uuid", bound_tenant=TENANT_A)
    sentinel = _signed_fixture(str(UUID(int=0)), bound_tenant=UUID(int=0))
    _raises(ProjectionTenantContextError, malformed.verify)
    _raises(ProjectionTenantContextError, sentinel.verify)

    fixture = _signed_fixture(str(TENANT_A), bound_tenant=TENANT_A)
    tampered = fixture.envelope.model_copy(update={"tenant_id": str(TENANT_B)})
    _raises(
        ProjectionTenantContextError,
        lambda: verify_signed_projection_tenant_authority(
            tampered,
            fixture.key_provider,
            _TenantBindingResolver(
                fixture.envelope.runtime_id,
                fixture.envelope.realm,
                fixture.envelope.bus_id,
                TENANT_B,
            ),
        ),
    )
    wrong_binding = _TenantBindingResolver(
        fixture.envelope.runtime_id,
        fixture.envelope.realm,
        fixture.envelope.bus_id,
        TENANT_B,
    )
    _raises(
        ProjectionTenantContextError,
        lambda: verify_signed_projection_tenant_authority(
            fixture.envelope,
            fixture.key_provider,
            wrong_binding,
        ),
    )


def main() -> None:
    _initialize_database()
    _assert_database_identities()
    tenant_target = _target(TENANT_TABLE, "tenant")
    internal_target = _target(INTERNAL_TABLE, "omninode_internal")
    catalog_target = _target(
        CATALOG_TABLE,
        "platform_catalog",
        access="read",
        catalog_read_binding="app_dashboard",
    )
    assert [domain.value for domain in tenant_target.domains] == ["TENANT"]
    assert [domain.value for domain in internal_target.domains] == ["OMNINODE_INTERNAL"]
    assert [domain.value for domain in catalog_target.domains] == ["PLATFORM_CATALOG"]

    _prove_signature_failures()
    _prove_real_rls_with_check()

    authority_a, event_a = _verified_dispatch(TENANT_A)
    tenant_a = _adapter(tenant_target, authority_a, event_a)
    correlation_a = uuid4()
    try:
        assert tenant_a.upsert(
            TENANT_TABLE,
            "correlation_id",
            {"correlation_id": correlation_a, "task_type": "signed-a"},
        )
        found = tenant_a.query(TENANT_TABLE, {"correlation_id": correlation_a})
        assert found[0]["correlation_id"] == correlation_a
        assert isinstance(found[0]["correlation_id"], UUID)
        assert found[0]["tenant_id"] == TENANT_A
    finally:
        tenant_a.close()

    authority_b, event_b = _verified_dispatch(TENANT_B)
    tenant_b = _adapter(tenant_target, authority_b, event_b)
    try:
        assert tenant_b.query(TENANT_TABLE, {"correlation_id": correlation_a}) == []
    finally:
        tenant_b.close()

    # OMN-16831 (operator ruling 2026-08-28, option D): a MISMATCHED authority is
    # still refused before a single connection is opened -- that half is unchanged
    # and stays at full strength, which `connect.assert_not_called()` below proves.
    # The no-authority case is no longer a refusal and is proven separately,
    # against the real database, by _prove_no_authority_records_the_claim().
    with patch("psycopg2.connect") as connect:
        mismatch = _adapter(tenant_target, authority_a, event_a)
        _raises(
            ProjectionTenantContextError,
            lambda: mismatch.upsert(
                TENANT_TABLE,
                "correlation_id",
                {
                    "correlation_id": uuid4(),
                    "task_type": "wrong-row-tenant",
                    "tenant_id": TENANT_B,
                },
            ),
        )
    connect.assert_not_called()

    _prove_no_authority_records_the_claim(tenant_target)

    _prove_rollback_clears_reused_connection(tenant_target)

    internal = _adapter(internal_target)
    internal_id = uuid4()
    try:
        assert internal.upsert(
            INTERNAL_TABLE,
            "correlation_id",
            {
                "correlation_id": internal_id,
                "source_tenant_id": TENANT_A,
                "status": "complete",
            },
        )
        assert (
            internal.query(INTERNAL_TABLE, {"correlation_id": internal_id})[0]["status"]
            == "complete"
        )
        internal_connection = next(iter(internal._connections.values()))
        with internal_connection.cursor() as cursor:
            cursor.execute("SELECT current_setting('app.tenant_id', true)")
            assert cursor.fetchone()[0] in (None, "")
        _raises(
            ValueError,
            lambda: internal.upsert(
                INTERNAL_TABLE,
                "correlation_id",
                {
                    "correlation_id": uuid4(),
                    "tenant_id": TENANT_A,
                    "status": "invalid",
                },
            ),
        )
    finally:
        internal.close()

    catalog = _adapter(catalog_target)
    try:
        assert catalog.query(CATALOG_TABLE)[0]["tier_id"] == "beta"
        _raises(
            PermissionError,
            lambda: catalog.upsert(
                CATALOG_TABLE,
                "tier_id",
                {"tier_id": "pro", "display_name": "Pro"},
            ),
        )
    finally:
        catalog.close()
    _raises(
        ValueError,
        lambda: _target(CATALOG_TABLE, "platform_catalog", access="write"),
    )

    for dsn, sql in (
        (TENANT_DSN, f"SELECT * FROM omninode_internal.{INTERNAL_TABLE}"),
        (INTERNAL_DSN, "SELECT * FROM tenant.future_tenant_projection"),
        (
            CATALOG_DSN,
            "INSERT INTO platform_catalog.plan_tiers VALUES ('x', 'X')",
        ),
    ):
        conn = psycopg2.connect(dsn)
        conn.autocommit = True
        try:
            _raises(
                psycopg2.errors.InsufficientPrivilege,
                lambda conn=conn, sql=sql: conn.cursor().execute(sql),
            )
        finally:
            conn.close()

    miswired = _build_projection_db_adapter(
        {"tenant_projection": INTERNAL_DSN},
        tenant_target,
        authority_a,
        event_a,
    )
    _raises(
        PermissionError,
        lambda: miswired.upsert(
            TENANT_TABLE,
            "correlation_id",
            {"correlation_id": uuid4(), "task_type": "miswired"},
        ),
    )

    rows = _admin_rows(
        "SELECT tenant_id FROM tenant.future_tenant_projection ORDER BY tenant_id"
    )
    assert TENANT_A in {row[0] for row in rows}
    assert TENANT_B in {row[0] for row in rows}
    sys.stdout.write("OMN-15421 PostgreSQL 16 rebuilt-container proof: PASS\n")


if __name__ == "__main__":
    main()
