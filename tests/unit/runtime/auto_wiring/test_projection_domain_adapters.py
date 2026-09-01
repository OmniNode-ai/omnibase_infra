# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Domain-bound PostgreSQL adapter tests for OMN-15421."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest

from omnibase_core.enums.enum_database_privilege import EnumDatabasePrivilege
from omnibase_core.models.contracts.subcontracts.model_db_table_declaration import (
    ModelDbTableDeclaration,
)
from omnibase_core.models.envelope.model_message_envelope import ModelMessageEnvelope
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.models.security.model_security_context import ModelSecurityContext
from omnibase_infra.errors.error_projection import ProjectionTenantContextError
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _build_projection_db_adapter,
    _resolve_projection_database_target,
)
from omnibase_infra.runtime.projection_tenant_authority import (
    VerifiedProjectionTenantAuthority,
    assert_projection_tenant_authority_matches_event,
    verify_signed_projection_tenant_authority,
)
from tests.helpers.application_db_topology import (
    application_topology,
    projection_database_target,
    projection_database_urls,
)
from tests.helpers.projection_tenant_authority import (
    StaticTenantBindingResolver,
    signed_tenant_authority_fixture,
    verified_tenant_dispatch,
)

pytestmark = pytest.mark.unit


def _connection(
    principal: str,
    database: str = "omnidash_analytics",
) -> tuple[MagicMock, MagicMock]:
    cursor = MagicMock()
    cursor.fetchone.return_value = (principal, database)
    cursor.fetchall.return_value = []
    cursor_context = MagicMock()
    cursor_context.__enter__.return_value = cursor
    conn = MagicMock()
    conn.closed = False
    conn.autocommit = True
    conn.cursor.return_value = cursor_context
    return conn, cursor


def _adapter(
    target: object,
    *,
    authority: VerifiedProjectionTenantAuthority | None = None,
    tenant_event: object | None = None,
    default_url: str = "postgresql://fixture",
    **binding_urls: str,
) -> object:
    return _build_projection_db_adapter(
        projection_database_urls(target, default_url, **binding_urls),  # type: ignore[arg-type]
        target,  # type: ignore[arg-type]
        authority,
        tenant_event,
    )


def _verified_adapter(target: object, tenant_id: UUID, **binding_urls: str) -> object:
    authority, event = verified_tenant_dispatch(tenant_id)
    return _adapter(
        target,
        authority=authority,
        tenant_event=event,
        **binding_urls,
    )


def _target(
    name: str,
    schema: str,
    access: Literal["read", "write", "read_write"],
) -> object:
    return projection_database_target(
        name,
        schema=schema,
        access=access,
    )


def test_signed_envelope_mints_uuid_tenant_authority() -> None:
    tenant_id = uuid4()
    fixture = signed_tenant_authority_fixture(tenant_id)

    authority = fixture.verify()

    assert authority.tenant_id == tenant_id
    assert isinstance(authority.tenant_id, UUID)
    assert authority.trace_id == fixture.envelope.trace_id
    assert isinstance(authority.trace_id, UUID)
    assert authority.runtime_id == fixture.envelope.runtime_id


def test_signed_envelope_json_round_trip_mints_and_binds_exact_event() -> None:
    fixture = signed_tenant_authority_fixture(uuid4(), payload={"value": "proof"})
    parsed = ModelMessageEnvelope[object].model_validate_json(
        fixture.envelope.model_dump_json()
    )

    authority = verify_signed_projection_tenant_authority(
        parsed,
        fixture.key_provider,
        fixture.binding_resolver,
    )
    typed_event = ModelEventEnvelope[object].model_validate(parsed.payload)

    assert_projection_tenant_authority_matches_event(authority, typed_event)


def test_verified_authority_cannot_authorize_a_different_dispatch_event() -> None:
    tenant_id = uuid4()
    authority, _event = verified_tenant_dispatch(tenant_id)
    different_event = ModelEventEnvelope[dict[str, object]](
        payload={"value": "different"},
        correlation_id=authority.trace_id,
    )
    target = projection_database_target("delegation_events", schema="tenant")

    with patch("psycopg2.connect") as connect:
        adapter = _adapter(
            target,
            authority=authority,
            tenant_event=different_event,
        )
        with pytest.raises(ProjectionTenantContextError, match="dispatched envelope"):
            adapter.upsert("delegation_events", "event_id", {"event_id": uuid4()})

    connect.assert_not_called()


def test_signed_tenant_or_payload_tampering_fails_verification() -> None:
    fixture = signed_tenant_authority_fixture(uuid4(), payload={"value": "original"})
    other_tenant = uuid4()
    resolver = StaticTenantBindingResolver(
        runtime_id=fixture.envelope.runtime_id,
        realm=fixture.envelope.realm,
        bus_id=fixture.envelope.bus_id,
        tenant_id=other_tenant,
    )

    tenant_tampered = fixture.envelope.model_copy(
        update={"tenant_id": str(other_tenant)}
    )
    payload_tampered = fixture.envelope.model_copy(
        update={"payload": {"value": "tampered"}}
    )

    with pytest.raises(ProjectionTenantContextError, match="verification failed"):
        verify_signed_projection_tenant_authority(
            tenant_tampered, fixture.key_provider, resolver
        )
    with pytest.raises(ProjectionTenantContextError, match="verification failed"):
        verify_signed_projection_tenant_authority(
            payload_tampered, fixture.key_provider, fixture.binding_resolver
        )


def test_red_control_mismatched_signer_binding() -> None:
    fixture = signed_tenant_authority_fixture(uuid4())
    wrong_binding = StaticTenantBindingResolver(
        runtime_id=fixture.envelope.runtime_id,
        realm=fixture.envelope.realm,
        bus_id=fixture.envelope.bus_id,
        tenant_id=uuid4(),
    )

    with pytest.raises(ProjectionTenantContextError, match="signer binding"):
        verify_signed_projection_tenant_authority(
            fixture.envelope,
            fixture.key_provider,
            wrong_binding,
        )


def test_red_control_untrusted_tenant_selection() -> None:
    """A self-asserted envelope tenant is never promoted to an authority.

    OMN-16831 (ruling 2026-08-28, option D) changed what this control asserts,
    and deliberately so. It previously proved the write was REFUSED outright,
    which conflated two different things: recording who an event belongs to
    (attribution) and being entitled to act as them (authorization). Refusing
    the write did not make the platform safer -- it destroyed the dimension on
    every event, on every lane, because no authority is ever bound.

    OMN-16976 sharpens it a second time, and the distinction is the whole
    point. The previous revision asserted that an unverified claim "never
    reaches ``set_config('app.tenant_id', ...)``" -- but that conflated the two
    things a GUC scope can do. Opening a session as a tenant in order to be
    ENTITLED to act as them is a capability. Scoping a single statement to the
    identity that statement is itself persisting is the isolation MECHANISM
    reading the attribution it has already been handed; it grants nothing the
    row did not already declare, and the write remains confined to it.

    Confining the write to that scope is not optional on this platform: every
    canonical tenant policy is
    ``tenant_id = current_setting('app.tenant_id', true)`` under FORCE ROW
    LEVEL SECURITY, and `check_migration_rls_policy_atomicity.py` plus
    `application_database_domain_enforcement` together forbid the alternative
    (an extra permissive policy admitting the writer role "widens access").
    With the GUC unset the comparison is against NULL, so the pre-OMN-16976
    bare INSERT was denied by the database on every tenant relation -- the
    OMN-16831 fix removed the Python refusal and left a Postgres one.

    So the invariant that actually matters survives at full strength and is
    what is asserted here, in its sharpened form: an unsigned envelope's
    payload/security-context tenant can NEVER mint a
    ``VerifiedProjectionTenantAuthority``, is never verified against a dispatch
    event, and never authorizes anything. The only scope opened on its word is
    the one the row itself declares -- asserted below, positionally, against
    the row's own ``tenant_id``.
    """
    tenant_id = uuid4()
    envelope = ModelEventEnvelope[dict[str, object]](
        payload={"tenant_id": str(tenant_id)},
        security_context=ModelSecurityContext(
            user_id=uuid4(),
            auth_method="oidc",
            auth_timestamp=datetime.now(UTC),
            security_labels={"tenant_id": str(tenant_id)},
        ),
    )
    target = projection_database_target("delegation_events", schema="tenant")
    conn, cursor = _connection("tenant_projection_writer")

    with patch("psycopg2.connect", return_value=conn):
        adapter = _adapter(target, authority=None)
        adapter.upsert(
            "delegation_events",
            "event_id",
            {"event_id": uuid4(), "tenant_id": str(tenant_id)},
        )

    # No capability was minted from the claim: the adapter holds no authority,
    # so nothing was ever verified against a dispatch event.
    assert adapter._optional_tenant_context() is None  # type: ignore[attr-defined]

    # The one scope that IS opened is the row's own declared tenant, and it is
    # transaction-LOCAL. Nothing widened: same value, both statements.
    scope = next(
        call for call in cursor.execute.call_args_list if "set_config" in call.args[0]
    )
    assert scope.args[1] == ("app.tenant_id", str(tenant_id))
    insert = cursor.execute.call_args_list[-1]
    assert insert.args[1]["tenant_id"] == str(tenant_id)
    assert conn.autocommit is True
    assert (
        envelope.security_context is not None
    )  # proves the untrusted shape was present


def test_capability_constructor_is_sealed() -> None:
    with pytest.raises(TypeError, match="only be minted"):
        VerifiedProjectionTenantAuthority(
            tenant_id=uuid4(),
            trace_id=uuid4(),
            runtime_id="forged",
            realm="test",
            bus_id="test",
            emitted_at=datetime.now(UTC),
            event_envelope_id=uuid4(),
            event_payload_hash="0" * 64,
            _mint=object(),
        )


def test_red_control_nonlocal_tenant_guc() -> None:
    tenant_id = uuid4()
    target = projection_database_target("delegation_events", schema="tenant")
    conn, cursor = _connection("tenant_projection_writer")

    with patch("psycopg2.connect", return_value=conn):
        adapter = _verified_adapter(target, tenant_id)
        assert adapter.upsert(
            "delegation_events", "event_id", {"event_id": uuid4(), "value": "ok"}
        )

    calls = cursor.execute.call_args_list
    assert calls[0].args == ("SELECT current_user, current_database()",)
    assert calls[1].args == (
        "SELECT set_config(%s, %s, true)",
        ("app.tenant_id", str(tenant_id)),
    )
    # OMN-16239: qualified with the PHYSICAL schema. delegation_events is still
    # under the OMN-15359 bridge and lives in public; the pre-fix assertion here
    # named "tenant", a schema the analytics database does not even have.
    assert 'INSERT INTO "public"."delegation_events"' in calls[2].args[0]
    assert calls[2].args[1]["tenant_id"] == tenant_id
    assert isinstance(calls[2].args[1]["tenant_id"], UUID)
    conn.commit.assert_called_once_with()
    assert conn.autocommit is True


def test_red_control_leaked_tenant_guc() -> None:
    tenant_id = uuid4()
    target = projection_database_target("delegation_events", schema="tenant")
    conn, cursor = _connection("tenant_projection_writer")
    cursor.execute.side_effect = (None, None, RuntimeError("write failed"))

    with patch("psycopg2.connect", return_value=conn):
        adapter = _verified_adapter(target, tenant_id)
        with pytest.raises(RuntimeError, match="write failed"):
            adapter.upsert(
                "delegation_events", "event_id", {"event_id": uuid4(), "value": "red"}
            )

    conn.rollback.assert_called_once_with()
    conn.commit.assert_not_called()
    assert conn.autocommit is True


def test_equal_canonical_row_string_is_assertion_not_authority() -> None:
    tenant_id = uuid4()
    target = projection_database_target("delegation_events", schema="tenant")
    conn, cursor = _connection("tenant_projection_writer")

    with patch("psycopg2.connect", return_value=conn):
        adapter = _verified_adapter(target, tenant_id)
        adapter.upsert(
            "delegation_events",
            "event_id",
            {"event_id": uuid4(), "tenant_id": str(tenant_id)},
        )

    insert = cursor.execute.call_args_list[2]
    assert insert.args[1]["tenant_id"] == tenant_id
    assert isinstance(insert.args[1]["tenant_id"], UUID)


@pytest.mark.parametrize("supplied", [uuid4(), "not-a-uuid", "", 7])
def test_wrong_or_malformed_row_tenant_fails_before_connect(supplied: object) -> None:
    tenant_id = uuid4()
    target = projection_database_target("delegation_events", schema="tenant")

    with patch("psycopg2.connect") as connect:
        adapter = _verified_adapter(target, tenant_id)
        with pytest.raises(ProjectionTenantContextError):
            adapter.upsert(
                "delegation_events",
                "event_id",
                {"event_id": uuid4(), "tenant_id": supplied},
            )

    connect.assert_not_called()


def test_connection_identity_and_database_are_attested_before_dml() -> None:
    target = projection_database_target("generation_events", schema="omninode_internal")
    wrong_conn, cursor = _connection("tenant_projection_writer", "retired_database")

    with patch("psycopg2.connect", return_value=wrong_conn):
        adapter = _adapter(target)
        with pytest.raises(PermissionError, match="expected"):
            adapter.upsert("generation_events", "event_id", {"event_id": uuid4()})

    assert cursor.execute.call_args.args == ("SELECT current_user, current_database()",)
    wrong_conn.close.assert_called_once_with()


def test_mixed_target_uses_distinct_domain_bindings_and_connections() -> None:
    tables = (
        ModelDbTableDeclaration(
            name="delegation_events",
            database_ref="application",
            schema="tenant",
            migration="proof/tenant.sql",
            access="read_write",
            role="delegation_events",
        ),
        ModelDbTableDeclaration(
            name="generation_events",
            database_ref="application",
            schema="omninode_internal",
            migration="proof/internal.sql",
            access="read_write",
            role="generation_events",
        ),
    )
    target = _resolve_projection_database_target(tables, application_topology())
    tenant_conn, tenant_cursor = _connection("tenant_projection_writer")
    internal_conn, internal_cursor = _connection("omninode_runtime")
    connections = {
        "postgresql://tenant": tenant_conn,
        "postgresql://internal": internal_conn,
    }

    with patch("psycopg2.connect", side_effect=lambda dsn: connections[dsn]) as connect:
        adapter = _verified_adapter(
            target,
            uuid4(),
            tenant_projection="postgresql://tenant",
            omninode_runtime_service="postgresql://internal",
        )
        adapter.upsert("delegation_events", "event_id", {"event_id": uuid4()})
        adapter.upsert("generation_events", "event_id", {"event_id": uuid4()})

    assert {call.args[0] for call in connect.call_args_list} == {
        "postgresql://tenant",
        "postgresql://internal",
    }
    assert any(
        "set_config" in call.args[0] for call in tenant_cursor.execute.call_args_list
    )
    assert not any(
        "set_config" in call.args[0] for call in internal_cursor.execute.call_args_list
    )


def test_internal_source_tenant_is_provenance_not_conflict_authority() -> None:
    target = projection_database_target("generation_events", schema="omninode_internal")
    adapter = _adapter(target)

    with pytest.raises(ValueError, match="provenance only"):
        adapter.upsert(
            "generation_events",
            "source_tenant_id",
            {"source_tenant_id": uuid4(), "status": "complete"},
        )


def test_red_control_internal_resolver_call() -> None:
    target = projection_database_target("generation_events", schema="omninode_internal")
    conn, cursor = _connection("omninode_runtime")

    with (
        patch("psycopg2.connect", return_value=conn),
        patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring."
            "assert_projection_tenant_authority_matches_event"
        ) as resolve_tenant,
    ):
        adapter = _adapter(target)
        assert adapter.upsert("generation_events", "event_id", {"event_id": uuid4()})

    resolve_tenant.assert_not_called()
    assert not any(
        "set_config" in call.args[0] for call in cursor.execute.call_args_list
    )


@pytest.mark.parametrize(
    ("schema", "table"),
    [("tenant", "delegation_events"), ("omninode_internal", "generation_events")],
)
def test_write_only_declaration_rejects_query_for_every_domain(
    schema: str, table: str
) -> None:
    # Relations are real shipped grants (OMN-15656): a write-only *declaration*
    # must refuse reads even when the principal separately holds SELECT.
    target = _target(table, schema, "write")
    adapter = _adapter(target)

    with pytest.raises(PermissionError, match="read refused"):
        adapter.query(table)


def test_catalog_reader_is_explicit_and_has_no_writer_operation() -> None:
    target = projection_database_target(
        "plan_tiers",
        schema="platform_catalog",
        access="read",
        catalog_read_binding="app_dashboard",
        unshipped_grant_principal="app_dashboard",
        unshipped_grant_reason=(
            "PLATFORM_CATALOG grants are not derivable from node contracts: no "
            "db_io.db_tables block declares a catalog relation, so the shipped "
            "topology carries none (OMN-15355/OMN-15424 own that grant set). "
            "This asserts the catalog binding mechanism, not catalog coverage."
        ),
    )
    reader_conn, reader_cursor = _connection("app_dashboard")

    with patch("psycopg2.connect", return_value=reader_conn) as connect:
        adapter = _adapter(
            target,
            app_dashboard="postgresql://catalog-read",
        )
        assert adapter.query("plan_tiers") == []
        with pytest.raises(PermissionError, match="write refused"):
            adapter.upsert("plan_tiers", "tier_id", {"tier_id": "beta"})

    connect.assert_called_once_with("postgresql://catalog-read")
    assert any(
        'SELECT * FROM "platform_catalog"."plan_tiers"' in call.args[0]
        for call in reader_cursor.execute.call_args_list
    )


def test_catalog_target_fails_without_required_explicit_binding() -> None:
    table = ModelDbTableDeclaration(
        name="plan_tiers",
        database_ref="application",
        schema="platform_catalog",
        migration="proof/catalog.sql",
        access="read",
        role="catalog",
    )

    with pytest.raises(ValueError, match="explicit reader"):
        _resolve_projection_database_target(
            (table,),
            application_topology(),
            catalog_read_binding=None,
        )


def test_catalog_write_fails_without_declared_writer_binding() -> None:
    table = ModelDbTableDeclaration(
        name="plan_tiers",
        database_ref="application",
        schema="platform_catalog",
        migration="proof/catalog.sql",
        access="write",
        role="catalog",
    )

    with pytest.raises(ValueError, match="explicit writer"):
        _resolve_projection_database_target((table,), application_topology())


def test_catalog_binding_must_declare_the_exact_table_read_grant() -> None:
    table = ModelDbTableDeclaration(
        name="plan_tiers",
        database_ref="application",
        schema="platform_catalog",
        migration="proof/catalog.sql",
        access="read",
        role="catalog",
    )

    with pytest.raises(ValueError, match="lacks declared read privileges"):
        _resolve_projection_database_target(
            (table,),
            application_topology(),
            catalog_read_binding="tenant_projection",
        )


def test_upsert_binding_requires_select_in_addition_to_insert_and_update() -> None:
    topology = application_topology()
    database = topology.databases["application"]
    principal = database.principals["omninode_runtime"]
    grants = tuple(
        grant.model_copy(
            update={
                "privileges": tuple(
                    privilege
                    for privilege in grant.privileges
                    if privilege is not EnumDatabasePrivilege.SELECT
                )
            }
        )
        for grant in principal.grants
    )
    principals = dict(database.principals)
    principals["omninode_runtime"] = principal.model_copy(update={"grants": grants})
    database = database.model_copy(update={"principals": principals})
    topology = topology.model_copy(update={"databases": {"application": database}})
    table = ModelDbTableDeclaration(
        name="generation_events",
        database_ref="application",
        schema="omninode_internal",
        migration="proof/internal.sql",
        access="write",
        role="internal",
    )

    with pytest.raises(ValueError, match="SELECT"):
        _resolve_projection_database_target((table,), topology)


def test_adapter_rejects_missing_or_extra_dsn_bindings() -> None:
    target = projection_database_target("generation_events", schema="omninode_internal")

    with pytest.raises(ValueError, match="exactly match"):
        _build_projection_db_adapter({}, target, None, None)
    with pytest.raises(ValueError, match="exactly match"):
        _build_projection_db_adapter(
            {
                "omninode_runtime_service": "postgresql://internal",
                "tenant_projection": "postgresql://wrong-extra",
            },
            target,
            None,
            None,
        )


def test_red_control_domain_blind_upsert() -> None:
    target = projection_database_target("generation_events", schema="omninode_internal")
    adapter = _adapter(target)

    with pytest.raises(ValueError, match="not declared"):
        adapter.upsert("delegation_events", "event_id", {"event_id": uuid4()})


def test_closed_adapter_cannot_reconnect_or_reuse_authority() -> None:
    target = projection_database_target("generation_events", schema="omninode_internal")
    conn, _ = _connection("omninode_runtime")

    with patch("psycopg2.connect", return_value=conn) as connect:
        adapter = _adapter(target)
        adapter.upsert("generation_events", "event_id", {"event_id": uuid4()})
        adapter.close()
        with pytest.raises(RuntimeError, match="closed"):
            adapter.upsert("generation_events", "event_id", {"event_id": uuid4()})

    connect.assert_called_once_with("postgresql://fixture")
    conn.close.assert_called_once_with()


def test_connection_is_closed_when_identity_attestation_raises() -> None:
    target = projection_database_target("generation_events", schema="omninode_internal")
    conn, cursor = _connection("omninode_runtime")
    cursor.execute.side_effect = RuntimeError("attestation unavailable")

    with patch("psycopg2.connect", return_value=conn):
        adapter = _adapter(target)
        with pytest.raises(RuntimeError, match="attestation unavailable"):
            adapter.query("generation_events")

    conn.close.assert_called_once_with()


# ---------------------------------------------------------------------------
# OMN-16831 item 3 (operator ruling 2026-08-28, option D): the verified tenant
# authority is an AUTHORIZATION artifact and must never be the SOURCE of
# tenant_id. Attribution is recorded by the producer at write time; the
# authority, when present, only VERIFIES it.
#
# Before this change `TenantProjectionTableOperation` called `self._context()`
# unconditionally, so with no authority bound -- which is every real dispatch
# on every lane, because `bind_projection_tenant_authority` has zero non-test
# call sites -- all 15 TENANT-classified relations were structurally
# unwritable and 100% of their events DLQ'd before any SQL was issued.
# ---------------------------------------------------------------------------


def test_unbound_authority_writes_the_events_own_tenant_unmodified() -> None:
    """No authority bound -> the row's own tenant is stored verbatim.

    The event carries a real verified slug. Nothing invents, defaults, or
    overwrites it: what the producer recorded is what lands.
    """
    target = projection_database_target("delegation_events", schema="tenant")
    conn, cursor = _connection("tenant_projection_writer")

    with patch("psycopg2.connect", return_value=conn):
        adapter = _adapter(target, authority=None)
        adapter.upsert(
            "delegation_events",
            "event_id",
            {"event_id": uuid4(), "tenant_id": "beta-business-proof"},
        )

    insert = cursor.execute.call_args_list[-1]
    assert insert.args[1]["tenant_id"] == "beta-business-proof"


def test_unbound_authority_scopes_the_write_to_the_recorded_tenant() -> None:
    """OMN-16976: the unbound write is CONFINED to the tenant it records.

    RED before this change: the unbound path issued a bare INSERT with no
    ``set_config`` at all. Every tenant-classified relation carries
    ``WITH CHECK (tenant_id = current_setting('app.tenant_id', true))`` under
    FORCE ROW LEVEL SECURITY and the runtime connects as a NOBYPASSRLS role,
    so an unset GUC compares against NULL and Postgres denied the row --
    observed live on the .201 dev lane as 40 InsufficientPrivilege /3h on
    ``projection_delegation_inference_response_text``, the one such relation
    currently taking traffic. The OMN-16831 fix removed the Python refusal and
    left a Postgres one in its place.

    The scope value is the row's own ``tenant_id``, so this is a confinement,
    not a grant: the statement cannot write outside the identity it declares.
    """
    target = projection_database_target("delegation_events", schema="tenant")
    conn, cursor = _connection("tenant_projection_writer")

    with patch("psycopg2.connect", return_value=conn):
        adapter = _adapter(target, authority=None)
        adapter.upsert(
            "delegation_events",
            "event_id",
            {"event_id": uuid4(), "tenant_id": "beta-outside-every-compiled-map"},
        )

    statements = [call.args[0] for call in cursor.execute.call_args_list]
    assert "SELECT set_config(%s, %s, true)" in statements
    scope = next(
        call for call in cursor.execute.call_args_list if "set_config" in call.args[0]
    )
    # is_local=True -- the scope dies with the transaction, never leaks to the
    # next event on this pooled connection.
    assert scope.args[1] == ("app.tenant_id", "beta-outside-every-compiled-map")
    insert = cursor.execute.call_args_list[-1]
    assert insert.args[1]["tenant_id"] == "beta-outside-every-compiled-map"
    conn.commit.assert_called_once_with()
    assert conn.autocommit is True


def test_unbound_write_opens_no_scope_for_a_tenantless_row() -> None:
    """A row with no tenant gets NO scope, so the database denies it.

    The refusal stays where OMN-16831 item 4 put it -- the producer's
    obligation, enforced by the relation's own policy -- rather than being
    re-imported into this seam as a Python raise. An unset GUC compares
    against NULL, so a tenant-less row cannot pass ``WITH CHECK``.
    """
    target = projection_database_target("delegation_events", schema="tenant")
    conn, cursor = _connection("tenant_projection_writer")

    with patch("psycopg2.connect", return_value=conn):
        adapter = _adapter(target, authority=None)
        adapter.upsert("delegation_events", "event_id", {"event_id": uuid4()})

    assert not any(
        "set_config" in call.args[0] for call in cursor.execute.call_args_list
    )
    conn.commit.assert_not_called()


@pytest.mark.parametrize("blank", ["", "   ", None])
def test_unbound_write_opens_no_scope_for_a_blank_tenant(blank: object) -> None:
    """A blank or null tenant is not a tenant; it opens no scope either."""
    target = projection_database_target("delegation_events", schema="tenant")
    conn, cursor = _connection("tenant_projection_writer")

    with patch("psycopg2.connect", return_value=conn):
        adapter = _adapter(target, authority=None)
        adapter.upsert(
            "delegation_events", "event_id", {"event_id": uuid4(), "tenant_id": blank}
        )

    assert not any(
        "set_config" in call.args[0] for call in cursor.execute.call_args_list
    )


def test_unbound_read_is_scoped_only_when_the_caller_named_a_tenant() -> None:
    """Reads are scoped by the caller's own filter, and never widened.

    ``_preserve_existing_evidence`` reads before every upsert. Under FORCE RLS
    an unscoped read returns [] even when the tenant's row exists, so the
    upsert that follows OVERWRITES accumulated evidence instead of merging it
    -- silently, and worse than the refusal it replaced. Scoping the read to
    the tenant the caller already filtered on fixes that without widening
    anything: a filter naming no tenant still opens no scope and the read
    stays blind.
    """
    target = projection_database_target(
        "delegation_events", schema="tenant", access="read_write"
    )
    conn, cursor = _connection("tenant_projection_writer")

    with patch("psycopg2.connect", return_value=conn):
        adapter = _adapter(target, authority=None)
        adapter.query("delegation_events", {"tenant_id": "beta-outside-the-map"})
        scoped = [
            call
            for call in cursor.execute.call_args_list
            if "set_config" in call.args[0]
        ]
        assert scoped and scoped[-1].args[1] == (
            "app.tenant_id",
            "beta-outside-the-map",
        )

        cursor.execute.reset_mock()
        adapter.query("delegation_events", {"correlation_id": "cid-1"})

    assert not any(
        "set_config" in call.args[0] for call in cursor.execute.call_args_list
    )


def test_unbound_scope_is_rolled_back_and_never_leaks_on_failure() -> None:
    """The unbound scope obeys the same leak control as the bound one."""
    target = projection_database_target("delegation_events", schema="tenant")
    conn, cursor = _connection("tenant_projection_writer")
    cursor.execute.side_effect = (None, None, RuntimeError("write failed"))

    with patch("psycopg2.connect", return_value=conn):
        adapter = _adapter(target, authority=None)
        with pytest.raises(RuntimeError, match="write failed"):
            adapter.upsert(
                "delegation_events",
                "event_id",
                {"event_id": uuid4(), "tenant_id": "beta-outside-the-map"},
            )

    conn.rollback.assert_called_once_with()
    conn.commit.assert_not_called()
    assert conn.autocommit is True


def test_internal_and_catalog_domains_still_open_no_tenant_scope() -> None:
    """OMN-16976 must not leak a tenant scope into a non-tenant domain.

    Only ``TenantProjectionTableOperation`` supplies a recorded scope. The
    INTERNAL and CATALOG operations pass none, so their statements stay
    exactly as they were -- which their own red controls already assert, and
    which this restates against the new seam directly.
    """
    target = projection_database_target("generation_events", schema="omninode_internal")
    conn, cursor = _connection("omninode_runtime")

    with patch("psycopg2.connect", return_value=conn):
        adapter = _adapter(target)
        adapter.upsert("generation_events", "event_id", {"event_id": uuid4()})

    assert not any(
        "set_config" in call.args[0] for call in cursor.execute.call_args_list
    )


def test_unbound_authority_never_invents_a_tenant() -> None:
    """No authority and no supplied tenant -> the key is not fabricated.

    The operation refuses to source an identity it was never given. Whether a
    row is allowed to reach here tenant-less is the PRODUCER's obligation
    (ruling item 4); this seam's obligation is only that it invents nothing.
    """
    target = projection_database_target("delegation_events", schema="tenant")
    conn, cursor = _connection("tenant_projection_writer")

    with patch("psycopg2.connect", return_value=conn):
        adapter = _adapter(target, authority=None)
        adapter.upsert("delegation_events", "event_id", {"event_id": uuid4()})

    insert = cursor.execute.call_args_list[-1]
    assert "tenant_id" not in insert.args[1]


def test_unbound_authority_query_is_not_refused_and_is_not_narrowed() -> None:
    """The READ seam that produced the observed OMN-16831 stack must open too.

    `_preserve_existing_evidence` issues a READ before every delegation
    upsert, so a refusal here alone was enough to DLQ the whole event.
    """
    target = projection_database_target(
        "delegation_events", schema="tenant", access="read_write"
    )
    conn, cursor = _connection("tenant_projection_writer")

    with patch("psycopg2.connect", return_value=conn):
        adapter = _adapter(target, authority=None)
        rows = adapter.query("delegation_events", {"correlation_id": "cid-1"})

    assert rows == []
    select = cursor.execute.call_args_list[-1]
    assert "tenant_id" not in select.args[0]


def test_bound_authority_still_overrides_nothing_but_verifies_everything() -> None:
    """Authority bound -> a matching supplied tenant is ACCEPTED, not replaced.

    Full strength is preserved: the stored value still equals the authority's
    tenant, but it gets there by verification rather than by substitution.
    """
    tenant_id = uuid4()
    target = projection_database_target("delegation_events", schema="tenant")
    conn, cursor = _connection("tenant_projection_writer")

    with patch("psycopg2.connect", return_value=conn):
        adapter = _verified_adapter(target, tenant_id)
        adapter.upsert(
            "delegation_events",
            "event_id",
            {"event_id": uuid4(), "tenant_id": str(tenant_id)},
        )

    insert = cursor.execute.call_args_list[2]
    assert insert.args[1]["tenant_id"] == tenant_id


def test_bound_authority_still_fails_closed_on_a_mismatched_tenant() -> None:
    """Regression guard: decoupling must not weaken the bound-authority path."""
    target = projection_database_target("delegation_events", schema="tenant")

    with patch("psycopg2.connect") as connect:
        adapter = _verified_adapter(target, uuid4())
        with pytest.raises(ProjectionTenantContextError):
            adapter.upsert(
                "delegation_events",
                "event_id",
                {"event_id": uuid4(), "tenant_id": str(uuid4())},
            )

    connect.assert_not_called()


def test_unbound_tenant_operation_still_honours_the_closed_adapter_guard() -> None:
    """Dropping the authority precondition must not drop the lifecycle one."""
    target = projection_database_target("delegation_events", schema="tenant")
    conn, _ = _connection("tenant_projection_writer")

    with patch("psycopg2.connect", return_value=conn):
        adapter = _adapter(target, authority=None)
        adapter.upsert("delegation_events", "event_id", {"event_id": uuid4()})
        adapter.close()
        with pytest.raises(RuntimeError, match="closed"):
            adapter.upsert("delegation_events", "event_id", {"event_id": uuid4()})
