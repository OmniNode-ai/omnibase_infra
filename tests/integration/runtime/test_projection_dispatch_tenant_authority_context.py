# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Live-dispatch regression for projection tenant authority (OMN-15421)."""

from __future__ import annotations

from unittest.mock import patch
from uuid import uuid4

import pytest

from omnibase_core.models.contracts.subcontracts.model_db_table_declaration import (
    ModelDbTableDeclaration,
)
from omnibase_core.models.dispatch.model_dispatch_route import ModelDispatchRoute
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumDispatchStatus, EnumMessageCategory
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _make_projection_dispatch_callback,
    _resolve_projection_database_target,
)
from omnibase_infra.runtime.dispatch_envelope_context import (
    bind_projection_tenant_authority,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
from tests.helpers.application_db_topology import (
    application_topology,
    projection_database_target,
)
from tests.helpers.projection_tenant_authority import signed_tenant_authority_fixture

pytestmark = pytest.mark.integration

TOPIC = "onex.evt.platform.tenant-proof.v1"


class _Cursor:
    def __init__(
        self,
        calls: list[tuple[str, object]],
        principal: str = "tenant_projection_writer",
    ) -> None:
        self._calls = calls
        self._principal = principal

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, sql: str, params: object = None) -> None:
        self._calls.append((sql, params))

    def fetchone(self) -> tuple[str, str]:
        return (self._principal, "omnidash_analytics")


class _Connection:
    closed = False

    def __init__(
        self,
        calls: list[tuple[str, object]],
        principal: str = "tenant_projection_writer",
    ) -> None:
        self.autocommit = True
        self._calls = calls
        self._principal = principal
        self.close_calls = 0

    def cursor(self, *_args: object, **_kwargs: object) -> _Cursor:
        return _Cursor(self._calls, self._principal)

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None

    def close(self) -> None:
        self.close_calls += 1
        self.closed = True


class _TenantProjectionHandler:
    def handle(self, input_data: dict[str, object]) -> dict[str, int]:
        database = input_data["_db"]
        database.upsert(  # type: ignore[union-attr]
            "delegation_events",
            "correlation_id",
            {"correlation_id": input_data["_envelope_id"], "value": "dispatched"},
        )
        return {"rows_upserted": 1}


class _InternalProjectionHandler:
    def handle(self, input_data: dict[str, object]) -> dict[str, int]:
        database = input_data["_db"]
        database.upsert(  # type: ignore[union-attr]
            "generation_events",
            "correlation_id",
            {"correlation_id": input_data["_envelope_id"], "status": "complete"},
        )
        return {"rows_upserted": 1}


async def test_dispatch_engine_keeps_verified_authority_out_of_band(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """JSON materialization preserves capability and typed transport identity."""
    tenant_id = uuid4()
    correlation_id = uuid4()
    calls: list[tuple[str, object]] = []
    connection = _Connection(calls)
    monkeypatch.setenv("OMNIDASH_ANALYTICS_DB_URL", "postgresql://fixture")
    callback = _make_projection_dispatch_callback(
        _TenantProjectionHandler(),
        projection_database_target("delegation_events", schema="tenant"),
        (TOPIC,),
    )
    engine = MessageDispatchEngine()
    engine.register_dispatcher(
        dispatcher_id="tenant-projection-proof",
        dispatcher=callback,
        category=EnumMessageCategory.EVENT,
    )
    engine.register_route(
        ModelDispatchRoute(
            route_id="tenant-projection-proof",
            topic_pattern=TOPIC,
            message_category=EnumMessageCategory.EVENT,
            dispatcher_id="tenant-projection-proof",
        )
    )
    engine.freeze()
    envelope = ModelEventEnvelope[dict[str, object]](
        payload={"value": "dispatched", "tenant_id": str(uuid4())},
        correlation_id=correlation_id,
        event_type=TOPIC,
    )
    authority = signed_tenant_authority_fixture(
        tenant_id,
        event_envelope=envelope,
    ).verify()

    with (
        bind_projection_tenant_authority(authority),
        patch("psycopg2.connect", return_value=connection),
    ):
        result = await engine.dispatch(topic=TOPIC, envelope=envelope)

    assert result.status == EnumDispatchStatus.SUCCESS
    assert calls[0] == ("SELECT current_user, current_database()", None)
    assert calls[1] == (
        "SELECT set_config(%s, %s, true)",
        ("app.tenant_id", str(tenant_id)),
    )
    # OMN-16239: physical schema, not the declared one -- delegation_events is
    # still bridged to public until OMN-15359 relocates the tenant family.
    assert 'INSERT INTO "public"."delegation_events"' in calls[2][0]
    assert calls[2][1]["correlation_id"] == envelope.envelope_id  # type: ignore[index]
    assert calls[2][1]["tenant_id"] == tenant_id  # type: ignore[index]
    assert connection.close_calls == 1


async def test_dispatch_without_verified_capability_records_but_never_selects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No bound authority: the write proceeds, the isolation context does not.

    OMN-16831 (operator ruling 2026-08-28, option D) inverts what this pins,
    deliberately. It previously asserted the dispatch failed BEFORE connecting
    whenever no verified capability was bound -- which is every real dispatch on
    every lane, because ``bind_projection_tenant_authority`` has zero non-test
    call sites. That refusal did not make the platform safer: it destroyed the
    tenant dimension on every event of all 15 TENANT-classified relations, and
    the event log is immutable, so nothing recovers it afterwards.

    The two halves the ruling separates are both asserted here:

    * **Attribution proceeds.** The absence of an *authorization* artifact is
      not an attribution failure, so the statement is actually issued and the
      producer's own tenant reaches the row unmodified. Falsified by
      ``connect`` not being called.
    * **Authorization does not.** The envelope's self-asserted tenant is a
      claim, not proof of entitlement, so it must NEVER be promoted into an
      isolation context -- no ``set_config('app.tenant_id', ...)`` is issued on
      its word. Falsified by any ``set_config`` in the captured statements.

    That second assertion is the security invariant that used to be implied by
    "we never connect at all", now stated directly instead of as a side effect
    of refusing the write.
    """
    monkeypatch.setenv("OMNIDASH_ANALYTICS_DB_URL", "postgresql://fixture")
    callback = _make_projection_dispatch_callback(
        _TenantProjectionHandler(),
        projection_database_target("delegation_events", schema="tenant"),
        (TOPIC,),
    )
    claimed_tenant = uuid4()
    envelope = ModelEventEnvelope[dict[str, object]](
        payload={"tenant_id": str(claimed_tenant)},
        event_type=TOPIC,
    )
    calls: list[tuple[str, object]] = []
    connection = _Connection(calls, principal="tenant_projection_writer")

    with patch("psycopg2.connect", return_value=connection) as connect:
        await callback(envelope)

    connect.assert_called_once_with("postgresql://fixture")
    assert all("set_config" not in sql for sql, _params in calls), (
        "an unverified, self-asserted envelope tenant must never become an RLS "
        "isolation context -- it is attribution, not authorization"
    )


async def test_dispatch_without_verified_capability_fails_at_db_role_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []
    connection = _Connection(calls, principal="unverified_projection_writer")
    monkeypatch.setenv("OMNIDASH_ANALYTICS_DB_URL", "postgresql://fixture")
    callback = _make_projection_dispatch_callback(
        _TenantProjectionHandler(),
        projection_database_target("delegation_events", schema="tenant"),
        (TOPIC,),
    )
    claimed_tenant = uuid4()
    envelope = ModelEventEnvelope[dict[str, object]](
        payload={"tenant_id": str(claimed_tenant)},
        event_type=TOPIC,
    )

    with patch("psycopg2.connect", return_value=connection) as connect:
        await callback(envelope)

    connect.assert_called_once_with("postgresql://fixture")
    assert calls == [("SELECT current_user, current_database()", None)]
    assert connection.close_calls == 1


async def test_mixed_target_internal_operation_does_not_resolve_tenant_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tables = (
        ModelDbTableDeclaration(
            name="delegation_events",
            database_ref="application",
            schema="tenant",
            migration="proof/tenant.sql",
            access="read_write",
            role="tenant",
        ),
        ModelDbTableDeclaration(
            name="generation_events",
            database_ref="application",
            schema="omninode_internal",
            migration="proof/internal.sql",
            access="read_write",
            role="internal",
        ),
    )
    target = _resolve_projection_database_target(tables, application_topology())
    monkeypatch.setenv("OMNIDASH_ANALYTICS_DB_URL", "postgresql://tenant")
    monkeypatch.setenv("OMNINODE_INTERNAL_DB_URL", "postgresql://internal")
    callback = _make_projection_dispatch_callback(
        _InternalProjectionHandler(), target, (TOPIC,)
    )
    calls: list[tuple[str, object]] = []
    connection = _Connection(calls, principal="omninode_runtime")
    envelope = ModelEventEnvelope[dict[str, object]](
        payload={"status": "complete"},
        correlation_id=uuid4(),
        event_type=TOPIC,
    )

    with patch("psycopg2.connect", return_value=connection) as connect:
        await callback(envelope)

    connect.assert_called_once_with("postgresql://internal")
    assert all("set_config" not in sql for sql, _params in calls)
    # OMN-16239: generation_events is still under the OMN-15359 bridge, so the
    # internal write resolves to public. The point of this assertion is that the
    # internal table was written at all on the internal binding -- the schema it
    # is qualified with must be the physical one.
    assert any('"public"."generation_events"' in sql for sql, _ in calls)
