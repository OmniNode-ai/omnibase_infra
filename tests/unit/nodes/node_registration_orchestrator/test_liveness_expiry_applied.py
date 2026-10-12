# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Liveness expiry must move the registration row, not only stamp a marker.

The runtime tick finds an ACTIVE registration whose liveness deadline has
passed, publishes ``node-liveness-expired`` and stamps
``liveness_timeout_emitted_at``. Before OMN-21028 that was all it did, so the
row stayed ``active`` for ever and the marker kept the scanner from looking at
it again. These tests pin the projection write at the seam that dropped it:
``ServiceTimeoutEmitter`` -> ``ProjectorShell.partial_update``.

Related Tickets:
    - OMN-21028: Apply the liveness expiry the timeout path already emits
    - OMN-932 (C2): Durable Timeout Handling
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from omnibase_core.container import ModelONEXContainer
from omnibase_core.enums import EnumNodeKind
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.enums import EnumRegistrationState
from omnibase_infra.models.projection import ModelRegistrationProjection
from omnibase_infra.models.registration import (
    ModelNodeCapabilities,
    ModelNodeIntrospectionEvent,
)
from omnibase_infra.models.registration.model_payload_postgres_upsert_registration import (
    ModelPayloadPostgresUpsertRegistration,
)
from omnibase_infra.nodes.node_registration_orchestrator.models.model_reducer_context import (
    ModelReducerContext,
)
from omnibase_infra.nodes.node_registration_orchestrator.services import (
    RegistrationReducerService,
)
from omnibase_infra.protocols import ProtocolEventBusLike
from omnibase_infra.services import (
    ModelTimeoutEmissionConfig,
    ModelTimeoutQueryResult,
    ServiceTimeoutEmitter,
)

LIVENESS_WINDOW_SECONDS = 90.0
"""Heartbeat extension used by the reducer under test."""


def _active_projection(
    *,
    entity_id: UUID,
    liveness_deadline: datetime,
    last_heartbeat_at: datetime | None = None,
    state: EnumRegistrationState = EnumRegistrationState.ACTIVE,
    liveness_timeout_emitted_at: datetime | None = None,
) -> ModelRegistrationProjection:
    """Build a projection (ACTIVE by default) with the given liveness deadline."""
    now = datetime.now(UTC)
    return ModelRegistrationProjection(
        entity_id=entity_id,
        domain="registration",
        current_state=state,
        liveness_timeout_emitted_at=liveness_timeout_emitted_at,
        node_type=EnumNodeKind.EFFECT,
        node_version=ModelSemVer.parse("1.0.0"),
        capabilities=ModelNodeCapabilities(),
        liveness_deadline=liveness_deadline,
        last_heartbeat_at=last_heartbeat_at,
        last_applied_event_id=uuid4(),
        last_applied_offset=100,
        registered_at=now,
        updated_at=now,
    )


def _emitter(
    *, overdue: list[ModelRegistrationProjection]
) -> tuple[ServiceTimeoutEmitter, AsyncMock, AsyncMock]:
    """Build an emitter whose scanner reports ``overdue`` liveness candidates."""
    scanner = AsyncMock()
    scanner.find_overdue_entities = AsyncMock(
        return_value=ModelTimeoutQueryResult(
            ack_timeouts=[],
            liveness_expirations=overdue,
            query_time=datetime.now(UTC),
            query_duration_ms=1.0,
        )
    )
    bus = AsyncMock(spec=ProtocolEventBusLike)
    bus.publish_envelope = AsyncMock()
    projector = AsyncMock()
    projector.partial_update = AsyncMock(return_value=True)
    emitter = ServiceTimeoutEmitter(
        container=MagicMock(spec=ModelONEXContainer),
        timeout_query=scanner,
        event_bus=bus,
        projector=projector,
        config=ModelTimeoutEmissionConfig(environment="test", namespace="omnitest"),
    )
    return emitter, bus, projector


@pytest.mark.unit
@pytest.mark.asyncio
async def test_liveness_expired_applied() -> None:
    """An ACTIVE row past its liveness deadline is written as liveness_expired."""
    now = datetime.now(UTC)
    node_id = uuid4()
    overdue = _active_projection(
        entity_id=node_id,
        liveness_deadline=now - timedelta(minutes=10),
    )
    emitter, bus, projector = _emitter(overdue=[overdue])

    result = await emitter.process_timeouts(
        now=now, tick_id=uuid4(), correlation_id=uuid4()
    )

    assert result.liveness_expirations_emitted == 1
    bus.publish_envelope.assert_called_once()
    projector.partial_update.assert_called_once()
    call = projector.partial_update.call_args
    assert call.kwargs["aggregate_id"] == node_id
    updates = call.kwargs["updates"]
    assert updates["current_state"] == EnumRegistrationState.LIVENESS_EXPIRED.value
    assert updates["liveness_timeout_emitted_at"] == now


@pytest.mark.unit
@pytest.mark.asyncio
async def test_liveness_heartbeat_keeps_active() -> None:
    """A registration that heartbeats before its deadline is never expired."""
    now = datetime.now(UTC)
    node_id = uuid4()
    reducer = RegistrationReducerService(
        liveness_window_seconds=LIVENESS_WINDOW_SECONDS
    )
    ctx = ModelReducerContext(correlation_id=uuid4(), now=now, tick_id=uuid4())

    # The node's deadline was about to pass; it heartbeats in time.
    about_to_expire = _active_projection(
        entity_id=node_id,
        liveness_deadline=now + timedelta(seconds=5),
    )
    decision = reducer.decide_heartbeat(
        projection=about_to_expire,
        node_id=node_id,
        heartbeat_timestamp=now,
        ctx=ctx,
    )
    assert decision.action == "emit"
    (intent,) = decision.intents
    heartbeat_updates = intent.payload.updates.model_dump()
    assert "current_state" not in heartbeat_updates
    assert heartbeat_updates["liveness_deadline"] == now + timedelta(
        seconds=LIVENESS_WINDOW_SECONDS
    )

    # The scanner therefore no longer lists it, and the emitter writes nothing.
    emitter, bus, projector = _emitter(overdue=[])
    result = await emitter.process_timeouts(
        now=now + timedelta(seconds=10), tick_id=uuid4(), correlation_id=uuid4()
    )

    assert result.liveness_expirations_emitted == 0
    bus.publish_envelope.assert_not_called()
    projector.partial_update.assert_not_called()
    extended = _active_projection(
        entity_id=node_id,
        liveness_deadline=heartbeat_updates["liveness_deadline"],
        last_heartbeat_at=now,
    )
    assert not extended.needs_liveness_timeout_event(now + timedelta(seconds=10))
    assert extended.current_state is EnumRegistrationState.ACTIVE


@pytest.mark.unit
def test_liveness_expired_reregistration_rearms_expiry() -> None:
    """A node that expired and re-registers can expire again.

    The re-registration upsert must clear the old emission marker; otherwise the
    scanner (which skips rows with a marker) never looks at the node again.
    """
    now = datetime.now(UTC)
    node_id = uuid4()
    expired = _active_projection(
        entity_id=node_id,
        liveness_deadline=now - timedelta(minutes=10),
        state=EnumRegistrationState.LIVENESS_EXPIRED,
        liveness_timeout_emitted_at=now - timedelta(minutes=9),
    )
    event = ModelNodeIntrospectionEvent(
        node_id=node_id,
        node_type=EnumNodeKind.EFFECT,
        correlation_id=uuid4(),
        timestamp=now,
    )

    decision = RegistrationReducerService().decide_introspection(
        projection=expired, event=event, correlation_id=uuid4(), now=now
    )

    (upsert,) = [
        i.payload
        for i in decision.intents
        if isinstance(i.payload, ModelPayloadPostgresUpsertRegistration)
    ]
    assert upsert.record.current_state == EnumRegistrationState.ACTIVE.value
    assert "liveness_timeout_emitted_at" in upsert.record.data
    assert upsert.record.data["liveness_timeout_emitted_at"] is None
