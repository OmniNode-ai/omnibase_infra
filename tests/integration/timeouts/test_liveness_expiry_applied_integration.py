# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration test: a runtime tick moves a lapsed registration to liveness_expired.

Wires the real ``ServiceTimeoutScanner`` and ``ServiceTimeoutEmitter`` over the
in-memory projection store and reads the stored rows back after the tick. A
row whose liveness deadline passed must be stored as ``liveness_expired`` with
its marker; a row that heartbeated in time, ticked in the same pass, must stay
``active`` with no marker. Before OMN-21028 the lapsed row kept
``current_state=active`` and only the marker changed.

Related Tickets:
    - OMN-21028: Apply the liveness expiry the timeout path already emits
    - OMN-932 (C2): Durable Timeout Handling
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from omnibase_core.container import ModelONEXContainer
from omnibase_core.enums import EnumNodeKind
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.enums import EnumRegistrationState
from omnibase_infra.models.projection import (
    ModelRegistrationProjection,
    ModelSequenceInfo,
)
from omnibase_infra.models.registration.model_node_capabilities import (
    ModelNodeCapabilities,
)
from omnibase_infra.models.registration.model_payload_postgres_update_registration import (
    ModelPayloadPostgresUpdateRegistration,
)
from omnibase_infra.nodes.node_registration_orchestrator.models.model_reducer_context import (
    ModelReducerContext,
)
from omnibase_infra.nodes.node_registration_orchestrator.services import (
    RegistrationReducerService,
)
from omnibase_infra.services import (
    ModelTimeoutEmissionConfig,
    ServiceTimeoutEmitter,
    ServiceTimeoutScanner,
)

if TYPE_CHECKING:
    from omnibase_core.protocols.event_bus.protocol_event_bus import ProtocolEventBus
    from omnibase_infra.projectors.projection_reader_registration import (
        ProjectionReaderRegistration,
    )
    from omnibase_infra.runtime.projector_shell import ProjectorShell
    from tests.integration.timeouts.conftest import (
        InMemoryProjectionStore,
        MockEventBus,
        MockProjector,
    )

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

LIVENESS_WINDOW_SECONDS = 90.0
"""Heartbeat extension used by the reducer under test."""


def _active_row(
    *,
    entity_id: UUID,
    liveness_deadline: datetime,
    last_heartbeat_at: datetime | None = None,
) -> ModelRegistrationProjection:
    now = datetime.now(UTC)
    return ModelRegistrationProjection(
        entity_id=entity_id,
        domain="registration",
        current_state=EnumRegistrationState.ACTIVE,
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


async def _store(
    store: InMemoryProjectionStore, projection: ModelRegistrationProjection
) -> None:
    await store.persist(
        projection=projection,
        entity_id=projection.entity_id,
        domain="registration",
        sequence_info=ModelSequenceInfo(sequence=100, offset=100, partition="0"),
    )


def _emitter(
    store: InMemoryProjectionStore, bus: MockEventBus, projector: MockProjector
) -> ServiceTimeoutEmitter:
    scanner = ServiceTimeoutScanner(
        container=MagicMock(spec=ModelONEXContainer),
        projection_reader=cast("ProjectionReaderRegistration", store),
    )
    return ServiceTimeoutEmitter(
        container=MagicMock(spec=ModelONEXContainer),
        timeout_query=scanner,
        event_bus=cast("ProtocolEventBus", bus),
        projector=cast("ProjectorShell", projector),
        config=ModelTimeoutEmissionConfig(environment="test", namespace="onex"),
    )


async def test_tick_expires_lapsed_row_and_keeps_heartbeating_row_active(
    mock_event_bus: MockEventBus,
    in_memory_store: InMemoryProjectionStore,
    mock_projector: MockProjector,
) -> None:
    """One tick: the lapsed row is stored expired, the heartbeating row stays active."""
    now = datetime.now(UTC)
    lapsed_id = uuid4()
    alive_id = uuid4()

    # The live node heartbeats just before its deadline; the reducer extends it.
    reducer = RegistrationReducerService(
        liveness_window_seconds=LIVENESS_WINDOW_SECONDS
    )
    decision = reducer.decide_heartbeat(
        projection=_active_row(
            entity_id=alive_id, liveness_deadline=now + timedelta(seconds=5)
        ),
        node_id=alive_id,
        heartbeat_timestamp=now,
        ctx=ModelReducerContext(correlation_id=uuid4(), now=now, tick_id=uuid4()),
    )
    (intent,) = decision.intents
    assert isinstance(intent.payload, ModelPayloadPostgresUpdateRegistration)
    extended_deadline = intent.payload.updates.model_dump()["liveness_deadline"]

    await _store(
        in_memory_store,
        _active_row(entity_id=lapsed_id, liveness_deadline=now - timedelta(minutes=10)),
    )
    await _store(
        in_memory_store,
        _active_row(
            entity_id=alive_id,
            liveness_deadline=extended_deadline,
            last_heartbeat_at=now,
        ),
    )

    tick_at = now + timedelta(seconds=10)
    emitter = _emitter(in_memory_store, mock_event_bus, mock_projector)
    result = await emitter.process_timeouts(
        now=tick_at, tick_id=uuid4(), correlation_id=uuid4()
    )

    assert result.liveness_expirations_emitted == 1
    assert mock_event_bus.count_events("liveness-expired") == 1

    lapsed = in_memory_store.projections[(lapsed_id, "registration")]
    assert lapsed.current_state is EnumRegistrationState.LIVENESS_EXPIRED
    assert lapsed.liveness_timeout_emitted_at == tick_at

    alive = in_memory_store.projections[(alive_id, "registration")]
    assert alive.current_state is EnumRegistrationState.ACTIVE
    assert alive.liveness_timeout_emitted_at is None

    # A second tick finds nothing new and leaves both rows as they are.
    again = await _emitter(
        in_memory_store, mock_event_bus, mock_projector
    ).process_timeouts(now=tick_at, tick_id=uuid4(), correlation_id=uuid4())
    assert again.liveness_expirations_emitted == 0
    assert mock_event_bus.count_events("liveness-expired") == 1
    assert (
        in_memory_store.projections[(lapsed_id, "registration")].current_state
        is EnumRegistrationState.LIVENESS_EXPIRED
    )
    assert (
        in_memory_store.projections[(alive_id, "registration")].current_state
        is EnumRegistrationState.ACTIVE
    )
