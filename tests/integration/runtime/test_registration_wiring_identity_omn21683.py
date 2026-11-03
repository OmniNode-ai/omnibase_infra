# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Registration wiring readback through the real container and handlers."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import cast
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from omnibase_core.container import ModelONEXContainer
from omnibase_core.enums import EnumNodeKind
from omnibase_infra.models.catalog.model_topic_catalog_response import (
    ModelTopicCatalogResponse,
)
from omnibase_infra.models.dispatch.model_injected_request import (
    ModelInjectedRequest,
)
from omnibase_infra.models.registration import ModelNodeIntrospectionEvent
from omnibase_infra.models.registration.model_event_bus_topic_entry import (
    ModelEventBusTopicEntry,
)
from omnibase_infra.models.registration.model_node_event_bus_config import (
    ModelNodeEventBusConfig,
)
from omnibase_infra.models.registration.model_topic_catalog_request import (
    ModelTopicCatalogRequest,
)
from omnibase_infra.nodes.node_registration_orchestrator.handlers import (
    HandlerCatalogRequest,
    HandlerNodeHeartbeat,
    HandlerNodeIntrospected,
)
from omnibase_infra.nodes.node_registration_orchestrator.wiring import (
    wire_registration_handlers,
)
from omnibase_infra.protocols.protocol_node_heartbeat import ProtocolNodeHeartbeat
from omnibase_infra.topics import SUFFIX_NODE_INTROSPECTION

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest.mark.parametrize("with_projector", [False, True])
async def test_introspection_topics_reach_catalog_through_real_registry(
    with_projector: bool,
) -> None:
    """Both optional-projector paths retain the introspection handler's store.

    Database access and the unused optional projector are stubbed. Registration,
    resolution, reduction, introspection handling, topic storage and catalog
    handling all run for real.
    """
    container = ModelONEXContainer()
    registry = container.service_registry
    assert registry is not None
    connection = AsyncMock()
    connection.fetchrow.return_value = None
    pool = MagicMock()
    pool.acquire.return_value.__aenter__.return_value = connection

    summary = await wire_registration_handlers(
        container, pool, projector=MagicMock() if with_projector else None
    )
    assert summary["status"] == "success"
    introspected = await registry.resolve_service(HandlerNodeIntrospected)
    catalog = await registry.resolve_service(HandlerCatalogRequest)
    assert isinstance(introspected, HandlerNodeIntrospected)
    assert isinstance(catalog, HandlerCatalogRequest)
    assert introspected._topic_store is catalog._topic_store
    registrations = await registry.get_registrations_by_interface(
        HandlerNodeIntrospected
    )
    assert len(registrations) == 1

    now = datetime(2026, 11, 1, tzinfo=UTC)
    correlation_id = uuid4()
    request: ModelInjectedRequest[ModelTopicCatalogRequest] = ModelInjectedRequest(
        event=ModelTopicCatalogRequest(
            correlation_id=correlation_id,
            requested_at=now,
            requester="registration-wiring-readback",
        ),
        now=now,
        input_id=uuid4(),
        correlation_id=correlation_id,
    )
    cold = await catalog.handle(request)
    assert len(cold.events) == 1
    assert cold.events[0].topics == ()

    topic = SUFFIX_NODE_INTROSPECTION
    event = ModelNodeIntrospectionEvent(
        node_id=uuid4(),
        node_type=EnumNodeKind.EFFECT,
        correlation_id=correlation_id,
        timestamp=now,
        event_bus=ModelNodeEventBusConfig(
            publish_topics=[ModelEventBusTopicEntry(topic=topic)]
        ),
    )
    await introspected.handle(
        ModelInjectedRequest(
            event=event,
            now=now,
            input_id=uuid4(),
            correlation_id=correlation_id,
        )
    )
    connection.fetchrow.assert_awaited_once()
    warm = await catalog.handle(request)
    assert len(warm.events) == 1
    response = warm.events[0]
    assert isinstance(response, ModelTopicCatalogResponse)
    assert [entry.topic_name for entry in response.topics] == [topic]
    assert response.correlation_id == correlation_id
    assert not response.warnings

    if with_projector:
        concrete = await registry.resolve_service(HandlerNodeHeartbeat)
        # Registry keys include protocol classes; they are never constructed.
        heartbeat_protocol = cast("type[ProtocolNodeHeartbeat]", ProtocolNodeHeartbeat)
        protocol = await registry.resolve_service(heartbeat_protocol)
        assert isinstance(concrete, HandlerNodeHeartbeat)
        assert protocol is concrete
        assert await registry.resolve_service(HandlerNodeHeartbeat) is concrete
        assert await registry.resolve_service(heartbeat_protocol) is concrete
    else:
        assert "HandlerNodeHeartbeat" not in summary["services"]

    print(
        f"registration-wiring-readback projector={with_projector}: "
        f"single introspection registration; catalog topics={[topic]}; "
        f"heartbeat identity={'shared' if with_projector else 'not registered'}"
    )
