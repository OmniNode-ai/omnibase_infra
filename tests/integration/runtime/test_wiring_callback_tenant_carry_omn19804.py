# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tenant carriage through the real engine behind the subcontract wiring (OMN-19804).

The unit test beside this one pins the callback with a stand-in engine. This one
runs the REAL ``MessageDispatchEngine`` -- its own ``bind_dispatch_envelope``
scope, its own route matching -- behind the real ``EventBusSubcontractWiring``
callback and the real ``DispatchResultApplier``, with only the handler and the
bus stubbed.

The defect lived in the gap between the two: the engine binds the consumed
envelope only while the dispatcher runs, and the callback applied the result
after that scope had closed. A re-routed delegation's ``routing-decision`` and
``quality-gate-result`` envelopes therefore recorded no tenant, while the
``delegation-request`` that began the chain did.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from pydantic import BaseModel

from omnibase_core.models.dispatch.model_dispatch_route import ModelDispatchRoute
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumDispatchStatus
from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.event_bus.models import ModelEventHeaders, ModelEventMessage
from omnibase_infra.models.dispatch import ModelDispatchResult
from omnibase_infra.protocols import ProtocolEventBusLike
from omnibase_infra.runtime.event_bus_subcontract_wiring import (
    EventBusSubcontractWiring,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
from omnibase_infra.runtime.service_dispatch_result_applier import (
    DispatchResultApplier,
)
from omnibase_infra.topics import topic_keys
from omnibase_infra.topics.service_topic_registry import ServiceTopicRegistry

pytestmark = pytest.mark.integration

_TENANT = "acme"
_TOPIC = "dev.test.events.v1"
_REGISTRY = ServiceTopicRegistry.from_defaults()
_RETRY_PATH_TOPIC_KEYS: tuple[str, ...] = (
    topic_keys.DELEGATION_ROUTING_REQUEST,
    topic_keys.DELEGATION_ROUTING_DECISION,
    topic_keys.DELEGATION_QUALITY_GATE_RESULT,
    topic_keys.DELEGATION_INFERENCE_RESPONSE,
)


class _RetryPathEvent(BaseModel):
    entity_id: str = "n-1"


def _message(*, tenant_id: str | None) -> ModelEventMessage:
    body: dict[str, object] = {
        "event_type": "test.event",
        "correlation_id": str(uuid4()),
        "payload": {"key": "value"},
    }
    if tenant_id is not None:
        body["tenant_id"] = tenant_id
    return ModelEventMessage(
        topic=_TOPIC,
        key=b"test-key",
        value=json.dumps(body).encode("utf-8"),
        headers=ModelEventHeaders(
            source="test-service",
            event_type="test.event",
            timestamp=datetime.now(UTC),
        ),
    )


def _wired(mock_bus: AsyncMock) -> EventBusSubcontractWiring:
    topic_router = {
        f"_RetryPathEvent{idx}": _REGISTRY.resolve(key)
        for idx, key in enumerate(_RETRY_PATH_TOPIC_KEYS)
    }
    event_classes = [type(name, (_RetryPathEvent,), {}) for name in topic_router]

    async def dispatcher(_envelope: object) -> ModelDispatchResult:
        return ModelDispatchResult(
            status=EnumDispatchStatus.SUCCESS,
            topic=_TOPIC,
            started_at=datetime.now(UTC),
            dispatcher_id="retry-path-dispatcher",
            output_events=[cls() for cls in event_classes],
        )

    engine = MessageDispatchEngine()
    engine.register_dispatcher(
        dispatcher_id="retry-path-dispatcher",
        dispatcher=dispatcher,
        category=EnumMessageCategory.EVENT,
        message_types=None,
    )
    engine.register_route(
        ModelDispatchRoute(
            route_id="route-retry-path",
            topic_pattern="*.test.events.*",
            message_category=EnumMessageCategory.EVENT,
            dispatcher_id="retry-path-dispatcher",
        )
    )
    engine.freeze()
    return EventBusSubcontractWiring(
        event_bus=mock_bus,
        dispatch_engine=engine,
        environment="dev",
        node_name="test-handler",
        service="test-service",
        version="v1",
        result_applier=DispatchResultApplier(
            event_bus=mock_bus,
            output_topic="fallback-topic",
            topic_router=topic_router,
        ),
    )


def _published(mock_bus: AsyncMock) -> dict[str, ModelEventEnvelope[BaseModel]]:
    return {
        call.kwargs["topic"]: call.kwargs["envelope"]
        for call in mock_bus.publish_envelope.call_args_list
    }


def _bus() -> AsyncMock:
    bus = AsyncMock(spec=ProtocolEventBusLike)
    bus.publish_envelope = AsyncMock()
    return bus


@pytest.mark.asyncio
async def test_real_engine_outputs_carry_tenant_and_causal_edge() -> None:
    mock_bus = _bus()
    callback = _wired(mock_bus)._create_dispatch_callback(_TOPIC, "dev.test-handler")

    await callback(_message(tenant_id=_TENANT))

    published = _published(mock_bus)
    expected_topics = {_REGISTRY.resolve(key) for key in _RETRY_PATH_TOPIC_KEYS}
    assert set(published) == expected_topics
    assert {env.tenant_id for env in published.values()} == {_TENANT}
    assert all(env.parent_envelope_id is not None for env in published.values())


@pytest.mark.asyncio
async def test_real_engine_no_tenant_consumed_publishes_none() -> None:
    """Negative control: carried, never sourced."""
    mock_bus = _bus()
    callback = _wired(mock_bus)._create_dispatch_callback(_TOPIC, "dev.test-handler")

    await callback(_message(tenant_id=None))

    published = _published(mock_bus)
    assert len(published) == len(_RETRY_PATH_TOPIC_KEYS)
    assert {env.tenant_id for env in published.values()} == {None}
