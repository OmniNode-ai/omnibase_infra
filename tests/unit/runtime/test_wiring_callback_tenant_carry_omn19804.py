# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The subcontract wiring callback carries the consumed tenant (OMN-19804).

``DispatchResultApplier`` records the consumed envelope's ``tenant_id`` and
``parent_envelope_id`` on every envelope it publishes, but it reads that
envelope from the ``current_dispatch_envelope()`` contextvar, which
``MessageDispatchEngine.dispatch`` binds only for the duration of the
dispatcher call. ``EventBusSubcontractWiring``'s callback applies the result
AFTER ``dispatch`` has returned, so on that path the contextvar was already
reset: the applier saw no consumed envelope and published every output with no
tenant and no causal edge. The two ``handler_wiring`` apply sites bind the
envelope themselves; this callback did not.

Live shape: a re-routed delegation's ``routing-decision`` and
``quality-gate-result`` rows recorded ``tenant_id`` at neither the envelope nor
the payload, while the ``delegation-request`` row did.

The stand-in engine below binds the envelope only inside ``dispatch`` and
resets it on return, which is exactly the real engine's contextvar lifetime.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumDispatchStatus
from omnibase_infra.event_bus.models import ModelEventHeaders, ModelEventMessage
from omnibase_infra.models.dispatch import ModelDispatchResult
from omnibase_infra.protocols import ProtocolEventBusLike
from omnibase_infra.runtime.dispatch_envelope_context import bind_dispatch_envelope
from omnibase_infra.runtime.event_bus_subcontract_wiring import (
    EventBusSubcontractWiring,
)
from omnibase_infra.runtime.service_dispatch_result_applier import (
    DispatchResultApplier,
)
from omnibase_infra.topics import topic_keys
from omnibase_infra.topics.service_topic_registry import ServiceTopicRegistry

pytestmark = pytest.mark.unit

_TENANT = "acme"
_REGISTRY = ServiceTopicRegistry.from_defaults()

# The four envelopes the brief names on a delegation's retry path, keyed by the
# topic each publishes to.
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
        topic="onex.evt.test.v1",
        key=b"test-key",
        value=json.dumps(body).encode("utf-8"),
        headers=ModelEventHeaders(
            source="test-service",
            event_type="test.event",
            timestamp=datetime.now(UTC),
        ),
    )


def _wiring_over_real_applier(
    mock_bus: AsyncMock,
) -> EventBusSubcontractWiring:
    """Wire the real applier behind an engine with the real contextvar lifetime."""
    topic_router = {
        f"_RetryPathEvent{idx}": _REGISTRY.resolve(key)
        for idx, key in enumerate(_RETRY_PATH_TOPIC_KEYS)
    }
    event_classes = [type(name, (_RetryPathEvent,), {}) for name in topic_router]

    async def dispatch(
        topic: str, envelope: ModelEventEnvelope[object]
    ) -> ModelDispatchResult:
        with bind_dispatch_envelope(envelope):
            result = ModelDispatchResult(
                status=EnumDispatchStatus.SUCCESS,
                topic=topic,
                started_at=datetime.now(UTC),
                correlation_id=envelope.correlation_id,
                dispatcher_id="retry-path-dispatcher",
                output_events=[cls() for cls in event_classes],
            )
        return result

    engine = AsyncMock()
    engine.dispatch = AsyncMock(side_effect=dispatch)
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
async def test_every_retry_path_output_carries_the_consumed_tenant() -> None:
    mock_bus = _bus()
    wiring = _wiring_over_real_applier(mock_bus)
    callback = wiring._create_dispatch_callback("onex.evt.test.v1", "dev.test-handler")

    await callback(_message(tenant_id=_TENANT))

    published = _published(mock_bus)
    expected_topics = {_REGISTRY.resolve(key) for key in _RETRY_PATH_TOPIC_KEYS}
    assert set(published) == expected_topics
    assert {topic: env.tenant_id for topic, env in published.items()} == dict.fromkeys(
        expected_topics, _TENANT
    )


@pytest.mark.asyncio
async def test_every_retry_path_output_records_the_consumed_envelope_as_parent() -> (
    None
):
    mock_bus = _bus()
    wiring = _wiring_over_real_applier(mock_bus)
    callback = wiring._create_dispatch_callback("onex.evt.test.v1", "dev.test-handler")

    await callback(_message(tenant_id=_TENANT))

    parents = {env.parent_envelope_id for env in _published(mock_bus).values()}
    assert len(parents) == 1
    (parent,) = parents
    assert isinstance(parent, UUID)


@pytest.mark.asyncio
async def test_a_consumed_envelope_with_no_tenant_publishes_none() -> None:
    """Negative control: carried, never sourced (OMN-16831 AC2)."""
    mock_bus = _bus()
    wiring = _wiring_over_real_applier(mock_bus)
    callback = wiring._create_dispatch_callback("onex.evt.test.v1", "dev.test-handler")

    await callback(_message(tenant_id=None))

    published = _published(mock_bus)
    assert len(published) == len(_RETRY_PATH_TOPIC_KEYS)
    assert {env.tenant_id for env in published.values()} == {None}
