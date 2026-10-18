# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Publisher declarations reach runtime dispatch through serialized bus envelopes."""

from __future__ import annotations

from pathlib import Path
from typing import cast
from uuid import uuid4

import pytest

from omnibase_core.models.contracts.subcontracts import ModelEventBusSubcontract
from omnibase_core.models.dispatch.model_dispatch_route import ModelDispatchRoute
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_core.protocols.event_bus.protocol_event_bus_subscriber import (
    ProtocolEventBusSubscriber,
)
from omnibase_infra.enums import EnumMessageCategory
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.runtime.dispatch_envelope_context import current_dispatch_envelope
from omnibase_infra.runtime.event_bus_subcontract_wiring import (
    EventBusSubcontractWiring,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
from omnibase_infra.testing.publisher_contract_fixture import (
    PublisherContractCorpus,
    PublisherTopicError,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "declaration",
    [
        "event_bus:\n  publish_topics: [TOPIC]\n",
        "event_bus:\n  publish:\n    success_topic: TOPIC\n",
        "event_bus:\n  publish:\n    failure_topic: TOPIC\n",
        "produced_events:\n  - topic: TOPIC\n",
        "publications: [TOPIC]\n",
        "topics:\n  produces: [TOPIC]\n",
        "runtime_dispatch:\n  command_topic: TOPIC\n",
    ],
    ids=[
        "publish-topics",
        "success",
        "failure",
        "produced",
        "publications",
        "produces",
        "command",
    ],
)
@pytest.mark.asyncio
async def test_declared_publisher_envelope_reaches_alias_only_dispatcher(
    tmp_path: Path, declaration: str
) -> None:
    # Each declaration stands alone: a second publish_topics entry must not hide
    # a reader that ignores the shape under test.
    kind = "cmd" if declaration.startswith("runtime_dispatch:") else "evt"
    topic = f"onex.{kind}.demo.publisher-dispatch.v1"
    category = EnumMessageCategory.from_topic(topic)
    assert category is not None
    contract = tmp_path / "contract.yaml"
    contract.write_text(
        "name: node_publisher_dispatch\n" + declaration.replace("TOPIC", topic),
        encoding="utf-8",
    )
    corpus = PublisherContractCorpus.from_repo_root(tmp_path)
    correlation_id = uuid4()
    payload = {"marker": str(uuid4())}
    envelope = corpus.publisher_envelope(
        topic, payload=payload, correlation_id=correlation_id
    )
    assert corpus.publishers_of(topic) == ("node_publisher_dispatch",)
    assert envelope.event_type == "demo.publisher-dispatch"

    received: list[ModelEventEnvelope[object]] = []

    async def consume(message: object) -> None:
        # The runtime materializes binding/debug metadata alongside the JSON
        # envelope before calling a dispatcher.
        assert isinstance(message, dict)
        assert message["payload"] == payload
        consumed = current_dispatch_envelope()
        assert consumed is not None
        received.append(consumed)

    engine = MessageDispatchEngine()
    engine.register_dispatcher(
        dispatcher_id="publisher-dispatch",
        dispatcher=consume,
        category=category,
        message_types={"demo.publisher-dispatch"},
    )
    engine.register_route(
        ModelDispatchRoute(
            route_id="publisher-route",
            topic_pattern=topic,
            message_category=category,
            handler_id="publisher-dispatch",
        )
    )
    engine.freeze()
    bus = EventBusInmemory(environment="test")
    wiring = EventBusSubcontractWiring(
        event_bus=cast("ProtocolEventBusSubscriber", bus),
        dispatch_engine=engine,
        environment="test",
        node_name="publisher-consumer",
        service="publisher-dispatch",
        version="v1",
    )
    await bus.start()
    try:
        await wiring.wire_subscriptions(
            ModelEventBusSubcontract(
                version=ModelSemVer(major=1, minor=0, patch=0),
                subscribe_topics=[topic],
            ),
            node_name="publisher-consumer",
        )
        await bus.publish_envelope(envelope, topic)
        assert len(received) == 1
        assert received[0].payload == payload
        assert received[0].correlation_id == correlation_id
        assert received[0].event_type == "demo.publisher-dispatch"

        # The positive delivery must not turn undeclared topics into publishers.
        with pytest.raises(PublisherTopicError, match="No contract"):
            corpus.publisher_envelope("onex.evt.demo.undeclared.v1", payload=payload)
        assert len(received) == 1
    finally:
        await wiring.cleanup()
        await bus.close()
