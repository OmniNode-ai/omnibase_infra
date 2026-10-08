# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18879: the consume boundary must honor the contract's DLQ address.

Drive manifest wiring, real dispatch and the real Kafka DLQ publisher together.
Only the broker producer is replaced. The local handler mirrors the census
fold's rejection so this regression runs without a sibling repository.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest
from pydantic import BaseModel

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config.model_kafka_event_bus_config import (
    ModelKafkaEventBusConfig,
)
from omnibase_infra.event_bus.models.model_event_headers import ModelEventHeaders
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.event_bus.topic_constants import get_dlq_topic_for_original
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    BoundaryDlqNotPersistedError,
    wire_from_manifest,
)
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
    ModelHandlerRef,
    ModelHandlerRouting,
    ModelHandlerRoutingEntry,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

pytestmark = pytest.mark.asyncio

SOURCE_TOPIC = "onex.evt.omnibase-infra.lane-census-observed.v1"
DECLARED_DLQ = "onex.dlq.omnimarket.projection-lab-lane-health-malformed.v1"
_MODULE = "tests.unit.runtime.auto_wiring.test_contract_declared_dlq_omn18879"


class ModelCensusRequest(BaseModel):
    lanes_checked: list[str]
    observed_at: str | None = None


class ModelCensusResult(BaseModel):
    observed_at: str


class LaneHealthFoldError(ValueError):
    """Mirror of the domain fold's error, without a cross-repo dependency."""


class HandlerCensus:
    def handle(self, request: ModelCensusRequest) -> ModelCensusResult:
        if request.observed_at is None:
            raise LaneHealthFoldError("observed_at is required and must be a timestamp")
        return ModelCensusResult(observed_at=request.observed_at)


class RecordingProducer:
    def __init__(self, fail_topic: str | None = None) -> None:
        self.sent: list[tuple[str, bytes]] = []
        self.attempted: list[str] = []
        self.fail_topic = fail_topic

    async def send_and_wait(
        self, topic: str, *, value: bytes, **kwargs: object
    ) -> object:
        self.attempted.append(topic)
        if topic == self.fail_topic:
            raise RuntimeError("declared DLQ unavailable")
        self.sent.append((topic, value))
        return object()


class RecordingKafkaBus(EventBusKafka):
    """Real publisher, with a recorded subscription and broker acknowledgment."""

    def __init__(
        self, producer: RecordingProducer, *, override: str | None = None
    ) -> None:
        super().__init__(
            config=ModelKafkaEventBusConfig(
                bootstrap_servers="localhost:9092", dead_letter_topic=override
            )
        )
        self._producer = producer
        self.callbacks: dict[str, Callable[..., Awaitable[None]]] = {}

    async def subscribe(
        self,
        topic: str,
        node_identity: object | None = None,
        on_message: Callable[[ModelEventMessage], Awaitable[None]] | None = None,
        **kwargs: object,
    ) -> Callable[[], Awaitable[None]]:
        assert on_message is not None
        self.callbacks[topic] = on_message

        async def unsubscribe() -> None:
            return None

        return unsubscribe


async def _wire(
    tmp_path: Path,
    bus: RecordingKafkaBus,
    *,
    typed_declaration: bool = True,
    declared: bool = True,
) -> Callable[..., Awaitable[None]]:
    path = tmp_path / "contract.yaml"
    dlq_yaml = f"  dlq_topics: [{DECLARED_DLQ}]\n" if declared else ""
    path.write_text(
        f"name: node_census_effect\nnode_type: EFFECT_GENERIC\nevent_bus:\n"
        f"  subscribe_topics: [{SOURCE_TOPIC}]\n  publish_topics: []\n{dlq_yaml}"
    )
    contract = ModelDiscoveredContract(
        name="node_census_effect",
        node_type="EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=path,
        entry_point_name="node_census_effect",
        package_name="omnimarket",
        event_bus=ModelEventBusWiring(
            subscribe_topics=(SOURCE_TOPIC,),
            dlq_topics=(DECLARED_DLQ,) if declared and typed_declaration else (),
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="payload_type_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(name="HandlerCensus", module=_MODULE),
                    event_model=ModelHandlerRef(
                        name="ModelCensusRequest", module=_MODULE
                    ),
                ),
            ),
        ),
    )
    engine = MessageDispatchEngine()
    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        return_value=HandlerCensus,
    ):
        report = await wire_from_manifest(
            ModelAutoWiringManifest(contracts=(contract,)),
            engine,
            event_bus=bus,
            environment="local",
        )
    assert report.total_failed == 0, report.model_dump()
    assert report.total_wired == 1
    engine.freeze()
    assert set(bus.callbacks) == {SOURCE_TOPIC}
    return bus.callbacks[SOURCE_TOPIC]


def _message() -> ModelEventMessage:
    correlation = uuid4()
    envelope = ModelEventEnvelope[object](
        payload={"lanes_checked": ["dev"]}, correlation_id=correlation
    )
    return ModelEventMessage(
        topic=SOURCE_TOPIC,
        key=None,
        value=envelope.model_dump_json().encode(),
        headers=ModelEventHeaders(
            timestamp=datetime.now(UTC),
            source="omn18879-test",
            event_type="lane-census-observed",
            correlation_id=correlation,
        ),
    )


@pytest.fixture(autouse=True)
def _dlq_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ONEX_BOUNDARY_DLQ_ENABLED", "true")


@pytest.mark.parametrize("typed_declaration", [True, False])
async def test_rejection_lands_once_on_declared_dlq_despite_reply_suppression(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, typed_declaration: bool
) -> None:
    producer = RecordingProducer()
    bus = RecordingKafkaBus(producer)
    callback = await _wire(tmp_path, bus, typed_declaration=typed_declaration)
    message = _message()
    with caplog.at_level(logging.ERROR):
        await callback(message)
    assert [topic for topic, _ in producer.sent] == [DECLARED_DLQ]
    dead_letter = json.loads(producer.sent[0][1])
    assert dead_letter["original_topic"] == SOURCE_TOPIC
    assert dead_letter["correlation_id"] == str(message.headers.correlation_id)
    assert "observed_at is required" in dead_letter["failure_reason"]
    assert "dlq_routed=true" in caplog.text
    assert f"dlq_topic={DECLARED_DLQ}" in caplog.text
    assert "Boundary failure terminal suppressed" in caplog.text


async def test_declared_dlq_failure_does_not_claim_category_fallback_success(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    producer = RecordingProducer(fail_topic=DECLARED_DLQ)
    callback = await _wire(tmp_path, RecordingKafkaBus(producer))
    with caplog.at_level(logging.ERROR), pytest.raises(BoundaryDlqNotPersistedError):
        await callback(_message())
    assert producer.sent == []
    assert producer.attempted == [DECLARED_DLQ]
    assert "dlq_routed=false" in caplog.text
    assert "dlq_routed=true" not in caplog.text


async def test_bus_default_cannot_override_contract_dlq(tmp_path: Path) -> None:
    producer = RecordingProducer()
    callback = await _wire(
        tmp_path,
        RecordingKafkaBus(producer, override=get_dlq_topic_for_original(SOURCE_TOPIC)),
    )
    await callback(_message())
    assert [topic for topic, _ in producer.sent] == [DECLARED_DLQ]


async def test_undeclared_contract_keeps_category_routing(tmp_path: Path) -> None:
    producer = RecordingProducer()
    callback = await _wire(tmp_path, RecordingKafkaBus(producer), declared=False)
    await callback(_message())
    assert [topic for topic, _ in producer.sent] == [
        get_dlq_topic_for_original(SOURCE_TOPIC)
    ]
