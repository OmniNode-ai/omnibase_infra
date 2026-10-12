# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21050: a command on the bus reaches a DUAL_WRITE lifecycle event on the bus.

Boots the REAL ``node_topic_migration_executor_effect`` contract from disk through
``wire_from_manifest`` (the call the kernel makes), delivers the command as a Kafka
message to the subscription the wiring registered, and reads back what the real
publisher sent. Only the broker producer and the two infrastructure collaborators
(topic provisioner, lag-observing drain gate) are replaced.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest

from omnibase_core.enums.enum_cutover_criterion import EnumCutoverCriterion
from omnibase_core.enums.enum_migration_phase import EnumMigrationPhase
from omnibase_core.models.contracts.model_topic_migration_contract import (
    ModelTopicMigrationContract,
)
from omnibase_core.models.contracts.model_topic_schema_binding import (
    ModelTopicSchemaBinding,
)
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config.model_kafka_event_bus_config import (
    ModelKafkaEventBusConfig,
)
from omnibase_infra.event_bus.models.model_event_headers import ModelEventHeaders
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.migration.service_drain_proof_gate import (
    ModelDrainProofDecision,
    ServiceDrainProofGate,
)
from omnibase_infra.nodes.node_topic_migration_executor_effect.models.model_topic_migration_command import (
    ModelTopicMigrationCommand,
)
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths
from omnibase_infra.runtime.auto_wiring.handler_wiring import wire_from_manifest
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
from omnibase_infra.topics.model_topic_spec import ModelTopicSpec

pytestmark = pytest.mark.asyncio

_CONTRACT_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "omnibase_infra"
    / "nodes"
    / "node_topic_migration_executor_effect"
    / "contract.yaml"
)
_COMMAND_TOPIC = "onex.cmd.omnibase-infra.topic-migration-execute.v1"
_LIFECYCLE_TOPIC = "onex.evt.omnibase-infra.topic-migration-lifecycle.v1"
_OLD_TOPIC = "onex.evt.labproof.omn21050-old.v1"
_NEW_TOPIC = "onex.evt.labproof.omn21050-new.v1"
_PATCH_DEPS = (
    "omnibase_infra.runtime.auto_wiring.handler_wiring."
    "_build_topic_migration_executor_dependencies"
)


class _RecordingProducer:
    """The aiokafka producer surface the publisher uses: ``send`` -> future of metadata."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, bytes]] = []

    async def send(
        self, topic: str, *, value: bytes, **kwargs: object
    ) -> asyncio.Future[object]:
        self.sent.append((topic, value))
        future: asyncio.Future[object] = asyncio.get_running_loop().create_future()
        future.set_result(SimpleNamespace(topic=topic, partition=0, offset=0))
        return future

    async def send_and_wait(
        self, topic: str, *, value: bytes, **kwargs: object
    ) -> object:
        self.sent.append((topic, value))
        return SimpleNamespace(topic=topic, partition=0, offset=0)


class _RecordingBus(EventBusKafka):
    """Real publisher, with a recorded subscription and broker acknowledgment."""

    def __init__(self, producer: _RecordingProducer) -> None:
        super().__init__(
            config=ModelKafkaEventBusConfig(bootstrap_servers="localhost:9092")
        )
        self._producer = producer
        self._started = True
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


class _RecordingProvisioner:
    def __init__(self) -> None:
        self.created: list[str] = []

    async def ensure_topic_exists(
        self,
        topic_name: str,
        config: object | None = None,
        correlation_id: UUID | None = None,
        *,
        spec: ModelTopicSpec | None = None,
    ) -> bool:
        self.created.append(topic_name)
        return True


class _FakeGate(ServiceDrainProofGate):
    def __init__(self) -> None:
        pass

    async def evaluate(
        self, contract: ModelTopicMigrationContract
    ) -> ModelDrainProofDecision:
        return ModelDrainProofDecision(
            migration_ticket=contract.ticket,
            old_consumer_group=contract.old_consumer_group,
            old_topic=contract.old_binding.topic,
            retirement_allowed=True,
            residual_lag=0,
            reason="fake gate decision",
        )


def _binding(topic: str) -> ModelTopicSchemaBinding:
    return ModelTopicSchemaBinding(
        topic=topic,
        event_name="OMN21050_PROOF",
        schema_version=ModelSemVer(major=1, minor=0, patch=0),
    )


def _dual_write_message() -> tuple[ModelEventMessage, UUID]:
    correlation = uuid4()
    command = ModelTopicMigrationCommand(
        correlation_id=correlation,
        contract=ModelTopicMigrationContract(
            contract_version=ModelSemVer(major=1, minor=0, patch=0),
            ticket="OMN-21050",
            old_binding=_binding(_OLD_TOPIC),
            new_binding=_binding(_NEW_TOPIC),
            old_consumer_group="local.labproof.omn21050-old.consume.v1",
            new_consumer_group="local.labproof.omn21050-new.consume.v1",
            compatibility_window_hours=1,
            cutover_criteria=(EnumCutoverCriterion.OLD_TOPIC_DRAINED,),
            phase=EnumMigrationPhase.PLANNED,
        ),
        target_phase=EnumMigrationPhase.DUAL_WRITE,
        new_topic_partitions=1,
    )
    envelope = ModelEventEnvelope[object](
        payload=command.model_dump(mode="json"), correlation_id=correlation
    )
    message = ModelEventMessage(
        topic=_COMMAND_TOPIC,
        key=None,
        value=envelope.model_dump_json().encode(),
        headers=ModelEventHeaders(
            timestamp=datetime.now(UTC),
            source="omn21050-test",
            event_type="execute_topic_migration",
            correlation_id=correlation,
        ),
    )
    return message, correlation


async def test_dual_write_command_on_the_bus_publishes_a_dual_write_lifecycle_event() -> (
    None
):
    producer = _RecordingProducer()
    bus = _RecordingBus(producer)
    provisioner = _RecordingProvisioner()
    manifest = discover_contracts_from_paths([_CONTRACT_PATH])
    assert not manifest.errors, manifest.errors

    engine = MessageDispatchEngine()
    with patch(
        _PATCH_DEPS,
        return_value={"provisioner": provisioner, "drain_proof_gate": _FakeGate()},
    ):
        report = await wire_from_manifest(
            manifest, engine, event_bus=bus, environment="local"
        )
    assert report.total_failed == 0, report.model_dump()
    assert report.total_wired == 1
    engine.freeze()
    assert _COMMAND_TOPIC in bus.callbacks

    message, correlation = _dual_write_message()
    await bus.callbacks[_COMMAND_TOPIC](message)

    assert provisioner.created == [_NEW_TOPIC]
    assert [topic for topic, _ in producer.sent] == [_LIFECYCLE_TOPIC]
    event = json.loads(producer.sent[0][1])["payload"]
    assert event["phase"] == EnumMigrationPhase.DUAL_WRITE.value
    assert event["correlation_id"] == str(correlation)
    assert event["new_topic"] == _NEW_TOPIC
    assert event["new_topic_provisioned"] is True
