# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-14936: a ``no_dispatcher`` drop must leave a durable DLQ record or rewind.

The wiring's own unit suite
(``tests/unit/runtime/test_event_bus_subcontract_wiring.py``) proves the callback
does not call ``commit_offset`` when the DLQ write is unconfirmed. That proves
nothing about the offset: ``EventBusKafka`` has no ``commit_offset`` method, and
its consumers run with ``enable_auto_commit=True`` (the config default), so the
client commits the FETCH POSITION on its own cadence. A callback that merely
declines to commit and returns normally IS an ACK (``_dispatch_to_subscriber``
reads "no exception" as success), and the record is committed past with no DLQ
copy -- the defect OMN-14936 was filed for, surviving its own fix.

These tests drive the artifact that runs: the real ``_consume_loop`` over a fake
consumer, the real ``EventBusSubcontractWiring`` dispatch callback, and the real
``EventBusKafka._publish_raw_to_dlq`` over a fake producer. The assertions are on
what the broker would see -- the records the producer was asked to write and the
fetch-position seeks the consumer received -- not on mock call counts.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from aiokafka.errors import UnknownTopicOrPartitionError
from aiokafka.structs import TopicPartition

import omnibase_infra.event_bus.event_bus_kafka as event_bus_kafka_module
from omnibase_infra.enums import EnumDispatchStatus
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.models.dispatch import ModelDispatchResult
from omnibase_infra.models.event_bus import ModelDlqConfig
from omnibase_infra.runtime.event_bus_subcontract_wiring import (
    EventBusSubcontractWiring,
)

pytestmark = pytest.mark.unit

# The live shape from the OMN-12912 P1 round-trip: a flat, un-namespaced
# event_type on a command topic, so the dispatch engine derives a per-event_type
# DLQ topic that nobody ever provisioned.
TEST_TOPIC: str = "onex.cmd.omnibase-infra.delegation-inference-request.v1"
NEVER_PROVISIONED_DLQ_TOPIC: str = (
    "onex.dlq.omnibase-infra.delegation-inference-request.v1"
)
CATEGORY_DLQ_TOPIC: str = "onex.dlq.omnibase-infra.commands.v1"
TEST_GROUP: str = "dev.test-handler"
TEST_PARTITION: int = 0
TEST_OFFSET: int = 200


class _FakeConsumer:
    """Async stand-in for ``AIOKafkaConsumer`` that records seeks."""

    def __init__(self, messages: list[Any]) -> None:
        self._messages = list(messages)
        self.seek_calls: list[tuple[TopicPartition, int]] = []
        self.on_drained: Callable[[], None] | None = None

    async def getmany(
        self,
        *partitions: TopicPartition,
        timeout_ms: int = 0,
        max_records: int | None = None,
    ) -> dict[TopicPartition, list[Any]]:
        if not self._messages:
            if self.on_drained is not None:
                self.on_drained()
            return {}
        return {TopicPartition(TEST_TOPIC, TEST_PARTITION): [self._messages.pop(0)]}

    def assignment(self) -> set[TopicPartition]:
        return set()

    def seek(self, partition: TopicPartition, offset: int) -> None:
        self.seek_calls.append((partition, offset))

    async def stop(self) -> None:
        return None


def _make_raw_msg() -> MagicMock:
    payload = {
        "event_type": "delegation-inference-request",
        "correlation_id": "3fc1fa18-ec2a-46bc-b840-ab0d9a2f0875",
        "payload": {"prompt": "hello"},
    }
    msg = MagicMock()
    msg.topic = TEST_TOPIC
    msg.partition = TEST_PARTITION
    msg.offset = TEST_OFFSET
    msg.timestamp = int(datetime.now(UTC).timestamp() * 1000)
    msg.key = None
    msg.headers = []
    msg.value = json.dumps(payload).encode("utf-8")
    return msg


def _no_dispatcher_result() -> ModelDispatchResult:
    return ModelDispatchResult(
        status=EnumDispatchStatus.NO_DISPATCHER,
        topic=TEST_TOPIC,
        started_at=datetime.now(UTC),
        correlation_id=uuid4(),
        error_message="No dispatcher found for delegation-inference-request",
        error_details={"failure_class": "no_dispatcher"},
        dlq_topic=NEVER_PROVISIONED_DLQ_TOPIC,
    )


async def _consume_one_unroutable_record(
    send_and_wait: AsyncMock,
) -> tuple[_FakeConsumer, AsyncMock]:
    """Drive the real consume loop over one record no dispatcher handles."""
    producer = AsyncMock()
    producer.start = AsyncMock()
    producer.stop = AsyncMock()
    producer.send_and_wait = send_and_wait

    engine = AsyncMock()
    engine.dispatch = AsyncMock(return_value=_no_dispatcher_result())

    config = ModelKafkaEventBusConfig(
        bootstrap_servers="localhost:9092", environment="dev"
    )
    assert config.enable_auto_commit is True, (
        "the production default; the whole point of these tests is that it holds"
    )

    with patch(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
        return_value=producer,
    ):
        event_bus = EventBusKafka(config=config)
        await event_bus.start()

        wiring = EventBusSubcontractWiring(
            event_bus=cast("Any", event_bus),
            dispatch_engine=engine,
            environment="dev",
            node_name="test-handler",
            service="test-service",
            version="v1",
            dlq_config=ModelDlqConfig(enabled=True, on_content_error="dlq_and_commit"),
        )
        callback = wiring._create_dispatch_callback(TEST_TOPIC, TEST_GROUP)

        consumer = _FakeConsumer([_make_raw_msg()])
        event_bus._group_consumers[(TEST_TOPIC, TEST_GROUP)] = cast("Any", consumer)
        consumer.on_drained = lambda: setattr(event_bus, "_shutdown", True)
        event_bus._subscribers[TEST_TOPIC] = [
            (TEST_GROUP, "sub-1", cast("Any", callback))
        ]

        with patch.object(
            event_bus_kafka_module, "DLQ_UNPERSISTED_REWIND_BACKOFF_SECONDS", 0.0
        ):
            await event_bus._consume_loop(TEST_TOPIC, TEST_GROUP, uuid4())

        await event_bus.close()

    return consumer, producer.send_and_wait


@pytest.mark.asyncio
async def test_no_dispatcher_with_blocked_dlq_write_rewinds_the_fetch_position() -> (
    None
):
    """AC2: DLQ write blocked -> the offset must NOT advance.

    The producer refuses every send. Before the fix the wiring callback logged
    ``dlq_publish_not_persisted`` and returned normally; the consume loop read
    that as success and left the fetch position past the record for
    auto-commit to commit -- zero seeks, zero DLQ records, offset committed.
    """
    send_and_wait = AsyncMock(
        side_effect=UnknownTopicOrPartitionError("DLQ topic does not exist")
    )

    consumer, sent = await _consume_one_unroutable_record(send_and_wait)

    assert sent.await_count >= 1, "the DLQ write was attempted"
    assert consumer.seek_calls == [
        (TopicPartition(TEST_TOPIC, TEST_PARTITION), TEST_OFFSET)
    ], (
        "OMN-14936: with the DLQ write unconfirmed the consumer must rewind to "
        "the failed record's own offset; under enable_auto_commit=True, "
        "declining to call commit_offset withholds nothing"
    )


@pytest.mark.asyncio
async def test_no_dispatcher_leaves_a_durable_dlq_record_then_advances() -> None:
    """AC1: an unroutable record is DLQ'd durably, then the offset may advance.

    The per-event_type DLQ topic the dispatch engine names does not exist on the
    broker (the live defect), so the write must land on the already-provisioned
    category topic, carry the topic, the original payload and the
    ``no_dispatcher`` failure class, and only then leave the position alone.
    """
    record_metadata = MagicMock()
    record_metadata.partition = 0
    record_metadata.offset = 7
    accepted: list[tuple[str, dict[str, Any]]] = []

    async def _send_and_wait(topic: str, *args: Any, **kwargs: Any) -> MagicMock:
        if topic == NEVER_PROVISIONED_DLQ_TOPIC:
            raise UnknownTopicOrPartitionError(f"{topic} does not exist")
        value = kwargs.get("value", args[0] if args else b"")
        accepted.append((topic, json.loads(value)))
        return record_metadata

    consumer, _sent = await _consume_one_unroutable_record(
        AsyncMock(side_effect=_send_and_wait)
    )

    assert len(accepted) == 1
    dlq_topic, dlq_record = accepted[0]
    assert dlq_topic == CATEGORY_DLQ_TOPIC
    assert dlq_record["original_topic"] == TEST_TOPIC
    assert dlq_record["failure_class"] == "no_dispatcher"
    assert "delegation-inference-request" in dlq_record["original_message"]["value"]
    assert consumer.seek_calls == [], (
        "a confirmed DLQ record is the condition under which the offset advances"
    )
