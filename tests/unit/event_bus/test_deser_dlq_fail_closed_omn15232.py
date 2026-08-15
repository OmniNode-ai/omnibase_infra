# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-15232: the deserialization DLQ path in ``EventBusKafka`` must be fail-closed.

Defect (same class as OMN-14936, at a call site that fix did not cover):
``EventBusKafka._consume_loop`` awaited ``_publish_raw_to_dlq(...)`` on the
deserialization-failure path and DISCARDED the returned ``bool``, then
``continue``d. The consumers built by this class run with
``enable_auto_commit=self._config.enable_auto_commit``, which defaults to
``True`` (``omnibase_core`` ``model_event_bus_config.py``), so the offset
advances on the poll cadence regardless of whether the DLQ write ever landed.
A failed DLQ publish on a poison message was therefore a silent, committed,
unrecoverable drop.

The OMN-14936 gate landed only in ``runtime/event_bus_subcontract_wiring.py``
(5x ``if not dlq_persisted: return``). That path is a *manual-commit* model —
withholding the commit is enough there. This loop has no commit call at all;
the client commits for it. So the fail-closed action here is a rewind of the
fetch position (``consumer.seek(tp, msg.offset)``), which is the same "does NOT
advance the committed offset" idiom ``KafkaTransport.nack`` already uses in
``event_bus/kafka_transport.py``. With auto-commit ON, the auto-committer
commits the *position*, so a rewound position cannot advance past the
un-persisted message; with auto-commit OFF it forces redelivery. Both models
end up fail-closed through one mechanism.

These tests drive the real ``_consume_loop`` against a fake consumer, so they
exercise the artifact that runs rather than a surrogate.
"""

from __future__ import annotations

import ast
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from aiokafka.structs import TopicPartition

import omnibase_infra.event_bus.event_bus_kafka as event_bus_kafka_module
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

TEST_TOPIC: str = "omn15232-source-topic"
TEST_GROUP: str = "test.test-service.omn15232.consume.v1"
TEST_PARTITION: int = 3
TEST_OFFSET: int = 4242


class _FakeConsumer:
    """Minimal async-iterable stand-in for ``AIOKafkaConsumer``.

    Records ``seek`` calls so a test can assert whether the loop rewound the
    fetch position (fail-closed) or let it advance (fail-open).
    """

    def __init__(self, messages: list[Any]) -> None:
        self._messages = list(messages)
        self.seek_calls: list[tuple[TopicPartition, int]] = []

    def __aiter__(self) -> _FakeConsumer:
        return self

    async def __anext__(self) -> Any:
        if not self._messages:
            raise StopAsyncIteration
        return self._messages.pop(0)

    def seek(self, topic_partition: TopicPartition, offset: int) -> None:
        self.seek_calls.append((topic_partition, offset))

    async def stop(self) -> None:
        """No-op; the loop's cleanup path calls this on shutdown."""


def _make_raw_msg() -> MagicMock:
    msg = MagicMock()
    msg.topic = TEST_TOPIC
    msg.partition = TEST_PARTITION
    msg.offset = TEST_OFFSET
    msg.key = b"omn15232-key"
    msg.value = b"\x00not-json"
    msg.headers = ()
    return msg


@pytest.fixture
def dlq_config() -> ModelKafkaEventBusConfig:
    return ModelKafkaEventBusConfig(
        bootstrap_servers="localhost:9092",
        environment="dev",
        dead_letter_topic="dlq-events",
    )


@pytest.fixture
def mock_producer() -> AsyncMock:
    producer = AsyncMock()
    producer.start = AsyncMock()
    producer.stop = AsyncMock()
    producer.send_and_wait = AsyncMock()
    producer._closed = False
    return producer


async def _run_consume_loop_with_dlq_result(
    dlq_config: ModelKafkaEventBusConfig,
    mock_producer: AsyncMock,
    *,
    dlq_result: bool,
) -> tuple[_FakeConsumer, list[Any]]:
    """Drive one poison message through the real ``_consume_loop``.

    Returns the fake consumer (for ``seek`` assertions) and the recorded
    ``_publish_raw_to_dlq`` call list.
    """
    with patch(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
        return_value=mock_producer,
    ):
        event_bus = EventBusKafka(config=dlq_config)
        await event_bus.start()

        consumer = _FakeConsumer([_make_raw_msg()])
        event_bus._group_consumers[(TEST_TOPIC, TEST_GROUP)] = consumer  # type: ignore[assignment]

        # Force the deserialization failure this path exists to handle.
        def _boom(_msg: Any, _topic: str) -> Any:
            raise ValueError("undeserializable payload")

        dlq_calls: list[Any] = []

        async def _fake_publish_raw_to_dlq(**kwargs: Any) -> bool:
            dlq_calls.append(kwargs)
            return dlq_result

        with (
            patch.object(event_bus, "_kafka_msg_to_model", side_effect=_boom),
            patch.object(
                event_bus, "_publish_raw_to_dlq", side_effect=_fake_publish_raw_to_dlq
            ),
            # Keep the fail-closed backoff from slowing the suite.
            patch.object(
                event_bus_kafka_module,
                "DLQ_UNPERSISTED_REWIND_BACKOFF_SECONDS",
                0.0,
                create=True,
            ),
        ):
            await event_bus._consume_loop(TEST_TOPIC, TEST_GROUP, uuid4())

        await event_bus.close()
        return consumer, dlq_calls


@pytest.mark.asyncio
async def test_unpersisted_dlq_write_rewinds_offset_instead_of_advancing(
    dlq_config: ModelKafkaEventBusConfig, mock_producer: AsyncMock
) -> None:
    """RED against dev: an unconfirmed DLQ write must not let the offset advance.

    ``_publish_raw_to_dlq`` returns ``False`` (producer unavailable / send
    rejected / DLQ topic missing — all real OMN-14936 scenarios). Before the
    fix the loop discarded that ``False`` and ``continue``d, so the auto-commit
    task committed a position past a message that exists nowhere durable. The
    fail-closed behaviour is a rewind to ``msg.offset`` on that message's own
    partition.
    """
    consumer, dlq_calls = await _run_consume_loop_with_dlq_result(
        dlq_config, mock_producer, dlq_result=False
    )

    assert len(dlq_calls) == 1, "the deserialization path must attempt a DLQ write"
    assert dlq_calls[0]["failure_type"] == "deserialization_error"

    assert consumer.seek_calls, (
        "OMN-15232: DLQ persistence was NOT confirmed, yet the consume loop "
        "let the fetch position advance past the message. With "
        "enable_auto_commit defaulting to True this commits an offset past a "
        "message that was never written to the DLQ — a silent, unrecoverable "
        "drop. The loop must rewind (seek) instead."
    )
    assert consumer.seek_calls == [
        (TopicPartition(TEST_TOPIC, TEST_PARTITION), TEST_OFFSET)
    ], (
        "the rewind must target the failed message's own partition and its own "
        "offset — not a sibling partition and not offset+1"
    )


@pytest.mark.asyncio
async def test_confirmed_dlq_write_lets_offset_advance(
    dlq_config: ModelKafkaEventBusConfig, mock_producer: AsyncMock
) -> None:
    """A confirmed durable DLQ record must NOT stall the partition.

    The fix must be a gate, not a blanket rewind: when the DLQ write is
    confirmed the poison message is durably captured, so the offset is allowed
    to advance and the loop moves on.
    """
    consumer, dlq_calls = await _run_consume_loop_with_dlq_result(
        dlq_config, mock_producer, dlq_result=True
    )

    assert len(dlq_calls) == 1
    assert consumer.seek_calls == [], (
        "a confirmed DLQ persist must let the offset advance; rewinding here "
        "would replay poison messages forever"
    )


@pytest.mark.asyncio
async def test_handler_path_unpersisted_dlq_also_rewinds(
    dlq_config: ModelKafkaEventBusConfig, mock_producer: AsyncMock
) -> None:
    """The module audit required by OMN-15232 found a second ungated site.

    ``_dispatch_to_subscriber`` discarded the DLQ result on the
    retries-exhausted branch, so a handler failure whose DLQ write never landed
    was dropped exactly the same way. The mixin's ``_publish_to_dlq`` did not
    even return a persistence signal; it does now, and this path gates on it.
    """
    with patch(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
        return_value=mock_producer,
    ):
        event_bus = EventBusKafka(config=dlq_config)
        await event_bus.start()

        consumer = _FakeConsumer([_make_raw_msg()])
        event_bus._group_consumers[(TEST_TOPIC, TEST_GROUP)] = consumer  # type: ignore[assignment]

        message = MagicMock()
        message.headers.retry_count = 5
        message.headers.max_retries = 5

        async def _failing_callback(_message: Any) -> None:
            raise RuntimeError("handler blew up")

        event_bus._subscribers[TEST_TOPIC] = [  # type: ignore[assignment]
            (TEST_GROUP, "sub-1", _failing_callback)
        ]

        async def _dlq_never_persists(**_kwargs: Any) -> bool:
            return False

        with (
            patch.object(event_bus, "_kafka_msg_to_model", return_value=message),
            patch.object(event_bus, "_publish_to_dlq", side_effect=_dlq_never_persists),
            patch.object(
                event_bus_kafka_module,
                "DLQ_UNPERSISTED_REWIND_BACKOFF_SECONDS",
                0.0,
            ),
        ):
            await event_bus._consume_loop(TEST_TOPIC, TEST_GROUP, uuid4())

        await event_bus.close()

    assert consumer.seek_calls == [
        (TopicPartition(TEST_TOPIC, TEST_PARTITION), TEST_OFFSET)
    ], (
        "OMN-15232: retries were exhausted and the DLQ write was not confirmed, "
        "so the offset must not advance past the message"
    )


@pytest.mark.asyncio
async def test_publish_to_dlq_reports_persistence_outcome(
    dlq_config: ModelKafkaEventBusConfig, mock_producer: AsyncMock
) -> None:
    """``MixinKafkaDlq._publish_to_dlq`` must return a real persistence bool.

    It previously returned ``None`` unconditionally, so no caller *could* gate
    on it. Callers can only be fail-closed if the signal exists.
    """
    from omnibase_infra.event_bus.models import ModelEventHeaders, ModelEventMessage

    with patch(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
        return_value=mock_producer,
    ):
        event_bus = EventBusKafka(config=dlq_config)
        await event_bus.start()

        failed_message = ModelEventMessage(
            topic=TEST_TOPIC,
            key=b"k",
            value=b"v",
            headers=ModelEventHeaders(
                source="dev",
                event_type="test_event",
                correlation_id=uuid4(),
                timestamp=datetime.now(UTC),
            ),
            partition=TEST_PARTITION,
            offset=str(TEST_OFFSET),
        )

        ok = await event_bus._publish_to_dlq(
            original_topic=TEST_TOPIC,
            failed_message=failed_message,
            error=ValueError("handler failed"),
            correlation_id=uuid4(),
            consumer_group=TEST_GROUP,
        )
        assert ok is True, "a confirmed send must report True"

        mock_producer.send_and_wait.side_effect = RuntimeError("broker down")
        not_ok = await event_bus._publish_to_dlq(
            original_topic=TEST_TOPIC,
            failed_message=failed_message,
            error=ValueError("handler failed"),
            correlation_id=uuid4(),
            consumer_group=TEST_GROUP,
        )
        assert not_ok is False, (
            "a failed send (including the category-topic fallback) must report "
            "False so the caller can withhold offset advancement"
        )

        await event_bus.close()


def _dlq_publish_calls_with_discarded_result(source: str) -> list[int]:
    """Return line numbers of DLQ-publish calls whose return value is discarded.

    A call is "discarded" when it is the whole expression of an
    ``ast.Expr`` statement — i.e. nothing binds, tests, or returns the
    persistence bool.
    """
    tree = ast.parse(source)
    dlq_publishers = {"_publish_raw_to_dlq", "_publish_to_dlq"}
    discarded: list[int] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Expr):
            continue
        value = node.value
        if isinstance(value, ast.Await):
            value = value.value
        if not isinstance(value, ast.Call):
            continue
        func = value.func
        if isinstance(func, ast.Attribute) and func.attr in dlq_publishers:
            discarded.append(node.lineno)

    return discarded


def test_no_dlq_publish_call_site_discards_its_persistence_result() -> None:
    """Static guard: every DLQ publish in ``event_bus_kafka.py`` is checked.

    Acceptance criterion 3 on OMN-15232. ``_publish_raw_to_dlq`` documents that
    "callers that gate an offset commit on durable DLQ persistence (OMN-14936)
    MUST check this return value". This ratchets that contract at the module
    level so a future call site cannot silently reintroduce the fail-open drop.
    """
    module_path = Path(event_bus_kafka_module.__file__)
    discarded = _dlq_publish_calls_with_discarded_result(
        module_path.read_text(encoding="utf-8")
    )

    assert discarded == [], (
        f"{module_path.name} discards the DLQ persistence result at line(s) "
        f"{discarded}. Bind the returned bool and gate offset advancement on "
        f"it (OMN-15232 / OMN-14936) — an unchecked DLQ publish followed by an "
        f"offset commit is a silent data-loss path."
    )
