# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17379: an unmaterialized projection must rewind, not advance the offset.

The sibling unit suite
(``tests/unit/runtime/auto_wiring/test_omn17379_projection_silent_ack.py``)
proves the dispatch callback now RAISES instead of swallowing. That alone fixes
nothing: ``_dispatch_to_subscriber`` is where an exception becomes — or fails to
become — a withheld offset, and its pre-existing arms both end in ``return
True``.

The load-bearing one is the ``retries_exhausted is False`` branch. It logs
"Handler failed but retries available (0/5)" and returns ``True``. There is no
requeue behind that sentence: nothing re-invokes the callback, so "retries
available" means the record is dropped with a WARNING and not even a dead-letter
copy. A ``ProjectionNotMaterializedError`` arriving with a fresh retry budget —
which is every first delivery — would take exactly that branch.

These tests drive the real ``_consume_loop`` against a fake consumer, so they
exercise the artifact that runs. The assertion is on ``consumer.seek_calls``,
because under ``enable_auto_commit=True`` a rewind of the fetch position is the
only action that withholds the offset; declining to commit does nothing, since
this loop never commits — the client does, on its own cadence.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from aiokafka.structs import TopicPartition

import omnibase_infra.event_bus.event_bus_kafka as event_bus_kafka_module
from omnibase_infra.errors import ProjectionNotMaterializedError
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

TEST_TOPIC: str = "onex.evt.github.pr-merged.v1"
TEST_GROUP: str = "local.omnimarket.pr_merged_projection.consume.1.0.0"
TEST_PARTITION: int = 0
# Offset 96 on the live dev-lane topic: PR #2159, merged 2026-08-27T04:27:37Z.
TEST_OFFSET: int = 96


class _FakeConsumer:
    """Async-iterable stand-in for ``AIOKafkaConsumer`` that records seeks."""

    def __init__(self, messages: list[Any]) -> None:
        self._messages = list(messages)
        self.seek_calls: list[tuple[TopicPartition, int]] = []

    def __aiter__(self) -> _FakeConsumer:
        return self

    async def __anext__(self) -> Any:
        if not self._messages:
            raise StopAsyncIteration
        return self._messages.pop(0)

    def seek(self, partition: TopicPartition, offset: int) -> None:
        self.seek_calls.append((partition, offset))

    async def stop(self) -> None:
        return None


def _make_raw_msg() -> MagicMock:
    msg = MagicMock()
    msg.topic = TEST_TOPIC
    msg.partition = TEST_PARTITION
    msg.offset = TEST_OFFSET
    msg.timestamp = int(datetime.now(UTC).timestamp() * 1000)
    msg.key = None
    msg.headers = []
    msg.value = b'{"pr_number": 2159, "ticket": "OMN-16589"}'
    return msg


@pytest.fixture
def kafka_config() -> ModelKafkaEventBusConfig:
    return ModelKafkaEventBusConfig(bootstrap_servers="localhost:9092")


@pytest.fixture
def mock_producer() -> AsyncMock:
    producer = AsyncMock()
    producer.start = AsyncMock()
    producer.stop = AsyncMock()
    producer.send_and_wait = AsyncMock()
    return producer


async def _run_loop_with_callback_error(
    config: ModelKafkaEventBusConfig,
    producer: AsyncMock,
    error: Exception,
    *,
    retry_count: int,
    max_retries: int,
) -> tuple[_FakeConsumer, list[dict[str, Any]]]:
    """Drive the real consume loop over one message whose callback raises."""
    dlq_calls: list[dict[str, Any]] = []

    with patch(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
        return_value=producer,
    ):
        event_bus = EventBusKafka(config=config)
        await event_bus.start()

        consumer = _FakeConsumer([_make_raw_msg()])
        event_bus._group_consumers[(TEST_TOPIC, TEST_GROUP)] = consumer  # type: ignore[assignment]

        message = MagicMock()
        message.headers.retry_count = retry_count
        message.headers.max_retries = max_retries

        async def _raising_callback(_message: Any) -> None:
            raise error

        event_bus._subscribers[TEST_TOPIC] = [  # type: ignore[assignment]
            (TEST_GROUP, "sub-1", _raising_callback)
        ]

        async def _record_dlq(**kwargs: Any) -> bool:
            dlq_calls.append(kwargs)
            return True

        with (
            patch.object(event_bus, "_kafka_msg_to_model", return_value=message),
            patch.object(event_bus, "_publish_to_dlq", side_effect=_record_dlq),
            patch.object(
                event_bus_kafka_module,
                "DLQ_UNPERSISTED_REWIND_BACKOFF_SECONDS",
                0.0,
            ),
        ):
            await event_bus._consume_loop(TEST_TOPIC, TEST_GROUP, uuid4())

        await event_bus.close()

    return consumer, dlq_calls


@pytest.mark.asyncio
async def test_unmaterialized_projection_rewinds_on_first_delivery(
    kafka_config: ModelKafkaEventBusConfig, mock_producer: AsyncMock
) -> None:
    """The exact live shape: first delivery, full retry budget, zero rows written.

    Before the fix this took the "retries available" branch and returned ``True``
    — the branch that let ``pr_merged_events`` acknowledge 230 merged PRs into
    nothing while its group reported Stable at TOTAL-LAG 0.
    """
    consumer, dlq_calls = await _run_loop_with_callback_error(
        kafka_config,
        mock_producer,
        ProjectionNotMaterializedError(
            "permission denied for sequence pr_merged_events_projection_cursor_seq"
        ),
        retry_count=0,
        max_retries=5,
    )

    assert consumer.seek_calls == [
        (TopicPartition(TEST_TOPIC, TEST_PARTITION), TEST_OFFSET)
    ], (
        "OMN-17379: a projection that wrote no row for a non-content reason must "
        "rewind to the failed message's own offset. Under auto-commit, letting "
        "this return True commits past an event that was never materialized."
    )
    assert dlq_calls == [], (
        "the record is preserved by withholding the offset, not by a dead-letter "
        "copy — it is still on its own topic, uncommitted"
    )


@pytest.mark.asyncio
async def test_unmaterialized_projection_rewinds_with_budget_exhausted(
    kafka_config: ModelKafkaEventBusConfig, mock_producer: AsyncMock
) -> None:
    """Offset-unsafe independently of the retry budget.

    The retry budget answers "should this be re-attempted in-process". It cannot
    answer "is this record safe to forget" — a broken write path stays broken
    until an operator repairs it, at which point redelivery materializes the row.
    """
    consumer, dlq_calls = await _run_loop_with_callback_error(
        kafka_config,
        mock_producer,
        ProjectionNotMaterializedError("connection refused"),
        retry_count=5,
        max_retries=5,
    )

    assert consumer.seek_calls == [
        (TopicPartition(TEST_TOPIC, TEST_PARTITION), TEST_OFFSET)
    ]
    assert dlq_calls == []


@pytest.mark.asyncio
async def test_ordinary_handler_failure_keeps_its_prior_semantics(
    kafka_config: ModelKafkaEventBusConfig, mock_producer: AsyncMock
) -> None:
    """The blast radius is bounded to the new type.

    A generic handler failure with retries exhausted still DLQs and, on a
    confirmed persist, still lets the offset advance (OMN-15232). This fix
    narrows one class; it does not turn every consumer failure into a stall.
    """
    consumer, dlq_calls = await _run_loop_with_callback_error(
        kafka_config,
        mock_producer,
        RuntimeError("handler blew up"),
        retry_count=5,
        max_retries=5,
    )

    assert len(dlq_calls) == 1
    assert consumer.seek_calls == []
