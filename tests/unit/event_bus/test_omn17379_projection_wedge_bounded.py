# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17379: the offset withhold is BOUNDED, so one poison record cannot wedge
a partition forever.

WHAT THE WITHHOLD IS FOR, AND WHERE IT OVERSHOT. OMN-17379 made a projection
that consumed an event and wrote no row rewind instead of acknowledging, because
`pr_merged_events` had acknowledged 230 merged PRs into nothing while its group
reported Stable at lag 0. That is correct for a TRANSIENT write-path failure --
a missing GRANT, a dead database, a not-yet-applied migration -- where the
record is still owed a row and redelivery materialises it once an operator
repairs the path.

It is NOT correct for a record that fails the SAME way every time. There was no
ceiling: ``_dispatch_to_subscriber`` returned ``False`` unconditionally for
``ProjectionNotMaterializedError``, so the same record was re-read, re-refused
and re-rewound forever, and every later record on that partition -- including
records that would have projected fine -- was never delivered at all.

MEASURED, NOT HYPOTHESISED. The onex-dev staging namespace, 2026-09-15T16:43Z. The delegation projection writer pod was head-of-line blocked
on two records at once, each re-refusing roughly once per second:

    delegation-completed.v1   partition 0 offset 286  TenantRegistryResolutionError
    quality-gate-result.v1    partition 0 offset 300  NotNullViolation on task_type

Both are properties of the EVENT or of a registry state no redelivery changes,
and both were classified POISON by omnimarket's own
``projection/error_classification.py`` -- which this deployment path never
consults, because the writer is dispatched by this runtime's auto-wiring rather
than by ``BaseProjectionRunner``. The consequence was not two lost rows. It was
a stopped plane: the OMN-15256 staging business proof failed its quality_gate
check on run 34989725632 and would have failed on every subsequent deploy.

THE FIX, AND ITS TWO HALVES. A record whose withhold repeats with the SAME
failure fingerprint more than ``projection_withhold_max_redeliveries`` times is
dead-lettered with a typed reason and the offset advances. A record whose
failure CHANGES, or that has not yet reached the bound, still rewinds exactly as
before. The second half is not decoration -- without it this change would
re-open OMN-17379 by turning every transient failure into a discarded record,
which is the defect the withhold exists to prevent. Every test below that
asserts the new behaviour has a paired negative control that asserts the old one
still holds.

The dead-letter is gated on CONFIRMED persistence, per OMN-15232: if the
quarantine write is not confirmed durable the partition keeps stalling, because
a record that exists nowhere is worse than a stalled feed.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from aiokafka.structs import TopicPartition

import omnibase_infra.event_bus.event_bus_kafka as event_bus_kafka_module
from omnibase_infra.enums import EnumDlqFailureClass
from omnibase_infra.errors import ProjectionNotMaterializedError
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

TEST_TOPIC: str = "onex.evt.omnibase-infra.quality-gate-result.v1"
TEST_GROUP: str = "local.omnimarket.delegation_projection.consume.1.0.0"
TEST_PARTITION: int = 0
#: The live wedged coordinate on onex-dev, 2026-09-15.
TEST_OFFSET: int = 300

#: The live failure at that offset, verbatim from the pod's own log line.
LIVE_NOT_NULL = (
    'null value in column "task_type" of relation "delegation_events" '
    "violates not-null constraint"
)
#: The live failure at delegation-completed.v1 offset 286, same pod, same second.
LIVE_TENANT_REFUSAL = (
    "OMN-16831: no row in tenant_registry_mirror whose tenant_uuid is "
    "'af432b87-4390-4c76-af97-ee9e158936e7'"
)


class _FakeConsumer:
    """Async-iterable stand-in for ``AIOKafkaConsumer`` that records seeks."""

    def __init__(self, messages: list[Any]) -> None:
        self._messages = list(messages)
        self.seek_calls: list[tuple[TopicPartition, int]] = []
        # OMN-18640: the loop now polls with a deadline instead of iterating,
        # because an iterator over a wedged consumer never returns. A real
        # ``getmany`` has no end-of-stream, so the driver supplies this hook to
        # end the loop once the fixture's records are drained.
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
        # One record per fetch. A rewind drops the rest of its partition's
        # batch, so a fixture that models redelivery as "the next record in
        # the list" must deliver each one in its own fetch -- which is also
        # what a real refetch after a seek does.
        return {TopicPartition(TEST_TOPIC, TEST_PARTITION): [self._messages.pop(0)]}

    def assignment(self) -> set[TopicPartition]:
        return set()

    def seek(self, partition: TopicPartition, offset: int) -> None:
        self.seek_calls.append((partition, offset))

    async def stop(self) -> None:
        return None


def _make_raw_msg(*, offset: int = TEST_OFFSET) -> MagicMock:
    msg = MagicMock()
    msg.topic = TEST_TOPIC
    msg.partition = TEST_PARTITION
    msg.offset = offset
    msg.timestamp = int(datetime.now(UTC).timestamp() * 1000)
    msg.key = None
    msg.headers = []
    msg.value = b'{"correlation_id": "fc3d267f-dcb2-44c8-ac6f-9285a9b5e827"}'
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


async def _redeliver(
    config: ModelKafkaEventBusConfig,
    producer: AsyncMock,
    errors: list[Exception],
    *,
    offsets: list[int] | None = None,
    dlq_persists: bool = True,
    retry_count: int = 0,
) -> tuple[_FakeConsumer, list[dict[str, Any]]]:
    """Drive the real consume loop over N deliveries of the SAME record.

    ``errors`` supplies one exception per delivery, so a test can change the
    failure mid-stream and observe whether the counter resets. Kafka's own
    redelivery after a rewind is modelled by enqueuing the same message again,
    which is exactly what the broker does with a rewound fetch position.
    """
    dlq_calls: list[dict[str, Any]] = []
    coords = offsets or [TEST_OFFSET] * len(errors)
    remaining = list(errors)

    with patch(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
        return_value=producer,
    ):
        event_bus = EventBusKafka(config=config)
        await event_bus.start()

        consumer = _FakeConsumer([_make_raw_msg(offset=o) for o in coords])
        event_bus._group_consumers[(TEST_TOPIC, TEST_GROUP)] = consumer  # type: ignore[assignment]
        consumer.on_drained = lambda: setattr(event_bus, "_shutdown", True)

        message = MagicMock()
        message.headers.retry_count = retry_count
        message.headers.max_retries = 5

        async def _raising_callback(_message: Any) -> None:
            exc = remaining.pop(0)
            if exc is not None:
                raise exc

        event_bus._subscribers[TEST_TOPIC] = [  # type: ignore[assignment]
            (TEST_GROUP, "sub-1", _raising_callback)
        ]

        async def _record_dlq(**kwargs: Any) -> bool:
            dlq_calls.append(kwargs)
            return dlq_persists

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


def _not_null() -> ProjectionNotMaterializedError:
    return ProjectionNotMaterializedError(
        f"projection handler HandlerProjectionDelegation consumed an event from "
        f"{TEST_TOPIC} and wrote no row because its write path failed "
        f"(NotNullViolation: {LIVE_NOT_NULL}); the offset must not advance "
        "(OMN-17379)",
        projection_type="HandlerProjectionDelegation",
    )


def _bound(config: ModelKafkaEventBusConfig) -> int:
    return config.projection_withhold_max_redeliveries


class TestTheBoundIsDeclaredNotMagic:
    def test_the_bound_is_a_contract_config_field(
        self, kafka_config: ModelKafkaEventBusConfig
    ) -> None:
        """A number compiled into the source is a number nobody can change on a
        lane that needs it different."""
        assert "projection_withhold_max_redeliveries" in type(kafka_config).model_fields
        assert _bound(kafka_config) >= 1

    def test_the_bound_is_configurable_upward(self) -> None:
        raised = ModelKafkaEventBusConfig(
            bootstrap_servers="localhost:9092",
            projection_withhold_max_redeliveries=9,
        )
        assert _bound(raised) == 9

    def test_the_bound_cannot_be_set_to_zero(self) -> None:
        """Zero would dead-letter on the FIRST refusal, which is OMN-17379 in
        reverse: a transient failure would discard a record still owed a row."""
        with pytest.raises(ValueError):
            ModelKafkaEventBusConfig(
                bootstrap_servers="localhost:9092",
                projection_withhold_max_redeliveries=0,
            )


class TestARepeatingRefusalIsEventuallyDeadLettered:
    @pytest.mark.asyncio
    async def test_the_partition_advances_after_the_bound(
        self, kafka_config: ModelKafkaEventBusConfig, mock_producer: AsyncMock
    ) -> None:
        """The wedge, ended. This is the live onex-dev shape replayed."""
        bound = _bound(kafka_config)
        consumer, dlq_calls = await _redeliver(
            kafka_config, mock_producer, [_not_null() for _ in range(bound + 1)]
        )
        assert len(dlq_calls) == 1, (
            f"after {bound + 1} identical refusals of the same record the "
            "partition must be released; it was rewound every time instead "
            f"(seeks={len(consumer.seek_calls)}, dlq={len(dlq_calls)})"
        )
        assert len(consumer.seek_calls) == bound, (
            "every delivery up to the bound must still rewind -- releasing "
            "earlier would discard a record a transient failure still owed"
        )

    @pytest.mark.asyncio
    async def test_the_dead_letter_carries_a_typed_reason(
        self, kafka_config: ModelKafkaEventBusConfig, mock_producer: AsyncMock
    ) -> None:
        """A dead-letter that does not say WHY reads like any other consumer
        error, and the wedge that produced it is invisible in the census."""
        bound = _bound(kafka_config)
        _, dlq_calls = await _redeliver(
            kafka_config, mock_producer, [_not_null() for _ in range(bound + 1)]
        )
        assert dlq_calls, "nothing was dead-lettered"
        call = dlq_calls[0]
        assert (
            call["failure_class"] == EnumDlqFailureClass.PROJECTION_WEDGE_EXHAUSTED
        ), call.get("failure_class")
        detail = str(call.get("validation_detail") or "")
        assert str(TEST_OFFSET) in detail and str(bound) in detail, (
            "the reason must name the coordinate and the bound it exhausted, "
            f"so the record is findable: {detail!r}"
        )

    @pytest.mark.asyncio
    async def test_an_unconfirmed_dead_letter_keeps_stalling(
        self, kafka_config: ModelKafkaEventBusConfig, mock_producer: AsyncMock
    ) -> None:
        """OMN-15232 discipline is not relaxed by this change.

        A record that is neither projected nor durably quarantined must not have
        its offset advanced, however many times it has failed. A stalled feed is
        recoverable and loud; a vanished record is neither.
        """
        bound = _bound(kafka_config)
        consumer, dlq_calls = await _redeliver(
            kafka_config,
            mock_producer,
            [_not_null() for _ in range(bound + 1)],
            dlq_persists=False,
        )
        assert len(dlq_calls) == 1
        assert len(consumer.seek_calls) == bound + 1, (
            "the delivery that attempted the dead-letter must ALSO rewind when "
            "the quarantine write was not confirmed durable"
        )


class TestATransientFailureStillRewindsForever:
    """The negative controls. Each one fails if the bound is applied too widely."""

    @pytest.mark.asyncio
    async def test_below_the_bound_nothing_is_dead_lettered(
        self, kafka_config: ModelKafkaEventBusConfig, mock_producer: AsyncMock
    ) -> None:
        bound = _bound(kafka_config)
        consumer, dlq_calls = await _redeliver(
            kafka_config, mock_producer, [_not_null() for _ in range(bound)]
        )
        assert dlq_calls == []
        assert len(consumer.seek_calls) == bound

    @pytest.mark.asyncio
    async def test_a_changing_failure_resets_the_counter(
        self, kafka_config: ModelKafkaEventBusConfig, mock_producer: AsyncMock
    ) -> None:
        """A write path being repaired in stages changes its error as it goes.

        A counter that ignored the failure's identity would dead-letter a record
        that was making progress toward materialising -- exactly the record
        OMN-17379 exists to keep.
        """
        bound = _bound(kafka_config)
        alternating: list[Exception] = []
        for index in range(bound * 2 + 2):
            alternating.append(
                _not_null()
                if index % 2 == 0
                else ProjectionNotMaterializedError(
                    f"(TenantRegistryResolutionError: {LIVE_TENANT_REFUSAL})",
                    projection_type="HandlerProjectionDelegation",
                )
            )
        consumer, dlq_calls = await _redeliver(kafka_config, mock_producer, alternating)
        assert dlq_calls == [], (
            "the failure changed on every delivery, so no single failure ever "
            "repeated to the bound; this record is still owed a row"
        )
        assert len(consumer.seek_calls) == len(alternating)

    @pytest.mark.asyncio
    async def test_a_successful_delivery_clears_the_count(
        self, kafka_config: ModelKafkaEventBusConfig, mock_producer: AsyncMock
    ) -> None:
        """A repaired write path must restore the full budget.

        Without this, a record that failed a few times during an outage would
        carry that history and be dead-lettered later on unrelated refusals.
        """
        bound = _bound(kafka_config)
        deliveries: list[Any] = [_not_null() for _ in range(bound)]
        deliveries.append(None)  # the repair
        deliveries.extend(_not_null() for _ in range(bound))
        consumer, dlq_calls = await _redeliver(kafka_config, mock_producer, deliveries)
        assert dlq_calls == [], (
            "the counter survived a successful projection of the same "
            "coordinate, so the post-repair budget was short"
        )
        assert len(consumer.seek_calls) == bound * 2

    @pytest.mark.asyncio
    async def test_the_count_is_per_record_not_per_partition(
        self, kafka_config: ModelKafkaEventBusConfig, mock_producer: AsyncMock
    ) -> None:
        """Different offsets are different records.

        A per-partition counter would dead-letter the (bound+1)th DIFFERENT
        record during an ordinary database outage, which is mass data loss
        dressed as a poison-record guard.
        """
        bound = _bound(kafka_config)
        count = bound + 2
        consumer, dlq_calls = await _redeliver(
            kafka_config,
            mock_producer,
            [_not_null() for _ in range(count)],
            offsets=[TEST_OFFSET + i for i in range(count)],
        )
        assert dlq_calls == []
        assert len(consumer.seek_calls) == count

    @pytest.mark.asyncio
    async def test_an_ordinary_handler_failure_is_untouched(
        self, kafka_config: ModelKafkaEventBusConfig, mock_producer: AsyncMock
    ) -> None:
        """Blast radius: this change narrows one class and widens none.

        A generic handler failure still follows the OMN-15232 path on its own
        retry budget, with no withhold counter involved at all.
        """
        consumer, dlq_calls = await _redeliver(
            kafka_config,
            mock_producer,
            [RuntimeError("handler blew up")],
            # Retries exhausted, which is the condition under which the generic
            # arm dead-letters. With budget remaining it takes the
            # "retries available" branch and neither DLQs nor rewinds, and this
            # control would pass for a reason that has nothing to do with the
            # withhold bound.
            retry_count=5,
        )
        assert len(dlq_calls) == 1
        assert consumer.seek_calls == []


class TestTheTrackerCannotGrowWithoutBound:
    def test_the_capacity_is_declared(
        self, kafka_config: ModelKafkaEventBusConfig
    ) -> None:
        assert (
            "projection_withhold_tracking_capacity" in type(kafka_config).model_fields
        )
        assert kafka_config.projection_withhold_tracking_capacity >= 1

    @pytest.mark.asyncio
    async def test_the_tracker_evicts_rather_than_accumulating(
        self, mock_producer: AsyncMock
    ) -> None:
        """A map keyed by coordinate is unbounded if nothing ever removes a key.

        Entries clear on success and on dead-letter, but a partition revoked
        mid-stall leaves its key behind. The capacity is what makes that a
        bounded leak instead of a slow one.
        """
        config = ModelKafkaEventBusConfig(
            bootstrap_servers="localhost:9092",
            projection_withhold_tracking_capacity=3,
        )
        count = 8
        with patch(
            "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
            return_value=mock_producer,
        ):
            event_bus = EventBusKafka(config=config)
            for index in range(count):
                event_bus._record_projection_withhold(
                    (TEST_TOPIC, TEST_PARTITION, TEST_OFFSET + index, "sub-1"),
                    "boom",
                )
            assert len(event_bus._projection_withholds) <= 3
