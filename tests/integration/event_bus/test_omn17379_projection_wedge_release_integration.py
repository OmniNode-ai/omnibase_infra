# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17379: the wedge and its release, across BOTH seams that produce it.

The unit suite
(``tests/unit/event_bus/test_omn17379_projection_wedge_bounded.py``) constructs a
``ProjectionNotMaterializedError`` by hand and feeds it to the consume loop. That
proves the bound, and it cannot prove that the error a real projection handler
produces ever reaches it. The wedge lives in the JOIN between two subsystems:

* ``runtime/auto_wiring/handler_wiring`` decides that a handler which wrote zero
  rows for a non-content reason must NOT be acknowledged, and raises.
* ``event_bus/event_bus_kafka`` decides what a raised
  ``ProjectionNotMaterializedError`` does to the partition's offset.

Neither module's own tests observe the other. This module wires the REAL
projection dispatch callback as the REAL consume loop's subscriber, so the
handler's refusal, the withhold, the bound and the release are one chain rather
than four separately-asserted facts.

THE SHAPE IS THE LIVE ONE. On the onex-dev staging namespace 2026-09-15 the
delegation projection writer refused ``quality-gate-result.v1`` partition 0
offset 300 with a NOT NULL violation on ``delegation_events.task_type``, about
once per second, and every later record on that partition was never delivered.
The handler here writes zero rows and raises the same class of driver error, so
the chain under test is the chain that ran.

The negative control is the half that matters: a MALFORMED event is a content
failure, is dead-lettered on its FIRST delivery by the auto-wiring seam, and
never reaches the withhold bound at all. If that ever stops holding, this change
has widened a poison-record bound into a data-loss path.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from aiokafka.structs import TopicPartition

import omnibase_infra.event_bus.event_bus_kafka as event_bus_kafka_module
from omnibase_infra.enums import EnumDlqFailureClass
from omnibase_infra.enums.enum_infra_transport_type import EnumInfraTransportType
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.event_bus.models.model_publish_receipt import ModelPublishReceipt
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    ProjectionDispatchSinks,
    _make_projection_dispatch_callback,
)
from tests.helpers.application_db_topology import (
    configure_projection_dsns,
    projection_database_target,
)

pytestmark = pytest.mark.integration

_PATCH_BUILD_ADAPTER = (
    "omnibase_infra.runtime.auto_wiring.handler_wiring._build_projection_db_adapter"
)
_PATCH_ENVIRON_GET = "omnibase_infra.runtime.auto_wiring.handler_wiring.os.environ.get"
_TEST_DSN = "postgresql://user:pw@host:5432/omnidash_analytics"

TOPIC = "onex.evt.omnibase-infra.quality-gate-result.v1"
GROUP = "local.omnimarket.delegation_projection.consume.1.0.0"
PARTITION = 0
OFFSET = 300

#: The live refusal at that coordinate, verbatim from the pod's own log line.
LIVE_NOT_NULL = (
    'null value in column "task_type" of relation "delegation_events" '
    "violates not-null constraint"
)


class NotNullViolationError(Exception):
    """Stand-in for ``psycopg2.errors.NotNullViolation`` (named with the
    ``Error`` suffix this repo's lint requires, which the driver's own class
    does not carry).

    Declared locally on purpose, exactly as the sibling unit module does for
    ``InsufficientPrivilege``: what is pinned is the runtime's CLASSIFICATION
    rule -- a failure it cannot positively identify as the event's own defect is
    treated as the runtime's -- not a psycopg2 type. Importing the driver class
    would let a fix pass by special-casing one library instead of closing the
    class.
    """


class _RefusingHandler:
    """A projection handler whose write path fails the same way every time."""

    def __init__(self) -> None:
        self.calls = 0

    def handle(self, input_data: dict[str, object]) -> dict[str, object]:
        self.calls += 1
        raise NotNullViolationError(LIVE_NOT_NULL)


class _FakeConsumer:
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


class _FakeEventBus:
    """Reports a durability coordinate, as both shipped buses do (OMN-17862)."""

    def __init__(self) -> None:
        self.published: list[tuple[str, object, bytes]] = []

    async def publish(
        self, topic: str, key: object, value: bytes
    ) -> ModelPublishReceipt:
        self.published.append((topic, key, value))
        return ModelPublishReceipt(
            topic=topic,
            partition=0,
            offset=len(self.published) - 1,
            cluster="test-cluster",
            produced_at=datetime.now(UTC),
            transport=EnumInfraTransportType.INMEMORY,
        )


@pytest.fixture(autouse=True)
def _configured_projection_dsns(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_projection_dsns(monkeypatch, url=_TEST_DSN)


def _raw_msg() -> MagicMock:
    msg = MagicMock()
    msg.topic = TOPIC
    msg.partition = PARTITION
    msg.offset = OFFSET
    msg.timestamp = int(datetime.now(UTC).timestamp() * 1000)
    msg.key = None
    msg.headers = []
    msg.value = b'{"correlation_id": "fc3d267f-dcb2-44c8-ac6f-9285a9b5e827"}'
    return msg


async def _run_chain(
    handler: object, *, deliveries: int, config: ModelKafkaEventBusConfig
) -> tuple[_FakeConsumer, list[dict[str, Any]], _FakeEventBus]:
    """Drive the REAL projection callback through the REAL consume loop."""
    dlq_calls: list[dict[str, Any]] = []
    projection_bus = _FakeEventBus()

    callback = _make_projection_dispatch_callback(
        handler,
        projection_database_target("delegation_events", schema="public"),
        (TOPIC,),
        sinks=ProjectionDispatchSinks(event_bus=projection_bus),
    )

    producer = AsyncMock()
    producer.start = AsyncMock()
    producer.stop = AsyncMock()
    producer.send_and_wait = AsyncMock()

    envelope = MagicMock()
    envelope.topic = TOPIC
    envelope.payload = {"correlation_id": "fc3d267f-dcb2-44c8-ac6f-9285a9b5e827"}
    envelope.correlation_id = "omn-17379-offset-300"
    envelope.headers.retry_count = 0
    envelope.headers.max_retries = 5

    with patch(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
        return_value=producer,
    ):
        event_bus = EventBusKafka(config=config)
        await event_bus.start()
        consumer = _FakeConsumer([_raw_msg() for _ in range(deliveries)])
        event_bus._group_consumers[(TOPIC, GROUP)] = consumer  # type: ignore[assignment]
        event_bus._subscribers[TOPIC] = [(GROUP, "sub-1", callback)]  # type: ignore[assignment]

        async def _record_dlq(**kwargs: Any) -> bool:
            dlq_calls.append(kwargs)
            return True

        with (
            patch.object(event_bus, "_kafka_msg_to_model", return_value=envelope),
            patch.object(event_bus, "_publish_to_dlq", side_effect=_record_dlq),
            patch.object(
                event_bus_kafka_module, "DLQ_UNPERSISTED_REWIND_BACKOFF_SECONDS", 0.0
            ),
            patch(_PATCH_ENVIRON_GET, return_value=_TEST_DSN),
            patch(_PATCH_BUILD_ADAPTER, return_value=MagicMock()),
        ):
            await event_bus._consume_loop(TOPIC, GROUP, uuid4())
        await event_bus.close()

    return consumer, dlq_calls, projection_bus


@pytest.mark.asyncio
async def test_a_repeating_write_path_refusal_wedges_then_releases() -> None:
    """The whole chain, end to end, on the live shape.

    Every delivery up to the bound rewinds -- that is OMN-17379 doing its job
    for a failure that might still be transient. The delivery after it is
    quarantined with a typed reason and the partition is released, which is what
    stops one record from stopping a plane.
    """
    config = ModelKafkaEventBusConfig(bootstrap_servers="localhost:9092")
    bound = config.projection_withhold_max_redeliveries
    handler = _RefusingHandler()

    consumer, dlq_calls, _ = await _run_chain(
        handler, deliveries=bound + 1, config=config
    )

    assert handler.calls == bound + 1, (
        "the handler must be invoked on every redelivery; if it is not, the "
        "chain under test is not the chain that ran"
    )
    assert len(consumer.seek_calls) == bound
    assert consumer.seek_calls[0] == (TopicPartition(TOPIC, PARTITION), OFFSET)
    assert len(dlq_calls) == 1
    assert (
        dlq_calls[0]["failure_class"] == EnumDlqFailureClass.PROJECTION_WEDGE_EXHAUSTED
    )
    assert LIVE_NOT_NULL in str(dlq_calls[0]["validation_detail"]), (
        "the quarantine must carry the refusal that caused it, or the record is "
        "durable but the reason is only in a container log"
    )


@pytest.mark.asyncio
async def test_below_the_bound_the_chain_only_withholds() -> None:
    """Negative control: a write-path failure is still owed a row.

    Without this, the test above would pass just as well against a change that
    dead-lettered on the first refusal -- which is OMN-17379 reopened.
    """
    config = ModelKafkaEventBusConfig(bootstrap_servers="localhost:9092")
    bound = config.projection_withhold_max_redeliveries
    consumer, dlq_calls, _ = await _run_chain(
        _RefusingHandler(), deliveries=bound, config=config
    )
    assert dlq_calls == []
    assert len(consumer.seek_calls) == bound


@pytest.mark.asyncio
async def test_a_content_failure_never_reaches_the_bound() -> None:
    """Negative control: the auto-wiring seam still dead-letters on delivery ONE.

    A malformed payload is the EVENT's defect, redelivery can never repair it,
    and it is quarantined by ``handler_wiring`` before the consume loop ever
    classifies an offset. The bound added here must not have moved that
    boundary: a content failure that started taking the withhold path would be
    N pointless redeliveries of a record that was already recoverable.
    """
    from pydantic import BaseModel, ValidationError

    class _Strict(BaseModel):
        required_field: str

    class _MalformedHandler:
        def __init__(self) -> None:
            self.calls = 0

        def handle(self, input_data: dict[str, object]) -> dict[str, object]:
            self.calls += 1
            try:
                _Strict.model_validate({})
            except ValidationError as exc:
                raise exc from None
            raise AssertionError("the strict model accepted an empty payload")

    config = ModelKafkaEventBusConfig(bootstrap_servers="localhost:9092")
    handler = _MalformedHandler()
    consumer, dlq_calls, projection_bus = await _run_chain(
        handler, deliveries=1, config=config
    )

    assert handler.calls == 1
    assert projection_bus.published, (
        "a content failure is quarantined by the auto-wiring seam on its first "
        "delivery, before the consume loop decides anything about the offset"
    )
    assert consumer.seek_calls == [], "a content failure must not withhold at all"
    assert dlq_calls == [], (
        "the consume loop's own dead-letter path is not involved: the callback "
        "returned normally after quarantining"
    )
