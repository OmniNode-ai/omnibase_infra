# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19355: a projection handler parked in its worker thread, end to end.

The unit suite (``tests/unit/event_bus/test_omn19355_dispatch_deadline.py``)
hangs a hand-written callback. The 2026-09-23 hang was not in a hand-written
callback: it was in a projection handler dispatched by the runtime's
auto-wiring, which runs ``handle()`` through ``asyncio.to_thread`` inside the
runtime-wide projection gate. This module wires the REAL projection dispatch
callback as the REAL consume loop's subscriber and parks the handler's worker
thread, which is the shape a ``/proc`` read found on the .201 dev lane (one
``to_thread`` worker in ``epoll_wait`` with timeout -1).

What it proves that the unit suite cannot:

* the deadline fires through the auto-wiring seam, and the record is
  quarantined by the consume loop rather than by the seam;
* the parked thread keeps its projection gate slot, which is why the orphan
  limit sits below ``PROJECTION_HANDLER_MAX_INFLIGHT`` -- an abandoned dispatch
  that released its slot would let the process lose count of the resource that
  is actually leaking;
* the next record projects normally through the same gate.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
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
    PROJECTION_HANDLER_MAX_INFLIGHT,
    ProjectionDispatchSinks,
    _make_projection_dispatch_callback,
    _projection_inflight_gate,
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
_TEST_DSN = "postgresql://user:***REDACTED***@host:5432/omnidash_analytics"

TOPIC = "onex.evt.omnibase-infra.lab-lane-health.v1"
GROUP = "local.omnimarket.lab_lane_health_projection.consume.1.0.0"
PARTITION = 0
HUNG = 51808


class _ParkingHandler:
    """A projection handler whose FIRST call parks its worker thread.

    Every later call writes one row, so the record after the hung one proves
    the gate and the loop are both still usable.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.parked = threading.Event()
        self.release = threading.Event()

    def handle(self, input_data: dict[str, object]) -> dict[str, object]:
        self.calls += 1
        if self.calls == 1:
            self.parked.set()
            self.release.wait()
        return {"rows_upserted": 1}


class _PartitionLog:
    def __init__(self, offsets: list[int]) -> None:
        self._records = [_raw_msg(offset) for offset in offsets]
        self.fetch_position = offsets[0]
        self.seeks: list[int] = []
        self.on_drained: Callable[[], None] | None = None

    async def getmany(
        self,
        *partitions: TopicPartition,
        timeout_ms: int = 0,
        max_records: int | None = None,
    ) -> dict[TopicPartition, list[Any]]:
        batch = [r for r in self._records if r.offset >= self.fetch_position]
        if not batch:
            if self.on_drained is not None:
                self.on_drained()
            return {}
        self.fetch_position = batch[-1].offset + 1
        return {TopicPartition(TOPIC, PARTITION): batch}

    def assignment(self) -> set[TopicPartition]:
        return set()

    def seek(self, partition: TopicPartition, offset: int) -> None:
        self.seeks.append(offset)
        self.fetch_position = offset

    async def stop(self) -> None:
        return None


class _FakeEventBus:
    """The projection sink bus; reports a durability coordinate (OMN-17862)."""

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


def _raw_msg(offset: int) -> MagicMock:
    msg = MagicMock()
    msg.topic = TOPIC
    msg.partition = PARTITION
    msg.offset = offset
    msg.timestamp = int(datetime.now(UTC).timestamp() * 1000)
    msg.key = None
    msg.headers = []
    msg.value = b'{"correlation_id": "fc3d267f-dcb2-44c8-ac6f-9285a9b5e827"}'
    return msg


@pytest.mark.asyncio
async def test_a_parked_projection_thread_is_abandoned_and_the_next_record_projects() -> (
    None
):
    handler = _ParkingHandler()
    callback = _make_projection_dispatch_callback(
        handler,
        projection_database_target("delegation_events", schema="public"),
        (TOPIC,),
        sinks=ProjectionDispatchSinks(event_bus=_FakeEventBus()),
    )
    config = ModelKafkaEventBusConfig(
        bootstrap_servers="localhost:9092",
        max_poll_interval_ms=10_000,
        session_timeout_ms=6_000,
        heartbeat_interval_ms=2_000,
        consumer_dispatch_deadline_seconds=0.5,
        consumer_dispatch_withhold_after_seconds=0.05,
    )

    envelope = MagicMock()
    envelope.topic = TOPIC
    envelope.payload = {"correlation_id": "fc3d267f-dcb2-44c8-ac6f-9285a9b5e827"}
    envelope.correlation_id = "omn-19355-offset-51808"
    envelope.headers.retry_count = 0
    envelope.headers.max_retries = 3

    producer = AsyncMock()
    producer._closed = False
    dlq_calls: list[dict[str, Any]] = []

    async def _record_dlq(**kwargs: Any) -> bool:
        dlq_calls.append(kwargs)
        return True

    log = _PartitionLog([HUNG, HUNG + 1])
    with patch(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
        return_value=producer,
    ):
        bus = EventBusKafka(config=config)
        await bus.start()
        bus._group_consumers[(TOPIC, GROUP)] = log  # type: ignore[assignment]
        log.on_drained = lambda: setattr(bus, "_shutdown", True)
        bus._subscribers[TOPIC] = [(GROUP, "sub-1", callback)]  # type: ignore[assignment]
        try:
            with (
                patch.object(bus, "_kafka_msg_to_model", return_value=envelope),
                patch.object(bus, "_publish_to_dlq", side_effect=_record_dlq),
                patch.object(
                    event_bus_kafka_module,
                    "DLQ_UNPERSISTED_REWIND_BACKOFF_SECONDS",
                    0.0,
                ),
                patch(_PATCH_ENVIRON_GET, return_value=_TEST_DSN),
                patch(_PATCH_BUILD_ADAPTER, return_value=MagicMock()),
            ):
                await asyncio.wait_for(
                    bus._consume_loop(TOPIC, GROUP, uuid4()), timeout=15.0
                )

            assert handler.parked.is_set()
            assert handler.calls == 2, (
                "the record after the parked one must reach the handler"
            )
            assert len(dlq_calls) == 1
            assert (
                dlq_calls[0]["failure_class"]
                == EnumDlqFailureClass.DISPATCH_DEADLINE_EXCEEDED
            )
            status = bus.dispatch_deadline_status()
            assert status.orphaned_dispatches == 1
            assert status.status == "degraded"
            gate = _projection_inflight_gate()
            assert gate._value == PROJECTION_HANDLER_MAX_INFLIGHT - 1, (  # type: ignore[attr-defined]
                "the parked thread must keep its projection gate slot; a slot "
                "released under a live thread hides the leak it represents"
            )
            assert (
                config.consumer_dispatch_orphan_limit < PROJECTION_HANDLER_MAX_INFLIGHT
            )

            handler.release.set()
            for _ in range(100):
                if bus.dispatch_deadline_status().orphaned_dispatches == 0:
                    break
                await asyncio.sleep(0.02)
            assert bus.dispatch_deadline_status().orphaned_dispatches == 0
            assert gate._value == PROJECTION_HANDLER_MAX_INFLIGHT  # type: ignore[attr-defined]
        finally:
            handler.release.set()
            await bus.close()
