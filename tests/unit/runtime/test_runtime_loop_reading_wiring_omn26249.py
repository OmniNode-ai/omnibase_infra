# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26249: the runtime member publishes its loop reading every window.

Validates:
- One reading per window, windows contiguous, published to the contracted topic
- The abandoned count comes from the bus's dispatch-deadline status
- A tick consumed on the tick topic reaches the sampler as tick lag, and a
  member wiring no tick subscription stays not-applicable
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
import yaml

from omnibase_core.protocols.event_bus.protocol_event_bus_subscriber import (
    ProtocolEventBusSubscriber,
)
from omnibase_infra.event_bus.models import ModelEventHeaders, ModelEventMessage
from omnibase_infra.handlers import handler_runtime_loop_lag_sampler as handler_module
from omnibase_infra.handlers.handler_runtime_loop_lag_sampler import (
    NOT_A_TICK_SUBSCRIBER_REASON,
    RuntimeLoopLagSampler,
    RuntimeLoopReadingPublisher,
)
from omnibase_infra.models import ModelNodeIdentity
from omnibase_infra.models.health.model_dispatch_deadline_status import (
    ModelDispatchDeadlineStatus,
)
from omnibase_infra.models.health.model_runtime_loop_reading import (
    ModelRuntimeLoopReading,
)
from omnibase_infra.runtime import event_bus_subcontract_wiring as wiring_module
from omnibase_infra.runtime.event_bus_subcontract_wiring import (
    EventBusSubcontractWiring,
)
from omnibase_infra.topics import SUFFIX_RUNTIME_TICK

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_NODE_CONTRACT = (
    _REPO_ROOT
    / "src/omnibase_infra/nodes/node_runtime_loop_reading_effect/contract.yaml"
)
_TOPIC = "onex.evt.omnibase-infra.runtime-loop-reading-recorded.v1"


class _BusWithOrphans:
    """A bus that reports two abandoned dispatches."""

    def __init__(self) -> None:
        self.publish_envelope = AsyncMock()

    def dispatch_deadline_status(self) -> ModelDispatchDeadlineStatus:
        return ModelDispatchDeadlineStatus(
            deadline_seconds=30.0,
            orphan_limit=5,
            orphaned_dispatches=2,
            degrading_orphans=2,
            deadline_expiries_total=2,
            oldest_orphan_age_seconds=1.0,
            orphans=("topic:0:1",),
        )


class _PlainBus:
    """A bus with no consume loop, so nothing to abandon."""

    def __init__(self) -> None:
        self.publish_envelope = AsyncMock()


_IDENTITY = ModelNodeIdentity(
    env="dev", service="omnibase-infra", node_name="runtime-main", version="v1"
)


async def _started(
    bus: object, window_seconds: float = 3600.0
) -> RuntimeLoopReadingPublisher:
    publisher = RuntimeLoopReadingPublisher()
    await publisher.start(
        cast("Any", bus), _IDENTITY, MagicMock(), window_seconds=window_seconds
    )
    return publisher


@pytest.fixture
def sampler(monkeypatch: pytest.MonkeyPatch) -> RuntimeLoopLagSampler:
    fresh = RuntimeLoopLagSampler()
    monkeypatch.setattr(handler_module, "_PROCESS_SAMPLER", fresh)
    return fresh


async def test_one_reading_per_window_with_contiguous_windows_on_the_contracted_topic(
    sampler: RuntimeLoopLagSampler,
) -> None:
    bus = _PlainBus()
    publisher = await _started(bus, window_seconds=0.1)
    try:
        await asyncio.sleep(0.55)
    finally:
        await publisher.stop()
    topics = {call.args[1] for call in bus.publish_envelope.await_args_list}
    readings = [
        ModelRuntimeLoopReading.model_validate(call.args[0])
        for call in bus.publish_envelope.await_args_list
    ]
    assert len(readings) >= 4
    assert topics == {_TOPIC}
    for earlier, later in pairwise(readings):
        assert later.window_start == earlier.window_end


async def test_reading_carries_member_host_and_the_bus_abandoned_count(
    sampler: RuntimeLoopLagSampler,
) -> None:
    bus = _BusWithOrphans()
    publisher = await _started(bus)
    try:
        await publisher.publish_reading(cast("Any", bus), _IDENTITY)
    finally:
        await publisher.stop()
    reading = ModelRuntimeLoopReading.model_validate(
        bus.publish_envelope.await_args.args[0]
    )
    assert reading.abandoned_dispatches == 2
    assert reading.runtime_member.startswith("dev.omnibase-infra.runtime-main.")
    assert reading.host
    assert reading.tick_subscriber is False
    assert reading.tick_lag_not_applicable_reason == NOT_A_TICK_SUBSCRIBER_REASON


async def test_bus_without_deadline_status_reports_zero_abandoned(
    sampler: RuntimeLoopLagSampler,
) -> None:
    bus = _PlainBus()
    publisher = await _started(bus)
    try:
        await publisher.publish_reading(cast("Any", bus), _IDENTITY)
    finally:
        await publisher.stop()
    assert bus.publish_envelope.await_args.args[0]["abandoned_dispatches"] == 0


async def test_publish_failure_does_not_stop_the_publisher(
    sampler: RuntimeLoopLagSampler,
) -> None:
    bus = _PlainBus()
    publisher = await _started(bus)
    try:
        bus.publish_envelope.side_effect = OSError("bus down")
        await publisher.publish_reading(cast("Any", bus), _IDENTITY)
        bus.publish_envelope.side_effect = None
        await publisher.publish_reading(cast("Any", bus), _IDENTITY)
    finally:
        await publisher.stop()
    assert bus.publish_envelope.await_count == 2


async def test_start_and_stop_run_and_stop_the_sampler(
    sampler: RuntimeLoopLagSampler,
) -> None:
    publisher = await _started(_PlainBus())
    assert sampler.is_running
    await publisher.stop()
    assert not sampler.is_running


async def test_publishing_before_start_is_refused(
    sampler: RuntimeLoopLagSampler,
) -> None:
    with pytest.raises(RuntimeError, match="before start"):
        await RuntimeLoopReadingPublisher().publish_reading(
            cast("Any", _PlainBus()), _IDENTITY
        )


def _tick_message(scheduled_at: datetime) -> ModelEventMessage:
    payload = {
        "event_type": "platform.runtime-tick",
        "correlation_id": str(uuid4()),
        "payload": {"scheduled_at": scheduled_at.isoformat()},
    }
    return ModelEventMessage(
        topic=SUFFIX_RUNTIME_TICK,
        key=b"tick",
        value=json.dumps(payload).encode("utf-8"),
        headers=ModelEventHeaders(
            source="runtime-scheduler",
            event_type="platform.runtime-tick",
            timestamp=datetime.now(UTC),
        ),
    )


def _wiring() -> EventBusSubcontractWiring:
    engine = AsyncMock()
    engine.dispatch = AsyncMock()
    return EventBusSubcontractWiring(
        event_bus=AsyncMock(spec=ProtocolEventBusSubscriber),
        dispatch_engine=engine,
        environment="dev",
        node_name="tick-subscriber",
        service="test-service",
        version="v1",
    )


async def test_tick_consumed_on_the_tick_topic_reaches_the_reading_as_tick_lag(
    sampler: RuntimeLoopLagSampler,
) -> None:
    callback = _wiring()._create_dispatch_callback(
        SUFFIX_RUNTIME_TICK, "dev.test-service.tick-subscriber.consume.v1"
    )
    await callback(_tick_message(datetime.now(UTC) - timedelta(milliseconds=700)))
    bus = _PlainBus()
    publisher = await _started(bus)
    try:
        await publisher.publish_reading(cast("Any", bus), _IDENTITY)
    finally:
        await publisher.stop()
    reading = ModelRuntimeLoopReading.model_validate(
        bus.publish_envelope.await_args.args[0]
    )
    assert reading.tick_subscriber is True
    assert reading.tick_lag_ms is not None
    assert reading.tick_lag_ms >= 700.0


async def test_callback_for_another_topic_leaves_the_member_a_non_subscriber(
    sampler: RuntimeLoopLagSampler,
) -> None:
    _wiring()._create_dispatch_callback(
        "onex.evt.test-service.other.v1", "dev.test-service.other.consume.v1"
    )
    bus = _PlainBus()
    publisher = await _started(bus)
    try:
        await publisher.publish_reading(cast("Any", bus), _IDENTITY)
    finally:
        await publisher.stop()
    assert bus.publish_envelope.await_args.args[0]["tick_subscriber"] is False


def test_unreadable_tick_payload_records_no_lag(
    sampler: RuntimeLoopLagSampler,
) -> None:
    sampler.observe_subscription(SUFFIX_RUNTIME_TICK)
    sampler.observe_message(SUFFIX_RUNTIME_TICK, {"scheduled_at": "not-a-time"}, 0)
    sampler.observe_message(
        SUFFIX_RUNTIME_TICK, {"scheduled_at": "2026-11-21T00:00:00"}, 0
    )
    sampler.observe_message(SUFFIX_RUNTIME_TICK, {}, 0)
    assert sampler._newest_tick_lag_ms is None
    assert sampler._tick_subscriber is True


def test_message_on_another_topic_is_not_a_tick(
    sampler: RuntimeLoopLagSampler,
) -> None:
    sampler.observe_message(
        "onex.evt.test-service.other.v1",
        {"scheduled_at": datetime.now(UTC).isoformat()},
        0,
    )
    assert sampler._tick_subscriber is False
    assert sampler._newest_tick_lag_ms is None


def test_contract_declares_the_topic_the_publisher_publishes_to() -> None:
    contract = yaml.safe_load(_NODE_CONTRACT.read_text())
    assert contract["event_bus"]["publish_topics"] == [_TOPIC]
