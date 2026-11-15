# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests: heartbeat health of the introspection mixin (OMN-24559).

A real ``ServiceNodeIntrospection`` (built from the registration orchestrator
contract the way the kernel builds it) runs on an in-memory bus. The tests
pin what a runtime says about its own heartbeat:

- ``heartbeat_health()`` is unhealthy while no heartbeat task runs, healthy
  after a published beat, and unhealthy exactly at the configured number of
  consecutive failures (one failure fewer is still healthy).
- ``publish_heartbeat_now()`` counts failures (a raising bus, no bus) and a
  later success resets the streak.
- The real ``_heartbeat_loop`` on a bus that rejects heartbeats publishes
  exactly one runtime-error refusal event for the outage; the control loop on
  a healthy bus publishes none (this is the "a loop that never publishes"
  case).

Exemplar: tests/integration/runtime/test_runtime_heartbeat_mandatory.py
(construction of the real service); it differs by using the unit marker and a
bus that rejects heartbeats.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from omnibase_infra.enums import EnumRuntimeHeartbeatRefusalReason
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.models.runtime_timeouts import RUNTIME_TIMEOUTS
from omnibase_infra.services.service_node_introspection import ServiceNodeIntrospection
from omnibase_infra.topics import SUFFIX_NODE_HEARTBEAT, SUFFIX_RUNTIME_ERROR

pytestmark = pytest.mark.unit

CONTRACTS_DIR = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "omnibase_infra"
    / "nodes"
    / "node_registration_orchestrator"
)
THRESHOLD = RUNTIME_TIMEOUTS.introspection_heartbeat_unhealthy_after_failures
LOOP_S = 0.01


class HeartbeatRejectingBus(EventBusInmemory):
    """In-memory bus that rejects only the node-heartbeat topic."""

    async def publish_envelope(self, *args: Any, **kwargs: Any) -> Any:
        topic = kwargs.get("topic") if "topic" in kwargs else args[1]
        if topic == SUFFIX_NODE_HEARTBEAT:
            raise RuntimeError("down")
        return await super().publish_envelope(*args, **kwargs)


def make_service(bus: EventBusInmemory) -> ServiceNodeIntrospection:
    """Build the introspection service the way the kernel builds it."""
    return ServiceNodeIntrospection.from_contract_dir(
        contracts_dir=CONTRACTS_DIR,
        event_bus=bus,
        node_name="heartbeat-health-test",
        environment="local",
    )


async def start_loop(service: ServiceNodeIntrospection) -> None:
    """Start the real heartbeat loop at a small injected period."""
    await service.start_introspection_tasks(
        enable_heartbeat=True,
        heartbeat_interval_seconds=LOOP_S,
        enable_registry_listener=False,
        enable_ack_listener=False,
    )


async def refusal_payloads(bus: EventBusInmemory) -> list[dict[str, Any]]:
    """Every payload published on the runtime-error topic."""
    history = await bus.get_event_history(topic=SUFFIX_RUNTIME_ERROR)
    return [json.loads(m.value)["payload"] for m in history]


@pytest.fixture
async def healthy_bus() -> AsyncIterator[EventBusInmemory]:
    bus = EventBusInmemory()
    await bus.start()
    yield bus
    await bus.close()


@pytest.fixture
async def rejecting_bus() -> AsyncIterator[HeartbeatRejectingBus]:
    bus = HeartbeatRejectingBus()
    await bus.start()
    yield bus
    await bus.close()


class TestHeartbeatHealthVerdict:
    """The typed verdict: stopped, healthy, then unhealthy at the threshold."""

    @pytest.mark.asyncio
    async def test_no_task_is_unhealthy_then_healthy_after_a_beat(
        self, healthy_bus: EventBusInmemory
    ) -> None:
        service = make_service(healthy_bus)
        stopped = service.heartbeat_health()
        assert stopped.healthy is False
        assert (
            stopped.reason is EnumRuntimeHeartbeatRefusalReason.HEARTBEAT_TASK_STOPPED
        )

        await service.start_heartbeat_task()
        try:
            assert await service.publish_heartbeat_now() is True
            health = service.heartbeat_health()
            assert health.healthy is True
            assert health.published_once is True
            assert health.consecutive_failures == 0
        finally:
            await service.stop_heartbeat_task()

    @pytest.mark.asyncio
    async def test_unhealthy_exactly_at_the_configured_failure_count(
        self, healthy_bus: EventBusInmemory
    ) -> None:
        service = make_service(healthy_bus)
        await service.start_heartbeat_task()
        try:
            assert await service.publish_heartbeat_now() is True
            rejecting = HeartbeatRejectingBus()
            await rejecting.start()
            service._introspection_event_bus = rejecting
            for _ in range(THRESHOLD - 1):
                assert await service.publish_heartbeat_now() is False
            assert service.heartbeat_health().healthy is True  # one fewer
            assert await service.publish_heartbeat_now() is False
            health = service.heartbeat_health()
            assert health.healthy is False
            assert health.reason is (
                EnumRuntimeHeartbeatRefusalReason.HEARTBEAT_PUBLISH_FAILING
            )
            assert health.consecutive_failures == THRESHOLD
            assert "down" in health.detail
            await rejecting.close()
        finally:
            await service.stop_heartbeat_task()


class TestPublishHeartbeatNowAccounting:
    """A failed publish is counted, never silent; a success resets the streak."""

    @pytest.mark.asyncio
    async def test_raising_bus_and_missing_bus_count_as_failures(
        self,
        healthy_bus: EventBusInmemory,
        rejecting_bus: HeartbeatRejectingBus,
    ) -> None:
        service = make_service(rejecting_bus)
        assert await service.publish_heartbeat_now() is False
        assert service.heartbeat_account.consecutive_failures == 1
        assert "down" in service.heartbeat_account.last_error

        service._introspection_event_bus = None
        assert await service.publish_heartbeat_now() is False
        assert service.heartbeat_account.consecutive_failures == 2
        assert service.heartbeat_account.last_error == "no event bus"

        service._introspection_event_bus = healthy_bus
        assert await service.publish_heartbeat_now() is True
        assert service.heartbeat_account.consecutive_failures == 0
        assert service.heartbeat_account.last_error == ""


class TestHeartbeatLoopRefusalEvent:
    """The loop that never publishes announces its outage exactly once."""

    @pytest.mark.asyncio
    async def test_failing_loop_publishes_one_refusal_event_and_reads_unhealthy(
        self, rejecting_bus: HeartbeatRejectingBus
    ) -> None:
        service = make_service(rejecting_bus)
        await start_loop(service)
        try:
            async with asyncio.timeout(5):
                while not await refusal_payloads(rejecting_bus):
                    await asyncio.sleep(LOOP_S)
            await asyncio.sleep(THRESHOLD * 5 * LOOP_S)  # failures keep coming
            events = await refusal_payloads(rejecting_bus)
            assert len(events) == 1
            assert "heartbeat_publish_failing" in events[0]["raw_message"]
            assert events[0]["severity"] == "critical"
            health = service.heartbeat_health()
            assert health.healthy is False
            assert health.consecutive_failures >= THRESHOLD
        finally:
            await service.stop_heartbeat_task()

    @pytest.mark.asyncio
    async def test_control_healthy_loop_publishes_no_refusal_event(
        self, healthy_bus: EventBusInmemory
    ) -> None:
        service = make_service(healthy_bus)
        await start_loop(service)
        try:
            await asyncio.sleep(THRESHOLD * 5 * LOOP_S)
            assert await refusal_payloads(healthy_bus) == []
            assert service.heartbeat_health().healthy is True
        finally:
            await service.stop_heartbeat_task()
