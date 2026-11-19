# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests: RuntimeHostProcess with a required heartbeat (OMN-24559).

A real ``RuntimeHostProcess`` with ``heartbeat_required=True`` on an in-memory
bus and a real ``ServiceNodeIntrospection``:

- a first heartbeat that never publishes refuses the start after exactly the
  configured attempts, and the refused runtime reads not running with its
  heartbeat task stopped (the rollback case);
- no introspection service refuses the start;
- ``/ready`` carries a ``heartbeat`` row: true while healthy, false with the
  reason once heartbeats fail (the "forced publish failure shows not ready"
  case);
- the control: a runtime built without ``heartbeat_required`` has no such row.

Exemplar: tests/integration/runtime/test_runtime_heartbeat_mandatory.py; it
differs by asserting the rollback and the readiness row.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from omnibase_infra.enums import EnumRuntimeHeartbeatRefusalReason
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.handlers import handler_runtime_introspection_startup as handler_mod
from omnibase_infra.handlers.handler_runtime_introspection_startup import (
    HandlerRuntimeIntrospectionStartup,
    RuntimeHeartbeatRequiredError,
)
from omnibase_infra.models.runtime import ModelRuntimeIntrospectionConfig
from omnibase_infra.models.runtime_timeouts import RUNTIME_TIMEOUTS
from omnibase_infra.runtime.runtime_host_process import RuntimeHostProcess
from omnibase_infra.services.service_node_introspection import ServiceNodeIntrospection
from tests.helpers.runtime_helpers import make_runtime_config, seed_mock_handlers

pytestmark = pytest.mark.unit

CONTRACTS_DIR = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "omnibase_infra"
    / "nodes"
    / "node_registration_orchestrator"
)
THRESHOLD = RUNTIME_TIMEOUTS.introspection_heartbeat_unhealthy_after_failures
ATTEMPTS = RUNTIME_TIMEOUTS.introspection_heartbeat_start_attempts


@pytest.fixture(autouse=True)
def _fast_retries_and_clean_counters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink the startup retry wait; clear the per-process refusal counters."""
    HandlerRuntimeIntrospectionStartup.reset_refusal_counts()
    monkeypatch.setattr(
        handler_mod,
        "RUNTIME_TIMEOUTS",
        RUNTIME_TIMEOUTS.model_copy(
            update={"introspection_heartbeat_retry_backoff_seconds": 0.001}
        ),
    )


@pytest.fixture
async def bus() -> AsyncIterator[EventBusInmemory]:
    event_bus = EventBusInmemory()
    yield event_bus
    await event_bus.close()


def supplemental(readiness: dict[str, object]) -> dict[str, dict[str, Any]]:
    """The supplemental readiness rows of a readiness answer."""
    return cast("dict[str, dict[str, Any]]", readiness["supplemental_readiness"])


def make_service(bus: EventBusInmemory) -> ServiceNodeIntrospection:
    return ServiceNodeIntrospection.from_contract_dir(
        contracts_dir=CONTRACTS_DIR,
        event_bus=bus,
        node_name="heartbeat-required-test",
        environment="local",
    )


def make_process(
    bus: EventBusInmemory,
    service: ServiceNodeIntrospection | None,
    *,
    required: bool,
) -> RuntimeHostProcess:
    process = RuntimeHostProcess(
        event_bus=bus,
        config=make_runtime_config(),
        introspection_service=service,
        introspection_config=ModelRuntimeIntrospectionConfig(
            enabled=True, jitter_max_ms=0, throttle_min_interval_s=10
        ),
        heartbeat_required=required,
    )
    seed_mock_handlers(process)
    return process


class TestRefusedStartRollsBack:
    """A refused start never reads running and leaves no heartbeat task."""

    @pytest.mark.asyncio
    async def test_first_heartbeat_never_publishes_refuses_and_rolls_back(
        self, bus: EventBusInmemory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        service = make_service(bus)
        never = AsyncMock(return_value=False)
        monkeypatch.setattr(service, "_publish_heartbeat", never)
        process = make_process(bus, service, required=True)

        with pytest.raises(RuntimeHeartbeatRequiredError) as exc_info:
            await process.start()

        assert exc_info.value.refusal.reason is (
            EnumRuntimeHeartbeatRefusalReason.HEARTBEAT_PUBLISH_FAILED
        )
        assert process.is_running is False
        task = service._heartbeat_task
        assert task is None or task.done()
        # The loop's own first beat may add one call; the bounded retry is the
        # attempts at startup, never fewer.
        assert never.await_count >= ATTEMPTS

    @pytest.mark.asyncio
    async def test_control_healthy_service_starts_running(
        self, bus: EventBusInmemory
    ) -> None:
        process = make_process(bus, make_service(bus), required=True)
        try:
            await process.start()
            assert process.is_running is True
        finally:
            await process.stop()

    @pytest.mark.asyncio
    async def test_no_introspection_service_refuses_the_start(
        self, bus: EventBusInmemory
    ) -> None:
        process = make_process(bus, None, required=True)

        with pytest.raises(RuntimeHeartbeatRequiredError) as exc_info:
            await process.start()

        assert exc_info.value.refusal.reason is (
            EnumRuntimeHeartbeatRefusalReason.INTROSPECTION_SERVICE_MISSING
        )
        assert process.is_running is False


class TestHeartbeatReadinessRow:
    """/ready names a failing heartbeat; a non-required runtime has no row."""

    @pytest.mark.asyncio
    async def test_ready_while_healthy_then_not_ready_with_the_reason(
        self, bus: EventBusInmemory
    ) -> None:
        service = make_service(bus)
        process = make_process(bus, service, required=True)
        try:
            await process.start()
            ready = await process.readiness_check()
            assert supplemental(ready)["heartbeat"]["ready"] is True
            assert ready["ready"] is True

            service.heartbeat_account.consecutive_failures = THRESHOLD
            service.heartbeat_account.last_error = "down"
            not_ready = await process.readiness_check()

            row = supplemental(not_ready)["heartbeat"]
            assert not_ready["ready"] is False
            assert row["ready"] is False
            assert row["reason"] == "heartbeat_publish_failing"
            assert "down" in row["summary"]
        finally:
            await process.stop()

    @pytest.mark.asyncio
    async def test_control_without_heartbeat_required_has_no_heartbeat_row(
        self, bus: EventBusInmemory
    ) -> None:
        process = make_process(bus, make_service(bus), required=False)
        try:
            await process.start()
            ready = await process.readiness_check()
            assert "heartbeat" not in supplemental(ready)
        finally:
            await process.stop()
