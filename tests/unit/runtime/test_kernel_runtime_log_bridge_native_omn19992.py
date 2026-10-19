# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The container kernel on the in-memory bus publishes runtime errors.

OMN-19992 / OB-3: ``RuntimeLogEventBridge`` was attached only when the kernel
resolved the Kafka transport, so a kernel booted on the in-memory bus published
no ``runtime-error`` event and the error-fingerprint projection had nothing to
fold. This drives the real ``bootstrap()`` on the in-memory bus, forces a
handler exception through the real projection dispatch callback while the
kernel runs, and asserts the event reaches the selected bus inside the bound.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_infra.errors import ProjectionNotMaterializedError
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.runtime.auto_wiring import handler_wiring
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    ProjectionDispatchSinks,
    _make_projection_dispatch_callback,
)
from omnibase_infra.runtime.service_kernel import bootstrap
from omnibase_infra.topics import topic_keys
from omnibase_infra.topics.service_topic_registry import ServiceTopicRegistry
from tests.conftest import check_service_registry_available
from tests.helpers.application_db_topology import (
    configure_projection_dsns,
    projection_database_target,
)
from tests.unit.runtime.conftest import force_inmemory_runtime_config

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        not check_service_registry_available(),
        reason="service_registry unavailable (omnibase_core circular import)",
    ),
]

# AC2: the runtime-error event must reach the in-memory bus within 10 s.
PUBLISH_BOUND_SECONDS = 10.0
FINGERPRINT_LENGTH = 16
FORCED_TOPIC = "onex.evt.test.forced-handler-failure.v1"
_PATCH_BUILD_ADAPTER = (
    "omnibase_infra.runtime.auto_wiring.handler_wiring._build_projection_db_adapter"
)


class _ForcedHandlerError(OSError):
    """The exception the forced handler raises."""


class _FailingProjectionHandler:
    def handle(self, input_data: dict[str, object]) -> dict[str, object]:
        raise _ForcedHandlerError("forced handler failure on the in-memory kernel")


async def _force_handler_exception(event_bus: EventBusInmemory) -> None:
    """Dispatch one event to a handler that raises, on the production callback."""
    callback = _make_projection_dispatch_callback(
        _FailingProjectionHandler(),
        projection_database_target("pr_merged_events", schema="omninode_internal"),
        (FORCED_TOPIC,),
        sinks=ProjectionDispatchSinks(event_bus=event_bus),
    )
    envelope = MagicMock()
    envelope.topic = FORCED_TOPIC
    envelope.payload = {"event_id": "omn-19992-forced", "topic": FORCED_TOPIC}
    envelope.correlation_id = "omn-19992-forced"
    with patch(_PATCH_BUILD_ADAPTER, return_value=MagicMock()):
        with pytest.raises(ProjectionNotMaterializedError):
            await callback(envelope)


@pytest.fixture
async def native_bus() -> AsyncGenerator[EventBusInmemory, None]:
    bus = EventBusInmemory(environment="local", group="omn19992")
    await bus.start()
    yield bus
    await bus.close()


async def test_inmemory_kernel_publishes_runtime_error_event(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    native_bus: EventBusInmemory,
    mock_wire_infrastructure: MagicMock,
) -> None:
    force_inmemory_runtime_config(monkeypatch, tmp_path)
    monkeypatch.delenv("OMNIBASE_INFRA_DB_URL", raising=False)
    monkeypatch.setenv("ENABLE_RUNTIME_LOG_BRIDGE", "true")
    topic = ServiceTopicRegistry.from_defaults().resolve(topic_keys.RUNTIME_ERROR)
    seen: list[dict[str, object]] = []
    elapsed: list[float] = []

    async def start_runtime_and_force_error() -> None:
        configure_projection_dsns(monkeypatch)
        started = time.monotonic()
        await _force_handler_exception(native_bus)
        deadline = started + PUBLISH_BOUND_SECONDS
        while time.monotonic() < deadline:
            history = await native_bus.get_event_history(topic=topic)
            events = [json.loads(message.value) for message in history]
            forced = [
                event
                for event in events
                if "Projection handler error" in str(event["raw_message"])
                and _ForcedHandlerError.__name__ in str(event["raw_message"])
            ]
            if forced:
                seen.extend(forced)
                elapsed.append(time.monotonic() - started)
                return
            await asyncio.sleep(0.1)

    async def noop() -> None:
        return None

    runtime = MagicMock()
    runtime.start = AsyncMock(side_effect=start_runtime_and_force_error)
    runtime.stop = AsyncMock(side_effect=noop)
    runtime.input_topic = "requests"
    runtime.output_topic = "responses"
    health = MagicMock()
    health.start = AsyncMock(side_effect=noop)
    health.stop = AsyncMock(side_effect=noop)

    with (
        patch(
            "omnibase_infra.backends.auto_configure.select_event_bus",
            return_value=native_bus,
        ),
        patch(
            "omnibase_infra.runtime.service_kernel.RuntimeHostProcess",
            return_value=runtime,
        ),
        patch(
            "omnibase_infra.services.health_checker.ServiceHealth",
            return_value=health,
        ),
        patch("omnibase_infra.runtime.service_kernel.asyncio.Event") as mock_event,
    ):
        mock_event.return_value.wait = AsyncMock(return_value=None)
        exit_code = await bootstrap()

    assert exit_code == 0
    assert seen, (
        f"no runtime-error event reached the in-memory bus on {topic} "
        f"within {PUBLISH_BOUND_SECONDS}s of a forced handler exception"
    )
    assert elapsed[0] < PUBLISH_BOUND_SECONDS
    event = seen[0]
    assert event["logger_family"] == handler_wiring.logger.name
    assert event["service_label"] == "onex-kernel"
    fingerprint = event["fingerprint"]
    assert isinstance(fingerprint, str)
    assert len(fingerprint) == FINGERPRINT_LENGTH
