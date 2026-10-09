# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The native (non-Kafka) kernel publishes runtime errors through the bridge.

OMN-19992 / OB-3: ``RuntimeLogEventBridge`` was attached only when the kernel
resolved the Kafka transport, so a runtime booted natively on the in-memory bus
published no ``runtime-error`` event and the error-fingerprint projection had
nothing to fold. This drives the real ``bootstrap()`` on the in-memory bus,
forces a handler exception through the logger the auto-wired handler path
reports on, and asserts the event reaches the selected bus inside the declared
bound.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.runtime.auto_wiring import handler_wiring
from omnibase_infra.runtime.service_kernel import bootstrap
from omnibase_infra.topics import topic_keys
from omnibase_infra.topics.service_topic_registry import ServiceTopicRegistry
from tests.conftest import check_service_registry_available
from tests.unit.runtime import conftest as runtime_conftest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not check_service_registry_available(),
        reason="service_registry unavailable (omnibase_core circular import)",
    ),
]

# AC1: the fingerprint row must exist within 60 s. The publish leg of that
# budget is far smaller; 10 s keeps a regression from hanging the suite.
PUBLISH_BOUND_SECONDS = 10.0
FINGERPRINT_LENGTH = 16


# Reuse the unit-suite fixture under a local name (fixtures register by attribute).
wire_infrastructure_mock = runtime_conftest.mock_wire_infrastructure


class _ForcedHandlerError(RuntimeError):
    """The exception the forced handler raises."""


async def _force_handler_exception() -> None:
    """Raise from a handler and report it the way handler wiring does."""
    try:
        raise _ForcedHandlerError("forced native handler failure")
    except _ForcedHandlerError as exc:
        handler_wiring.logger.error(
            "Projection handler error: handler=%s topic=%s error_type=%s error=%s",
            "ForcedHandler",
            "onex.evt.test.forced.v1",
            type(exc).__name__,
            exc,
            exc_info=exc,
        )


@pytest.fixture
async def native_bus() -> AsyncGenerator[EventBusInmemory, None]:
    bus = EventBusInmemory(environment="local", group="omn19992")
    await bus.start()
    yield bus
    await bus.close()


async def test_native_kernel_publishes_runtime_error_event(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    native_bus: EventBusInmemory,
    wire_infrastructure_mock: MagicMock,
) -> None:
    runtime_conftest.force_inmemory_runtime_config(monkeypatch, tmp_path)
    monkeypatch.delenv("OMNIBASE_INFRA_DB_URL", raising=False)
    monkeypatch.setenv("ENABLE_RUNTIME_LOG_BRIDGE", "true")
    topic = ServiceTopicRegistry.from_defaults().resolve(topic_keys.RUNTIME_ERROR)
    seen: list[dict[str, object]] = []

    async def start_runtime_and_force_error() -> None:
        await _force_handler_exception()
        deadline = time.monotonic() + PUBLISH_BOUND_SECONDS
        while time.monotonic() < deadline:
            history = await native_bus.get_event_history(topic=topic)
            if history:
                seen.extend(json.loads(message.value) for message in history)
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
    event = seen[0]
    assert event["exception_type"] == "_ForcedHandlerError"
    assert event["logger_family"] == handler_wiring.logger.name
    fingerprint = event["fingerprint"]
    assert isinstance(fingerprint, str)
    assert len(fingerprint) == FINGERPRINT_LENGTH
