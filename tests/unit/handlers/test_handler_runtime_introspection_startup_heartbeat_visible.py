# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests: a refused heartbeat is visible on the bus and in readiness.

OMN-24559: the mandatory heartbeat (OMN-24496) must never run silent. Every
refusal of :class:`HandlerRuntimeIntrospectionStartup` is announced as a
runtime-error event on the bus (best-effort) before it propagates, a first
heartbeat that cannot be published within the configured attempts refuses the
startup, and the heartbeat readiness row reads false with a typed reason.

Test Coverage:
- ``handle`` with ``heartbeat_required``: service ``None``, introspection
  disabled, a first heartbeat that never lands and a service without
  ``publish_heartbeat_now`` all refuse instead of skipping; the same calls
  without ``heartbeat_required`` keep the old skip behaviour (controls)
- bounded retry: a startup publish that always fails refuses after the
  configured attempts, one that fails once then succeeds publishes
- throttling: a required runtime still starts and confirms its heartbeat
- ``publish_refusal_event``: envelope payload on the runtime error topic,
  raw-JSON fallback, best-effort on a failing or missing bus
- ``heartbeat_readiness``: missing service, disabled config, unhealthy and
  untyped verdicts read not ready, never assumed healthy

Related:
    - OMN-24559: a runtime that cannot heartbeat is visibly not healthy
    - OMN-24496: Runtime heartbeat is mandatory
    - contracts/OMN-24559.yaml
    - src/omnibase_infra/handlers/handler_runtime_introspection_startup.py
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from omnibase_core.enums import EnumNodeKind
from omnibase_infra.enums import EnumRuntimeHeartbeatRefusalReason
from omnibase_infra.handlers.handler_runtime_introspection_startup import (
    HandlerRuntimeIntrospectionStartup,
    RuntimeHeartbeatRequiredError,
)
from omnibase_infra.models.health.enum_runtime_error_category import (
    EnumRuntimeErrorCategory,
)
from omnibase_infra.models.health.enum_runtime_error_severity import (
    EnumRuntimeErrorSeverity,
)
from omnibase_infra.models.health.model_runtime_error_event import (
    ModelRuntimeErrorEvent,
)
from omnibase_infra.models.model_runtime_heartbeat_health import (
    ModelRuntimeHeartbeatHealth,
)
from omnibase_infra.models.model_runtime_heartbeat_refusal import (
    ModelRuntimeHeartbeatRefusal,
)
from omnibase_infra.models.model_runtime_introspection_startup_request import (
    ModelRuntimeIntrospectionStartupRequest,
)
from omnibase_infra.models.registration import ModelNodeIntrospectionEvent
from omnibase_infra.models.runtime import ModelRuntimeIntrospectionConfig
from omnibase_infra.models.runtime_timeouts import RUNTIME_TIMEOUTS
from omnibase_infra.protocols.protocol_event_bus_like import ProtocolEventBusLike
from omnibase_infra.topics import SUFFIX_RUNTIME_ERROR

pytestmark = pytest.mark.unit

ANNOUNCED_AT = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _reset_refusal_counts() -> None:
    """Reset the handler's class-level refusal counters around every test."""
    HandlerRuntimeIntrospectionStartup.reset_refusal_counts()


@pytest.fixture(autouse=True)
def _no_retry_waits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the handler's sleeps so retry budgets need no real waiting."""
    monkeypatch.setattr(
        "omnibase_infra.handlers.handler_runtime_introspection_startup.asyncio.sleep",
        AsyncMock(),
    )


# =============================================================================
# Helpers
# =============================================================================


def make_service() -> MagicMock:
    """Create a mock introspection service implementing the protocol.

    ``get_introspection_data`` returns a real introspection event so the
    announced-node path can extract the runtime identity from it;
    ``publish_heartbeat_now`` succeeds so the golden path confirms its
    first heartbeat.
    """
    service = MagicMock()
    service.publish_introspection = AsyncMock()
    service.publish_heartbeat_now = AsyncMock(return_value=True)
    service.start_heartbeat_task = AsyncMock()
    service.stop_heartbeat_task = AsyncMock()
    service.get_introspection_data = AsyncMock(
        return_value=ModelNodeIntrospectionEvent(
            node_id=uuid4(),
            node_type=EnumNodeKind.EFFECT,
            correlation_id=uuid4(),
            timestamp=ANNOUNCED_AT,
            endpoints={"health": "http://localhost:8085/health"},
        )
    )
    return service


def make_config() -> ModelRuntimeIntrospectionConfig:
    """Deterministic introspection config: no jitter, long throttle."""
    return ModelRuntimeIntrospectionConfig(
        enabled=True, jitter_max_ms=0, throttle_min_interval_s=10
    )


def make_bus() -> MagicMock:
    """Create a mock event bus accepting envelope publishes."""
    bus = MagicMock(spec=ProtocolEventBusLike)
    bus.publish_envelope = AsyncMock()
    bus.publish = AsyncMock()
    return bus


def make_request(
    last_introspection_time: float | None = None,
    heartbeat_required: bool = False,
) -> ModelRuntimeIntrospectionStartupRequest:
    """Build one startup introspection request."""
    return ModelRuntimeIntrospectionStartupRequest(
        correlation_id=uuid4(),
        announcements=(),
        last_introspection_time=last_introspection_time,
        heartbeat_required=heartbeat_required,
    )


def make_refusal(**overrides: Any) -> ModelRuntimeHeartbeatRefusal:
    """Build one valid refusal, with fields overridden per test."""
    values: dict[str, Any] = {
        "runtime_label": "runtime-a",
        "reason": EnumRuntimeHeartbeatRefusalReason.HEARTBEAT_START_FAILED,
        "detail": "RuntimeError: boom",
        "correlation_id": uuid4(),
        "count": 1,
    }
    values.update(overrides)
    return ModelRuntimeHeartbeatRefusal(**values)


def read_envelope_payload(bus: MagicMock) -> ModelRuntimeErrorEvent:
    """Return the typed runtime-error event the bus received by envelope."""
    call = bus.publish_envelope.await_args
    assert call is not None, "the bus received no envelope publish"
    assert call.kwargs["topic"] == SUFFIX_RUNTIME_ERROR
    payload = call.kwargs["envelope"].payload
    assert isinstance(payload, ModelRuntimeErrorEvent)
    return payload


# =============================================================================
# handle: a required heartbeat refuses instead of skipping
# =============================================================================


class TestRequiredHeartbeatRefusesInsteadOfSkipping:
    """The two HIGH cases: required refusals that dev used to skip silently."""

    @pytest.mark.asyncio
    async def test_service_none_refuses_instead_of_skipping(self) -> None:
        """No service with a required heartbeat refuses; no service without
        it still skips quietly (control)."""
        handler = HandlerRuntimeIntrospectionStartup()

        with pytest.raises(RuntimeHeartbeatRequiredError) as exc_info:
            await handler.handle(
                make_request(heartbeat_required=True),
                service=None,
                config=make_config(),
                event_bus=make_bus(),
                build_events=MagicMock(name="build_events"),
            )

        assert exc_info.value.refusal.reason is (
            EnumRuntimeHeartbeatRefusalReason.INTROSPECTION_SERVICE_MISSING
        )

    @pytest.mark.asyncio
    async def test_service_none_without_required_skips(self) -> None:
        """Control: the same call without the requirement returns, not raises."""
        handler = HandlerRuntimeIntrospectionStartup()

        result = await handler.handle(
            make_request(),
            service=None,
            config=make_config(),
            event_bus=make_bus(),
            build_events=MagicMock(name="build_events"),
        )

        assert result.published is False

    @pytest.mark.asyncio
    async def test_disabled_config_refuses_instead_of_skipping(self) -> None:
        """Disabled introspection with a required heartbeat refuses without
        starting a heartbeat task."""
        service = make_service()
        handler = HandlerRuntimeIntrospectionStartup()

        with pytest.raises(RuntimeHeartbeatRequiredError) as exc_info:
            await handler.handle(
                make_request(heartbeat_required=True),
                service=service,
                config=ModelRuntimeIntrospectionConfig(enabled=False),
                event_bus=make_bus(),
                build_events=MagicMock(name="build_events"),
            )

        assert exc_info.value.refusal.reason is (
            EnumRuntimeHeartbeatRefusalReason.INTROSPECTION_DISABLED
        )
        service.start_heartbeat_task.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_disabled_config_without_required_skips(self) -> None:
        """Control: disabled introspection without the requirement skips."""
        service = make_service()
        handler = HandlerRuntimeIntrospectionStartup()

        result = await handler.handle(
            make_request(),
            service=service,
            config=ModelRuntimeIntrospectionConfig(enabled=False),
            event_bus=make_bus(),
            build_events=MagicMock(name="build_events"),
        )

        assert result.published is False

    @pytest.mark.asyncio
    async def test_first_heartbeat_that_never_publishes_refuses(self) -> None:
        """A heartbeat task that starts but never lands a publish refuses
        after exactly the configured attempts, with no announced node."""
        service = make_service()
        service.publish_heartbeat_now = AsyncMock(return_value=False)
        handler = HandlerRuntimeIntrospectionStartup()
        build_events = MagicMock(name="build_events")

        with pytest.raises(RuntimeHeartbeatRequiredError) as exc_info:
            await handler.handle(
                make_request(heartbeat_required=True),
                service=service,
                config=make_config(),
                event_bus=make_bus(),
                build_events=build_events,
            )

        assert exc_info.value.refusal.reason is (
            EnumRuntimeHeartbeatRefusalReason.HEARTBEAT_PUBLISH_FAILED
        )
        assert (
            service.publish_heartbeat_now.await_count
            == RUNTIME_TIMEOUTS.introspection_heartbeat_start_attempts
        )
        build_events.assert_not_called()

    @pytest.mark.asyncio
    async def test_first_heartbeat_that_publishes_starts_the_runtime(
        self,
    ) -> None:
        """Control: a first heartbeat that lands publishes the startup."""
        service = make_service()
        handler = HandlerRuntimeIntrospectionStartup()

        result = await handler.handle(
            make_request(heartbeat_required=True),
            service=service,
            config=make_config(),
            event_bus=make_bus(),
            build_events=MagicMock(name="build_events"),
        )

        assert result.published is True
        service.publish_heartbeat_now.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_service_without_publish_heartbeat_now_refuses(self) -> None:
        """A service that cannot publish a heartbeat on demand refuses."""
        service = make_service()
        del service.publish_heartbeat_now
        handler = HandlerRuntimeIntrospectionStartup()

        with pytest.raises(RuntimeHeartbeatRequiredError) as exc_info:
            await handler.handle(
                make_request(heartbeat_required=True),
                service=service,
                config=make_config(),
                event_bus=None,
                build_events=MagicMock(name="build_events"),
            )

        assert exc_info.value.refusal.reason is (
            EnumRuntimeHeartbeatRefusalReason.HEARTBEAT_PUBLISH_FAILED
        )

    @pytest.mark.asyncio
    async def test_startup_publish_always_failing_refuses_after_budget(
        self,
    ) -> None:
        """A startup publish that keeps failing refuses after the configured
        attempts, before any heartbeat task starts."""
        service = make_service()
        service.publish_introspection = AsyncMock(side_effect=RuntimeError("bus down"))
        handler = HandlerRuntimeIntrospectionStartup()

        with pytest.raises(RuntimeHeartbeatRequiredError) as exc_info:
            await handler.handle(
                make_request(heartbeat_required=True),
                service=service,
                config=make_config(),
                event_bus=None,
                build_events=MagicMock(name="build_events"),
            )

        assert exc_info.value.refusal.reason is (
            EnumRuntimeHeartbeatRefusalReason.STARTUP_PUBLISH_FAILED
        )
        assert (
            service.publish_introspection.await_count
            == RUNTIME_TIMEOUTS.introspection_heartbeat_start_attempts
        )
        service.start_heartbeat_task.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_startup_publish_failure_without_required_tries_once(
        self,
    ) -> None:
        """Control: without the requirement one failed attempt skips."""
        service = make_service()
        service.publish_introspection = AsyncMock(side_effect=RuntimeError("bus down"))
        handler = HandlerRuntimeIntrospectionStartup()

        result = await handler.handle(
            make_request(),
            service=service,
            config=make_config(),
            event_bus=None,
            build_events=MagicMock(name="build_events"),
        )

        assert result.published is False
        service.publish_introspection.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_startup_publish_failing_once_then_succeeding(self) -> None:
        """The bounded retry works: one failure then success publishes."""
        service = make_service()
        service.publish_introspection = AsyncMock(
            side_effect=[RuntimeError("bus down"), None]
        )
        handler = HandlerRuntimeIntrospectionStartup()

        result = await handler.handle(
            make_request(heartbeat_required=True),
            service=service,
            config=make_config(),
            event_bus=None,
            build_events=MagicMock(name="build_events"),
        )

        assert result.published is True
        service.publish_introspection.assert_awaited()  # exactly two attempts
        assert service.publish_introspection.await_count == 2

    @pytest.mark.asyncio
    async def test_throttled_required_request_still_confirms_its_heartbeat(
        self,
    ) -> None:
        """Throttling suppresses only the publish, never the heartbeat."""
        service = make_service()
        handler = HandlerRuntimeIntrospectionStartup()

        result = await handler.handle(
            make_request(
                last_introspection_time=time.monotonic(), heartbeat_required=True
            ),
            service=service,
            config=make_config(),
            event_bus=None,
            build_events=MagicMock(name="build_events"),
        )

        assert result.published is False
        service.start_heartbeat_task.assert_awaited_once()
        service.publish_heartbeat_now.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_throttled_request_without_required_starts_nothing(
        self,
    ) -> None:
        """Control: a throttled request without the requirement skips both."""
        service = make_service()
        handler = HandlerRuntimeIntrospectionStartup()

        result = await handler.handle(
            make_request(last_introspection_time=time.monotonic()),
            service=service,
            config=make_config(),
            event_bus=None,
            build_events=MagicMock(name="build_events"),
        )

        assert result.published is False
        service.start_heartbeat_task.assert_not_awaited()
        service.publish_heartbeat_now.assert_not_awaited()


# =============================================================================
# handle / publish_refusal_event: the refusal lands on the bus
# =============================================================================


class TestRefusalEventOnTheBus:
    """Every refusal is announced as a runtime-error event, best-effort."""

    @pytest.mark.asyncio
    async def test_refusal_event_rides_an_envelope(self) -> None:
        """An envelope bus receives one typed CRITICAL runtime-error event."""
        refusal = make_refusal()
        bus = make_bus()
        handler = HandlerRuntimeIntrospectionStartup()

        await handler.publish_refusal_event(refusal, bus)

        bus.publish_envelope.assert_awaited_once()
        event = read_envelope_payload(bus)
        assert event.severity is EnumRuntimeErrorSeverity.CRITICAL
        assert event.error_category is EnumRuntimeErrorCategory.RUNTIME
        assert refusal.reason.value in event.raw_message
        assert refusal.runtime_label in event.raw_message

    @pytest.mark.asyncio
    async def test_refusal_event_falls_back_to_raw_publish(self) -> None:
        """A bus without ``publish_envelope`` gets JSON bytes on ``publish``."""
        refusal = make_refusal()
        bus = MagicMock(spec=["publish"])
        bus.publish = AsyncMock()
        handler = HandlerRuntimeIntrospectionStartup()

        await handler.publish_refusal_event(refusal, bus)

        bus.publish.assert_awaited_once()
        call = bus.publish.await_args
        assert call.kwargs["topic"] == SUFFIX_RUNTIME_ERROR
        payload = json.loads(call.kwargs["value"])
        assert refusal.reason.value in payload["raw_message"]

    @pytest.mark.asyncio
    async def test_refusal_event_is_best_effort(self) -> None:
        """A rejecting bus and a missing bus both leave the refusal intact."""
        refusal = make_refusal()
        bus = make_bus()
        bus.publish_envelope = AsyncMock(side_effect=RuntimeError("bus down"))
        handler = HandlerRuntimeIntrospectionStartup()

        await handler.publish_refusal_event(refusal, bus)
        await handler.publish_refusal_event(refusal, None)

    @pytest.mark.asyncio
    async def test_handle_announces_a_missing_service_refusal(self) -> None:
        """A refused ``handle`` publishes once on the bus and still raises."""
        bus = make_bus()
        handler = HandlerRuntimeIntrospectionStartup()

        with pytest.raises(RuntimeHeartbeatRequiredError):
            await handler.handle(
                make_request(heartbeat_required=True),
                service=None,
                config=make_config(),
                event_bus=bus,
                build_events=MagicMock(name="build_events"),
            )

        bus.publish_envelope.assert_awaited_once()
        event = read_envelope_payload(bus)
        assert "introspection_service_missing" in event.raw_message

    @pytest.mark.asyncio
    async def test_empty_runtime_name_announces_then_refuses(self) -> None:
        """An empty runtime name publishes its refusal, then raises."""
        bus = make_bus()
        build = MagicMock(name="build")
        handler = HandlerRuntimeIntrospectionStartup()

        with pytest.raises(RuntimeHeartbeatRequiredError) as exc_info:
            await handler.build_required_service(
                runtime_name="",
                correlation_id=uuid4(),
                build=build,
                event_bus=bus,
            )

        assert exc_info.value.refusal.reason is (
            EnumRuntimeHeartbeatRefusalReason.RUNTIME_NAME_EMPTY
        )
        bus.publish_envelope.assert_awaited_once()
        event = read_envelope_payload(bus)
        assert "runtime_name_empty" in event.raw_message
        build.assert_not_called()


# =============================================================================
# heartbeat_readiness: the runtime's heartbeat readiness row
# =============================================================================


class TestHeartbeatReadinessRow:
    """The row reads false with a reason, never assumed healthy."""

    def test_missing_service_reads_not_ready(self) -> None:
        """No service reads false with the missing-service reason."""
        handler = HandlerRuntimeIntrospectionStartup()

        ready, detail = handler.heartbeat_readiness(service=None, config=make_config())

        assert ready is False
        assert detail["reason"] == "introspection_service_missing"
        assert isinstance(detail["summary"], str)

    def test_disabled_config_reads_not_ready(self) -> None:
        """Disabled introspection reads false with the disabled reason."""
        handler = HandlerRuntimeIntrospectionStartup()

        ready, detail = handler.heartbeat_readiness(
            service=make_service(),
            config=ModelRuntimeIntrospectionConfig(enabled=False),
        )

        assert ready is False
        assert detail["reason"] == "introspection_disabled"

    def test_unhealthy_verdict_reads_not_ready_with_its_reason(self) -> None:
        """An unhealthy typed verdict keeps the row false; a healthy typed
        verdict flips it true (control)."""
        handler = HandlerRuntimeIntrospectionStartup()
        service = make_service()
        service.heartbeat_health.return_value = ModelRuntimeHeartbeatHealth(
            healthy=False,
            reason=EnumRuntimeHeartbeatRefusalReason.HEARTBEAT_PUBLISH_FAILING,
            detail="boom",
            consecutive_failures=3,
        )

        ready, detail = handler.heartbeat_readiness(
            service=service, config=make_config()
        )

        assert ready is False
        assert detail["reason"] == "heartbeat_publish_failing"
        assert "boom" in detail["summary"]

        service.heartbeat_health.return_value = ModelRuntimeHeartbeatHealth(
            healthy=True
        )
        ready, _detail = handler.heartbeat_readiness(
            service=service, config=make_config()
        )
        assert ready is True

    def test_an_untyped_verdict_is_never_assumed_healthy(self) -> None:
        """A service whose health is not the typed model reads false."""
        handler = HandlerRuntimeIntrospectionStartup()
        service = make_service()  # heartbeat_health returns a plain MagicMock

        ready, detail = handler.heartbeat_readiness(
            service=service, config=make_config()
        )

        assert ready is False
        assert detail["reason"] == "heartbeat_task_stopped"
