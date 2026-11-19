# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests: the startup introspection handler refuses without a heartbeat.

OMN-24496: the heartbeat is mandatory. ``HandlerRuntimeIntrospectionStartup``
no longer degrades: a runtime that cannot build its introspection service or
cannot start its heartbeat task refuses to start with a typed, counted
``RuntimeHeartbeatRequiredError``, and no announced node is published for it.

Test Coverage:
- ``build_required_service``: golden build, empty-name refusal, unbuildable
  service refusal (typed ``__cause__``, counted per reason)
- ``start_heartbeat``: heartbeat-start failure refuses; service ``None`` and
  introspection disabled stay silent non-kernel paths
- ``handle``: refusal ordering — the runtime's own presence is published
  first, then the heartbeat starts, then announced nodes are built; a
  heartbeat that cannot start means no announced node is built at all
- ``ModelRuntimeHeartbeatRefusal``: frozen, closed, count >= 1
- Source-level kernel seam: the kernel builds its service only through
  ``build_required_service`` (matching the pattern of
  ``tests/unit/runtime/test_service_kernel_introspection_wiring.py``)

Related:
    - OMN-24496: Runtime heartbeat is mandatory
    - src/omnibase_infra/handlers/handler_runtime_introspection_startup.py
    - src/omnibase_infra/runtime/service_kernel.py
"""

from __future__ import annotations

import ast
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from omnibase_core.enums import EnumNodeKind
from omnibase_core.enums.enum_announced_node_kind import EnumAnnouncedNodeKind
from omnibase_core.enums.enum_announcement_operation import (
    EnumAnnouncementOperation,
)
from omnibase_core.models.events.contract_registration.model_announced_package import (
    ModelAnnouncedPackage,
)
from omnibase_core.models.events.contract_registration.model_node_announcement import (
    ModelNodeAnnouncement,
)
from omnibase_infra.enums import EnumRuntimeHeartbeatRefusalReason
from omnibase_infra.handlers.handler_runtime_introspection_startup import (
    HandlerRuntimeIntrospectionStartup,
    RuntimeHeartbeatRequiredError,
)
from omnibase_infra.models.model_runtime_heartbeat_refusal import (
    ModelRuntimeHeartbeatRefusal,
)
from omnibase_infra.models.model_runtime_introspection_startup_request import (
    ModelRuntimeIntrospectionStartupRequest,
)
from omnibase_infra.models.registration import ModelNodeIntrospectionEvent
from omnibase_infra.models.runtime import ModelRuntimeIntrospectionConfig
from omnibase_infra.protocols.protocol_event_bus_like import ProtocolEventBusLike

pytestmark = pytest.mark.unit

ANNOUNCER_ID = UUID("00000000-0000-4000-8000-000000000001")
ANNOUNCED_AT = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)

KERNEL_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "omnibase_infra"
    / "runtime"
    / "service_kernel.py"
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _reset_refusal_counts() -> None:
    """Reset the handler's class-level refusal counters around every test."""
    HandlerRuntimeIntrospectionStartup.reset_refusal_counts()


# =============================================================================
# Helpers
# =============================================================================


def make_service() -> MagicMock:
    """Create a mock introspection service implementing the protocol.

    ``get_introspection_data`` returns a real introspection event so the
    announced-node path can extract the runtime identity from it.
    """
    service = MagicMock()
    service.publish_introspection = AsyncMock()
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


def make_announcement(
    entity_id: str = "node_example_effect",
) -> ModelNodeAnnouncement:
    """Build one node announcement to pass in the startup request."""
    return ModelNodeAnnouncement(
        entity_id=entity_id,
        operation=EnumAnnouncementOperation.ANNOUNCE,
        content_hash="a" * 64,
        package=ModelAnnouncedPackage(distribution="omnibase-example", version="1.2.3"),
        node_kind=EnumAnnouncedNodeKind.NODE,
        declared_topics=("example.topic.v1",),
        announcer_id=ANNOUNCER_ID,
        generation=4,
        announced_at=ANNOUNCED_AT,
    )


def make_request(
    announcements: tuple[ModelNodeAnnouncement, ...] = (),
    last_introspection_time: float | None = None,
) -> ModelRuntimeIntrospectionStartupRequest:
    """Build one startup introspection request."""
    return ModelRuntimeIntrospectionStartupRequest(
        correlation_id=uuid4(),
        announcements=announcements,
        last_introspection_time=last_introspection_time,
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


def read_kernel_source() -> str:
    """Read service_kernel.py source for seam analysis."""
    assert KERNEL_PATH.exists(), f"service_kernel.py not found at {KERNEL_PATH}"
    return KERNEL_PATH.read_text(encoding="utf-8")


# =============================================================================
# build_required_service
# =============================================================================


class TestBuildRequiredService:
    """The refusing constructor: no service, no runtime."""

    @pytest.mark.asyncio
    async def test_build_required_service_returns_the_built_service(self) -> None:
        """A passing ``build`` callable's service is returned unchanged."""
        handler = HandlerRuntimeIntrospectionStartup()
        sentinel = MagicMock(name="introspection_service")
        build = MagicMock(name="build", return_value=sentinel)

        service = await handler.build_required_service(
            runtime_name="runtime-a",
            correlation_id=uuid4(),
            build=build,
            event_bus=None,
        )

        assert service is sentinel
        build.assert_called_once_with()
        assert (
            HandlerRuntimeIntrospectionStartup.refusal_count(
                EnumRuntimeHeartbeatRefusalReason.INTROSPECTION_SERVICE_UNBUILDABLE
            )
            == 0
        )

    @pytest.mark.parametrize("runtime_name", ["", "   "])
    @pytest.mark.asyncio
    async def test_empty_runtime_name_refuses(self, runtime_name: str) -> None:
        """An empty or whitespace runtime name refuses before building."""
        handler = HandlerRuntimeIntrospectionStartup()
        build = MagicMock(name="build")

        with pytest.raises(RuntimeHeartbeatRequiredError) as exc_info:
            await handler.build_required_service(
                runtime_name=runtime_name,
                correlation_id=uuid4(),
                build=build,
                event_bus=None,
            )

        refusal = exc_info.value.refusal
        assert refusal.reason is (EnumRuntimeHeartbeatRefusalReason.RUNTIME_NAME_EMPTY)
        assert refusal.runtime_label == runtime_name
        assert refusal.count == 1
        build.assert_not_called()
        assert (
            HandlerRuntimeIntrospectionStartup.refusal_count(
                EnumRuntimeHeartbeatRefusalReason.RUNTIME_NAME_EMPTY
            )
            == 1
        )

    @pytest.mark.asyncio
    async def test_empty_runtime_name_refusals_count_up(self) -> None:
        """Each refused start increments the reason's counter: 1 then 2."""
        handler = HandlerRuntimeIntrospectionStartup()
        build = MagicMock(name="build")

        for expected_count in (1, 2):
            with pytest.raises(RuntimeHeartbeatRequiredError) as exc_info:
                await handler.build_required_service(
                    runtime_name="",
                    correlation_id=uuid4(),
                    build=build,
                    event_bus=None,
                )
            assert exc_info.value.refusal.count == expected_count

        assert (
            HandlerRuntimeIntrospectionStartup.refusal_count(
                EnumRuntimeHeartbeatRefusalReason.RUNTIME_NAME_EMPTY
            )
            == 2
        )
        build.assert_not_called()

    @pytest.mark.asyncio
    async def test_unbuildable_service_refuses_with_typed_cause(self) -> None:
        """A ``build`` failure refuses with the cause chained and counted."""
        handler = HandlerRuntimeIntrospectionStartup()
        failure = ValueError("no contract")
        build = MagicMock(name="build", side_effect=failure)

        with pytest.raises(RuntimeHeartbeatRequiredError) as exc_info:
            await handler.build_required_service(
                runtime_name="runtime-a",
                correlation_id=uuid4(),
                build=build,
                event_bus=None,
            )

        refusal = exc_info.value.refusal
        assert refusal.reason is (
            EnumRuntimeHeartbeatRefusalReason.INTROSPECTION_SERVICE_UNBUILDABLE
        )
        assert "ValueError: no contract" in refusal.detail
        assert exc_info.value.__cause__ is failure
        assert refusal.count == 1
        assert (
            HandlerRuntimeIntrospectionStartup.refusal_count(
                EnumRuntimeHeartbeatRefusalReason.INTROSPECTION_SERVICE_UNBUILDABLE
            )
            == 1
        )


# =============================================================================
# start_heartbeat
# =============================================================================


class TestStartHeartbeat:
    """The heartbeat start: mandatory for kernel runtimes."""

    @pytest.mark.asyncio
    async def test_heartbeat_start_failure_refuses(self) -> None:
        """A failing ``start_heartbeat_task`` is a typed, counted refusal.

        Control: the old code swallowed the failure and returned ``None``,
        leaving the runtime running without a heartbeat.
        """
        service = make_service()
        service.start_heartbeat_task.side_effect = RuntimeError("boom")
        handler = HandlerRuntimeIntrospectionStartup()

        with pytest.raises(RuntimeHeartbeatRequiredError) as exc_info:
            await handler.start_heartbeat(
                uuid4(), service=service, config=make_config()
            )

        refusal = exc_info.value.refusal
        assert refusal.reason is (
            EnumRuntimeHeartbeatRefusalReason.HEARTBEAT_START_FAILED
        )
        assert refusal.count == 1
        service.start_heartbeat_task.assert_awaited_once()
        assert (
            HandlerRuntimeIntrospectionStartup.refusal_count(
                EnumRuntimeHeartbeatRefusalReason.HEARTBEAT_START_FAILED
            )
            == 1
        )

    @pytest.mark.asyncio
    async def test_no_service_returns_none_without_refusing(self) -> None:
        """Service ``None`` is a non-kernel path: silent, not a refusal."""
        handler = HandlerRuntimeIntrospectionStartup()

        result = await handler.start_heartbeat(
            uuid4(), service=None, config=make_config()
        )

        assert result is None
        assert (
            HandlerRuntimeIntrospectionStartup.refusal_count(
                EnumRuntimeHeartbeatRefusalReason.HEARTBEAT_START_FAILED
            )
            == 0
        )

    @pytest.mark.asyncio
    async def test_disabled_config_returns_none_without_starting(self) -> None:
        """Introspection disabled by config: silent, heartbeat not started."""
        service = make_service()
        handler = HandlerRuntimeIntrospectionStartup()

        result = await handler.start_heartbeat(
            uuid4(),
            service=service,
            config=ModelRuntimeIntrospectionConfig(enabled=False),
        )

        assert result is None
        service.start_heartbeat_task.assert_not_awaited()
        assert (
            HandlerRuntimeIntrospectionStartup.refusal_count(
                EnumRuntimeHeartbeatRefusalReason.HEARTBEAT_START_FAILED
            )
            == 0
        )


# =============================================================================
# handle: refusal ordering
# =============================================================================


class TestHandleRefusesWithoutHeartbeat:
    """The heartbeat starts after the runtime's own presence, before any
    announced node is built or published."""

    @pytest.mark.asyncio
    async def test_failing_heartbeat_refuses_after_own_presence(
        self,
    ) -> None:
        """A heartbeat that cannot start builds no announced node.

        Verifies:
        - the runtime's own ``publish_introspection`` was awaited once first
        - the refusal raises out of ``handle``
        - ``build_events`` was never called: no announced node is built or
          published for a runtime without a heartbeat
        """
        service = make_service()
        service.start_heartbeat_task.side_effect = RuntimeError("boom")
        handler = HandlerRuntimeIntrospectionStartup()
        build_events = MagicMock(name="build_events")

        with pytest.raises(RuntimeHeartbeatRequiredError):
            await handler.handle(
                make_request((make_announcement(),)),
                service=service,
                config=make_config(),
                event_bus=make_bus(),
                build_events=build_events,
            )

        service.publish_introspection.assert_awaited_once()
        build_events.assert_not_called()

    @pytest.mark.asyncio
    async def test_working_service_orders_publish_then_heartbeat_then_events(
        self,
    ) -> None:
        """Golden order: publish, start heartbeat, then build events."""
        service = make_service()
        handler = HandlerRuntimeIntrospectionStartup()

        manager = MagicMock(name="call-order")
        manager.attach_mock(service.publish_introspection, "publish_introspection")
        manager.attach_mock(service.start_heartbeat_task, "start_heartbeat_task")
        build_events = MagicMock(name="build_events", return_value=())
        manager.attach_mock(build_events, "build_events")

        result = await handler.handle(
            make_request((make_announcement(),)),
            service=service,
            config=make_config(),
            event_bus=make_bus(),
            build_events=build_events,
        )

        assert result.published
        tracked = {
            "publish_introspection",
            "start_heartbeat_task",
            "build_events",
        }
        observed = [
            name.split(".")[0]
            for name, _args, _kwargs in manager.mock_calls
            if name.split(".")[0] in tracked
        ]
        assert observed == [
            "publish_introspection",
            "start_heartbeat_task",
            "build_events",
        ]

    @pytest.mark.asyncio
    async def test_publish_failure_keeps_heartbeat_from_starting(self) -> None:
        """Existing behaviour kept: a failed publish never starts a heartbeat."""
        service = make_service()
        service.publish_introspection.side_effect = RuntimeError("bus down")
        handler = HandlerRuntimeIntrospectionStartup()

        result = await handler.handle(
            make_request(),
            service=service,
            config=make_config(),
            event_bus=None,
            build_events=MagicMock(name="build_events"),
        )

        assert result.published is False
        assert result.last_introspection_time is None
        service.start_heartbeat_task.assert_not_awaited()
        assert (
            HandlerRuntimeIntrospectionStartup.refusal_count(
                EnumRuntimeHeartbeatRefusalReason.HEARTBEAT_START_FAILED
            )
            == 0
        )

    @pytest.mark.asyncio
    async def test_throttled_request_publishes_and_starts_nothing(self) -> None:
        """A request inside the throttle window publishes and starts nothing."""
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
        service.publish_introspection.assert_not_awaited()
        service.start_heartbeat_task.assert_not_awaited()


# =============================================================================
# ModelRuntimeHeartbeatRefusal
# =============================================================================


class TestRefusalModel:
    """The refusal payload is frozen, closed, and counts from one."""

    def test_refusal_is_frozen(self) -> None:
        """Assigning a field on a refusal is rejected."""
        refusal = make_refusal()
        with pytest.raises(ValidationError):
            refusal.__setattr__("detail", "changed")

    def test_refusal_rejects_extra_fields(self) -> None:
        """Unknown fields are rejected (extra="forbid")."""
        with pytest.raises(ValidationError):
            make_refusal(surprise="unexpected")

    def test_refusal_rejects_count_below_one(self) -> None:
        """A refusal always carries a count of at least one."""
        with pytest.raises(ValidationError):
            make_refusal(count=0)


# =============================================================================
# Kernel seam
# =============================================================================


class TestKernelRefusesWithoutHeartbeat:
    """Source-level seam: the kernel builds its service only through the
    refusing constructor (OMN-24496), matching the AST/text pattern used by
    ``tests/unit/runtime/test_service_kernel_introspection_wiring.py``."""

    def test_kernel_builds_service_through_build_required_service(self) -> None:
        """service_kernel.py must call ``build_required_service(``.

        The kernel has no None fallback: the only way it obtains its
        introspection service is through the constructor that refuses to
        start a runtime without one.
        """
        source = read_kernel_source()
        tree = ast.parse(source)

        # The call must be real code, not a mention in a string or comment:
        # match ``<something>.build_required_service(`` / bare call sites in
        # the AST, then require the text too.
        found_call = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            attr_name = func.attr if isinstance(func, ast.Attribute) else None
            if attr_name == "build_required_service":
                found_call = True
                break

        assert found_call, (
            "service_kernel.py no longer calls build_required_service() — "
            "the kernel would build its introspection service without the "
            "mandatory-heartbeat refusal (OMN-24496)"
        )
        assert "build_required_service(" in source

    def test_kernel_no_longer_announces_introspection_disabled(self) -> None:
        """The old degrade message must be gone from service_kernel.py.

        ``"introspection will be disabled"`` was the log line the kernel
        emitted when it fell back to no introspection service. The rule is
        now mandatory: there is no fallback, so the string must not return.
        """
        source = read_kernel_source()
        assert "introspection will be disabled" not in source, (
            "service_kernel.py still carries the old 'introspection will be "
            "disabled' fallback wording — the heartbeat is mandatory, the "
            "kernel must refuse instead of degrading (OMN-24496)"
        )
