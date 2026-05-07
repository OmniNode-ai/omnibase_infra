# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from omnibase_core.models.dispatch.model_dispatch_bus_command import (
    ModelDispatchBusCommand,
)
from omnibase_core.models.dispatch.model_dispatch_bus_terminal_result import (
    ModelDispatchBusTerminalResult,
)
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.runtime.runtime_local_ingress import RuntimeLocalIngressRoute
from omnibase_infra.runtime.service_pattern_b_broker import RuntimePatternBBroker
from omnibase_infra.runtime.service_runtime_host_process import RuntimeHostProcess

pytestmark = pytest.mark.unit


def _route() -> RuntimeLocalIngressRoute:
    return RuntimeLocalIngressRoute(
        node_name="node_session_orchestrator",
        contract_name="session_orchestrator",
        command_topic="onex.cmd.omnimarket.session-orchestrator-start.v1",
        event_type="omnimarket.session-orchestrator-start",
        terminal_event="onex.evt.omnimarket.session-orchestrator-completed.v1",
        contract_path="/tmp/node_session_orchestrator/contract.yaml",  # noqa: S108
        package_name="omnimarket",
    )


async def _collect_terminal_result(
    bus: EventBusInmemory,
    response_topic: str,
) -> asyncio.Queue[ModelDispatchBusTerminalResult]:
    queue: asyncio.Queue[ModelDispatchBusTerminalResult] = asyncio.Queue(maxsize=1)

    async def on_message(message: ModelEventMessage) -> None:
        envelope = ModelEventEnvelope[
            ModelDispatchBusTerminalResult
        ].model_validate_json(message.value)
        if queue.empty():
            await queue.put(envelope.payload)

    await bus.subscribe(
        response_topic,
        group_id=f"collector-{uuid4()}",
        on_message=on_message,
    )
    return queue


@pytest.mark.asyncio
async def test_service_pattern_b_broker_round_trips_terminal_event() -> None:
    bus = EventBusInmemory(environment="test", group="pattern-b")
    await bus.start()

    route = _route()
    broker = RuntimePatternBBroker(
        bus,
        command_topic="onex.cmd.omnibase-infra.pattern-b-dispatch.v1",
        routes={"session_orchestrator": route},
    )
    await broker.start()

    async def worker(message: ModelEventMessage) -> None:
        envelope = ModelEventEnvelope[object].model_validate_json(message.value)
        terminal_envelope = ModelEventEnvelope[object](
            payload={"status": "complete", "dispatch_count": 5},
            correlation_id=envelope.correlation_id,
            envelope_timestamp=datetime.now(UTC),
            event_type=route.terminal_event,
            source_tool="session_orchestrator",
        )
        await bus.publish(
            route.terminal_event or "unknown",
            None,
            terminal_envelope.model_dump_json().encode("utf-8"),
            None,
        )

    await bus.subscribe(route.command_topic, group_id="worker", on_message=worker)

    response_topic = "onex.evt.pattern-b.dispatch-completed.v1"
    results = await _collect_terminal_result(bus, response_topic)

    command = ModelDispatchBusCommand(
        command_name="session_orchestrator",
        requester="codex",
        payload={"dry_run": True},
        response_topic=response_topic,
        timeout_seconds=1,
    )
    envelope = ModelEventEnvelope[ModelDispatchBusCommand](
        payload=command,
        correlation_id=command.correlation_id,
        envelope_timestamp=datetime.now(UTC),
        event_type="onex.cmd.omnibase-infra.pattern-b-dispatch.v1",
        source_tool="codex",
    )
    await bus.publish(
        "onex.cmd.omnibase-infra.pattern-b-dispatch.v1",
        None,
        envelope.model_dump_json().encode("utf-8"),
        None,
    )

    result = await asyncio.wait_for(results.get(), timeout=2)

    assert result.status == "completed"
    assert result.payload == {"status": "complete", "dispatch_count": 5}

    await broker.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_service_pattern_b_broker_publishes_timeout_result() -> None:
    bus = EventBusInmemory(environment="test", group="pattern-b")
    await bus.start()

    route = _route()
    broker = RuntimePatternBBroker(
        bus,
        command_topic="onex.cmd.omnibase-infra.pattern-b-dispatch.v1",
        routes={"session_orchestrator": route},
    )
    await broker.start()

    response_topic = "onex.evt.pattern-b.dispatch-completed.v1"
    results = await _collect_terminal_result(bus, response_topic)

    command = ModelDispatchBusCommand(
        command_name="session_orchestrator",
        requester="codex",
        payload={"dry_run": True},
        response_topic=response_topic,
        timeout_seconds=1,
    )
    envelope = ModelEventEnvelope[ModelDispatchBusCommand](
        payload=command,
        correlation_id=command.correlation_id,
        envelope_timestamp=datetime.now(UTC),
        event_type="onex.cmd.omnibase-infra.pattern-b-dispatch.v1",
        source_tool="codex",
    )
    await bus.publish(
        "onex.cmd.omnibase-infra.pattern-b-dispatch.v1",
        None,
        envelope.model_dump_json().encode("utf-8"),
        None,
    )

    result = await asyncio.wait_for(results.get(), timeout=2)

    assert result.status == "timeout"
    assert result.error_message is not None

    await broker.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_service_pattern_b_broker_publishes_failed_result_for_unknown_route() -> (
    None
):
    bus = EventBusInmemory(environment="test", group="pattern-b")
    await bus.start()

    broker = RuntimePatternBBroker(
        bus,
        command_topic="onex.cmd.omnibase-infra.pattern-b-dispatch.v1",
        routes={},
    )
    await broker.start()

    response_topic = "onex.evt.pattern-b.dispatch-completed.v1"
    results = await _collect_terminal_result(bus, response_topic)

    command = ModelDispatchBusCommand(
        command_name="missing_route",
        requester="codex",
        payload={"dry_run": True},
        response_topic=response_topic,
        timeout_seconds=1,
    )
    envelope = ModelEventEnvelope[ModelDispatchBusCommand](
        payload=command,
        correlation_id=command.correlation_id,
        envelope_timestamp=datetime.now(UTC),
        event_type="onex.cmd.omnibase-infra.pattern-b-dispatch.v1",
        source_tool="codex",
    )
    await bus.publish(
        "onex.cmd.omnibase-infra.pattern-b-dispatch.v1",
        None,
        envelope.model_dump_json().encode("utf-8"),
        None,
    )

    result = await asyncio.wait_for(results.get(), timeout=1)

    assert result.status == "failed"
    assert result.error_message is not None
    assert "Unknown Pattern B route" in result.error_message

    await broker.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_service_pattern_b_broker_sanitizes_dispatch_exception() -> None:
    class FailingPublishTransport:
        async def publish(
            self,
            _topic: str,
            _key: bytes | None,
            _value: bytes,
            _headers: object | None = None,
        ) -> None:
            raise RuntimeError("failed to connect to postgres://user:pass@db:5432/app")

        async def subscribe(
            self,
            _topic: str,
            *_args: object,
            **_kwargs: object,
        ) -> object:
            async def _unsubscribe() -> None:
                return None

            return _unsubscribe

    route = _route()
    broker = RuntimePatternBBroker(
        FailingPublishTransport(),
        command_topic="onex.cmd.omnibase-infra.pattern-b-dispatch.v1",
        routes={"session_orchestrator": route},
    )

    command = ModelDispatchBusCommand(
        command_name="session_orchestrator",
        requester="codex",
        payload={"dry_run": True},
        response_topic="onex.evt.pattern-b.dispatch-completed.v1",
        timeout_seconds=1,
    )
    resolved_route, result = await broker.dispatch_request(command)

    assert resolved_route == route
    assert result.status == "failed"
    assert (
        result.error_message == "RuntimeError: [REDACTED - potentially sensitive data]"
    )


@pytest.mark.asyncio
async def test_runtime_host_process_starts_pattern_b_broker_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = _route()
    captured: dict[str, object] = {}

    class FakeBroker:
        def __init__(
            self,
            event_bus: object,
            *,
            command_topic: str,
            routes: object,
        ) -> None:
            captured["event_bus"] = event_bus
            captured["command_topic"] = command_topic
            captured["routes"] = routes
            self.start = AsyncMock()
            self.stop = AsyncMock()
            captured["start_mock"] = self.start

    monkeypatch.setattr(
        "omnibase_infra.runtime.service_runtime_host_process.discover_runtime_local_ingress_routes",
        lambda packages: captured.setdefault("packages", packages)
        and {"session_orchestrator": route},
    )
    monkeypatch.setattr(
        "omnibase_infra.runtime.service_runtime_host_process.RuntimePatternBBroker",
        FakeBroker,
    )

    process = RuntimeHostProcess(
        event_bus=EventBusInmemory(environment="test", group="pattern-b"),
        config={
            "service_name": "test-service",
            "node_name": "test-node",
            "env": "test",
            "version": "v1",
            "pattern_b_broker": {
                "enabled": True,
                "command_topic": "onex.cmd.omnibase-infra.pattern-b-dispatch.v1",
                "package_names": ["omnibase_infra", "omnimarket"],
            },
        },
        dispatch_engine=AsyncMock(),
    )

    await process._start_pattern_b_broker()

    assert process._pattern_b_broker is not None
    assert captured["command_topic"] == "onex.cmd.omnibase-infra.pattern-b-dispatch.v1"
    assert captured["packages"] == ("omnibase_infra", "omnimarket")
    assert captured["routes"] == {"session_orchestrator": route}
    start_mock = captured["start_mock"]
    assert isinstance(start_mock, AsyncMock)
    start_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_runtime_host_process_broker_package_names_ignore_active_runtime_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = _route()
    captured: dict[str, object] = {}

    class FakeBroker:
        def __init__(
            self,
            event_bus: object,
            *,
            command_topic: str,
            routes: object,
        ) -> None:
            captured["routes"] = routes
            self.start = AsyncMock()

    monkeypatch.setenv("ONEX_ACTIVE_RUNTIME_PACKAGES", "omnibase_infra")
    monkeypatch.setattr(
        "omnibase_infra.runtime.service_runtime_host_process.discover_runtime_local_ingress_routes",
        lambda packages: captured.setdefault("packages", packages)
        and {"session_orchestrator": route},
    )
    monkeypatch.setattr(
        "omnibase_infra.runtime.service_runtime_host_process.RuntimePatternBBroker",
        FakeBroker,
    )

    process = RuntimeHostProcess(
        event_bus=EventBusInmemory(environment="test", group="pattern-b"),
        config={
            "service_name": "test-service",
            "node_name": "test-node",
            "env": "test",
            "version": "v1",
            "pattern_b_broker": {
                "enabled": True,
                "package_names": ["omnimarket"],
            },
        },
        dispatch_engine=AsyncMock(),
    )

    await process._start_pattern_b_broker()

    assert captured["packages"] == ("omnimarket",)
    assert captured["routes"] == {"session_orchestrator": route}


@pytest.mark.asyncio
async def test_runtime_host_process_reuses_local_ingress_routes_for_pattern_b_broker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = _route()
    captured: dict[str, object] = {}

    class FakeBroker:
        def __init__(
            self,
            event_bus: object,
            *,
            command_topic: str,
            routes: object,
        ) -> None:
            captured["event_bus"] = event_bus
            captured["command_topic"] = command_topic
            captured["routes"] = routes
            self.start = AsyncMock()

    monkeypatch.setattr(
        "omnibase_infra.runtime.service_runtime_host_process.discover_runtime_local_ingress_routes",
        lambda _packages: pytest.fail("broker should reuse local ingress routes"),
    )
    monkeypatch.setattr(
        "omnibase_infra.runtime.service_runtime_host_process.RuntimePatternBBroker",
        FakeBroker,
    )

    process = RuntimeHostProcess(
        event_bus=EventBusInmemory(environment="test", group="pattern-b"),
        config={
            "service_name": "test-service",
            "node_name": "test-node",
            "env": "test",
            "version": "v1",
            "local_ingress": {"enabled": True},
            "pattern_b_broker": {"enabled": True},
        },
        dispatch_engine=AsyncMock(),
    )
    process._local_ingress_routes = {"session_orchestrator": route}

    await process._start_pattern_b_broker()

    assert captured["routes"] == {"session_orchestrator": route}


@pytest.mark.asyncio
async def test_runtime_host_process_rejects_enabled_ingress_without_pattern_b_broker() -> (
    None
):
    process = RuntimeHostProcess(
        event_bus=EventBusInmemory(environment="test", group="pattern-b"),
        config={
            "service_name": "test-service",
            "node_name": "test-node",
            "env": "test",
            "version": "v1",
            "local_ingress": {"enabled": True},
            "pattern_b_broker": {"enabled": False},
        },
        dispatch_engine=AsyncMock(),
    )

    with pytest.raises(
        ProtocolConfigurationError,
        match=r"local runtime ingress requires pattern_b_broker\.enabled=true",
    ):
        await process._start_pattern_b_broker()
