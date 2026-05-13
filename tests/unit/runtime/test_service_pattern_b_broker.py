# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
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


def _route_with_failure_terminal() -> RuntimeLocalIngressRoute:
    route = _route()
    return RuntimeLocalIngressRoute(
        node_name=route.node_name,
        contract_name=route.contract_name,
        command_topic=route.command_topic,
        event_type=route.event_type,
        terminal_event=route.terminal_event,
        contract_path=route.contract_path,
        package_name=route.package_name,
        terminal_events=(
            "onex.evt.omnimarket.session-orchestrator-completed.v1",
            "onex.evt.omnimarket.session-orchestrator-failed.v1",
        ),
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
async def test_service_pattern_b_broker_returns_failed_for_failure_terminal() -> None:
    bus = EventBusInmemory(environment="test", group="pattern-b")
    await bus.start()

    route = _route_with_failure_terminal()
    failure_topic = route.terminal_events[1]
    broker = RuntimePatternBBroker(
        bus,
        command_topic="onex.cmd.omnibase-infra.pattern-b-dispatch.v1",
        routes={"session_orchestrator": route},
    )
    await broker.start()

    async def worker(message: ModelEventMessage) -> None:
        envelope = ModelEventEnvelope[object].model_validate_json(message.value)
        terminal_envelope = ModelEventEnvelope[object](
            payload={"payload": {"failure_reason": "configured endpoint missing"}},
            correlation_id=envelope.correlation_id,
            envelope_timestamp=datetime.now(UTC),
            event_type=failure_topic,
            source_tool="session_orchestrator",
        )
        await bus.publish(
            failure_topic,
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

    assert result.status == "failed"
    assert result.error_message == "configured endpoint missing"

    await broker.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_service_pattern_b_broker_kafka_waiter_seeks_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = _route()
    created_consumers: list[FakeAIOKafkaConsumer] = []

    class FakeAIOKafkaConsumer:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.messages: asyncio.Queue[SimpleNamespace] = asyncio.Queue()
            self.seeked_to_end = False
            self.started = False
            self.stopped = False
            created_consumers.append(self)

        async def start(self) -> None:
            self.started = True

        def assignment(self) -> set[str]:
            return {"partition-0"} if self.started else set()

        async def seek_to_end(self, *_assignment: object) -> None:
            self.seeked_to_end = True

        async def getone(self) -> SimpleNamespace:
            return await self.messages.get()

        async def stop(self) -> None:
            self.stopped = True

    class FakeKafkaTransport:
        config = SimpleNamespace(
            session_timeout_ms=45000,
            heartbeat_interval_ms=15000,
            max_poll_interval_ms=300000,
            reconnect_backoff_ms=2000,
        )
        _bootstrap_servers = "redpanda:9092"

        def _build_auth_kwargs(self) -> dict[str, object]:
            return {}

        async def publish(
            self,
            _topic: str,
            _key: bytes | None,
            value: bytes,
            _headers: object | None = None,
        ) -> None:
            assert created_consumers
            assert created_consumers[-1].seeked_to_end is True
            command_envelope = ModelEventEnvelope[object].model_validate_json(value)
            terminal_envelope = ModelEventEnvelope[object](
                payload={"status": "complete", "dispatch_count": 5},
                correlation_id=command_envelope.correlation_id,
                envelope_timestamp=datetime.now(UTC),
                event_type=route.terminal_event,
                source_tool="session_orchestrator",
            )
            await created_consumers[-1].messages.put(
                SimpleNamespace(value=terminal_envelope.model_dump_json().encode())
            )

        async def subscribe(
            self,
            *_args: object,
            **_kwargs: object,
        ) -> object:
            pytest.fail("Kafka-backed terminal waits should use a direct consumer")

    monkeypatch.setattr(
        "omnibase_infra.runtime.service_pattern_b_broker.AIOKafkaConsumer",
        FakeAIOKafkaConsumer,
    )

    broker = RuntimePatternBBroker(
        FakeKafkaTransport(),
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
    assert result.status == "completed"
    assert result.payload == {"status": "complete", "dispatch_count": 5}
    assert created_consumers[0].stopped is True


@pytest.mark.asyncio
async def test_service_pattern_b_broker_kafka_waiter_consumes_failure_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = _route_with_failure_terminal()
    failure_topic = route.terminal_events[1]
    created_consumers: list[FakeAIOKafkaConsumer] = []

    class FakeAIOKafkaConsumer:
        def __init__(self, *topics: object, **_kwargs: object) -> None:
            self.topics = topics
            self.messages: asyncio.Queue[SimpleNamespace] = asyncio.Queue()
            self.started = False
            self.stopped = False
            created_consumers.append(self)

        async def start(self) -> None:
            self.started = True

        def assignment(self) -> set[str]:
            return {"partition-0"} if self.started else set()

        async def seek_to_end(self, *_assignment: object) -> None:
            return None

        async def getone(self) -> SimpleNamespace:
            return await self.messages.get()

        async def stop(self) -> None:
            self.stopped = True

    class FakeKafkaTransport:
        config = SimpleNamespace(
            session_timeout_ms=45000,
            heartbeat_interval_ms=15000,
            max_poll_interval_ms=300000,
            reconnect_backoff_ms=2000,
        )
        _bootstrap_servers = "redpanda:9092"

        def _build_auth_kwargs(self) -> dict[str, object]:
            return {}

        async def publish(
            self,
            _topic: str,
            _key: bytes | None,
            value: bytes,
            _headers: object | None = None,
        ) -> None:
            command_envelope = ModelEventEnvelope[object].model_validate_json(value)
            terminal_envelope = ModelEventEnvelope[object](
                payload={"payload": {"failure_reason": "routing contract missing"}},
                correlation_id=command_envelope.correlation_id,
                envelope_timestamp=datetime.now(UTC),
                event_type=failure_topic,
                source_tool="session_orchestrator",
            )
            await created_consumers[-1].messages.put(
                SimpleNamespace(
                    topic=failure_topic,
                    value=terminal_envelope.model_dump_json().encode(),
                )
            )

        async def subscribe(
            self,
            *_args: object,
            **_kwargs: object,
        ) -> object:
            pytest.fail("Kafka-backed terminal waits should use a direct consumer")

    monkeypatch.setattr(
        "omnibase_infra.runtime.service_pattern_b_broker.AIOKafkaConsumer",
        FakeAIOKafkaConsumer,
    )

    broker = RuntimePatternBBroker(
        FakeKafkaTransport(),
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
    assert created_consumers[0].topics[:2] == route.terminal_events
    assert result.status == "failed"
    assert result.error_message == "routing contract missing"
    assert created_consumers[0].stopped is True


@pytest.mark.asyncio
async def test_service_pattern_b_broker_preserves_result_when_kafka_stop_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = _route()
    created_consumers: list[FakeAIOKafkaConsumer] = []

    class FakeAIOKafkaConsumer:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.messages: asyncio.Queue[SimpleNamespace] = asyncio.Queue()
            self.started = False
            created_consumers.append(self)

        async def start(self) -> None:
            self.started = True

        def assignment(self) -> set[str]:
            return {"partition-0"} if self.started else set()

        async def seek_to_end(self, *_assignment: object) -> None:
            return None

        async def getone(self) -> SimpleNamespace:
            return await self.messages.get()

        async def stop(self) -> None:
            raise RuntimeError("failed to close terminal consumer")

    class FakeKafkaTransport:
        config = SimpleNamespace(
            session_timeout_ms=45000,
            heartbeat_interval_ms=15000,
            max_poll_interval_ms=300000,
            reconnect_backoff_ms=2000,
        )
        _bootstrap_servers = "redpanda:9092"

        def _build_auth_kwargs(self) -> dict[str, object]:
            return {}

        async def publish(
            self,
            _topic: str,
            _key: bytes | None,
            value: bytes,
            _headers: object | None = None,
        ) -> None:
            command_envelope = ModelEventEnvelope[object].model_validate_json(value)
            terminal_envelope = ModelEventEnvelope[object](
                payload={"status": "complete", "dispatch_count": 5},
                correlation_id=command_envelope.correlation_id,
                envelope_timestamp=datetime.now(UTC),
                event_type=route.terminal_event,
                source_tool="session_orchestrator",
            )
            await created_consumers[-1].messages.put(
                SimpleNamespace(value=terminal_envelope.model_dump_json().encode())
            )

        async def subscribe(
            self,
            *_args: object,
            **_kwargs: object,
        ) -> object:
            pytest.fail("Kafka-backed terminal waits should use a direct consumer")

    monkeypatch.setattr(
        "omnibase_infra.runtime.service_pattern_b_broker.AIOKafkaConsumer",
        FakeAIOKafkaConsumer,
    )

    broker = RuntimePatternBBroker(
        FakeKafkaTransport(),
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
    assert result.status == "completed"
    assert result.payload == {"status": "complete", "dispatch_count": 5}


def test_service_pattern_b_broker_decodes_target_runtime_address_during_core_skew(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(
        ModelDispatchBusCommand.model_fields,
        "target_runtime_address",
        raising=False,
    )
    command = RuntimePatternBBroker._decode_dispatch_command_payload(
        {
            "command_name": "session_orchestrator",
            "requester": "codex",
            "payload": {"dry_run": True},
            "response_topic": "onex.evt.pattern-b.dispatch-completed.v1",
            "timeout_seconds": 1,
            "target_runtime_address": "runtime://omninode-pc/stability-test/main",
        }
    )

    assert command.command_name == "session_orchestrator"
    assert command.requester == "codex"


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
async def test_runtime_host_process_skips_pattern_b_broker_for_disallowed_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RUNTIME_PROFILE", "effects")
    monkeypatch.setattr(
        "omnibase_infra.runtime.service_runtime_host_process.discover_runtime_local_ingress_routes",
        lambda _packages: pytest.fail("effects profile must not discover routes"),
    )
    monkeypatch.setattr(
        "omnibase_infra.runtime.service_runtime_host_process.RuntimePatternBBroker",
        lambda *_args, **_kwargs: pytest.fail(
            "effects profile must not start Pattern B broker"
        ),
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
                "enabled_profiles": ["main"],
                "package_names": ["omnibase_infra", "omnimarket"],
            },
        },
        dispatch_engine=AsyncMock(),
    )

    await process._start_pattern_b_broker()

    assert process._pattern_b_broker is None


def test_runtime_host_process_treats_blank_runtime_profile_as_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RUNTIME_PROFILE", "   ")

    process = RuntimeHostProcess(
        event_bus=EventBusInmemory(environment="test", group="pattern-b"),
        config={
            "service_name": "test-service",
            "node_name": "test-node",
            "env": "test",
            "version": "v1",
            "pattern_b_broker": {
                "enabled": True,
                "enabled_profiles": ["default"],
            },
        },
        dispatch_engine=AsyncMock(),
    )

    assert process._is_pattern_b_broker_effectively_enabled() is True


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
        match=r"local runtime ingress requires pattern_b_broker to be effectively enabled",
    ):
        await process._start_pattern_b_broker()
