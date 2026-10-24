# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-21341: receipt-mode exchanges against the real in-memory bus."""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from pydantic import JsonValue, ValidationError

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.cli import delegate_lane, delegate_lane_credentials
from omnibase_infra.cli.delegate_lane_credentials import DelegateLaneCredentialError
from omnibase_infra.cli.model_delegate_lane_selection import ModelDelegateLaneSelection
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.lane_client_transport_binding import (
    bound_lane_client_transport,
)
from omnibase_infra.event_bus.model_lane_client_transport import (
    ModelLaneClientTransport,
)
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.handlers import (
    handler_receipt_mode_publish_await as handler_module,
)
from omnibase_infra.handlers.handler_receipt_mode_publish_await import (
    HandlerReceiptModePublishAwait,
)
from omnibase_infra.models.model_receipt_mode_event import ModelReceiptModeEvent

pytestmark = pytest.mark.unit

COMMAND_TOPIC = "onex.cmd.receipt-mode.requested.v1"
PROGRESS_TOPIC = "onex.evt.receipt-mode.progress.v1"
ACTION_TOPIC = "onex.evt.receipt-mode.action.v1"
COMMAND_TYPE = "receipt-mode.requested"


def _respond(
    bus: EventBusInmemory,
    responder: Callable[[ModelEventMessage], Awaitable[None]],
) -> None:
    asyncio.run(
        bus.subscribe(COMMAND_TOPIC, on_message=responder, group_id="test-responder")
    )


def _call(
    bus: EventBusInmemory,
    *,
    payload: dict[str, JsonValue] | None = None,
    deadline_ms: int = 1000,
) -> tuple[ModelReceiptModeEvent, ...]:
    return HandlerReceiptModePublishAwait(lambda: bus).publish_and_await(
        topic=COMMAND_TOPIC,
        command_type=COMMAND_TYPE,
        payload=payload if payload is not None else {"invocation_id": "invocation-1"},
        awaited_events=(PROGRESS_TOPIC, ACTION_TOPIC),
        deadline_ms=deadline_ms,
    )


async def _reply(
    bus: EventBusInmemory,
    topic: str,
    correlation_id: str,
    payload: dict[str, JsonValue],
) -> None:
    await bus.publish(
        topic,
        key=None,
        value=json.dumps(
            {"correlation_id": correlation_id, "payload": payload}
        ).encode(),
    )


def test_action_is_last_after_progress_on_another_topic() -> None:
    bus = EventBusInmemory()
    commands: list[ModelEventEnvelope[dict[str, object]]] = []

    async def responder(message: ModelEventMessage) -> None:
        command = ModelEventEnvelope[dict[str, object]].model_validate_json(
            message.value
        )
        commands.append(command)
        assert message.key == str(command.correlation_id).encode("utf-8")
        await _reply(
            bus, PROGRESS_TOPIC, str(command.correlation_id), {"status": "ready"}
        )
        await _reply(
            bus, ACTION_TOPIC, str(command.correlation_id), {"action": "allow"}
        )
        await _reply(
            bus, PROGRESS_TOPIC, str(command.correlation_id), {"status": "late"}
        )

    _respond(bus, responder)
    events = _call(bus)

    assert events == (
        ModelReceiptModeEvent(topic=PROGRESS_TOPIC, payload={"status": "ready"}),
        ModelReceiptModeEvent(topic=ACTION_TOPIC, payload={"action": "allow"}),
    )
    assert len(commands) == 1
    assert commands[0].event_type == COMMAND_TYPE
    assert commands[0].payload == {"invocation_id": "invocation-1"}
    assert asyncio.run(bus.health_check())["started"] is False


def test_uncorrelated_reply_is_ignored_until_deadline() -> None:
    bus = EventBusInmemory()

    async def responder(message: ModelEventMessage) -> None:
        await _reply(
            bus,
            ACTION_TOPIC,
            str(uuid4()),
            {"invocation_id": "another-invocation", "action": "allow"},
        )

    _respond(bus, responder)
    started = time.monotonic()
    assert _call(bus, deadline_ms=200) == ()
    assert time.monotonic() - started >= 0.18
    assert asyncio.run(bus.health_check())["started"] is False


def test_invocation_echo_alone_correlates_reply() -> None:
    bus = EventBusInmemory()

    async def responder(message: ModelEventMessage) -> None:
        command = ModelEventEnvelope[dict[str, object]].model_validate_json(
            message.value
        )
        await _reply(
            bus,
            ACTION_TOPIC,
            str(uuid4()),
            {"invocation_id": str(command.payload["invocation_id"]), "action": "allow"},
        )

    _respond(bus, responder)
    events = _call(bus)
    assert len(events) == 1
    assert events[0].payload == {"invocation_id": "invocation-1", "action": "allow"}


def test_every_subscription_exists_before_synchronous_publish_reply() -> None:
    bus = EventBusInmemory()
    subscriber_counts: list[int] = []
    groups: list[dict[tuple[str, str], str]] = []

    async def responder(message: ModelEventMessage) -> None:
        subscriber_counts.extend(
            [
                await bus.get_subscriber_count(PROGRESS_TOPIC),
                await bus.get_subscriber_count(ACTION_TOPIC),
            ]
        )
        groups.append(bus.get_consumer_groups())
        command = ModelEventEnvelope[dict[str, object]].model_validate_json(
            message.value
        )
        await _reply(bus, ACTION_TOPIC, str(command.correlation_id), {"action": None})

    _respond(bus, responder)
    events = _call(bus)
    assert subscriber_counts == [1, 1]
    assert events == (
        ModelReceiptModeEvent(topic=ACTION_TOPIC, payload={"action": None}),
    )
    receipt_groups = {
        group for topic, group in groups[0] if topic in (PROGRESS_TOPIC, ACTION_TOPIC)
    }
    assert len(receipt_groups) == 1
    group = receipt_groups.pop()
    assert group.startswith("receipt-mode-client.")
    assert len(group.removeprefix("receipt-mode-client.")) == 12


def test_deadline_returns_collected_progress_and_ignores_malformed_messages() -> None:
    bus = EventBusInmemory()

    async def responder(message: ModelEventMessage) -> None:
        command = ModelEventEnvelope[dict[str, object]].model_validate_json(
            message.value
        )
        for value in (b"not json", b"\xff", b"[]", b"null", b'{"payload": []}'):
            await bus.publish(ACTION_TOPIC, key=None, value=value)
        await _reply(
            bus, PROGRESS_TOPIC, str(command.correlation_id), {"status": "ready"}
        )

    _respond(bus, responder)
    assert _call(bus, deadline_ms=200) == (
        ModelReceiptModeEvent(topic=PROGRESS_TOPIC, payload={"status": "ready"}),
    )


@pytest.mark.parametrize("payload", [{}, {"invocation_id": None}])
def test_absent_invocation_does_not_correlate_unrelated_action(
    payload: dict[str, JsonValue],
) -> None:
    bus = EventBusInmemory()

    async def responder(message: ModelEventMessage) -> None:
        await _reply(bus, ACTION_TOPIC, str(uuid4()), {"action": "allow"})

    _respond(bus, responder)
    assert _call(bus, payload=payload, deadline_ms=200) == ()


def test_deadline_includes_startup_and_still_closes_bus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus = EventBusInmemory()
    close = AsyncMock(wraps=bus.close)

    async def slow_start() -> None:
        await asyncio.sleep(10)

    monkeypatch.setattr(bus, "start", slow_start)
    monkeypatch.setattr(bus, "close", close)
    started = time.monotonic()
    assert _call(bus, deadline_ms=200) == ()
    assert time.monotonic() - started < 2
    close.assert_awaited_once()


def test_publish_failure_unsubscribes_every_topic_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus = EventBusInmemory()
    original_subscribe = bus.subscribe
    unsubscribes: list[AsyncMock] = []
    close = AsyncMock(wraps=bus.close)

    async def subscribe(
        topic: str,
        *,
        on_message: Callable[[ModelEventMessage], Awaitable[None]],
        group_id: str,
    ) -> Callable[[], Awaitable[None]]:
        unsubscribe = AsyncMock(
            wraps=await original_subscribe(
                topic, on_message=on_message, group_id=group_id
            )
        )
        unsubscribes.append(unsubscribe)
        return unsubscribe

    monkeypatch.setattr(bus, "subscribe", subscribe)
    monkeypatch.setattr(
        bus, "publish", AsyncMock(side_effect=RuntimeError("publish failed"))
    )
    monkeypatch.setattr(bus, "close", close)
    with pytest.raises(RuntimeError, match="publish failed"):
        _call(bus)
    assert len(unsubscribes) == 2
    for unsubscribe in unsubscribes:
        unsubscribe.assert_awaited_once()
    close.assert_awaited_once()


def test_default_factory_refuses_without_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OMNIBASE_PATH", raising=False)
    monkeypatch.delenv("OMNI_HOME", raising=False)
    with pytest.raises(RuntimeError, match="no workspace root is set"):
        HandlerReceiptModePublishAwait().publish_and_await(
            topic=COMMAND_TOPIC,
            command_type=COMMAND_TYPE,
            payload={},
            awaited_events=(ACTION_TOPIC,),
            deadline_ms=200,
        )


@pytest.mark.parametrize("has_omnibase_path", [True, False])
@pytest.mark.parametrize("bind_transport", [True, False])
def test_default_lane_binding_covers_construction_through_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    has_omnibase_path: bool,
    bind_transport: bool,
) -> None:
    workspace = tmp_path / "workspace"
    monkeypatch.setenv("OMNI_HOME", str(workspace))
    if has_omnibase_path:
        workspace = tmp_path / "preferred-workspace"
        monkeypatch.setenv("OMNIBASE_PATH", str(workspace))
    else:
        monkeypatch.delenv("OMNIBASE_PATH", raising=False)
    selection = ModelDelegateLaneSelection(
        lane="dev",
        bootstrap_servers="receipt-test.example:9092",
        security_protocol="PLAINTEXT",
        declared_in=workspace / "lanes.yaml",
    )
    transport = (
        ModelLaneClientTransport(**selection.model_dump()) if bind_transport else None
    )
    resolve = Mock(return_value=selection)
    credentials = Mock(return_value=transport)
    monkeypatch.setattr(delegate_lane, "resolve_lane_selection_at", resolve)
    monkeypatch.setattr(
        delegate_lane_credentials, "resolve_lane_client_transport_for", credentials
    )
    bus = EventBusInmemory()
    original_start = bus.start
    original_close = bus.close
    original_subscribe = bus.subscribe
    stages: list[str] = []

    def from_bootstrap(bootstrap: str) -> EventBusInmemory:
        assert bootstrap == selection.bootstrap_servers
        assert bound_lane_client_transport() is transport
        stages.append("construct")
        return bus

    async def start() -> None:
        assert bound_lane_client_transport() is transport
        stages.append("start")
        await original_start()

    async def close() -> None:
        assert bound_lane_client_transport() is transport
        stages.append("close")
        await original_close()

    async def subscribe(
        topic: str,
        *,
        on_message: Callable[[ModelEventMessage], Awaitable[None]],
        group_id: str,
        auto_offset_reset: str | None = None,
    ) -> Callable[[], Awaitable[None]]:
        assert auto_offset_reset == "latest"
        assert bound_lane_client_transport() is transport
        unsubscribe = await original_subscribe(
            topic, on_message=on_message, group_id=group_id
        )

        async def checked_unsubscribe() -> None:
            assert bound_lane_client_transport() is transport
            stages.append("unsubscribe")
            await unsubscribe()

        return checked_unsubscribe

    async def responder(message: ModelEventMessage) -> None:
        assert bound_lane_client_transport() is transport
        command = ModelEventEnvelope[dict[str, object]].model_validate_json(
            message.value
        )
        await _reply(
            bus, ACTION_TOPIC, str(command.correlation_id), {"action": "allow"}
        )

    _respond(bus, responder)
    monkeypatch.setattr(handler_module.EventBusKafka, "from_bootstrap", from_bootstrap)
    monkeypatch.setattr(bus, "start", start)
    monkeypatch.setattr(bus, "close", close)
    monkeypatch.setattr(bus, "subscribe", subscribe)
    events = HandlerReceiptModePublishAwait().publish_and_await(
        topic=COMMAND_TOPIC,
        command_type=COMMAND_TYPE,
        payload={},
        awaited_events=(PROGRESS_TOPIC, ACTION_TOPIC),
        deadline_ms=1000,
    )
    assert events == (
        ModelReceiptModeEvent(topic=ACTION_TOPIC, payload={"action": "allow"}),
    )
    assert stages == ["construct", "start", "unsubscribe", "unsubscribe", "close"]
    resolve.assert_called_once_with("dev", workspace)
    credentials.assert_called_once_with(
        lane_target=selection, onex_home=Path.home() / ".onex", environ=os.environ
    )
    assert bound_lane_client_transport() is None


def test_credential_refusal_becomes_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    selection = ModelDelegateLaneSelection(
        lane="dev",
        bootstrap_servers="receipt-test.example:9092",
        security_protocol="PLAINTEXT",
        declared_in=tmp_path / "lanes.yaml",
    )
    monkeypatch.setattr(
        delegate_lane, "resolve_lane_selection_at", Mock(return_value=selection)
    )
    error = DelegateLaneCredentialError("lane credential refused")
    monkeypatch.setattr(
        delegate_lane_credentials,
        "resolve_lane_client_transport_for",
        Mock(side_effect=error),
    )
    with pytest.raises(RuntimeError, match=r"^lane credential refused$") as exc:
        HandlerReceiptModePublishAwait(lane="dev").publish_and_await(
            topic=COMMAND_TOPIC,
            command_type=COMMAND_TYPE,
            payload={},
            awaited_events=(ACTION_TOPIC,),
            deadline_ms=200,
        )
    assert exc.value.__cause__ is error


def test_event_is_frozen_and_forbids_extra_fields() -> None:
    event = ModelReceiptModeEvent(topic=ACTION_TOPIC, payload={"action": "allow"})
    with pytest.raises(ValidationError, match="frozen"):
        event.topic = PROGRESS_TOPIC
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ModelReceiptModeEvent.model_validate(
            {"topic": ACTION_TOPIC, "payload": {}, "extra": True}
        )
