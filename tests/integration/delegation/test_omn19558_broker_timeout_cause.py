# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-19558: the real Pattern B broker reports a typed timeout cause."""

from __future__ import annotations

import asyncio
import json
from uuid import UUID, uuid4

import pytest

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.event_bus.topic_constants import derive_event_type_alias_for_topic
from omnibase_infra.runtime.runtime_local_ingress import ModelRuntimeLocalIngressRoute
from omnibase_infra.runtime.service_delegation_dispatch_port import (
    RuntimeDelegationDispatchPort,
)

pytestmark = pytest.mark.integration


def _route(
    *, package_name: str, terminal_events: tuple[str, ...]
) -> ModelRuntimeLocalIngressRoute:
    contract_name = "node_delegation_orchestrator"
    return ModelRuntimeLocalIngressRoute(
        node_name=contract_name,
        contract_name=contract_name,
        command_topic=f"onex.cmd.{package_name}.delegation-request.v1",
        event_type=f"{package_name}.delegation-request",
        terminal_event=terminal_events[0],
        terminal_events=terminal_events,
        contract_path=f"/contracts/{package_name}/{contract_name}.yaml",
        package_name=package_name,
    )


@pytest.mark.asyncio
async def test_missing_terminal_is_a_typed_timeout_without_redispatch() -> None:
    """OMN-19558: the real broker waits once and reports its timeout cause."""
    route = _route(
        package_name="omnimarket",
        terminal_events=(
            "onex.evt.omnimarket.delegation-completed.v1",
            "onex.evt.omnimarket.delegation-failed.v1",
        ),
    )
    correlation_id = uuid4()
    commands: list[dict[str, object]] = []
    bus = EventBusInmemory(environment="test", group="missing-terminal")
    await bus.start()

    async def hold_terminal(message: ModelEventMessage) -> None:
        commands.append(json.loads(message.value))
        # The command was accepted, but its inference terminal never arrives.

    await bus.subscribe(
        route.command_topic, group_id="hold-terminal", on_message=hold_terminal
    )
    port = RuntimeDelegationDispatchPort(bus, routes={"delegation.orchestrate": route})
    try:
        result = await asyncio.wait_for(
            port.dispatch(
                prompt="held inference terminal",
                task_type="reasoning",
                correlation_id=correlation_id,
                max_tokens=None,
                source_file_path=None,
                source_session_id=None,
                wait=True,
                execution_timeout_seconds=1,
                terminal_delivery_margin_seconds=1,
            ),
            timeout=5,
        )
    finally:
        await bus.close()

    assert len(commands) == 1
    assert commands[0]["correlation_id"] == str(correlation_id)
    assert result["status"] == "timeout"
    assert result["terminal_failure_cause"] == "timeout"


@pytest.mark.asyncio
@pytest.mark.parametrize("late_status", ["completed", "failed"])
async def test_late_terminal_after_timeout_does_not_redispatch_or_complete_next_request(
    late_status: str,
) -> None:
    """A stale terminal leaves the timeout final and the next command isolated."""
    route = _route(
        package_name="omnimarket",
        terminal_events=(
            "onex.evt.omnimarket.delegation-completed.v1",
            "onex.evt.omnimarket.delegation-failed.v1",
        ),
    )
    timed_out_id, next_id = uuid4(), uuid4()
    commands: list[dict[str, object]] = []
    accepted: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    bus = EventBusInmemory(environment="test", group="late-terminal")
    await bus.start()

    async def hold_terminal(message: ModelEventMessage) -> None:
        command = json.loads(message.value)
        commands.append(command)
        await accepted.put(command)

    await bus.subscribe(
        route.command_topic, group_id="hold-terminal", on_message=hold_terminal
    )
    port = RuntimeDelegationDispatchPort(bus, routes={"delegation.orchestrate": route})

    async def dispatch(correlation_id: UUID) -> dict[str, object]:
        return await port.dispatch(
            prompt="held inference terminal",
            task_type="reasoning",
            correlation_id=correlation_id,
            max_tokens=None,
            source_file_path=None,
            source_session_id=None,
            wait=True,
            execution_timeout_seconds=1,
            terminal_delivery_margin_seconds=1,
        )

    pending: asyncio.Task[dict[str, object]] | None = None
    try:
        timed_out = await asyncio.wait_for(dispatch(timed_out_id), timeout=5)
        assert timed_out["status"] == "timeout"
        assert timed_out["terminal_failure_cause"] == "timeout"
        assert (await asyncio.wait_for(accepted.get(), timeout=5))[
            "correlation_id"
        ] == str(timed_out_id)
        assert (await bus.health_check())["subscriber_count"] == 1

        pending = asyncio.create_task(dispatch(next_id))
        assert (await asyncio.wait_for(accepted.get(), timeout=5))[
            "correlation_id"
        ] == str(next_id)
        assert (await bus.health_check())["subscriber_count"] == 3

        late_topic = route.terminal_events[0 if late_status == "completed" else 1]
        await bus.publish_envelope(
            ModelEventEnvelope[object](
                event_type=derive_event_type_alias_for_topic(late_topic),
                correlation_id=timed_out_id,
                payload={"content": "stale", "status": late_status},
            ),
            late_topic,
        )
        # In-memory publication awaits all callbacks, so the stale terminal has
        # been consumed before the valid terminal is published.
        assert not pending.done()
        await bus.publish_envelope(
            ModelEventEnvelope[object](
                event_type=derive_event_type_alias_for_topic(route.terminal_events[0]),
                correlation_id=next_id,
                payload={"content": "next request completed"},
            ),
            route.terminal_events[0],
        )
        completed = await asyncio.wait_for(pending, timeout=5)
        assert completed["status"] == "completed"
        assert completed["content"] == "next request completed"
        assert "terminal_failure_cause" not in completed
        assert timed_out["status"] == "timeout"
        assert timed_out["terminal_failure_cause"] == "timeout"
        assert [command["correlation_id"] for command in commands] == [
            str(timed_out_id),
            str(next_id),
        ]
        assert (await bus.health_check())["subscriber_count"] == 1
    finally:
        if pending is not None:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        await bus.close()
