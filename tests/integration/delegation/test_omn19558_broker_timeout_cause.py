# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-19558: the real Pattern B broker reports a typed timeout cause."""

from __future__ import annotations

import asyncio
import json
from uuid import uuid4

import pytest

from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
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
