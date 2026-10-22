# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Terminal measurement kwargs survive the runtime dispatch boundary (OMN-17427)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TypedDict
from uuid import UUID, uuid4

import pytest

from omnibase_core.models.delegation.wire import ModelDelegationRequest
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.runtime.runtime_local_ingress import ModelRuntimeLocalIngressRoute
from omnibase_infra.runtime.service_delegation_dispatch_port import (
    RuntimeDelegationDispatchPort,
)

pytestmark = pytest.mark.integration


class TerminalMeasureKwargs(TypedDict, total=False):
    trace_id: UUID | None
    queue_wait_ms: int | None


@pytest.mark.asyncio
@pytest.mark.parametrize("measurements", ["present", "none", "omitted"])
async def test_terminal_measurements_allow_a_valid_bus_request_and_terminal(
    measurements: str,
) -> None:
    """Exercise the real port, Pattern-B broker, wire model and terminal readback."""
    route = ModelRuntimeLocalIngressRoute(
        node_name="node_delegation_orchestrator",
        contract_name="node_delegation_orchestrator",
        command_topic="onex.cmd.omnimarket.delegation-request.v1",
        event_type="omnimarket.delegation-request",
        terminal_event="onex.evt.omnimarket.delegation-completed.v1",
        terminal_events=(
            "onex.evt.omnimarket.delegation-completed.v1",
            "onex.evt.omnimarket.delegation-failed.v1",
        ),
        contract_path="/contracts/omnimarket/node_delegation_orchestrator/contract.yaml",
        package_name="omnimarket",
    )
    correlation_id = uuid4()
    bus = EventBusInmemory(environment="test", group="terminal-measure-readback")
    await bus.start()
    payloads: list[dict[str, object]] = []

    async def consumer(message: ModelEventMessage) -> None:
        envelope = ModelEventEnvelope[dict[str, object]].model_validate_json(
            message.value
        )
        payloads.append(envelope.payload)
        request = ModelDelegationRequest.model_validate(envelope.payload)
        assert request.correlation_id == correlation_id
        assert request.prompt == "terminal measurement probe"
        assert request.requested_timeout_seconds == 5
        terminal = ModelEventEnvelope[dict[str, object]](
            payload={"content": "measurement-runtime-ok", "quality_passed": True},
            correlation_id=request.correlation_id,
            envelope_timestamp=datetime.now(UTC),
            event_type=route.terminal_events[0],
            source_tool="node_delegation_orchestrator",
        )
        await bus.publish(
            route.terminal_events[0], None, terminal.model_dump_json().encode(), None
        )

    unsubscribe = await bus.subscribe(
        route.command_topic, group_id="terminal-measure-consumer", on_message=consumer
    )
    port = RuntimeDelegationDispatchPort(
        event_bus=bus, routes={"delegation.orchestrate": route}
    )
    kwargs: TerminalMeasureKwargs = {}
    trace_id = uuid4()
    if measurements == "present":
        kwargs = {"trace_id": trace_id, "queue_wait_ms": 42}
    elif measurements == "none":
        kwargs = {"trace_id": None, "queue_wait_ms": None}
    try:
        result = await port.dispatch(
            prompt="terminal measurement probe",
            task_type="summarization",
            correlation_id=correlation_id,
            max_tokens=None,
            source_file_path=None,
            source_session_id=None,
            wait=True,
            execution_timeout_seconds=5,
            terminal_delivery_margin_seconds=1,
            **kwargs,
        )
    finally:
        await unsubscribe()
        await bus.close()

    assert len(payloads) == 1
    # OMN-17427 (#5214): a present trace_id rides the wire; an absent one is
    # omitted. queue_wait_ms is never published.
    if measurements == "present":
        assert payloads[0]["trace_id"] == str(trace_id)
    else:
        assert "trace_id" not in payloads[0]
    assert "queue_wait_ms" not in payloads[0]
    assert result["status"] == "completed"
    assert result["content"] == "measurement-runtime-ok"
    assert result["quality_passed"] is True
