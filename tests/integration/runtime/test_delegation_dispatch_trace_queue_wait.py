# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Trace and queue-wait compatibility through the runtime bus (OMN-17427)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TypedDict
from uuid import UUID, uuid4

import pytest

from omnibase_core.models.delegation.wire import ModelDelegationRequest
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.runtime.protocols.protocol_delegation_dispatch_port import (
    ProtocolDelegationDispatchPort,
)
from omnibase_infra.runtime.runtime_local_ingress import ModelRuntimeLocalIngressRoute
from omnibase_infra.runtime.service_delegation_dispatch_port import (
    RuntimeDelegationDispatchPort,
)

pytestmark = pytest.mark.integration


class DispatchTelemetry(TypedDict, total=False):
    trace_id: UUID | None
    queue_wait_ms: int | None


@pytest.mark.asyncio
@pytest.mark.parametrize("telemetry", ["present", "none", "omitted"])
async def test_dispatch_telemetry_round_trip(telemetry: str) -> None:
    """The consumer decodes the real command and returns a correlated terminal."""
    trace_id = uuid4() if telemetry == "present" else None
    correlation_id = uuid4()
    kwargs: DispatchTelemetry = (
        {"trace_id": trace_id, "queue_wait_ms": 42 if trace_id else None}
        if telemetry != "omitted"
        else {}
    )
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
    bus = EventBusInmemory(environment="test", group="dispatch-telemetry")
    await bus.start()
    payloads: list[dict[str, object]] = []

    async def consumer(message: ModelEventMessage) -> None:
        envelope = ModelEventEnvelope[dict[str, object]].model_validate_json(
            message.value
        )
        payloads.append(envelope.payload)
        request = ModelDelegationRequest.model_validate(envelope.payload)
        terminal = ModelEventEnvelope[dict[str, object]](
            payload={
                "content": str(request.trace_id),
                "quality_passed": True,
            },
            correlation_id=request.correlation_id,
            envelope_timestamp=datetime.now(UTC),
            event_type=route.terminal_events[0],
            source_tool="node_delegation_orchestrator",
        )
        await bus.publish(
            route.terminal_events[0], None, terminal.model_dump_json().encode(), None
        )

    unsubscribe = await bus.subscribe(
        route.command_topic, group_id="dispatch-telemetry-consumer", on_message=consumer
    )
    port: ProtocolDelegationDispatchPort = RuntimeDelegationDispatchPort(
        event_bus=bus, routes={"delegation.orchestrate": route}
    )
    try:
        result = await port.dispatch(
            prompt="Read back dispatch telemetry",
            task_type="summarization",
            correlation_id=correlation_id,
            max_tokens=None,
            source_file_path=None,
            source_session_id=None,
            wait=True,
            execution_timeout_seconds=5,
            terminal_delivery_margin_seconds=1,
            quality_contract_mode="extend_task_class",
            acceptance_criteria=(),
            **kwargs,
        )
    finally:
        await unsubscribe()
        await bus.close()

    assert len(payloads) == 1
    assert payloads[0]["correlation_id"] == str(correlation_id)
    assert "queue_wait_ms" not in payloads[0]
    if trace_id is None:
        assert "trace_id" not in payloads[0]
    else:
        assert payloads[0]["trace_id"] == str(trace_id)
    assert result["status"] == "completed"
    assert result["content"] == str(trace_id)
    assert result["quality_gate_passed"] is True
    print(
        json.dumps(
            {
                "readback": "dispatch-telemetry-round-trip",
                "transport": "EventBusInmemory",
                "telemetry": telemetry,
                "correlation_id": str(correlation_id),
                "trace_id": None if trace_id is None else str(trace_id),
                "status": result["status"],
                "content": result["content"],
                "queue_wait_on_wire": "queue_wait_ms" in payloads[0],
            }
        )
    )
