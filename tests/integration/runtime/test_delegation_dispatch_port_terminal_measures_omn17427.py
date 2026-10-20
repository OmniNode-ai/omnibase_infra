# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Consumer trace and measured queue wait survive the runtime bus boundary."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

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


@pytest.mark.asyncio
@pytest.mark.parametrize("queue_wait_ms", [None, 0, 125])
async def test_terminal_measures_preserve_a_consumer_decodable_request(
    queue_wait_ms: int | None,
) -> None:
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
    bus = EventBusInmemory(environment="test", group="terminal-measures")
    await bus.start()
    payloads: list[dict[str, object]] = []
    trace_id = None if queue_wait_ms is None else uuid4()

    async def consumer(message: ModelEventMessage) -> None:
        envelope = ModelEventEnvelope[dict[str, object]].model_validate_json(
            message.value
        )
        payloads.append(envelope.payload)
        request = ModelDelegationRequest.model_validate(envelope.payload)
        assert request.trace_id == trace_id
        terminal = ModelEventEnvelope[dict[str, object]](
            payload={"content": "runtime-ok", "quality_passed": True},
            correlation_id=request.correlation_id,
            envelope_timestamp=datetime.now(UTC),
            event_type=route.terminal_events[0],
            source_tool="node_delegation_orchestrator",
        )
        await bus.publish(
            route.terminal_events[0], None, terminal.model_dump_json().encode(), None
        )

    unsubscribe = await bus.subscribe(
        route.command_topic, group_id="terminal-measures-consumer", on_message=consumer
    )
    port = RuntimeDelegationDispatchPort(
        event_bus=bus, routes={"delegation.orchestrate": route}
    )
    try:
        result = await port.dispatch(
            prompt="runtime probe",
            task_type="summarization",
            correlation_id=uuid4(),
            max_tokens=None,
            source_file_path=None,
            source_session_id=None,
            wait=True,
            execution_timeout_seconds=5,
            terminal_delivery_margin_seconds=1,
            trace_id=trace_id,
            queue_wait_ms=queue_wait_ms,
        )
    finally:
        await unsubscribe()
        await bus.close()

    assert result["status"] == "completed"
    assert len(payloads) == 1
    assert "queue_wait_ms" not in payloads[0]
    if queue_wait_ms is None:
        assert "trace_id" not in payloads[0]
        assert "queue_wait_ms" not in result
    else:
        assert payloads[0]["trace_id"] == str(trace_id)
        assert result["queue_wait_ms"] == queue_wait_ms
