# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""trace_id rides the bus command; queue_wait_ms is accepted and not published (OMN-17427, OMN-19448)."""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from uuid import uuid4

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


@asynccontextmanager
async def _dispatch_bus() -> AsyncIterator[
    tuple[
        RuntimeDelegationDispatchPort,
        list[dict[str, object]],
        list[ModelDelegationRequest],
    ]
]:
    """Record consumer-decoded commands and complete them on the runtime bus."""
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
    bus = EventBusInmemory(environment="test", group="trace-queue-wait")
    payloads: list[dict[str, object]] = []
    requests: list[ModelDelegationRequest] = []

    async def consumer(message: ModelEventMessage) -> None:
        envelope = ModelEventEnvelope[dict[str, object]].model_validate_json(
            message.value
        )
        payloads.append(envelope.payload)
        request = ModelDelegationRequest.model_validate(envelope.payload)
        requests.append(request)
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

    try:
        await bus.start()
        unsubscribe = await bus.subscribe(
            route.command_topic,
            group_id="trace-queue-wait-consumer",
            on_message=consumer,
        )
        try:
            port = RuntimeDelegationDispatchPort(
                event_bus=bus, routes={"delegation.orchestrate": route}
            )
            yield port, payloads, requests
        finally:
            await unsubscribe()
    finally:
        await bus.close()


@pytest.mark.parametrize(
    "dispatch_method",
    [ProtocolDelegationDispatchPort.dispatch, RuntimeDelegationDispatchPort.dispatch],
)
def test_trace_id_and_queue_wait_ms_are_optional_keyword_only_parameters(
    dispatch_method: Callable[..., Awaitable[dict[str, object]]],
) -> None:
    parameters = inspect.signature(dispatch_method).parameters
    for name in ("trace_id", "queue_wait_ms"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters[name].default is None


@pytest.mark.asyncio
async def test_trace_id_rides_the_bus_and_queue_wait_ms_is_not_published() -> None:
    trace_id = uuid4()
    async with _dispatch_bus() as (port, payloads, requests):
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
            queue_wait_ms=37,
        )

    assert result["status"] == "completed"
    assert len(payloads) == 1
    assert payloads[0]["trace_id"] == str(trace_id)
    assert len(requests) == 1
    assert requests[0].trace_id == trace_id
    assert "queue_wait_ms" not in payloads[0]


@pytest.mark.asyncio
async def test_omitted_trace_id_and_queue_wait_ms_are_not_published() -> None:
    async with _dispatch_bus() as (port, payloads, requests):
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
        )

    assert result["status"] == "completed"
    assert len(payloads) == 1
    assert "trace_id" not in payloads[0]
    assert "queue_wait_ms" not in payloads[0]
    assert len(requests) == 1
    assert requests[0].trace_id is None
