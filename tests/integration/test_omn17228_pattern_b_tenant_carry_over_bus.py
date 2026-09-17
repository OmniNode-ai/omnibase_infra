# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17228: the tenant survives a real publish/subscribe round trip.

The unit twin (``tests/unit/runtime/test_omn17228_pattern_b_tenant_carry.py``)
asserts what the broker hands the bus. This one asserts what a CONSUMER reads
back off the bus, because that is the property the defect actually turned on:
the delegation projection writer does not see the object the publisher built, it
sees the bytes that survived serialisation, transport and re-parse, and it reads
``ModelEventEnvelope.tenant_id`` off those bytes.

A stamp that is set on the model and lost on the wire would pass the unit test
and fail on the lane in exactly the way this whole chain already failed three
times: the verdict row written under the house tenant
``820272f9-4aaf-5add-a2df-0af942852ab2`` while the reader queried the submitting
tenant, on deploy-onex-staging runs 35063077145, 35079154240 and 35086243365.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime

import pytest

from omnibase_core.models.dispatch.model_dispatch_bus_command import (
    ModelDispatchBusCommand,
)
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.runtime.dispatch_envelope_context import bind_dispatch_envelope
from omnibase_infra.runtime.runtime_local_ingress import ModelRuntimeLocalIngressRoute
from omnibase_infra.runtime.service_pattern_b_broker import RuntimePatternBBroker

_BETA_TENANT_UUID = "91c74442-1233-4c97-b191-911a10346fdf"
_HOUSE_TENANT_UUID = "820272f9-4aaf-5add-a2df-0af942852ab2"


def _route() -> ModelRuntimeLocalIngressRoute:
    return ModelRuntimeLocalIngressRoute(
        node_name="node_session_orchestrator",
        contract_name="session_orchestrator",
        command_topic="onex.cmd.omnimarket.session-orchestrator-start.v1",
        event_type="omnimarket.session-orchestrator-start",
        terminal_event="onex.evt.omnimarket.session-orchestrator-completed.v1",
        contract_path="/tmp/node_session_orchestrator/contract.yaml",  # noqa: S108
        package_name="omnimarket",
    )


@pytest.mark.asyncio
async def test_worker_command_tenant_survives_the_bus_round_trip() -> None:
    """Published, transported, re-parsed by a subscriber — still attributed."""
    bus = EventBusInmemory(environment="test", group="omn17228-integration")
    await bus.start()
    try:
        route = _route()
        broker = RuntimePatternBBroker(
            bus,
            command_topic="onex.cmd.omnibase-infra.pattern-b-dispatch.v1",
            routes={"session_orchestrator": route},
        )

        seen: asyncio.Queue[ModelEventEnvelope[object]] = asyncio.Queue()

        async def worker(message: ModelEventMessage) -> None:
            await seen.put(
                ModelEventEnvelope[object].model_validate_json(message.value)
            )

        await bus.subscribe(route.command_topic, group_id="worker", on_message=worker)

        consumed = ModelEventEnvelope[object](
            envelope_id=uuid.uuid4(),
            payload={"prompt": "summarise the deploy"},
            correlation_id=uuid.uuid4(),
            envelope_timestamp=datetime.now(UTC),
            event_type="omnibase-infra.delegation-request",
            source_tool="onex-api",
            tenant_id=_BETA_TENANT_UUID,
        )
        command = ModelDispatchBusCommand(
            command_name="session_orchestrator",
            requester="business-proof",
            payload={"prompt": "summarise the deploy"},
            response_topic="onex.evt.pattern-b.dispatch-completed.v1",
            timeout_seconds=1,
        )

        with bind_dispatch_envelope(consumed):
            await broker._publish_worker_command(command, route)

        received = await asyncio.wait_for(seen.get(), timeout=5)

        assert received.tenant_id == _BETA_TENANT_UUID, (
            "the tenant did not survive the wire, so a consumer of this command "
            "is unattributed and a quality verdict derived from it reaches the "
            "projection writer with nothing to attribute it to (it then lands "
            f"under the house tenant {_HOUSE_TENANT_UUID})"
        )
        assert received.parent_envelope_id == consumed.envelope_id
        assert received.correlation_id == command.correlation_id
    finally:
        await bus.close()


@pytest.mark.asyncio
async def test_unattributed_cause_stays_unattributed_across_the_bus() -> None:
    """The negative control, over the same real transport.

    Without it, a publisher that stamped a constant would pass the test above.
    """
    bus = EventBusInmemory(environment="test", group="omn17228-integration-none")
    await bus.start()
    try:
        route = _route()
        broker = RuntimePatternBBroker(
            bus,
            command_topic="onex.cmd.omnibase-infra.pattern-b-dispatch.v1",
            routes={"session_orchestrator": route},
        )

        seen: asyncio.Queue[ModelEventEnvelope[object]] = asyncio.Queue()

        async def worker(message: ModelEventMessage) -> None:
            await seen.put(
                ModelEventEnvelope[object].model_validate_json(message.value)
            )

        await bus.subscribe(route.command_topic, group_id="worker", on_message=worker)

        command = ModelDispatchBusCommand(
            command_name="session_orchestrator",
            requester="business-proof",
            payload={"prompt": "summarise the deploy"},
            response_topic="onex.evt.pattern-b.dispatch-completed.v1",
            timeout_seconds=1,
        )

        # No bound dispatch envelope: this command genuinely is a chain head.
        await broker._publish_worker_command(command, route)

        received = await asyncio.wait_for(seen.get(), timeout=5)
        assert received.tenant_id is None
        assert received.parent_envelope_id is None
    finally:
        await bus.close()
