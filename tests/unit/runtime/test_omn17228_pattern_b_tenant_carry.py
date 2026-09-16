# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17228: the Pattern B broker carries the tenant off the consumed envelope.

``DispatchResultApplier`` has carried ``ModelEventEnvelope.tenant_id`` from the
consumed envelope onto everything a node publishes in response since OMN-16831,
and says why: omnimarket's delegation projection writer reads exactly that
field, and a writer under FORCE ROW LEVEL SECURITY cannot discover a row's
tenant by reading. ``service_pattern_b_broker`` publishes on the same terms and
was carrying the CAUSAL half (``parent_envelope_id``, OMN-18419) while dropping
the tenant -- its own docstring already points at the applier as the site it
mirrors.

WHY IT MATTERS FOR EXACTLY ONE EVENT. Every delegation event except the quality
verdict carries its tenant in its own payload. ``ModelQualityGateResult`` is
``frozen`` / ``extra="forbid"`` with no tenant field, so the envelope stamp is
its ONLY possible attribution. Unstamped, the verdict row was written under the
house tenant ``820272f9-4aaf-5add-a2df-0af942852ab2`` -- which is
``uuid5(NAMESPACE_DNS, "house-tenant.omninode.ai")``, a different tenant, not
another representation of the submitting one -- and the submitting tenant's
reader could never see its own row. That is the ``quality_gate`` leg of the
terminal business proof failing on deploy-onex-staging runs 35063077145,
35079154240 and 35086243365, the last two with the omnimarket-side and
gateway-side producer fixes already live on the plane.

CARRIED, NEVER SOURCED. ``None`` stays ``None``: from the HTTP ingress there is
no consumed envelope and nothing is recorded, which is the same checkable
statement the absent ``parent_message_id`` already makes there. Nothing is
defaulted, derived or invented at this site.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from omnibase_core.models.dispatch.model_dispatch_bus_command import (
    ModelDispatchBusCommand,
)
from omnibase_core.models.dispatch.model_dispatch_bus_terminal_result import (
    ModelDispatchBusTerminalResult,
)
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.runtime.dispatch_envelope_context import bind_dispatch_envelope
from omnibase_infra.runtime.runtime_local_ingress import (
    ModelRuntimeLocalIngressRoute,
)
from omnibase_infra.runtime.service_pattern_b_broker import RuntimePatternBBroker

_BETA_TENANT_UUID = "91c74442-1233-4c97-b191-911a10346fdf"
_HOUSE_TENANT_UUID = "820272f9-4aaf-5add-a2df-0af942852ab2"
_RESPONSE_TOPIC = "onex.evt.pattern-b.dispatch-completed.v1"


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


class _CapturingBus:
    """Records exactly what the broker handed the bus, unparsed."""

    def __init__(self) -> None:
        self.published: list[tuple[str, bytes]] = []

    async def publish(
        self,
        topic: str,
        key: object,
        value: bytes,
        headers: object = None,
    ) -> None:
        self.published.append((topic, value))

    def envelope_for(self, topic: str) -> ModelEventEnvelope[object]:
        for published_topic, value in self.published:
            if published_topic == topic:
                return ModelEventEnvelope[object].model_validate_json(value)
        raise AssertionError(f"nothing published to {topic}: {self.published!r}")


def _broker(bus: object) -> RuntimePatternBBroker:
    return RuntimePatternBBroker(
        bus,  # type: ignore[arg-type]
        command_topic="onex.cmd.omnibase-infra.pattern-b-dispatch.v1",
        routes={"session_orchestrator": _route()},
    )


def _consumed(tenant_id: str | None) -> ModelEventEnvelope[object]:
    return ModelEventEnvelope[object](
        envelope_id=uuid.uuid4(),
        payload={"prompt": "summarise the deploy"},
        correlation_id=uuid.uuid4(),
        envelope_timestamp=datetime.now(UTC),
        event_type="omnibase-infra.delegation-request",
        source_tool="onex-api",
        tenant_id=tenant_id,
    )


def _command() -> ModelDispatchBusCommand:
    return ModelDispatchBusCommand(
        command_name="session_orchestrator",
        requester="business-proof",
        payload={"prompt": "summarise the deploy"},
        response_topic=_RESPONSE_TOPIC,
        timeout_seconds=1,
    )


@pytest.mark.asyncio
class TestWorkerCommandCarriesTheTenant:
    """RED before this change: `tenant_id` was `None` on both envelopes."""

    async def test_worker_command_carries_the_consumed_tenant(self) -> None:
        bus = _CapturingBus()
        broker = _broker(bus)
        route = _route()

        with bind_dispatch_envelope(_consumed(_BETA_TENANT_UUID)):
            await broker._publish_worker_command(_command(), route)

        published = bus.envelope_for(route.command_topic)
        assert published.tenant_id == _BETA_TENANT_UUID, (
            "the worker command drops the tenant the consumed envelope "
            "recorded, so every event published downstream of it is "
            "unattributed and a quality verdict reaches the projection writer "
            f"with nothing to attribute it to (house: {_HOUSE_TENANT_UUID})"
        )

    async def test_worker_command_records_nothing_with_no_consumed_envelope(
        self,
    ) -> None:
        """From the HTTP ingress the command genuinely is a chain head."""
        bus = _CapturingBus()
        broker = _broker(bus)
        route = _route()

        await broker._publish_worker_command(_command(), route)

        published = bus.envelope_for(route.command_topic)
        assert published.tenant_id is None
        assert published.parent_envelope_id is None

    async def test_worker_command_records_nothing_when_the_cause_recorded_nothing(
        self,
    ) -> None:
        """Carried, never sourced: an unattributed cause stays unattributed."""
        bus = _CapturingBus()
        broker = _broker(bus)
        route = _route()

        with bind_dispatch_envelope(_consumed(None)):
            await broker._publish_worker_command(_command(), route)

        assert bus.envelope_for(route.command_topic).tenant_id is None

    async def test_the_causal_edge_still_rides_alongside_it(self) -> None:
        """The OMN-18419 carry is unchanged; this adds a dimension, not a swap."""
        bus = _CapturingBus()
        broker = _broker(bus)
        route = _route()
        consumed = _consumed(_BETA_TENANT_UUID)

        with bind_dispatch_envelope(consumed):
            await broker._publish_worker_command(_command(), route)

        published = bus.envelope_for(route.command_topic)
        assert published.parent_envelope_id == consumed.envelope_id
        assert published.tenant_id == _BETA_TENANT_UUID


@pytest.mark.asyncio
class TestTerminalResultCarriesTheTenant:
    async def test_terminal_result_carries_the_consumed_tenant(self) -> None:
        bus = _CapturingBus()
        broker = _broker(bus)
        result = ModelDispatchBusTerminalResult(
            correlation_id=uuid.uuid4(),
            status="completed",
            payload={"status": "complete"},
        )

        with bind_dispatch_envelope(_consumed(_BETA_TENANT_UUID)):
            await broker._publish_terminal_result(_RESPONSE_TOPIC, result)

        assert bus.envelope_for(_RESPONSE_TOPIC).tenant_id == _BETA_TENANT_UUID

    async def test_terminal_result_records_nothing_with_no_consumed_envelope(
        self,
    ) -> None:
        bus = _CapturingBus()
        broker = _broker(bus)
        result = ModelDispatchBusTerminalResult(
            correlation_id=uuid.uuid4(),
            status="completed",
            payload={"status": "complete"},
        )

        await broker._publish_terminal_result(_RESPONSE_TOPIC, result)

        assert bus.envelope_for(_RESPONSE_TOPIC).tenant_id is None


@pytest.mark.asyncio
async def test_inmemory_bus_round_trip_still_works_with_the_stamp() -> None:
    """The stamp does not change the wire shape the bus accepts."""
    bus = EventBusInmemory(environment="test", group="pattern-b-omn17228")
    await bus.start()
    try:
        broker = _broker(bus)
        route = _route()
        with bind_dispatch_envelope(_consumed(_BETA_TENANT_UUID)):
            await broker._publish_worker_command(_command(), route)
    finally:
        await bus.close()
