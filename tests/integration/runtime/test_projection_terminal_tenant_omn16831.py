# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The projection terminal reaches a real subscriber attributed (OMN-16831 item 2).

The sibling seam test in this directory covers the dispatch path: a tenant
recorded inbound survives into what ``DispatchResultApplier`` publishes. This
covers the path that deliberately does NOT go through that applier.

``_emit_projection_terminal_event`` publishes a projection handler's own
DECLARED output directly onto the bus -- OMN-17214 Defect B, noted in the
function itself -- which is exactly why the OMN-16831 carriage fix in
``omnibase_infra#3573`` never reached it. A tenant's projection terminal was
published unattributed, and the only reader of a tenant in the fleet
(``envelope_tenant_identity`` in omnimarket) reads the envelope dimension.

This drives a real ``EventBusInmemory``: a real subscription, a real publish,
real JSON serialization over the wire shape, and an assertion on what a
SUBSCRIBER actually received rather than on what the producer intended. The
unit test beside it asserts the construction; this asserts the delivery, and
the two fail for different reasons -- a serialization path that dropped the
field would pass the first and fail this one.

The negative control is not optional. OMN-16831 AC2 and OMN-16804 AC3 forbid
the runtime inventing or defaulting a tenant, so a propagation that also
SOURCED one would satisfy the positive case while destroying the property the
projection writer's fail-closed refusal exists to protect.
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

import pytest

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _emit_projection_terminal_event,
)

_TERMINAL_TOPIC = "onex.evt.omnibase-infra.projection-applied.v1"  # onex-topic-allow: terminal address under test
_TENANT = "acme"


async def _terminal_seen_by_a_subscriber(
    tenant_id: str | None,
) -> ModelEventEnvelope[Any]:
    """Publish one projection terminal on a real bus and return what arrived."""
    received: list[ModelEventEnvelope[Any]] = []
    arrived = asyncio.Event()

    async def _collect(message: ModelEventMessage) -> None:
        received.append(ModelEventEnvelope[object].model_validate_json(message.value))
        arrived.set()

    bus = EventBusInmemory(environment="test", group="omn16831-projection-terminal")
    await bus.start()
    try:
        await bus.subscribe(
            _TERMINAL_TOPIC,
            group_id="omn16831-projection-terminal",
            on_message=_collect,
        )

        # The wire shape a Kafka delivery hands the runtime: a mapping, not a
        # ModelEventEnvelope instance. The tenant is on the envelope dimension,
        # where the gateway forwarder records it.
        source: dict[str, object] = {
            "payload": {"task_type": "test"},
            "correlation_id": str(uuid4()),
            "event_type": "omnimarket.delegate-skill-completed",
        }
        if tenant_id is not None:
            source["tenant_id"] = tenant_id

        await _emit_projection_terminal_event(
            bus, _TERMINAL_TOPIC, source, {"rows_upserted": 1}
        )
        await asyncio.wait_for(arrived.wait(), timeout=10)
    finally:
        await bus.close()

    assert len(received) == 1
    return received[0]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_projection_terminal_arrives_attributed() -> None:
    """A subscriber receives the terminal carrying the source record's tenant."""
    delivered = await _terminal_seen_by_a_subscriber(_TENANT)

    assert delivered.tenant_id == _TENANT


@pytest.mark.integration
@pytest.mark.asyncio
async def test_projection_terminal_arrives_unattributed_when_nothing_was_recorded() -> (
    None
):
    """Negative control: an unattributed source stays unattributed on delivery.

    This is the behaviour the projection writer's refusal depends on -- an
    event nobody attributed must reach the writer still unattributed, so the
    refusal fires instead of a fabricated identity being written.
    """
    delivered = await _terminal_seen_by_a_subscriber(None)

    assert delivered.tenant_id is None
