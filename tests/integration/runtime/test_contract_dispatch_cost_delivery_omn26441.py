# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage: a projection writer under cost measurement gets its delivery (OMN-26441).

``attach_contract_dispatcher`` wraps every attached contract's dispatcher to
charge its time to the contract (OMN-26250). The wrapper took only the envelope,
so the dispatch engine's signature probe found no keyword-only ``delivery``
parameter and called a projection writer without ``_partition`` and ``_offset``;
the dev runtime then raised ``ProjectionDeliveryContextMissingError`` on every
record and rewound the same offset.

This drives a projection-writer-shaped dispatcher through the real attach seam,
registers it on a real ``MessageDispatchEngine`` with an owner contract label
(cost measurement on) and dispatches one envelope carrying delivery coordinates.
The writer builds its ``_partition`` and ``_offset`` columns from the delivery
it receives and refuses to build them from none, as a projection writer does.
No service is needed.
"""

from __future__ import annotations

import pytest

from omnibase_core.models.dispatch.model_dispatch_route import ModelDispatchRoute
from omnibase_core.models.dispatch.model_message_delivery_context import (
    ModelMessageDeliveryContext,
)
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums.enum_dispatch_status import EnumDispatchStatus
from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.handlers.contract_dispatch_cost import attach_contract_dispatcher
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

pytestmark = pytest.mark.integration

_TOPIC = "test.projection.writer.events.v1"
_EVENT_TYPE = "ProjectionProbeEvent"
_DISPATCHER_ID = "projection-writer-probe"
_OWNER = "projection_writer_cost_probe"


class _RecordingProjectionWriter:
    """Records the columns a projection writer derives from its delivery."""

    def __init__(self) -> None:
        self.rows: list[dict[str, int]] = []

    async def __call__(
        self,
        envelope: ModelEventEnvelope[object],
        *,
        delivery: ModelMessageDeliveryContext | None = None,
    ) -> None:
        if delivery is None:
            raise AssertionError("dispatched with no delivery context")
        self.rows.append({"_partition": delivery.partition, "_offset": delivery.offset})


async def test_projection_writer_under_cost_measurement_receives_delivery() -> None:
    writer = _RecordingProjectionWriter()
    measured = attach_contract_dispatcher(_OWNER, writer)
    assert measured is not writer, (
        "cost measurement is off: the dispatcher is unwrapped"
    )
    engine = MessageDispatchEngine()
    engine.register_dispatcher(
        dispatcher_id=_DISPATCHER_ID,
        dispatcher=measured,
        category=EnumMessageCategory.EVENT,
        message_types={_EVENT_TYPE},
        owner_contract_name=_OWNER,
    )
    engine.register_route(
        ModelDispatchRoute(
            route_id="projection-writer-probe-route",
            topic_pattern=_TOPIC,
            message_category=EnumMessageCategory.EVENT,
            handler_id=_DISPATCHER_ID,
        )
    )
    engine.freeze()
    delivery = ModelMessageDeliveryContext(topic=_TOPIC, partition=3, offset=16698)
    envelope = ModelEventEnvelope[object](payload={"k": "v"}, event_type=_EVENT_TYPE)

    result = await engine.dispatch(_TOPIC, envelope, delivery=delivery)

    assert result.status is EnumDispatchStatus.SUCCESS, result
    assert writer.rows == [{"_partition": 3, "_offset": 16698}]
