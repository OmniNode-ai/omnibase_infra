# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for post-freeze dynamic dispatch registration."""

from __future__ import annotations

import pytest

from omnibase_core.enums.enum_handler_resolution_outcome import (
    EnumHandlerResolutionOutcome,
)
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.models.dispatch.model_dispatch_route import ModelDispatchRoute
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    PreparedWiring,
    _commit_handler_wiring,
)
from omnibase_infra.runtime.service_message_dispatch_engine import (
    MessageDispatchEngine,
)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_dynamic_handler_wiring_dispatches_after_engine_freeze() -> None:
    """Authorized materialization can add a dispatcher and route after freeze."""
    engine = MessageDispatchEngine()
    engine.freeze()

    handled_payloads: list[object] = []

    async def handler(payload: object) -> None:
        handled_payloads.append(payload)

    route = ModelDispatchRoute(
        route_id="omn-11246-dynamic-route",
        topic_pattern="onex.evt.omnibase-infra.dynamic-registration.v1",
        message_category=EnumMessageCategory.EVENT,
        dispatcher_id="omn-11246-dynamic-dispatcher",
    )
    prepared = PreparedWiring(
        dispatcher_id="omn-11246-dynamic-dispatcher",
        dispatcher=handler,
        category=EnumMessageCategory.EVENT,
        message_types={"omnibase-infra.dynamic-registration"},
        routes=[route],
        route_ids=[route.route_id],
        resolution_outcome=EnumHandlerResolutionOutcome.RESOLVED_VIA_CONTAINER,
    )

    dispatcher_id, route_ids = _commit_handler_wiring(
        prepared,
        engine,
        dynamic_materialization_authorized=True,
    )

    assert dispatcher_id == "omn-11246-dynamic-dispatcher"
    assert route_ids == ["omn-11246-dynamic-route"]

    envelope = ModelEventEnvelope(
        payload={"registered_after_freeze": True},
        event_type="omnibase-infra.dynamic-registration",
    )
    await engine.dispatch("onex.evt.omnibase-infra.dynamic-registration.v1", envelope)

    assert handled_payloads == [
        {
            "payload": {"registered_after_freeze": True},
            "__bindings": {},
            "__debug_trace": {
                "event_type": "omnibase-infra.dynamic-registration",
                "correlation_id": None,
                "trace_id": None,
                "causation_id": None,
                "topic": "onex.evt.omnibase-infra.dynamic-registration.v1",
                "timestamp": None,
                "partition_key": None,
            },
        }
    ]
