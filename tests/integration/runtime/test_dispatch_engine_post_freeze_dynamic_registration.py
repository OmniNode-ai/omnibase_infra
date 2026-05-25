# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for post-freeze dynamic dispatch registration."""

from __future__ import annotations

import pytest

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.models.dispatch.model_dispatch_route import ModelDispatchRoute
from omnibase_infra.runtime.message_dispatch_engine import (
    MessageDispatchEngine,
)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_dynamic_handler_wiring_dispatches_after_engine_freeze() -> None:
    """A dynamically registered post-freeze route can dispatch an event."""
    engine = MessageDispatchEngine()
    engine.freeze()

    handled_events: list[object] = []

    async def handler(event: object) -> None:
        handled_events.append(event)

    engine._register_dispatcher_dynamic(
        dispatcher_id="omn-11246-handler",
        dispatcher=handler,
        category=EnumMessageCategory.EVENT,
        message_types={"omnibase-infra.dynamic-registration"},
    )
    engine._register_route_dynamic(
        ModelDispatchRoute(
            route_id="omn-11246-route",
            topic_pattern="onex.evt.omnibase-infra.dynamic-registration.v1",
            message_category=EnumMessageCategory.EVENT,
            dispatcher_id="omn-11246-handler",
        )
    )

    envelope = ModelEventEnvelope(
        payload={"registered_after_freeze": True},
        event_type="omnibase-infra.dynamic-registration",
    )
    await engine.dispatch("onex.evt.omnibase-infra.dynamic-registration.v1", envelope)

    assert len(handled_events) == 1
    assert handled_events[0]["payload"] == {"registered_after_freeze": True}
    assert handled_events[0]["__debug_trace"]["topic"] == (
        "onex.evt.omnibase-infra.dynamic-registration.v1"
    )
