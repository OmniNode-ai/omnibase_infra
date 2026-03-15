# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression test: DispatchResultApplier must publish ModelNodeRegistrationAccepted
to onex.evt.platform.node-registration-accepted.v1, not to the fallback "responses" topic.

This test catches any future regression where the topic_router is not wired or
the contract published_events mapping drifts from the runtime.

OMN-4884 / OMN-4880
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import yaml

from omnibase_infra.enums.enum_dispatch_status import EnumDispatchStatus
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.models.registration.events.model_node_registration_accepted import (
    ModelNodeRegistrationAccepted,
)
from omnibase_infra.runtime.contract_topic_router import (
    build_topic_router_from_contract,
)
from omnibase_infra.runtime.service_dispatch_result_applier import DispatchResultApplier

_CONTRACT_DATA = yaml.safe_load(
    (
        Path(__file__).parents[4]
        / "src/omnibase_infra/nodes/node_registration_orchestrator/contract.yaml"
    ).read_text()
)
_EXPECTED_ACCEPTED_TOPIC = "onex.evt.platform.node-registration-accepted.v1"


def _make_accepted_event() -> ModelNodeRegistrationAccepted:
    now = datetime.now(UTC)
    return ModelNodeRegistrationAccepted(
        entity_id=uuid.uuid4(),
        node_id=uuid.uuid4(),
        correlation_id=uuid.uuid4(),
        causation_id=uuid.uuid4(),
        emitted_at=now,
        ack_deadline=now + timedelta(seconds=90),
    )


@pytest.mark.asyncio
@pytest.mark.unit
async def test_registration_accepted_event_routes_to_declared_topic() -> None:
    """ModelNodeRegistrationAccepted must never be published to the fallback topic."""
    bus = AsyncMock()
    router = build_topic_router_from_contract(_CONTRACT_DATA)
    applier = DispatchResultApplier(
        event_bus=bus,
        output_topic="responses",  # fallback — must NOT be used for this event
        topic_router=router,
    )

    result = ModelDispatchResult(
        status=EnumDispatchStatus.SUCCESS,
        topic="onex.evt.platform.node-introspection.v1",
        started_at=datetime.now(UTC),
        output_events=[_make_accepted_event()],
    )

    await applier.apply(result)

    assert bus.publish_envelope.call_count == 1
    published_topic = bus.publish_envelope.call_args.kwargs["topic"]
    assert published_topic == _EXPECTED_ACCEPTED_TOPIC, (
        f"ModelNodeRegistrationAccepted was published to '{published_topic}' "
        f"instead of '{_EXPECTED_ACCEPTED_TOPIC}'. "
        "This means the topic_router is not wired or the contract mapping is wrong."
    )
    assert published_topic != "responses", (
        "ModelNodeRegistrationAccepted must never be published to the fallback 'responses' topic."
    )
