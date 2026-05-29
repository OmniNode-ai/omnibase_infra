# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit test for the OMN-12409 delegation-worker result_applier wiring.

The service kernel wires a DispatchResultApplier for the delegation chain's
worker contracts (routing reducer, quality-gate reducer, LLM call effect) by
loading each contract's published_events map and routing the handler's returned
model to the declared topic. Without this, an auto-wired worker handler's return
is dropped (no publish) and the chain stalls — on the effects instance the call
effect's inference-response never reached the orchestrator.

This test reproduces the kernel's exact construction
(load_published_events_map -> DispatchResultApplier(output_topic_map=...)) against
a synthetic contract and asserts the returned model routes to the correct topic.

Related:
    - OMN-12409: EFFECT/reducer result_applier gap
    - service_kernel.py auto_wiring_result_appliers delegation block
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from pydantic import BaseModel

from omnibase_infra.enums import EnumDispatchStatus
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.runtime.event_bus_subcontract_wiring import (
    load_published_events_map,
)
from omnibase_infra.runtime.service_dispatch_result_applier import (
    DispatchResultApplier,
)

pytestmark = pytest.mark.unit

_CONTRACT_YAML = """\
name: node_delegation_routing_reducer
node_type: REDUCER_GENERIC
event_bus:
  publish_topics:
    - "onex.evt.omnibase-infra.routing-decision.v1"
published_events:
  - event_type: "RoutingDecision"
    topic: "onex.evt.omnibase-infra.routing-decision.v1"
"""


class ModelRoutingDecision(BaseModel):
    """Stub mirroring the real returned model's class short-name (RoutingDecision)."""

    correlation_id: str = "c-1"


def test_published_events_map_loads_for_delegation_worker(tmp_path: Path) -> None:
    contract = tmp_path / "contract.yaml"
    contract.write_text(_CONTRACT_YAML, encoding="utf-8")

    pe_map = load_published_events_map(contract)

    assert pe_map == {"RoutingDecision": "onex.evt.omnibase-infra.routing-decision.v1"}


@pytest.mark.asyncio
async def test_applier_routes_returned_model_to_published_events_topic(
    tmp_path: Path,
) -> None:
    # Reproduce the kernel's exact construction (OMN-12409).
    contract = tmp_path / "contract.yaml"
    contract.write_text(_CONTRACT_YAML, encoding="utf-8")
    pe_map = load_published_events_map(contract)
    topics = list(pe_map.values())

    event_bus = AsyncMock()
    applier = DispatchResultApplier(
        event_bus=event_bus,
        output_topic=topics[0],
        output_topic_map=pe_map,
        allowed_output_topics=topics,
    )

    # A dispatch result carrying the worker's returned model as an output event.
    result = ModelDispatchResult(
        status=EnumDispatchStatus.SUCCESS,
        topic="onex.cmd.omnibase-infra.delegation-routing-request.v1",
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        output_count=1,
        output_events=[ModelRoutingDecision()],
        correlation_id=uuid4(),
    )

    await applier.apply(result, uuid4())

    # The returned ModelRoutingDecision must publish to routing-decision.v1.
    assert event_bus.publish_envelope.await_count == 1
    _, kwargs = event_bus.publish_envelope.await_args
    assert kwargs["topic"] == "onex.evt.omnibase-infra.routing-decision.v1"
