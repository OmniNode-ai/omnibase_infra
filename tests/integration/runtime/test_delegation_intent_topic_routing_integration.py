# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for OMN-11095 delegation intent topic routing.

End-to-end regression coverage that intermediate delegation intent payloads
(routing, inference, quality-gate, baseline, remote-agent invocation) are
published to their per-class command topics rather than the terminal
delegation-completed fallback topic.

Acceptance criteria (from OMN-11095):

* Intermediate intents are not published to ``onex.evt.omnibase-infra.delegation-completed.v1``.
* The delegation-completed topic contains only the terminal
  ``ModelDelegationResult`` envelope for the correlation.
* Focused tests cover dispatch-result topic selection for routing, inference,
  quality, baseline, and terminal delegation events.

These tests exercise the full DispatchResultApplier publish path with an
in-memory capturing event bus to assert the resolved topic per output event.

Related:
    - OMN-11095: Fix delegation-completed topic contamination.
    - src/omnibase_infra/runtime/service_dispatch_result_applier.py
    - tests/unit/runtime/test_dispatch_result_applier_topic_routing.py
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumDispatchStatus
from omnibase_infra.enums.generated.enum_omnibase_infra_topic import (
    EnumOmnibaseInfraTopic,
)
from omnibase_infra.event_bus.topic_constants import (
    TOPIC_DELEGATION_BASELINE_COMPARISON,
    TOPIC_DELEGATION_INFERENCE_REQUEST,
    TOPIC_DELEGATION_QUALITY_GATE_REQUEST,
    TOPIC_DELEGATION_ROUTING_REQUEST,
)
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.runtime.service_dispatch_result_applier import (
    DispatchResultApplier,
)

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Stand-in payload models
# ---------------------------------------------------------------------------


class ModelRoutingIntent(BaseModel):
    """Stand-in matching the omnimarket reducer routing intent class name."""

    intent: str = "routing_reducer"
    correlation_id: UUID


class ModelInferenceIntent(BaseModel):
    intent: str = "llm_inference"
    correlation_id: UUID


class ModelInvocationCommand(BaseModel):
    intent: str = "remote_agent_invoke"
    correlation_id: UUID


class ModelQualityGateIntent(BaseModel):
    intent: str = "quality_gate"
    correlation_id: UUID


class ModelBaselineIntent(BaseModel):
    intent: str = "baseline_comparison"
    correlation_id: UUID


class ModelDelegationResult(BaseModel):
    """Stand-in for the terminal delegation result payload."""

    correlation_id: UUID
    status: str = "completed"
    quality_score: float = 1.0


# ---------------------------------------------------------------------------
# In-memory capturing event bus
# ---------------------------------------------------------------------------


class _CapturingEventBus:
    """Captures every publish_envelope call for later assertions."""

    def __init__(self) -> None:
        self.published: list[dict[str, Any]] = []

    async def publish_envelope(
        self,
        *,
        envelope: ModelEventEnvelope[Any],
        topic: str,
        key: bytes | None = None,
    ) -> None:
        self.published.append(
            {
                "envelope": envelope,
                "topic": topic,
                "key": key,
                "captured_at": datetime.now(UTC),
            }
        )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_result(
    *,
    correlation_id: UUID,
    output_events: list[BaseModel],
) -> ModelDispatchResult:
    return ModelDispatchResult(
        status=EnumDispatchStatus.SUCCESS,
        topic="test.topic",
        started_at=datetime.now(UTC),
        correlation_id=correlation_id,
        dispatcher_id="integration-test-dispatcher",
        output_events=output_events,
    )


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestDelegationIntentTopicRoutingIntegration:
    """OMN-11095 regression: intermediate intents do not contaminate terminal topic."""

    @pytest.mark.asyncio
    async def test_intermediate_delegation_intents_route_to_command_topics(
        self,
    ) -> None:
        """All intermediate intent classes publish to their per-class command topics.

        The applier is configured with the terminal completed topic as
        ``output_topic`` (mirroring production wiring). Each intermediate
        intent must select its own command topic from
        ``_DELEGATION_INTENT_TOPIC_BY_CLASS`` rather than falling back to the
        completed terminal topic.
        """
        bus = _CapturingEventBus()
        completed_topic = "onex.evt.omnibase-infra.delegation-completed.v1"
        applier = DispatchResultApplier(
            event_bus=bus,
            output_topic=completed_topic,
        )

        correlation_id = uuid4()
        result = _make_result(
            correlation_id=correlation_id,
            output_events=[
                ModelRoutingIntent(correlation_id=correlation_id),
                ModelInvocationCommand(correlation_id=correlation_id),
                ModelInferenceIntent(correlation_id=correlation_id),
                ModelQualityGateIntent(correlation_id=correlation_id),
                ModelBaselineIntent(correlation_id=correlation_id),
            ],
        )

        await applier.apply(result)

        topics = [record["topic"] for record in bus.published]
        assert topics == [
            TOPIC_DELEGATION_ROUTING_REQUEST,
            EnumOmnibaseInfraTopic.CMD_REMOTE_AGENT_INVOKE_V1.value,
            TOPIC_DELEGATION_INFERENCE_REQUEST,
            TOPIC_DELEGATION_QUALITY_GATE_REQUEST,
            TOPIC_DELEGATION_BASELINE_COMPARISON,
        ]
        # The terminal topic must not appear for any intermediate intent.
        assert completed_topic not in topics

    @pytest.mark.asyncio
    async def test_terminal_delegation_result_publishes_to_completed_topic(
        self,
    ) -> None:
        """Terminal ``ModelDelegationResult`` payloads still flow through the
        configured fallback topic."""
        bus = _CapturingEventBus()
        completed_topic = "onex.evt.omnibase-infra.delegation-completed.v1"
        applier = DispatchResultApplier(
            event_bus=bus,
            output_topic=completed_topic,
        )

        correlation_id = uuid4()
        result = _make_result(
            correlation_id=correlation_id,
            output_events=[ModelDelegationResult(correlation_id=correlation_id)],
        )

        await applier.apply(result)

        assert len(bus.published) == 1
        published = bus.published[0]
        assert published["topic"] == completed_topic
        assert isinstance(published["envelope"].payload, ModelDelegationResult)
        assert published["envelope"].correlation_id == correlation_id

    @pytest.mark.asyncio
    async def test_mixed_intermediate_and_terminal_payloads_route_independently(
        self,
    ) -> None:
        """A single dispatch result mixing intermediate + terminal payloads
        routes each to its own topic without contamination."""
        bus = _CapturingEventBus()
        completed_topic = "onex.evt.omnibase-infra.delegation-completed.v1"
        applier = DispatchResultApplier(
            event_bus=bus,
            output_topic=completed_topic,
        )

        correlation_id = uuid4()
        result = _make_result(
            correlation_id=correlation_id,
            output_events=[
                ModelRoutingIntent(correlation_id=correlation_id),
                ModelDelegationResult(correlation_id=correlation_id),
            ],
        )

        await applier.apply(result)

        topics = [record["topic"] for record in bus.published]
        assert topics == [
            TOPIC_DELEGATION_ROUTING_REQUEST,
            completed_topic,
        ]
        # The intermediate intent must not appear on the completed topic.
        terminal_records = [
            record for record in bus.published if record["topic"] == completed_topic
        ]
        assert len(terminal_records) == 1
        assert isinstance(
            terminal_records[0]["envelope"].payload, ModelDelegationResult
        )
