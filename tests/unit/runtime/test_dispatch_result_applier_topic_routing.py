# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for DispatchResultApplier output_topic_map routing.

Tests the _resolve_output_topic() method and the end-to-end publish path
when output_topic_map is configured.

Related:
    - OMN-5132: Contract-aware topic routing
    - src/omnibase_infra/runtime/service_dispatch_result_applier.py
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from pydantic import BaseModel

from omnibase_infra.enums import EnumDispatchStatus
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.runtime.service_dispatch_result_applier import (
    DispatchResultApplier,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class ModelNodeBecameActive(BaseModel):
    """Stub event mimicking the real model class name."""

    entity_id: str = "node-123"


class ModelNodeRegistrationAccepted(BaseModel):
    """Stub event mimicking registration accepted."""

    entity_id: str = "node-456"


class UnmappedEvent(BaseModel):
    """Event not present in any topic map."""

    value: str = "x"


def _make_result(**overrides: object) -> ModelDispatchResult:
    defaults: dict[str, object] = {
        "status": EnumDispatchStatus.SUCCESS,
        "topic": "test.topic",
        "started_at": datetime.now(UTC),
        "correlation_id": uuid4(),
        "dispatcher_id": "test-dispatcher",
    }
    defaults.update(overrides)
    return ModelDispatchResult(**defaults)


# ---------------------------------------------------------------------------
# _resolve_output_topic unit tests
# ---------------------------------------------------------------------------


class TestResolveOutputTopic:
    """Tests for _resolve_output_topic()."""

    def test_returns_fallback_when_map_empty(self) -> None:
        applier = DispatchResultApplier(
            event_bus=AsyncMock(),
            output_topic="fallback-topic",
        )
        event = ModelNodeBecameActive()
        assert applier._resolve_output_topic(event) == "fallback-topic"

    def test_resolves_via_short_name(self) -> None:
        applier = DispatchResultApplier(
            event_bus=AsyncMock(),
            output_topic="fallback-topic",
            output_topic_map={
                "NodeBecameActive": "onex.evt.platform.node-became-active.v1",
            },
        )
        event = ModelNodeBecameActive()
        assert (
            applier._resolve_output_topic(event)
            == "onex.evt.platform.node-became-active.v1"
        )

    def test_resolves_via_full_class_name(self) -> None:
        applier = DispatchResultApplier(
            event_bus=AsyncMock(),
            output_topic="fallback-topic",
            output_topic_map={
                "ModelNodeBecameActive": "onex.evt.platform.node-became-active.v1",
            },
        )
        event = ModelNodeBecameActive()
        assert (
            applier._resolve_output_topic(event)
            == "onex.evt.platform.node-became-active.v1"
        )

    def test_short_name_takes_precedence_over_full(self) -> None:
        applier = DispatchResultApplier(
            event_bus=AsyncMock(),
            output_topic="fallback-topic",
            output_topic_map={
                "NodeBecameActive": "short-topic",
                "ModelNodeBecameActive": "full-topic",
            },
        )
        event = ModelNodeBecameActive()
        assert applier._resolve_output_topic(event) == "short-topic"

    def test_falls_back_to_output_topic_when_not_in_map(self) -> None:
        applier = DispatchResultApplier(
            event_bus=AsyncMock(),
            output_topic="fallback-topic",
            output_topic_map={
                "SomeOtherEvent": "other-topic",
            },
        )
        event = UnmappedEvent()
        assert applier._resolve_output_topic(event) == "fallback-topic"

    def test_non_model_prefixed_class_uses_full_name(self) -> None:
        """Classes without Model prefix: short_name == class_name."""
        applier = DispatchResultApplier(
            event_bus=AsyncMock(),
            output_topic="fallback-topic",
            output_topic_map={
                "UnmappedEvent": "mapped-topic",
            },
        )
        event = UnmappedEvent()
        assert applier._resolve_output_topic(event) == "mapped-topic"


# ---------------------------------------------------------------------------
# Publish path integration tests
# ---------------------------------------------------------------------------


class TestPublishPathTopicRouting:
    """Tests that apply() publishes to the correct resolved topic."""

    @pytest.mark.asyncio
    async def test_publish_uses_output_topic_map(self) -> None:
        mock_bus = AsyncMock()
        applier = DispatchResultApplier(
            event_bus=mock_bus,
            output_topic="fallback-topic",
            output_topic_map={
                "NodeBecameActive": "onex.evt.platform.node-became-active.v1",
            },
        )
        result = _make_result(
            output_events=[ModelNodeBecameActive(entity_id="n-1")],
        )
        await applier.apply(result)

        mock_bus.publish_envelope.assert_called_once()
        call_kwargs = mock_bus.publish_envelope.call_args
        assert call_kwargs.kwargs["topic"] == "onex.evt.platform.node-became-active.v1"

    @pytest.mark.asyncio
    async def test_publish_falls_back_to_output_topic(self) -> None:
        mock_bus = AsyncMock()
        applier = DispatchResultApplier(
            event_bus=mock_bus,
            output_topic="fallback-topic",
            output_topic_map={
                "SomeOtherEvent": "other-topic",
            },
        )
        result = _make_result(
            output_events=[UnmappedEvent()],
        )
        await applier.apply(result)

        mock_bus.publish_envelope.assert_called_once()
        call_kwargs = mock_bus.publish_envelope.call_args
        assert call_kwargs.kwargs["topic"] == "fallback-topic"

    @pytest.mark.asyncio
    async def test_topic_router_takes_precedence_over_output_topic_map(self) -> None:
        """topic_router (OMN-4881) takes priority over output_topic_map (OMN-5132)."""
        mock_bus = AsyncMock()
        applier = DispatchResultApplier(
            event_bus=mock_bus,
            output_topic="fallback-topic",
            topic_router={
                "ModelNodeBecameActive": "router-topic",
            },
            output_topic_map={
                "NodeBecameActive": "map-topic",
            },
        )
        result = _make_result(
            output_events=[ModelNodeBecameActive(entity_id="n-1")],
        )
        await applier.apply(result)

        mock_bus.publish_envelope.assert_called_once()
        call_kwargs = mock_bus.publish_envelope.call_args
        assert call_kwargs.kwargs["topic"] == "router-topic"

    @pytest.mark.asyncio
    async def test_multiple_events_route_independently(self) -> None:
        mock_bus = AsyncMock()
        applier = DispatchResultApplier(
            event_bus=mock_bus,
            output_topic="fallback-topic",
            output_topic_map={
                "NodeBecameActive": "active-topic",
                "NodeRegistrationAccepted": "accepted-topic",
            },
        )
        result = _make_result(
            output_events=[
                ModelNodeBecameActive(entity_id="n-1"),
                ModelNodeRegistrationAccepted(entity_id="n-2"),
            ],
        )
        await applier.apply(result)

        assert mock_bus.publish_envelope.call_count == 2
        topics = [c.kwargs["topic"] for c in mock_bus.publish_envelope.call_args_list]
        assert topics == ["active-topic", "accepted-topic"]


# ---------------------------------------------------------------------------
# Integration tests with real contract
# ---------------------------------------------------------------------------


class TestRealContractIntegration:
    """Verify that the real registration orchestrator contract maps critical events."""

    def test_contract_maps_critical_event_types(self) -> None:
        """The real contract.yaml must map NodeBecameActive and others."""
        from pathlib import Path

        from omnibase_infra.runtime.event_bus_subcontract_wiring import (
            load_published_events_map,
        )

        contract_path = (
            Path(__file__).resolve().parents[3]
            / "src"
            / "omnibase_infra"
            / "nodes"
            / "node_registration_orchestrator"
            / "contract.yaml"
        )
        assert contract_path.exists(), f"Contract not found at {contract_path}"

        topic_map = load_published_events_map(contract_path)

        # Critical event types that MUST be mapped
        assert "NodeBecameActive" in topic_map
        assert (
            topic_map["NodeBecameActive"] == "onex.evt.platform.node-became-active.v1"
        )

        assert "NodeRegistrationAccepted" in topic_map
        assert (
            topic_map["NodeRegistrationAccepted"]
            == "onex.evt.platform.node-registration-accepted.v1"
        )

        assert "NodeRegistrationRejected" in topic_map
        assert (
            topic_map["NodeRegistrationRejected"]
            == "onex.evt.platform.node-registration-rejected.v1"
        )

    @pytest.mark.asyncio
    async def test_applier_resolves_real_model_to_correct_topic(self) -> None:
        """DispatchResultApplier resolves a real ModelNodeBecameActive instance."""
        from datetime import UTC, datetime
        from pathlib import Path
        from uuid import uuid4 as _uuid4

        from omnibase_infra.models.registration.events.model_node_became_active import (
            ModelNodeBecameActive as RealModelNodeBecameActive,
        )
        from omnibase_infra.models.registration.model_node_capabilities import (
            ModelNodeCapabilities,
        )
        from omnibase_infra.runtime.event_bus_subcontract_wiring import (
            load_published_events_map,
        )

        contract_path = (
            Path(__file__).resolve().parents[3]
            / "src"
            / "omnibase_infra"
            / "nodes"
            / "node_registration_orchestrator"
            / "contract.yaml"
        )
        topic_map = load_published_events_map(contract_path)

        mock_bus = AsyncMock()
        applier = DispatchResultApplier(
            event_bus=mock_bus,
            output_topic="fallback-topic",
            output_topic_map=topic_map,
        )

        node_id = _uuid4()
        event = RealModelNodeBecameActive(
            entity_id=node_id,
            node_id=node_id,
            correlation_id=_uuid4(),
            causation_id=_uuid4(),
            emitted_at=datetime.now(UTC),
            capabilities=ModelNodeCapabilities(postgres=True, read=True),
        )
        result = _make_result(output_events=[event])
        await applier.apply(result)

        mock_bus.publish_envelope.assert_called_once()
        call_kwargs = mock_bus.publish_envelope.call_args
        assert call_kwargs.kwargs["topic"] == "onex.evt.platform.node-became-active.v1"
