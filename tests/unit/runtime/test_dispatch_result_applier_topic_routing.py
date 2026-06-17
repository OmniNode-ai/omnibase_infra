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

import os
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


class TopicCarryingEvent(BaseModel):
    """Event whose typed payload declares its own topic."""

    topic: str
    value: str = "x"


class ModelRoutingIntent(BaseModel):
    """Stub delegation routing intent from omnimarket."""

    intent: str = "routing_reducer"


class ModelInferenceIntent(BaseModel):
    """Stub delegation inference intent from omnimarket."""

    intent: str = "llm_inference"


class ModelInvocationCommand(BaseModel):
    """Stub delegation remote-agent invocation command from omnibase_core."""

    intent: str = "remote_agent_invoke"


class ModelQualityGateIntent(BaseModel):
    """Stub delegation quality-gate intent from omnimarket."""

    intent: str = "quality_gate"


class ModelBaselineIntent(BaseModel):
    """Stub delegation baseline intent from omnimarket."""

    intent: str = "baseline_comparison"


class ModelDelegationResult(BaseModel):
    """Stub terminal delegation result payload."""

    correlation_id: str = "cid-123"
    content: str = "done"


class ModelDelegationEventEnvelope(BaseModel):
    """Stub topic-bearing delegation event envelope."""

    topic: str
    payload: ModelDelegationResult


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

    def test_embedded_topic_takes_precedence(self) -> None:
        applier = DispatchResultApplier(
            event_bus=AsyncMock(),
            output_topic="fallback-topic",
            output_topic_map={
                "TopicCarryingEvent": "mapped-topic",
                "FailedTopicCarryingEvent": "onex.evt.example.failed.v1",
            },
        )
        event = TopicCarryingEvent(topic="onex.evt.example.failed.v1")
        assert applier._resolve_output_topic(event) == "onex.evt.example.failed.v1"

    def test_embedded_topic_can_use_publish_topic_allowlist(self) -> None:
        applier = DispatchResultApplier(
            event_bus=AsyncMock(),
            output_topic="fallback-topic",
            output_topic_map={
                "TopicCarryingEvent": "onex.evt.example.completed.v1",
            },
            allowed_output_topics={
                "onex.evt.example.completed.v1",
                "onex.evt.example.failed.v1",
            },
        )
        event = TopicCarryingEvent(topic="onex.evt.example.failed.v1")
        assert applier._resolve_output_topic(event) == "onex.evt.example.failed.v1"

    def test_undeclared_embedded_topic_falls_back_to_contract_topic(self) -> None:
        applier = DispatchResultApplier(
            event_bus=AsyncMock(),
            output_topic="fallback-topic",
            output_topic_map={
                "TopicCarryingEvent": "mapped-topic",
            },
        )
        event = TopicCarryingEvent(topic="onex.evt.example.failed.v1")
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
    async def test_embedded_topic_takes_precedence_over_topic_router(self) -> None:
        mock_bus = AsyncMock()
        applier = DispatchResultApplier(
            event_bus=mock_bus,
            output_topic="fallback-topic",
            topic_router={
                "TopicCarryingEvent": "router-topic",
            },
            output_topic_map={
                "TopicCarryingEvent": "onex.evt.example.completed.v1",
            },
            allowed_output_topics={
                "onex.evt.example.completed.v1",
                "onex.evt.example.failed.v1",
            },
        )
        result = _make_result(
            output_events=[TopicCarryingEvent(topic="onex.evt.example.failed.v1")],
        )
        await applier.apply(result)

        mock_bus.publish_envelope.assert_called_once()
        call_kwargs = mock_bus.publish_envelope.call_args
        assert call_kwargs.kwargs["topic"] == "onex.evt.example.failed.v1"

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


class TestDelegationIntentTopicRouting:
    """Regression coverage for OMN-11095 delegation terminal contamination."""

    @pytest.mark.asyncio
    async def test_delegation_intermediate_intents_do_not_fall_back_to_completed_topic(
        self,
    ) -> None:
        mock_bus = AsyncMock()
        completed_topic = "onex.evt.omnibase-infra.delegation-completed.v1"
        applier = DispatchResultApplier(
            event_bus=mock_bus,
            output_topic=completed_topic,
        )
        result = _make_result(
            output_events=[
                ModelRoutingIntent(),
                ModelInvocationCommand(),
                ModelInferenceIntent(),
                ModelQualityGateIntent(),
                ModelBaselineIntent(),
            ],
        )

        await applier.apply(result)

        topics = [c.kwargs["topic"] for c in mock_bus.publish_envelope.call_args_list]
        assert topics == [
            "onex.cmd.omnibase-infra.delegation-routing-request.v1",
            "onex.cmd.omnibase-infra.remote-agent-invoke.v1",
            "onex.cmd.omnibase-infra.delegation-inference-request.v1",
            "onex.cmd.omnibase-infra.delegation-quality-gate-request.v1",
            "onex.cmd.omnibase-infra.baseline-comparison-request.v1",
        ]
        assert completed_topic not in topics

    @pytest.mark.asyncio
    async def test_delegation_terminal_topic_keeps_terminal_payload_only(self) -> None:
        mock_bus = AsyncMock()
        completed_topic = "onex.evt.omnibase-infra.delegation-completed.v1"
        applier = DispatchResultApplier(
            event_bus=mock_bus,
            output_topic=completed_topic,
        )
        result = _make_result(output_events=[ModelDelegationResult()])

        await applier.apply(result)

        mock_bus.publish_envelope.assert_called_once()
        call_kwargs = mock_bus.publish_envelope.call_args.kwargs
        assert call_kwargs["topic"] == completed_topic
        assert isinstance(call_kwargs["envelope"].payload, ModelDelegationResult)

    @pytest.mark.asyncio
    async def test_delegation_topic_envelope_publishes_inner_terminal_payload(
        self,
    ) -> None:
        mock_bus = AsyncMock()
        completed_topic = "onex.evt.omnibase-infra.delegation-completed.v1"
        applier = DispatchResultApplier(
            event_bus=mock_bus,
            output_topic=completed_topic,
        )
        terminal_payload = ModelDelegationResult(correlation_id="cid-456")
        result = _make_result(
            output_events=[
                ModelDelegationEventEnvelope(
                    topic=completed_topic,
                    payload=terminal_payload,
                )
            ],
        )

        await applier.apply(result)

        mock_bus.publish_envelope.assert_called_once()
        call_kwargs = mock_bus.publish_envelope.call_args.kwargs
        assert call_kwargs["topic"] == completed_topic
        assert call_kwargs["envelope"].payload == terminal_payload


class TestDelegationTopicRegistryNoDrift:
    """OMN-13191: the applier resolves delegation topics from ServiceTopicRegistry
    (contract-sourced) instead of importing TOPIC_* string constants.

    Proves the three-way equality the migration depends on:
    registry-resolved string == legacy TOPIC_* constant == contract-declared string.
    A mismatch on any leg means a topic drifted and the migration is unsafe.
    """

    # Logical key -> legacy constant name (the 4 topics the applier resolves
    # plus the wider delegation family added to the registry for completeness).
    _KEY_TO_CONSTANT: dict[str, str] = {
        "DELEGATION_REQUEST": "TOPIC_DELEGATION_REQUEST",
        "DELEGATION_ROUTING_DECISION": "TOPIC_DELEGATION_ROUTING_DECISION",
        "DELEGATION_COMPLETED": "TOPIC_DELEGATION_COMPLETED",
        "DELEGATION_FAILED": "TOPIC_DELEGATION_FAILED",
        "DELEGATION_QUALITY_GATE_RESULT": "TOPIC_DELEGATION_QUALITY_GATE_RESULT",
        "DELEGATION_ROUTING_REQUEST": "TOPIC_DELEGATION_ROUTING_REQUEST",
        "DELEGATION_INVOCATION_COMMAND": "TOPIC_DELEGATION_INVOCATION_COMMAND",
        "DELEGATION_AGENT_TASK_LIFECYCLE": "TOPIC_DELEGATION_AGENT_TASK_LIFECYCLE",
        "DELEGATION_QUALITY_GATE_REQUEST": "TOPIC_DELEGATION_QUALITY_GATE_REQUEST",
        "DELEGATION_INFERENCE_REQUEST": "TOPIC_DELEGATION_INFERENCE_REQUEST",
        "DELEGATION_INFERENCE_RESPONSE": "TOPIC_DELEGATION_INFERENCE_RESPONSE",
        "DELEGATION_TASK_DELEGATED": "TOPIC_DELEGATION_TASK_DELEGATED",
        "DELEGATION_BASELINE_COMPARISON": "TOPIC_DELEGATION_BASELINE_COMPARISON",
    }

    # The 4 topics the infra applier actually resolves, mapped to the owning
    # publisher contract that declares them.
    _APPLIER_KEYS: tuple[str, ...] = (
        "DELEGATION_BASELINE_COMPARISON",
        "DELEGATION_INFERENCE_REQUEST",
        "DELEGATION_QUALITY_GATE_REQUEST",
        "DELEGATION_ROUTING_REQUEST",
    )

    def test_registry_resolution_equals_legacy_constant(self) -> None:
        """Every delegation key resolves to the same string as its TOPIC_* constant."""
        from omnibase_infra.event_bus import topic_constants
        from omnibase_infra.topics import topic_keys
        from omnibase_infra.topics.service_topic_registry import ServiceTopicRegistry

        registry = ServiceTopicRegistry.from_defaults()
        for key_name, const_name in self._KEY_TO_CONSTANT.items():
            key = getattr(topic_keys, key_name)
            resolved = registry.resolve(key)
            constant = getattr(topic_constants, const_name)
            assert resolved == constant, (
                f"drift: registry.resolve({key_name})={resolved!r} != "
                f"{const_name}={constant!r}"
            )

    def test_applier_resolved_topics_match_owning_contract(self) -> None:
        """The 4 applier topics equal the strings declared in the publisher contract.

        The delegation orchestrator (omnimarket node_delegation_orchestrator) is the
        publisher that declares all four command topics in its contract.yaml. This
        anchors the registry value to a contract source, not just a Python constant.
        """
        from pathlib import Path

        from omnibase_infra.topics import topic_keys
        from omnibase_infra.topics.service_topic_registry import ServiceTopicRegistry

        # The omnimarket sibling may live next to this repo (canonical layout) or,
        # in a per-ticket worktree, only in the canonical omni_home clone. Try the
        # in-tree sibling first, then fall back to $OMNI_HOME/omnimarket.
        rel = Path(
            "omnimarket/src/omnimarket/nodes/node_delegation_orchestrator/contract.yaml"
        )
        repo_root = Path(__file__).resolve().parents[3]
        candidates = [repo_root.parent / rel]
        omni_home = os.environ.get("OMNI_HOME")  # ONEX_EXCLUDE: env
        if omni_home:
            candidates.append(Path(omni_home) / rel)
        contract_path = next((p for p in candidates if p.exists()), None)
        if contract_path is None:
            pytest.skip(
                "omnimarket node_delegation_orchestrator/contract.yaml not found in "
                f"any of: {[str(c) for c in candidates]}"
            )

        contract_text = contract_path.read_text(encoding="utf-8")
        registry = ServiceTopicRegistry.from_defaults()
        for key_name in self._APPLIER_KEYS:
            resolved = registry.resolve(getattr(topic_keys, key_name))
            assert resolved in contract_text, (
                f"applier topic {resolved!r} (key {key_name}) is not declared in "
                f"the owning publisher contract {contract_path}"
            )

    def test_applier_imports_no_topic_constants(self) -> None:
        """Regression guard: the applier module must not import TOPIC_* constants."""
        from pathlib import Path

        applier_src = (
            Path(__file__).resolve().parents[3]
            / "src"
            / "omnibase_infra"
            / "runtime"
            / "service_dispatch_result_applier.py"
        ).read_text(encoding="utf-8")
        assert "from omnibase_infra.event_bus.topic_constants import" not in applier_src
        assert "TOPIC_DELEGATION_" not in applier_src


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
