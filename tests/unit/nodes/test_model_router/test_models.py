# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for model router Pydantic models — frozen, extra=forbid."""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.nodes.node_model_health_effect.models.model_endpoint_health import (
    ModelEndpointHealth,
)
from omnibase_infra.nodes.node_model_health_effect.models.model_health_probe_target import (
    ModelHealthProbeTarget,
)
from omnibase_infra.nodes.node_model_health_effect.models.model_health_request import (
    ModelHealthRequest,
)
from omnibase_infra.nodes.node_model_health_effect.models.model_health_snapshot import (
    ModelHealthSnapshot,
)
from omnibase_infra.nodes.node_model_router_compute.models.enum_task_type import (
    EnumTaskType,
)
from omnibase_infra.nodes.node_model_router_compute.models.model_registry_entry import (
    ModelRegistryEntry,
)
from omnibase_infra.nodes.node_model_router_compute.models.model_routing_constraints import (
    ModelRoutingConstraints,
)
from omnibase_infra.nodes.node_model_router_compute.models.model_routing_decision import (
    ModelRoutingDecision,
)
from omnibase_infra.nodes.node_model_router_compute.models.model_routing_request import (
    ModelRoutingRequest,
)
from omnibase_infra.nodes.node_routing_orchestrator.models.model_routing_command import (
    ModelRoutingCommand,
)
from omnibase_infra.nodes.node_routing_orchestrator.models.model_routing_result import (
    ModelRoutingResult,
)
from omnibase_infra.nodes.node_routing_score_reducer.models.model_routing_outcome import (
    ModelRoutingOutcome,
)


@pytest.mark.unit
class TestModelsFrozenAndForbid:
    """All models should be frozen (immutable) and forbid extra fields."""

    def test_routing_request_frozen(self) -> None:
        req = ModelRoutingRequest(
            correlation_id=uuid4(),
            task_description="test",
            task_type=EnumTaskType.CODE_GENERATION,
        )
        with pytest.raises(ValidationError):
            req.task_description = "mutated"  # type: ignore[misc]

    def test_routing_request_forbids_extra(self) -> None:
        with pytest.raises(ValidationError):
            ModelRoutingRequest(
                correlation_id=uuid4(),
                task_description="test",
                task_type=EnumTaskType.CODE_GENERATION,
                bogus_field="nope",  # type: ignore[call-arg]
            )

    def test_routing_constraints_defaults(self) -> None:
        c = ModelRoutingConstraints()
        assert c.max_cost_per_1k == 0.10
        assert c.prefer_local is True
        assert c.needs_vision is False

    def test_health_probe_target(self) -> None:
        t = ModelHealthProbeTarget(model_key="test", base_url="http://localhost:8000")
        assert t.transport == "http"

    def test_endpoint_health(self) -> None:
        h = ModelEndpointHealth(model_key="test", healthy=True)
        assert h.latency_ms == 0
        assert h.error_message == ""

    def test_health_snapshot(self) -> None:
        s = ModelHealthSnapshot(correlation_id=uuid4())
        assert s.success is True
        assert len(s.endpoints) == 0

    def test_registry_entry(self) -> None:
        e = ModelRegistryEntry(
            model_key="test",
            provider="local",
            transport="http",
            capabilities=("code_generation",),
        )
        assert e.tier == "local"

    def test_routing_decision(self) -> None:
        d = ModelRoutingDecision(
            correlation_id=uuid4(),
            selected_model_key="test",
            selected_endpoint_env="LLM_TEST",
            rationale="test",
        )
        assert d.success is True

    def test_routing_command(self) -> None:
        c = ModelRoutingCommand(
            correlation_id=uuid4(),
            task_description="test",
            task_type=EnumTaskType.CODE_GENERATION,
        )
        assert c.chain_hit is False

    def test_routing_result(self) -> None:
        r = ModelRoutingResult(
            correlation_id=uuid4(),
            selected_model_key="test",
            selected_endpoint_env="LLM_TEST",
            rationale="test",
        )
        assert r.success is True

    def test_routing_outcome(self) -> None:
        o = ModelRoutingOutcome(
            correlation_id=uuid4(),
            model_key="test",
            task_type=EnumTaskType.CODE_GENERATION,
            success=True,
        )
        assert o.actual_cost == 0.0

    def test_enum_task_type_values(self) -> None:
        assert EnumTaskType.CODE_GENERATION.value == "code_generation"
        assert EnumTaskType.VISION.value == "vision"
        assert EnumTaskType.DEEP_REASONING.value == "deep_reasoning"
