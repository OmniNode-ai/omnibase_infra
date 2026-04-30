# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration proof for OMN-7841 model-router registry scoring.

This exercises the real ``docker/catalog/model_registry.yaml`` through the
routing API loader and the scoring handler. It guards the production catalog
against reintroducing seed cost/speed fields while proving measured eval
metrics, not registry bootstrap estimates, control the speed-sensitive score.

Integration Test Coverage gate: OMN-7005 (hard gate since 2026-04-13).
"""

from __future__ import annotations

from uuid import uuid4

import pytest
import yaml

from omnibase_infra.nodes.node_model_router_compute.handlers.handler_score_models import (
    HandlerScoreModels,
)
from omnibase_infra.nodes.node_model_router_compute.models.enum_task_type import (
    EnumTaskType,
)
from omnibase_infra.nodes.node_model_router_compute.models.model_live_metrics import (
    ModelLiveMetrics,
)
from omnibase_infra.nodes.node_model_router_compute.models.model_routing_constraints import (
    ModelRoutingConstraints,
)
from omnibase_infra.nodes.node_model_router_compute.models.model_scoring_input import (
    ModelScoringInput,
)
from omnibase_infra.services.routing_api import routes as routing_routes


@pytest.mark.integration
def test_registry_loads_without_seed_fields_and_routes_by_live_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(routing_routes, "_registry_cache", ())

    raw_registry = yaml.safe_load(routing_routes._REGISTRY_PATH.read_text())
    raw_models = raw_registry["models"]
    assert raw_models
    for model in raw_models:
        assert "seed_cost_per_1k_tokens" not in model
        assert "seed_tokens_per_sec" not in model

    registry = routing_routes._load_registry()
    assert registry
    for entry in registry:
        assert not hasattr(entry, "seed_cost_per_1k_tokens")
        assert not hasattr(entry, "seed_tokens_per_sec")

    measured_model = "deepseek-r1-32b"
    scoring_input = ModelScoringInput(
        correlation_id=uuid4(),
        task_type=EnumTaskType.CODE_REVIEW,
        constraints=ModelRoutingConstraints(max_cost_per_1k=0.0),
        context_length_estimate=4096,
        registry=registry,
        live_metrics=(
            ModelLiveMetrics(
                model_key=measured_model,
                task_type=EnumTaskType.CODE_REVIEW,
                success_rate=0.96,
                sample_count=25,
                avg_latency_ms=300,
                avg_tokens_per_sec=180.0,
                graduated=True,
            ),
        ),
    )

    decision = HandlerScoreModels().score_candidates(scoring_input)

    assert decision.success
    assert decision.selected_model_key == measured_model
    assert decision.estimated_cost == 0.0
    unmeasured_scores = {
        model_key: score
        for model_key, score in decision.scores.items()
        if model_key != measured_model
    }
    assert max(unmeasured_scores.values()) <= 0.5
