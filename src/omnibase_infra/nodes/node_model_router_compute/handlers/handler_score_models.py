# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pure scoring handler for model routing.

This is a COMPUTE handler — NO I/O, NO state mutation.
Scores candidate models by: quality * w + (1-cost) * w + speed * w + chain_bonus.

Models without sufficient live metrics (>=20 samples) receive lowest routing
priority. Seed/bootstrap estimates have been removed (OMN-7841) — only measured
eval data drives routing decisions.
"""

from __future__ import annotations

import logging

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_model_health_effect.models.model_endpoint_health import (
    ModelEndpointHealth,
)
from omnibase_infra.nodes.node_model_router_compute.models.model_live_metrics import (
    ModelLiveMetrics,
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
from omnibase_infra.nodes.node_model_router_compute.models.model_scoring_input import (
    ModelScoringInput,
)

logger = logging.getLogger(__name__)

_LIVE_METRICS_THRESHOLD = 20

_MAX_QUEUE_DEPTH = 50

_COLD_START_QUALITY = 0.3

_UNMEASURED_SPEED_SCORE = 0.0

_LOCAL_COST_PER_1K = 0.0

_FRONTIER_COST_PER_1K = 0.015


def _capability_match(model: ModelRegistryEntry, task_type: str) -> bool:
    return task_type in model.capabilities


def _effective_cost_per_1k(model: ModelRegistryEntry) -> float:
    if model.tier == "local":
        return _LOCAL_COST_PER_1K
    return _FRONTIER_COST_PER_1K


def _passes_hard_constraints(
    model: ModelRegistryEntry,
    constraints: ModelRoutingConstraints,
    health_map: dict[str, ModelEndpointHealth],
) -> bool:
    if model.context_window < constraints.min_context_window:
        return False

    cost = _effective_cost_per_1k(model)
    if cost > constraints.max_cost_per_1k:
        return False

    if constraints.needs_vision and "vision" not in model.capabilities:
        return False

    if constraints.needs_computer_use and "computer_use" not in model.capabilities:
        return False

    if constraints.needs_tool_use and "tool_use" not in model.capabilities:
        return False

    health = health_map.get(model.model_key)
    if health is not None and not health.healthy:
        return False

    return True


class ScoringContext:
    __slots__ = (
        "chain_hit_model_key",
        "health_map",
        "metrics_map",
        "task_type",
        "weights",
    )

    def __init__(
        self,
        task_type: str,
        health_map: dict[str, ModelEndpointHealth],
        metrics_map: dict[str, ModelLiveMetrics],
        chain_hit_model_key: str | None,
        weights: dict[str, float],
    ) -> None:
        self.task_type = task_type
        self.health_map = health_map
        self.metrics_map = metrics_map
        self.chain_hit_model_key = chain_hit_model_key
        self.weights = weights


_MAX_COST_NORM = 0.02

_MAX_SPEED_NORM = 250.0


def _compute_score(
    model: ModelRegistryEntry,
    ctx: ScoringContext,
) -> float:
    task_type = ctx.task_type
    health_map = ctx.health_map
    metrics_map = ctx.metrics_map
    chain_hit_model_key = ctx.chain_hit_model_key
    weights = ctx.weights
    live = metrics_map.get(model.model_key)

    if live and live.sample_count >= _LIVE_METRICS_THRESHOLD:
        quality = live.success_rate
        if live.graduated:
            quality = min(1.0, quality + 0.1)
    elif live and live.sample_count > 0:
        ratio = live.sample_count / _LIVE_METRICS_THRESHOLD
        seed_quality = (
            0.5 if _capability_match(model, task_type) else _COLD_START_QUALITY
        )
        quality = ratio * live.success_rate + (1 - ratio) * seed_quality
    else:
        quality = 0.5 if _capability_match(model, task_type) else _COLD_START_QUALITY

    cost = _effective_cost_per_1k(model)
    cost_norm = min(cost / _MAX_COST_NORM, 1.0)
    cost_score = 1.0 - cost_norm

    if live and live.sample_count > 0 and live.avg_tokens_per_sec > 0:
        speed_score = min(live.avg_tokens_per_sec / _MAX_SPEED_NORM, 1.0)
    else:
        speed_score = _UNMEASURED_SPEED_SCORE

    chain_bonus = 1.0 if chain_hit_model_key == model.model_key else 0.0

    health = health_map.get(model.model_key)
    if health is None or health.healthy:
        queue = health.queue_depth if health else 0
        availability = 1.0 if queue < _MAX_QUEUE_DEPTH else 0.0
    else:
        availability = 0.0

    score = (
        weights.get("quality", 0.4) * quality
        + weights.get("cost", 0.3) * cost_score
        + weights.get("speed", 0.2) * speed_score
        + weights.get("chain_bonus", 0.1) * chain_bonus
    ) * availability

    return round(score, 4)


class HandlerScoreModels:
    """Pure scoring handler — no I/O, deterministic for same inputs."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    def score_candidates(
        self, scoring_input: ModelScoringInput
    ) -> ModelRoutingDecision:
        health_map: dict[str, ModelEndpointHealth] = {
            e.model_key: e for e in scoring_input.health
        }
        metrics_map: dict[str, ModelLiveMetrics] = {
            m.model_key: m for m in scoring_input.live_metrics
        }

        task_type_str = scoring_input.task_type.value

        candidates = [
            m
            for m in scoring_input.registry
            if _passes_hard_constraints(m, scoring_input.constraints, health_map)
        ]

        if not candidates:
            return ModelRoutingDecision(
                correlation_id=scoring_input.correlation_id,
                selected_model_key="",
                selected_endpoint_env="",
                rationale="No models pass hard constraints.",
                success=False,
                error_message="No eligible models after constraint filtering.",
            )

        ctx = ScoringContext(
            task_type=task_type_str,
            health_map=health_map,
            metrics_map=metrics_map,
            chain_hit_model_key=(
                scoring_input.chain_hit_model_key if scoring_input.chain_hit else None
            ),
            weights=scoring_input.scoring_weights,
        )
        scores: dict[str, float] = {}
        for model in candidates:
            scores[model.model_key] = _compute_score(model, ctx)

        def sort_key(model: ModelRegistryEntry) -> tuple[float, int]:
            score = scores[model.model_key]
            local_bonus = (
                1
                if (scoring_input.constraints.prefer_local and model.tier == "local")
                else 0
            )
            return (score, local_bonus)

        candidates.sort(key=sort_key, reverse=True)

        selected = candidates[0]
        fallback = candidates[1] if len(candidates) > 1 else None

        parts = [
            f"Selected {selected.model_key} (score={scores[selected.model_key]:.3f})",
            f"task_type={task_type_str}",
            f"tier={selected.tier}",
        ]
        if (
            scoring_input.chain_hit
            and scoring_input.chain_hit_model_key == selected.model_key
        ):
            parts.append("chain_hit_bonus_applied")

        logger.info(
            "Routing decision: %s (correlation_id=%s)",
            selected.model_key,
            scoring_input.correlation_id,
        )

        endpoint_env = selected.base_url_env or selected.api_key_env

        selected_live = metrics_map.get(selected.model_key)
        estimated_cost = _effective_cost_per_1k(selected) * (
            scoring_input.context_length_estimate / 1000
        )
        estimated_speed = (
            selected_live.avg_tokens_per_sec
            if selected_live
            and selected_live.sample_count > 0
            and selected_live.avg_tokens_per_sec > 0
            else 1.0
        )
        estimated_latency_ms = int(
            (scoring_input.context_length_estimate / estimated_speed) * 1000
        )

        return ModelRoutingDecision(
            correlation_id=scoring_input.correlation_id,
            selected_model_key=selected.model_key,
            selected_endpoint_env=endpoint_env,
            fallback_model_key=fallback.model_key if fallback else None,
            rationale="; ".join(parts),
            scores=scores,
            estimated_cost=estimated_cost,
            estimated_latency_ms=estimated_latency_ms,
        )
