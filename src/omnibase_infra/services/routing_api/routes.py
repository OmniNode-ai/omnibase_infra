# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Routing API Routes.

FastAPI route handlers for the Intelligent Model Router.

Endpoint Summary:
    POST /routing/route              - Request a routing decision
    GET  /routing/health             - Service health check
    GET  /routing/models             - List registered models
    GET  /routing/scores             - Current capability scores
    GET  /routing/outcomes           - Recent routing outcomes

Related Tickets:
    - OMN-7278: Routing API endpoint
    - OMN-7264: Intelligent Model Router MVP
"""

from __future__ import annotations

import logging
from pathlib import Path
from uuid import uuid4

import yaml
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_model_router_compute.handlers.handler_score_models import (
    HandlerScoreModels,
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
from omnibase_infra.nodes.node_model_router_compute.models.model_scoring_input import (
    ModelScoringInput,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/routing", tags=["routing"])

# Registry cache — loaded once at startup
_registry_cache: tuple[ModelRegistryEntry, ...] = ()

# Path to model registry relative to repo root
_REGISTRY_PATH = (
    Path(__file__).parents[4] / "docker" / "catalog" / "model_registry.yaml"
)


class ModelRouteRequestBody(BaseModel):
    """REST request body for /routing/route."""

    model_config = ConfigDict(extra="forbid")

    task_description: str = Field(..., description="Human-readable task description.")
    task_type: EnumTaskType = Field(..., description="Classified task type.")
    constraints: ModelRoutingConstraints = Field(
        default_factory=ModelRoutingConstraints,
    )
    context_length_estimate: int = Field(default=4096)
    chain_hit: bool = Field(default=False)
    chain_hit_model_key: str | None = Field(default=None)


def _load_registry() -> tuple[ModelRegistryEntry, ...]:
    """Load model registry from YAML."""
    global _registry_cache  # noqa: PLW0603
    if _registry_cache:
        return _registry_cache

    if not _REGISTRY_PATH.exists():
        logger.warning("Model registry not found at %s", _REGISTRY_PATH)
        return ()

    with open(_REGISTRY_PATH) as f:
        data = yaml.safe_load(f)

    entries: list[ModelRegistryEntry] = []
    for m in data.get("models", []):
        entries.append(
            ModelRegistryEntry(
                model_key=m["id"],
                provider=m.get("provider", "local"),
                transport=m.get("transport", "http"),
                base_url_env=m.get("base_url_env", ""),
                api_key_env=m.get("api_key_env", ""),
                capabilities=tuple(m.get("capabilities", [])),
                context_window=m.get("context_window", 4096),
                seed_cost_per_1k_tokens=m.get("seed_cost_per_1k_tokens", 0.0),
                seed_tokens_per_sec=m.get("seed_tokens_per_sec", 0.0),
                tier=m.get("tier", "local"),
            )
        )

    _registry_cache = tuple(entries)
    logger.info("Loaded %d models from registry", len(_registry_cache))
    return _registry_cache


@router.get("/health")
async def health() -> dict[str, str]:
    """Service health check."""
    return {"status": "ok", "service": "routing-api"}


@router.get("/models")
async def list_models() -> dict[str, list[dict]]:
    """List all registered models."""
    registry = _load_registry()
    return {
        "models": [
            {
                "model_key": m.model_key,
                "provider": m.provider,
                "tier": m.tier,
                "capabilities": list(m.capabilities),
                "context_window": m.context_window,
                "seed_cost_per_1k_tokens": m.seed_cost_per_1k_tokens,
                "seed_tokens_per_sec": m.seed_tokens_per_sec,
            }
            for m in registry
        ]
    }


@router.post("/route", response_model=ModelRoutingDecision)
async def route_request(body: ModelRouteRequestBody) -> ModelRoutingDecision:
    """Request an intelligent routing decision.

    Synchronous scoring — probes are skipped for MVP; health data
    is assumed healthy for all models. Live metrics are empty until
    the reducer accumulates data.
    """
    registry = _load_registry()
    if not registry:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model registry not loaded.",
        )

    correlation_id = uuid4()

    # Build scoring input — MVP: no live health or metrics
    scoring_input = ModelScoringInput(
        correlation_id=correlation_id,
        task_type=body.task_type,
        task_description=body.task_description,
        constraints=body.constraints,
        context_length_estimate=body.context_length_estimate,
        chain_hit=body.chain_hit,
        chain_hit_model_key=body.chain_hit_model_key,
        registry=registry,
        health=(),
        live_metrics=(),
    )

    handler = HandlerScoreModels()
    decision = handler.score_candidates(scoring_input)

    if not decision.success:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=decision.error_message,
        )

    return decision
