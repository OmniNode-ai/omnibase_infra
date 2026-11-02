# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""FastAPI routes for LLM cost and savings summaries."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, Query

from omnibase_infra.handlers.handler_cost_api_dependencies import (
    HandlerCostApiDependencies,
)
from omnibase_infra.models.model_cost_snapshot_topics import ModelCostSnapshotTopics
from omnibase_infra.services.cost_api.handlers import (
    fetch_cost_by_model,
    fetch_cost_by_repo,
    fetch_cost_summary,
    fetch_cost_trend,
    fetch_savings_summary,
    fetch_token_usage,
)
from omnibase_infra.services.cost_api.models import (
    AggregationWindow,
    ModelCostBreakdown,
    ModelCostSummary,
    ModelCostTrend,
    ModelSavingsSummary,
    ModelTokenUsage,
    TrendBucket,
)

router = APIRouter(tags=["costs"])
get_cost_api_pool = HandlerCostApiDependencies.get_pool
get_cost_snapshot_topics = HandlerCostApiDependencies.get_snapshot_topics
SnapshotTopics = Annotated[ModelCostSnapshotTopics, Depends(get_cost_snapshot_topics)]


def install_cost_routes(app: FastAPI, pool: object) -> None:
    """Bind cost dependencies before installing their routes on the registry app."""
    HandlerCostApiDependencies.handle(app, pool)
    app.include_router(router)


@router.get(
    "/api/costs/summary",
    response_model=ModelCostSummary,
    summary="LLM Cost Summary",
)
async def get_cost_summary(
    pool: Annotated[object, Depends(get_cost_api_pool)],
    snapshot_topics: SnapshotTopics,
    window: Annotated[
        AggregationWindow,
        Query(description="Rolling aggregate window to read."),
    ] = "24h",
) -> ModelCostSummary:
    # Why: Runtime wiring validates and narrows this payload shape before use.
    return await fetch_cost_summary(  # type: ignore[arg-type]
        pool, window=window, snapshot_topics=snapshot_topics
    )


@router.get(
    "/api/costs/trend",
    response_model=ModelCostTrend,
    summary="LLM Cost Trend",
)
async def get_cost_trend(
    pool: Annotated[object, Depends(get_cost_api_pool)],
    bucket: Annotated[
        TrendBucket,
        Query(description="Event-time bucket size for raw call metrics."),
    ] = "day",
    days: Annotated[
        int,
        Query(ge=1, le=365, description="Number of trailing days to include."),
    ] = 30,
) -> ModelCostTrend:
    # Why: Runtime wiring validates and narrows this payload shape before use.
    return await fetch_cost_trend(pool, bucket=bucket, days=days)  # type: ignore[arg-type]


@router.get(
    "/api/costs/by-model",
    response_model=ModelCostBreakdown,
    summary="LLM Cost By Model",
)
async def get_cost_by_model(
    pool: Annotated[object, Depends(get_cost_api_pool)],
    window: Annotated[
        AggregationWindow,
        Query(description="Rolling aggregate window to read."),
    ] = "24h",
) -> ModelCostBreakdown:
    # Why: Runtime wiring validates and narrows this payload shape before use.
    return await fetch_cost_by_model(pool, window=window)  # type: ignore[arg-type]


@router.get(
    "/api/costs/by-repo",
    response_model=ModelCostBreakdown,
    summary="LLM Cost By Repo",
)
async def get_cost_by_repo(
    pool: Annotated[object, Depends(get_cost_api_pool)],
    snapshot_topics: SnapshotTopics,
    window: Annotated[
        AggregationWindow,
        Query(description="Rolling aggregate window to read."),
    ] = "24h",
) -> ModelCostBreakdown:
    # Why: Runtime wiring validates and narrows this payload shape before use.
    return await fetch_cost_by_repo(  # type: ignore[arg-type]
        pool, window=window, snapshot_topics=snapshot_topics
    )


@router.get(
    "/api/costs/token-usage",
    response_model=ModelTokenUsage,
    summary="LLM Token Usage",
)
async def get_token_usage(
    pool: Annotated[object, Depends(get_cost_api_pool)],
    snapshot_topics: SnapshotTopics,
    window: Annotated[
        AggregationWindow,
        Query(description="Rolling aggregate window to read."),
    ] = "24h",
) -> ModelTokenUsage:
    # Why: Runtime wiring validates and narrows this payload shape before use.
    return await fetch_token_usage(  # type: ignore[arg-type]
        pool, window=window, snapshot_topics=snapshot_topics
    )


@router.get(
    "/api/savings/summary",
    response_model=ModelSavingsSummary,
    summary="Savings Summary",
)
async def get_savings_summary(
    pool: Annotated[object, Depends(get_cost_api_pool)],
    window: Annotated[
        AggregationWindow,
        Query(description="Trailing savings window to read."),
    ] = "24h",
) -> ModelSavingsSummary:
    # Why: Runtime wiring validates and narrows this payload shape before use.
    return await fetch_savings_summary(pool, window=window)  # type: ignore[arg-type]
