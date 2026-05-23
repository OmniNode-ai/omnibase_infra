# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""FastAPI routes for LLM cost and savings summaries."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext
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


def _request_correlation_id(request: Request) -> UUID:
    raw_correlation = request.headers.get("X-Correlation-ID")
    if raw_correlation:
        try:
            return UUID(raw_correlation)
        except ValueError:
            pass
    return uuid4()


def get_cost_api_pool(request: Request) -> object:
    """Return the asyncpg pool configured by the registry app factory."""
    pool = getattr(request.app.state, "cost_api_pool", None)
    if pool is None:
        context = ModelInfraErrorContext.with_correlation(
            correlation_id=_request_correlation_id(request),
            transport_type=EnumInfraTransportType.DATABASE,
            operation="cost_api_pool_lookup",
            target_name="registry_api.cost_api_pool",
            suggested_resolution="Configure cost_api_pool on the registry API app state.",
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "message": "Cost API database pool not configured",
                "context": context.model_dump(mode="json"),
            },
        )
    return pool


@router.get(
    "/api/costs/summary",
    response_model=ModelCostSummary,
    summary="LLM Cost Summary",
)
async def get_cost_summary(
    pool: Annotated[object, Depends(get_cost_api_pool)],
    window: Annotated[
        AggregationWindow,
        Query(description="Rolling aggregate window to read."),
    ] = "24h",
) -> ModelCostSummary:
    # Why: Runtime wiring validates and narrows this payload shape before use.
    return await fetch_cost_summary(pool, window=window)  # type: ignore[arg-type]


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
    window: Annotated[
        AggregationWindow,
        Query(description="Rolling aggregate window to read."),
    ] = "24h",
) -> ModelCostBreakdown:
    # Why: Runtime wiring validates and narrows this payload shape before use.
    return await fetch_cost_by_repo(pool, window=window)  # type: ignore[arg-type]


@router.get(
    "/api/costs/token-usage",
    response_model=ModelTokenUsage,
    summary="LLM Token Usage",
)
async def get_token_usage(
    pool: Annotated[object, Depends(get_cost_api_pool)],
    window: Annotated[
        AggregationWindow,
        Query(description="Rolling aggregate window to read."),
    ] = "24h",
) -> ModelTokenUsage:
    # Why: Runtime wiring validates and narrows this payload shape before use.
    return await fetch_token_usage(pool, window=window)  # type: ignore[arg-type]


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
