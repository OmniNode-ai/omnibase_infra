# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Bind contract topics and database dependencies for the registry cost API."""

from __future__ import annotations

from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Request, status

from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
)
from omnibase_infra.errors import ModelInfraErrorContext
from omnibase_infra.models.model_cost_snapshot_topics import ModelCostSnapshotTopics


class HandlerCostApiDependencies:
    """Install cost routes with their validated contract configuration."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    @staticmethod
    def handle(app: FastAPI, pool: object) -> None:
        """Bind dependencies before exposing the cost API routes."""
        from omnibase_infra.services.cost_api.router import router

        topics = ModelCostSnapshotTopics.from_contract()
        app.state.cost_api_pool = pool
        app.state.cost_snapshot_topics = topics
        app.include_router(router)

    @staticmethod
    def _request_correlation_id(request: Request) -> UUID:
        raw_correlation = request.headers.get("X-Correlation-ID")
        if raw_correlation:
            try:
                return UUID(raw_correlation)
            except ValueError:
                pass
        return uuid4()

    @staticmethod
    def get_pool(request: Request) -> object:
        """Return the asyncpg pool configured by the registry app factory."""
        pool = getattr(request.app.state, "cost_api_pool", None)
        if pool is None:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=HandlerCostApiDependencies._request_correlation_id(
                    request
                ),
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

    @staticmethod
    def get_snapshot_topics(request: Request) -> ModelCostSnapshotTopics:
        """Return the contract-declared snapshot topics bound by the registry app factory."""
        topics: ModelCostSnapshotTopics = request.app.state.cost_snapshot_topics
        return topics
