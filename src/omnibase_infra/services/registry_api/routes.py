# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registry API Routes.

FastAPI route handlers for the Registry API. Routes are defined as an
APIRouter for easy mounting into the main FastAPI application.

Endpoint Summary:
    GET /registry/discovery     - Full dashboard payload
    GET /registry/nodes         - Node list with pagination
    GET /registry/nodes/{id}    - Single node detail
    GET /registry/instances     - Live Consul instances
    GET /registry/widgets/mapping - Widget mapping configuration
    GET /registry/health        - Service health check

Related Tickets:
    - OMN-1278: Contract-Driven Dashboard - Registry Discovery
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status

from omnibase_infra.enums import EnumRegistrationState
from omnibase_infra.services.registry_api.models import (
    ModelRegistryDiscoveryResponse,
    ModelRegistryHealthResponse,
    ModelRegistryNodeView,
    ModelResponseListInstances,
    ModelResponseListNodes,
    ModelWidgetMapping,
)

if TYPE_CHECKING:
    from omnibase_infra.services.registry_api.service import ServiceRegistryDiscovery


def get_correlation_id(
    x_correlation_id: Annotated[
        str | None,
        Header(
            alias="X-Correlation-ID",
            description="Correlation ID for distributed tracing. Must be a valid UUID if provided.",
        ),
    ] = None,
) -> UUID:
    """FastAPI dependency to extract and validate correlation ID from HTTP header.

    Extracts the X-Correlation-ID header value and validates it as a UUID.
    If no header is provided, generates a new UUID for the request.

    Args:
        x_correlation_id: Optional correlation ID string from X-Correlation-ID header.

    Returns:
        Parsed UUID from header or newly generated UUID if not provided.

    Raises:
        HTTPException: 400 Bad Request if correlation ID is provided but not a valid UUID.

    Example:
        Valid header: X-Correlation-ID: 550e8400-e29b-41d4-a716-446655440000
        Invalid header: X-Correlation-ID: not-a-uuid (returns 400)
        Missing header: Generates new UUID automatically
    """
    if x_correlation_id is None:
        return uuid4()
    try:
        return UUID(x_correlation_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid X-Correlation-ID header format: '{x_correlation_id}'. Must be a valid UUID (e.g., '550e8400-e29b-41d4-a716-446655440000').",
        ) from None


# Create router with prefix
router = APIRouter(
    prefix="/registry",
    tags=["registry"],
    responses={
        400: {"description": "Bad request (e.g., invalid correlation ID format)"},
        500: {"description": "Internal server error"},
        503: {"description": "Service unavailable"},
    },
)


def get_service(request: Request) -> ServiceRegistryDiscovery:
    """Dependency to get the registry discovery service from app state.

    Args:
        request: FastAPI request object.

    Returns:
        ServiceRegistryDiscovery instance from app state.

    Raises:
        HTTPException: If service is not configured in app state.
    """
    service: ServiceRegistryDiscovery | None = getattr(
        request.app.state, "registry_service", None
    )
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Registry service not configured",
        )
    return service


@router.get(
    "/discovery",
    response_model=ModelRegistryDiscoveryResponse,
    summary="Full Dashboard Payload",
    description=(
        "Returns the complete dashboard payload including nodes, live instances, "
        "and summary statistics. This is the primary endpoint for dashboard "
        "consumption, providing all needed data in a single request."
    ),
    responses={
        400: {"description": "Bad request (e.g., invalid correlation ID format)"},
        200: {
            "description": "Successful response with full discovery data",
            "content": {
                "application/json": {
                    "example": {
                        "timestamp": "2025-01-21T10:00:00Z",
                        "warnings": [],
                        "summary": {
                            "total_nodes": 10,
                            "active_nodes": 8,
                            "healthy_instances": 5,
                            "unhealthy_instances": 2,
                            "by_node_type": {"EFFECT": 5, "COMPUTE": 3, "REDUCER": 2},
                            "by_state": {"active": 8, "pending_registration": 2},
                        },
                        "nodes": [],
                        "live_instances": [],
                        "pagination": {
                            "total": 10,
                            "limit": 100,
                            "offset": 0,
                            "has_more": False,
                        },
                    }
                }
            },
        },
    },
)
async def get_discovery(
    service: Annotated[ServiceRegistryDiscovery, Depends(get_service)],
    correlation_id: Annotated[UUID, Depends(get_correlation_id)],
    limit: Annotated[
        int,
        Query(ge=1, le=1000, description="Maximum number of nodes to return"),
    ] = 100,
    offset: Annotated[
        int,
        Query(ge=0, description="Number of nodes to skip for pagination"),
    ] = 0,
) -> ModelRegistryDiscoveryResponse:
    """Get full dashboard payload with nodes, instances, and summary."""
    response = await service.get_discovery(
        limit=limit,
        offset=offset,
        correlation_id=correlation_id,
    )

    return response


@router.get(
    "/nodes",
    response_model=ModelResponseListNodes,
    summary="List Registered Nodes",
    description=(
        "Returns a paginated list of registered nodes from the PostgreSQL "
        "projection store. Supports filtering by state and node type."
    ),
    responses={
        400: {"description": "Bad request (e.g., invalid correlation ID format)"},
        200: {
            "description": "Successful response with node list",
        },
    },
)
async def list_nodes(
    service: Annotated[ServiceRegistryDiscovery, Depends(get_service)],
    correlation_id: Annotated[UUID, Depends(get_correlation_id)],
    limit: Annotated[
        int,
        Query(ge=1, le=1000, description="Maximum number of nodes to return"),
    ] = 100,
    offset: Annotated[
        int,
        Query(ge=0, description="Number of nodes to skip for pagination"),
    ] = 0,
    state: Annotated[
        str | None,
        Query(
            description="Filter by registration state (e.g., 'active', 'pending_registration')"
        ),
    ] = None,
    node_type: Annotated[
        str | None,
        Query(
            description="Filter by node type (effect, compute, reducer, orchestrator)"
        ),
    ] = None,
) -> ModelResponseListNodes:
    """List registered nodes with pagination and optional filtering."""
    # Parse state filter
    state_filter: EnumRegistrationState | None = None
    if state is not None:
        try:
            state_filter = EnumRegistrationState(state)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid state value: {state}. Valid values: {[s.value for s in EnumRegistrationState]}",
            ) from None

    nodes, pagination, warnings = await service.list_nodes(
        limit=limit,
        offset=offset,
        state=state_filter,
        node_type=node_type,
        correlation_id=correlation_id,
    )

    return ModelResponseListNodes(
        nodes=nodes,
        pagination=pagination,
        warnings=warnings,
    )


@router.get(
    "/nodes/{node_id}",
    response_model=ModelRegistryNodeView,
    summary="Get Node Details",
    description="Returns detailed information for a single registered node by ID.",
    responses={
        400: {"description": "Bad request (e.g., invalid correlation ID format)"},
        200: {"description": "Successful response with node details"},
        404: {"description": "Node not found"},
    },
)
async def get_node(
    node_id: UUID,
    service: Annotated[ServiceRegistryDiscovery, Depends(get_service)],
    correlation_id: Annotated[UUID, Depends(get_correlation_id)],
) -> ModelRegistryNodeView:
    """Get a single node by ID."""

    node, warnings = await service.get_node(
        node_id=node_id,
        correlation_id=correlation_id,
    )

    if node is None:
        # Check if it was a service error or genuinely not found
        if warnings:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Service error: {warnings[0].message}",
            )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Node not found: {node_id}",
        )

    return node


@router.get(
    "/instances",
    response_model=ModelResponseListInstances,
    summary="List Live Consul Instances",
    description=(
        "Returns a list of live service instances from Consul. "
        "Includes health status and metadata for each instance."
    ),
    responses={
        400: {"description": "Bad request (e.g., invalid correlation ID format)"},
        200: {"description": "Successful response with instance list"},
    },
)
async def list_instances(
    service: Annotated[ServiceRegistryDiscovery, Depends(get_service)],
    correlation_id: Annotated[UUID, Depends(get_correlation_id)],
    service_name: Annotated[
        str | None,
        Query(description="Filter by service name"),
    ] = None,
    include_unhealthy: Annotated[
        bool,
        Query(description="Include unhealthy instances in results"),
    ] = False,
) -> ModelResponseListInstances:
    """List live Consul service instances."""
    instances, warnings = await service.list_instances(
        service_name=service_name,
        include_unhealthy=include_unhealthy,
        correlation_id=correlation_id,
    )

    return ModelResponseListInstances(
        instances=instances,
        warnings=warnings,
    )


@router.get(
    "/widgets/mapping",
    response_model=ModelWidgetMapping,
    summary="Widget Mapping Configuration",
    description=(
        "Returns the capability-to-widget mapping configuration. "
        "Used by dashboards to determine which widget type to render "
        "for each node capability."
    ),
    responses={
        200: {"description": "Successful response with widget mapping"},
        503: {"description": "Configuration not available"},
    },
)
async def get_widget_mapping(
    service: Annotated[ServiceRegistryDiscovery, Depends(get_service)],
) -> ModelWidgetMapping:
    """Get widget mapping configuration."""
    mapping, warnings = service.get_widget_mapping()

    if mapping is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Widget mapping not available: {warnings[0].message if warnings else 'Unknown error'}",
        )

    return mapping


@router.get(
    "/health",
    response_model=ModelRegistryHealthResponse,
    summary="Service Health Check",
    description=(
        "Performs a health check on all backend components (PostgreSQL, Consul, config) "
        "and returns the overall service health status."
    ),
    responses={
        400: {"description": "Bad request (e.g., invalid correlation ID format)"},
        200: {"description": "Health check response (may indicate degraded/unhealthy)"},
    },
)
async def health_check(
    service: Annotated[ServiceRegistryDiscovery, Depends(get_service)],
    correlation_id: Annotated[UUID, Depends(get_correlation_id)],
) -> ModelRegistryHealthResponse:
    """Perform health check on all backend components."""
    response = await service.health_check(correlation_id=correlation_id)

    return response


__all__ = ["router", "get_service", "get_correlation_id"]
