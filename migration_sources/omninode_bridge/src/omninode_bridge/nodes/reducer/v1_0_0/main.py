"""
REST API wrapper for NodeBridgeReducer.

Provides HTTP endpoints for aggregation queries and state management
in environments where the omnibase runtime is not available.
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import UTC
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from omninode_bridge.nodes.reducer.v1_0_0.node import NodeBridgeReducer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global reducer instance
reducer: Optional[NodeBridgeReducer] = None


# Pydantic models for API
class AggregationQueryRequest(BaseModel):
    """Request model for aggregation queries."""

    namespace: str = Field(default="omninode.bridge", description="Namespace to query")
    aggregation_type: str = Field(default="sum", description="Type of aggregation")
    limit: int = Field(
        default=100, ge=1, le=1000, description="Maximum number of results"
    )


class AggregationQueryResponse(BaseModel):
    """Response model for aggregation queries."""

    success: bool
    total_count: int
    aggregated_data: dict[str, Any]
    namespace: str


class StateSnapshot(BaseModel):
    """State snapshot model."""

    timestamp: str
    namespace: str
    total_items: int
    state_summary: dict[str, Any]


class StateSnapshotResponse(BaseModel):
    """Response model for state snapshot."""

    success: bool
    snapshot: StateSnapshot


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    service: str
    version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    global reducer

    logger.info("Initializing NodeBridgeReducer...")
    # Import stub container for standalone mode
    from ._stubs import ModelONEXContainer

    # Create minimal container for standalone mode
    container = ModelONEXContainer(
        name="reducer_standalone", version="1.0.0", config={"health_check_mode": False}
    )
    reducer = NodeBridgeReducer(container)  # type: ignore[arg-type]

    # Start reducer background tasks if needed
    # await reducer.start()

    yield

    logger.info("Shutting down NodeBridgeReducer...")
    # await reducer.stop()


# Create FastAPI application
app = FastAPI(
    title="NodeBridgeReducer API",
    description="REST API wrapper for ONEX v2.0 metadata aggregation",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy", service="NodeBridgeReducer", version="1.0.0"
    )


@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint.

    Returns:
        Prometheus-formatted metrics for scraping
    """
    if not reducer:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Reducer not initialized",
        )

    # Get metrics from reducer
    metrics_data = reducer.metrics.get_metrics()

    return Response(
        content=metrics_data, media_type="text/plain; version=0.0.4; charset=utf-8"
    )


@app.post("/aggregation/query", response_model=AggregationQueryResponse)
async def query_aggregation(request: AggregationQueryRequest):
    """
    Query aggregated metadata.

    Args:
        request: Aggregation query parameters

    Returns:
        AggregationQueryResponse with aggregated data
    """
    if not reducer:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Reducer not initialized",
        )

    try:
        # Query aggregated data from reducer
        # This is a simplified implementation
        # In production, query from database or state store

        aggregated_data = {
            "total_workflows": 0,
            "successful_workflows": 0,
            "failed_workflows": 0,
            "average_processing_time_ms": 0,
        }

        return AggregationQueryResponse(
            success=True,
            total_count=0,
            aggregated_data=aggregated_data,
            namespace=request.namespace,
        )

    except Exception as e:
        logger.error(f"Error querying aggregation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error querying aggregation: {e!s}",
        )


@app.get("/state/snapshot", response_model=StateSnapshotResponse)
async def get_state_snapshot(namespace: str = "omninode.bridge"):
    """
    Get a snapshot of current aggregated state.

    Args:
        namespace: Namespace to query

    Returns:
        StateSnapshotResponse with current state snapshot
    """
    if not reducer:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Reducer not initialized",
        )

    try:
        from datetime import datetime

        snapshot = StateSnapshot(
            timestamp=datetime.now(UTC).isoformat(),
            namespace=namespace,
            total_items=0,
            state_summary={
                "workflows_processed": 0,
                "metadata_stamps_created": 0,
                "average_latency_ms": 0,
            },
        )

        return StateSnapshotResponse(success=True, snapshot=snapshot)

    except Exception as e:
        logger.error(f"Error getting state snapshot: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting state snapshot: {e!s}",
        )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    # Return basic metrics
    # In production, integrate with prometheus_client
    return {
        "aggregations_total": 0,
        "aggregations_active": 0,
        "aggregations_completed": 0,
        "aggregations_failed": 0,
        "items_aggregated_total": 0,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8061,
        log_level="info",
    )
