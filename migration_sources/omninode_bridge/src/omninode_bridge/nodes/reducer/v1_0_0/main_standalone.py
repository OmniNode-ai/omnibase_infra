"""
Standalone REST API for NodeBridgeReducer - Simplified version without omnibase runtime.

This is a minimal implementation for demo/bridge environments.
"""

import logging
import os
from datetime import UTC
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Direct import - omnibase_core is required
from omnibase_core.models.container import ModelONEXContainer
from pydantic import BaseModel, Field

# Import node and container
from .node import NodeBridgeReducer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global node instance (will be initialized on startup)
node_instance: Optional[NodeBridgeReducer] = None


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
    mode: str


# Create FastAPI application
app = FastAPI(
    title="NodeBridgeReducer API (Standalone)",
    description="Standalone REST API for ONEX v2.0 metadata aggregation",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize node and start introspection on application startup."""
    global node_instance

    try:
        # Create ONEX DI container with configuration
        # Read from environment variables with fallback defaults
        container = ModelONEXContainer(
            config={
                "postgres": {
                    "host": os.getenv("POSTGRES_HOST", "localhost"),
                    "port": int(os.getenv("POSTGRES_PORT", "5432")),
                    "database": os.getenv("POSTGRES_DATABASE", "omninode_bridge"),
                },
                "kafka_broker_url": os.getenv(
                    "KAFKA_BOOTSTRAP_SERVERS", "omninode-bridge-redpanda:9092"
                ),
                "default_namespace": os.getenv("DEFAULT_NAMESPACE", "omninode.bridge"),
                "api_port": int(os.getenv("API_PORT", "8061")),
                "metrics_port": int(os.getenv("METRICS_PORT", "9091")),
                "environment": os.getenv("ENVIRONMENT", "development"),
                # Consul configuration
                "consul_host": os.getenv("CONSUL_HOST", "localhost"),
                "consul_port": int(os.getenv("CONSUL_PORT", "8500")),
                "consul_enable_registration": os.getenv(
                    "CONSUL_ENABLE_REGISTRATION", "true"
                ).lower()
                == "true",
            }
        )

        # Initialize container (creates and connects KafkaClient)
        # Check if container has initialize method (stub containers have it, omnibase_core may not)
        if hasattr(container, "initialize") and callable(container.initialize):
            await container.initialize()
            logger.info("Container initialized with KafkaClient")
        else:
            logger.info("Container does not have initialize method, skipping")

        # Initialize node
        node_instance = NodeBridgeReducer(container)

        # Start node (publishes introspection and starts background tasks)
        await node_instance.startup()

        logger.info("NodeBridgeReducer initialized and ready")

    except Exception as e:
        logger.error(f"Failed to initialize node: {e}", exc_info=True)
        # Continue running in degraded mode
        node_instance = None


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown node and stop introspection tasks."""
    global node_instance

    if node_instance:
        try:
            await node_instance.shutdown()
            logger.info("NodeBridgeReducer shutdown complete")
        except Exception as e:
            logger.error(f"Error during node shutdown: {e}", exc_info=True)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        service="NodeBridgeReducer",
        version="1.0.0",
        mode="standalone",
    )


@app.post("/aggregation/query", response_model=AggregationQueryResponse)
async def query_aggregation(request: AggregationQueryRequest):
    """
    Query aggregated metadata (Standalone mode - placeholder).

    Args:
        request: Aggregation query parameters

    Returns:
        AggregationQueryResponse
    """
    logger.info(
        f"Aggregation query (standalone mode): namespace={request.namespace}, type={request.aggregation_type}"
    )

    aggregated_data = {
        "total_workflows": 0,
        "successful_workflows": 0,
        "failed_workflows": 0,
        "average_processing_time_ms": 0,
        "note": "Standalone mode - full implementation requires omnibase runtime",
    }

    return AggregationQueryResponse(
        success=True,
        total_count=0,
        aggregated_data=aggregated_data,
        namespace=request.namespace,
    )


@app.get("/state/snapshot", response_model=StateSnapshotResponse)
async def get_state_snapshot(namespace: str = "omninode.bridge"):
    """
    Get state snapshot (Standalone mode - placeholder).

    Args:
        namespace: Namespace to query

    Returns:
        StateSnapshotResponse
    """
    from datetime import datetime

    snapshot = StateSnapshot(
        timestamp=datetime.now(UTC).isoformat(),
        namespace=namespace,
        total_items=0,
        state_summary={
            "workflows_processed": 0,
            "metadata_stamps_created": 0,
            "average_latency_ms": 0,
            "note": "Standalone mode",
        },
    )

    return StateSnapshotResponse(success=True, snapshot=snapshot)


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return {
        "aggregations_total": 0,
        "aggregations_active": 0,
        "aggregations_completed": 0,
        "aggregations_failed": 0,
        "items_aggregated_total": 0,
        "mode": "standalone",
    }


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "NodeBridgeReducer",
        "version": "1.0.0",
        "mode": "standalone",
        "status": "running",
        "message": "This is a standalone REST API wrapper. Full aggregation requires omnibase runtime.",
        "endpoints": {
            "health": "/health",
            "query_aggregation": "/aggregation/query",
            "state_snapshot": "/state/snapshot",
            "metrics": "/metrics",
            "docs": "/docs",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8061,
        log_level="info",
    )
