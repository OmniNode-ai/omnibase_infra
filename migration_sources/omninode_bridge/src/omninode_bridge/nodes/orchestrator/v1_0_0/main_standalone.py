"""
Standalone REST API for NodeBridgeOrchestrator - Simplified version without omnibase runtime.

This is a minimal implementation for demo/bridge environments.
"""

import logging
import os
from typing import Any, Optional

from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Try to import the real node, fall back to None if omnibase_core is not available
# Tests can mock this import regardless
try:
    from .node import NodeBridgeOrchestrator
except ImportError:
    # If omnibase_core is not available, set to None
    # This allows tests to mock it and the startup event will handle the error gracefully
    NodeBridgeOrchestrator = None  # type: ignore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global node instance (will be initialized on startup)
node_instance: Optional[NodeBridgeOrchestrator] = None


# Configuration model for standalone mode
class StandaloneConfig(BaseModel):
    """Configuration for standalone orchestrator mode."""

    metadata_stamping_service_url: str
    onextree_service_url: str
    kafka_broker_url: str
    default_namespace: str
    api_port: int
    metrics_port: int
    environment: str


# Pydantic models for API
class WorkflowSubmissionRequest(BaseModel):
    """Request model for workflow submission."""

    content: str = Field(..., description="Content to be stamped")
    correlation_id: Optional[str] = Field(None, description="Optional correlation ID")
    namespace: str = Field(
        default="omninode.bridge", description="Namespace for operation"
    )


class WorkflowSubmissionResponse(BaseModel):
    """Response model for workflow submission."""

    success: bool
    workflow_id: str
    state: str
    message: Optional[str] = None


class WorkflowStatusResponse(BaseModel):
    """Response model for workflow status."""

    workflow_id: str
    state: str
    current_step: Optional[str] = None
    result: Optional[dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    service: str
    version: str
    mode: str


# Create FastAPI application
app = FastAPI(
    title="NodeBridgeOrchestrator API (Standalone)",
    description="Standalone REST API for ONEX v2.0 workflow orchestration",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize node and start introspection on application startup."""
    global node_instance

    try:
        # Create standalone configuration from environment
        config = StandaloneConfig(
            metadata_stamping_service_url=os.getenv(
                "METADATA_STAMPING_SERVICE_URL", "http://metadata-stamping:8053"
            ),
            onextree_service_url=os.getenv(
                "ONEXTREE_SERVICE_URL", "http://onextree:8080"
            ),
            kafka_broker_url=os.getenv(
                "KAFKA_BOOTSTRAP_SERVERS", "omninode-bridge-redpanda:9092"
            ),
            default_namespace=os.getenv("DEFAULT_NAMESPACE", "omninode.bridge"),
            api_port=int(os.getenv("API_PORT", "8060")),
            metrics_port=int(os.getenv("METRICS_PORT", "9090")),
            environment=os.getenv("ENVIRONMENT", "development"),
        )

        logger.info(f"Standalone configuration loaded: {config.model_dump()}")

        # Note: In standalone mode, we run in degraded mode without full node initialization
        # The node requires omnibase runtime which is not available in standalone deployment
        # This allows the API to run for basic health checks and status endpoints

        logger.info("NodeBridgeOrchestrator API running in standalone mode")
        logger.info("Full workflow orchestration requires omnibase runtime")

    except Exception as e:
        logger.error(f"Failed to initialize standalone mode: {e}", exc_info=True)
        # Continue running in degraded mode
        node_instance = None


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown node and stop introspection tasks."""
    global node_instance

    if node_instance:
        try:
            await node_instance.shutdown()
            logger.info("NodeBridgeOrchestrator shutdown complete")
        except Exception as e:
            logger.error(f"Error during node shutdown: {e}", exc_info=True)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        service="NodeBridgeOrchestrator",
        version="1.0.0",
        mode="standalone",
    )


@app.post(
    "/workflow/submit",
    response_model=WorkflowSubmissionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def submit_workflow(request: WorkflowSubmissionRequest):
    """
    Submit a new workflow for metadata stamping (Standalone mode - placeholder).

    Args:
        request: Workflow submission request

    Returns:
        WorkflowSubmissionResponse
    """
    import uuid

    workflow_id = str(uuid.uuid4())

    logger.info(f"Workflow submitted (standalone mode): {workflow_id}")
    logger.info(f"Content: {request.content[:100]}...")
    logger.info(f"Namespace: {request.namespace}")

    return WorkflowSubmissionResponse(
        success=True,
        workflow_id=workflow_id,
        state="queued",
        message="Workflow queued in standalone mode (full implementation requires omnibase runtime)",
    )


@app.get("/workflow/{workflow_id}/status", response_model=WorkflowStatusResponse)
async def get_workflow_status(workflow_id: str):
    """
    Get workflow status (Standalone mode - placeholder).

    Args:
        workflow_id: UUID of the workflow

    Returns:
        WorkflowStatusResponse
    """
    return WorkflowStatusResponse(
        workflow_id=workflow_id,
        state="queued",
        current_step="pending",
        result={
            "note": "Standalone mode - full implementation requires omnibase runtime"
        },
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return {
        "workflows_total": 0,
        "workflows_active": 0,
        "workflows_completed": 0,
        "workflows_failed": 0,
        "mode": "standalone",
    }


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "NodeBridgeOrchestrator",
        "version": "1.0.0",
        "mode": "standalone",
        "status": "running",
        "message": "This is a standalone REST API wrapper. Full workflow orchestration requires omnibase runtime.",
        "endpoints": {
            "health": "/health",
            "submit_workflow": "/workflow/submit",
            "workflow_status": "/workflow/{workflow_id}/status",
            "metrics": "/metrics",
            "docs": "/docs",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8060,
        log_level="info",
    )
