"""
REST API wrapper for NodeBridgeOrchestrator.

Provides HTTP endpoints for workflow submission and status queries
in environments where the omnibase runtime is not available.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from omnibase_core.models.container import ModelONEXContainer
from pydantic import BaseModel, Field

from omninode_bridge.nodes.orchestrator.v1_0_0.models import ModelStampRequestInput
from omninode_bridge.nodes.orchestrator.v1_0_0.node import NodeBridgeOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global orchestrator instance
orchestrator: Optional[NodeBridgeOrchestrator] = None


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    global orchestrator

    try:
        logger.info("Initializing NodeBridgeOrchestrator...")

        # Validate required environment variables for MVP deployment
        required_env_vars = {
            "POSTGRES_PASSWORD": "Database password (security requirement)",  # pragma: allowlist secret
            "KAFKA_BOOTSTRAP_SERVERS": "Kafka broker connection (event sourcing)",
        }

        missing_vars = []
        for var, description in required_env_vars.items():
            if not os.getenv(var):
                missing_vars.append(f"  - {var}: {description}")

        if missing_vars:
            error_msg = "Missing required environment variables:\n" + "\n".join(
                missing_vars
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("Environment validation passed - all required variables present")

        # Create ONEX DI container with configuration
        container = ModelONEXContainer(
            config={
                "metadata_stamping_service_url": os.getenv(
                    "METADATA_STAMPING_SERVICE_URL", "http://metadata-stamping:8053"
                ),
                "onextree_service_url": os.getenv(
                    "ONEXTREE_SERVICE_URL", "http://onextree:8080"
                ),
                "kafka_broker_url": os.getenv(
                    "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
                ),
                "default_namespace": os.getenv("DEFAULT_NAMESPACE", "omninode.bridge"),
                "health_check_mode": False,
                # Consul configuration
                "consul_host": os.getenv("CONSUL_HOST", "localhost"),
                "consul_port": int(os.getenv("CONSUL_PORT", "8500")),
                "consul_enable_registration": os.getenv(
                    "CONSUL_ENABLE_REGISTRATION", "true"
                ).lower()
                == "true",
            }
        )

        # Initialize container (creates and connects KafkaClient if available)
        if hasattr(container, "initialize") and callable(container.initialize):
            await container.initialize()
            logger.info("Container initialized")

        # Initialize orchestrator with container
        orchestrator = NodeBridgeOrchestrator(container)

        # Start orchestrator background tasks if available
        if hasattr(orchestrator, "startup") and callable(orchestrator.startup):
            await orchestrator.startup()
            logger.info("NodeBridgeOrchestrator startup complete")

    except Exception as e:
        logger.error(f"Failed to initialize orchestrator: {e}", exc_info=True)
        # Continue in degraded mode
        orchestrator = None

    yield

    logger.info("Shutting down NodeBridgeOrchestrator...")
    if (
        orchestrator
        and hasattr(orchestrator, "shutdown")
        and callable(orchestrator.shutdown)
    ):
        try:
            await orchestrator.shutdown()
            logger.info("NodeBridgeOrchestrator shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)


# Create FastAPI application
app = FastAPI(
    title="NodeBridgeOrchestrator API",
    description="REST API wrapper for ONEX v2.0 workflow orchestration",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS - Security: Environment-based origin whitelist
# SECURITY RATIONALE:
# - allow_origins=["*"] with allow_credentials=True is a critical security vulnerability
# - Browsers reject wildcard origins when credentials are enabled per CORS spec
# - Use CORS_ALLOWED_ORIGINS env var to whitelist trusted origins only
# - Default allows localhost development origins, production must override
allowed_origins = os.getenv(
    "CORS_ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:8000",  # Default for local dev only
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy", service="NodeBridgeOrchestrator", version="1.0.0"
    )


@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint.

    Returns:
        Prometheus-formatted metrics for scraping
    """
    if not orchestrator:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Orchestrator not initialized",
        )

    # Get metrics from orchestrator
    metrics_data = orchestrator.metrics.get_metrics()

    return Response(
        content=metrics_data, media_type="text/plain; version=0.0.4; charset=utf-8"
    )


@app.post(
    "/workflow/submit",
    response_model=WorkflowSubmissionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def submit_workflow(request: WorkflowSubmissionRequest):
    """
    Submit a new workflow for metadata stamping.

    Args:
        request: Workflow submission request with content and parameters

    Returns:
        WorkflowSubmissionResponse with workflow ID and initial state
    """
    if not orchestrator:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Orchestrator not initialized",
        )

    try:
        # Create stamp request input
        # Map API request fields to ModelStampRequestInput requirements
        stamp_input = ModelStampRequestInput(
            file_path="<api-submitted>",  # Placeholder for API-submitted content
            file_content=request.content.encode("utf-8"),
            content_type="text/plain",
            namespace=request.namespace,
        )

        # Process through orchestrator
        # Note: This is a simplified implementation
        # In production, you would queue the workflow and return immediately
        result = await orchestrator.handle_workflow_start(stamp_input)

        if result:
            return WorkflowSubmissionResponse(
                success=True,
                workflow_id=str(result.get("workflow_id", "unknown")),
                state=result.get("state", "processing"),
                message="Workflow submitted successfully",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to start workflow",
            )

    except HTTPException:
        # Re-raise HTTPException without wrapping
        raise
    except Exception as e:
        logger.error(f"Error submitting workflow: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing workflow: {e!s}",
        )


@app.get("/workflow/{workflow_id}/status", response_model=WorkflowStatusResponse)
async def get_workflow_status(workflow_id: str):
    """
    Get the status of a workflow.

    Args:
        workflow_id: UUID of the workflow to query

    Returns:
        WorkflowStatusResponse with current state and result
    """
    if not orchestrator:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Orchestrator not initialized",
        )

    try:
        # Query workflow status from orchestrator
        # This would typically query a database or state store
        return WorkflowStatusResponse(
            workflow_id=workflow_id, state="unknown", current_step=None, result=None
        )

    except Exception as e:
        logger.error(f"Error querying workflow status: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error querying workflow: {e!s}",
        )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    # Return basic metrics
    # In production, integrate with prometheus_client
    return {
        "workflows_total": 0,
        "workflows_active": 0,
        "workflows_completed": 0,
        "workflows_failed": 0,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8060,
        log_level="info",
    )
