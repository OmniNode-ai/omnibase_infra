"""
Standalone REST API for NodeBridgeRegistry - Simplified version without omnibase runtime.

This is a minimal implementation for demo/bridge environments.
"""

import logging
import os
from typing import Any, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# Direct import - omnibase_core is required
from omnibase_core.models.container.model_onex_container import ModelONEXContainer
from pydantic import BaseModel, Field

# Import node and container
from .node import NodeBridgeRegistry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global node instance (will be initialized on startup)
node_instance: Optional[NodeBridgeRegistry] = None


# Pydantic models for API
class RegistrationRequest(BaseModel):
    """Request model for manual node registration."""

    node_id: str = Field(..., description="Node identifier")
    node_type: str = Field(..., description="Node type (orchestrator, reducer, etc.)")
    capabilities: dict[str, Any] = Field(
        default_factory=dict, description="Node capabilities"
    )
    endpoints: dict[str, str] = Field(
        default_factory=dict, description="Node endpoints"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class RegistrationResponse(BaseModel):
    """Response model for registration."""

    success: bool
    registered_node_id: str
    consul_registered: bool
    postgres_registered: bool
    registration_time_ms: float
    message: Optional[str] = None


class RegistryMetricsResponse(BaseModel):
    """Response model for registry metrics."""

    total_registrations: int
    successful_registrations: int
    failed_registrations: int
    consul_registrations: int
    postgres_registrations: int
    registered_nodes_count: int
    registered_nodes: list[str]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    service: str
    version: str
    mode: str
    components: dict[str, Any] = Field(default_factory=dict)


# Create FastAPI application
app = FastAPI(
    title="NodeBridgeRegistry API (Standalone)",
    description="Standalone REST API for ONEX v2.0 node discovery and dual registration",
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
        # Create ONEX container (without config parameter for compatibility)
        container = ModelONEXContainer(enable_service_registry=True)

        # Store configuration as custom attribute (config property is read-only)
        container._custom_config = {
            "kafka_broker_url": os.getenv(
                "KAFKA_BOOTSTRAP_SERVERS", "omninode-bridge-redpanda:9092"
            ),
            "consul_host": os.getenv("CONSUL_HOST", "consul"),
            "consul_port": int(os.getenv("CONSUL_PORT", "8500")),
            "postgres_host": os.getenv("POSTGRES_HOST", "postgres"),
            "postgres_port": int(os.getenv("POSTGRES_PORT", "5432")),
            "postgres_db": os.getenv("POSTGRES_DATABASE", "omninode_bridge"),
            "postgres_user": os.getenv("POSTGRES_USER", "postgres"),
            "postgres_password": os.getenv("POSTGRES_PASSWORD"),
            "registry_id": os.getenv("REGISTRY_ID", f"registry-{uuid4().hex[:8]}"),
            "default_namespace": "omninode.bridge",
            "api_port": int(os.getenv("REGISTRY_PORT", "8062")),
            "metrics_port": 9093,
            "environment": os.getenv("ENVIRONMENT", "development"),
        }

        # Add simple service storage for string-based lookup
        # ModelONEXContainer.service_registry uses protocol types, but we need simple string keys
        container._service_instances = {}

        def _register_service(name: str, instance: Any) -> None:
            """Register service by name for simple string-based lookup."""
            container._service_instances[name] = instance

        def _get_service(name: str) -> Any:
            """Get service by name, returns None if not found."""
            return container._service_instances.get(name)

        # Monkey-patch these methods onto container
        container.register_service = _register_service
        container.get_service = _get_service

        # Make _custom_config accessible via .config for compatibility
        type(container).config = property(
            lambda self: self._custom_config if hasattr(self, "_custom_config") else {}
        )

        logger.info("Container created successfully")

        # Initialize services
        from ....services.kafka_client import KafkaClient
        from ....services.metadata_stamping.registry.consul_client import (
            RegistryConsulClient,
        )
        from ....services.postgres_client import PostgresClient

        kafka_client = KafkaClient(
            bootstrap_servers=container.config.get(
                "kafka_broker_url", "omninode-bridge-redpanda:9092"
            ),
            enable_dead_letter_queue=True,
            max_retry_attempts=3,
            timeout_seconds=30,
        )

        # Connect Kafka client
        await kafka_client.connect()
        logger.info("Kafka client connected successfully")

        consul_client = RegistryConsulClient(
            consul_host=container.config.get("consul_host", "consul"),
            consul_port=container.config.get("consul_port", 8500),
        )

        postgres_client = PostgresClient(
            host=container.config.get("postgres_host", "postgres"),
            port=container.config.get("postgres_port", 5432),
            database=container.config.get("postgres_db", "omninode_bridge"),
            user=container.config.get("postgres_user", "postgres"),
            password=container.config.get("postgres_password"),
        )

        # Register services with container
        container.register_service("kafka_client", kafka_client)
        container.register_service("consul_client", consul_client)
        container.register_service("postgres_client", postgres_client)

        logger.info("Services initialized and registered with container")

        # Initialize node
        node_instance = NodeBridgeRegistry(container)

        # Start node (publishes introspection request and starts consuming)
        await node_instance.on_startup()

        logger.info("NodeBridgeRegistry initialized and ready")

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
            await node_instance.on_shutdown()
            logger.info("NodeBridgeRegistry shutdown complete")
        except Exception as e:
            logger.error(f"Error during node shutdown: {e}", exc_info=True)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint with component status.

    Returns detailed health information about the registry and its dependencies.
    """
    if node_instance:
        try:
            # Get detailed health check from node
            health_result = await node_instance.check_health()

            # Convert dataclass to dict (health_result is NodeHealthCheckResult)
            health_dict = health_result.to_dict()

            # Convert components list to dict keyed by component name
            components_dict = {
                comp["name"]: comp for comp in health_dict.get("components", [])
            }

            return HealthResponse(
                status=health_dict["overall_status"],
                service="NodeBridgeRegistry",
                version="1.0.0",
                mode="standalone",
                components=components_dict,
            )
        except Exception as e:
            logger.error(f"Health check failed: {e}", exc_info=True)
            return HealthResponse(
                status="unhealthy",
                service="NodeBridgeRegistry",
                version="1.0.0",
                mode="standalone",
                components={"error": str(e)},
            )
    else:
        return HealthResponse(
            status="degraded",
            service="NodeBridgeRegistry",
            version="1.0.0",
            mode="standalone",
            components={"node_instance": "not_initialized"},
        )


@app.post(
    "/registry/register",
    response_model=RegistrationResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register_node(request: RegistrationRequest):
    """
    Manually register a node (for testing or manual registration).

    In normal operation, nodes self-register by publishing introspection events.
    This endpoint allows manual registration for testing or recovery scenarios.

    Args:
        request: Node registration request

    Returns:
        RegistrationResponse
    """
    if not node_instance:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Registry node not initialized",
        )

    try:
        from ...orchestrator.v1_0_0.models.model_node_introspection_event import (
            ModelNodeIntrospectionEvent,
        )

        # Create introspection event from request
        introspection = ModelNodeIntrospectionEvent(
            node_id=request.node_id,
            node_type=request.node_type,
            capabilities=request.capabilities,
            endpoints=request.endpoints,
            metadata=request.metadata,
            correlation_id=uuid4(),
        )

        # Perform dual registration
        result = await node_instance.dual_register(introspection)

        return RegistrationResponse(
            success=result["status"] == "success",
            registered_node_id=result["registered_node_id"],
            consul_registered=result["consul_registered"],
            postgres_registered=result["postgres_registered"],
            registration_time_ms=result["registration_time_ms"],
            message=result.get("error"),
        )

    except Exception as e:
        logger.error(f"Manual registration failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {e!s}",
        )


@app.get("/registry/metrics", response_model=RegistryMetricsResponse)
async def get_registry_metrics():
    """
    Get registry metrics.

    Returns statistics about registration operations and registered nodes.

    Returns:
        RegistryMetricsResponse
    """
    if not node_instance:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Registry node not initialized",
        )

    try:
        metrics = node_instance.get_registration_metrics()

        return RegistryMetricsResponse(
            total_registrations=metrics["total_registrations"],
            successful_registrations=metrics["successful_registrations"],
            failed_registrations=metrics["failed_registrations"],
            consul_registrations=metrics["consul_registrations"],
            postgres_registrations=metrics["postgres_registrations"],
            registered_nodes_count=metrics["registered_nodes_count"],
            registered_nodes=metrics["registered_nodes"],
        )

    except Exception as e:
        logger.error(f"Failed to get metrics: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve metrics: {e!s}",
        )


@app.post("/registry/request-introspection")
async def request_introspection():
    """
    Request all nodes to broadcast introspection events.

    Useful for refreshing registry state or recovering from failures.

    Returns:
        Success status
    """
    if not node_instance:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Registry node not initialized",
        )

    try:
        from ...orchestrator.v1_0_0.models.model_registry_request_event import (
            EnumIntrospectionReason,
        )

        success = await node_instance._request_node_introspection(
            EnumIntrospectionReason.MANUAL
        )

        return {
            "success": success,
            "message": (
                "Introspection request published"
                if success
                else "Failed to publish request"
            ),
        }

    except Exception as e:
        logger.error(f"Failed to request introspection: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to request introspection: {e!s}",
        )


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    if node_instance:
        metrics = node_instance.get_registration_metrics()
        return {
            "registrations_total": metrics["total_registrations"],
            "registrations_successful": metrics["successful_registrations"],
            "registrations_failed": metrics["failed_registrations"],
            "consul_registrations_total": metrics["consul_registrations"],
            "postgres_registrations_total": metrics["postgres_registrations"],
            "registered_nodes_count": metrics["registered_nodes_count"],
            "mode": "standalone",
        }
    else:
        return {
            "registrations_total": 0,
            "mode": "standalone",
            "status": "not_initialized",
        }


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "NodeBridgeRegistry",
        "version": "1.0.0",
        "mode": "standalone",
        "status": "running" if node_instance else "not_initialized",
        "message": "This is a standalone REST API wrapper for node discovery and dual registration.",
        "endpoints": {
            "health": "/health",
            "register_node": "/registry/register",
            "registry_metrics": "/registry/metrics",
            "request_introspection": "/registry/request-introspection",
            "prometheus_metrics": "/metrics",
            "docs": "/docs",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8062,
        log_level="info",
    )
