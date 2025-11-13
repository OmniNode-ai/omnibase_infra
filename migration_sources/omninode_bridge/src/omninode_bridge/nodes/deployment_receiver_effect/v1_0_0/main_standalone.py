"""
Standalone REST API for NodeDeploymentReceiverEffect - Remote deployment receiver service.

This service receives Docker images and deployment requests from remote senders,
validates security credentials (HMAC + BLAKE3 + IP whitelisting), and deploys
containers to the local Docker daemon.

Performance Targets:
- Image load: <3s
- Container start: <2s
- Health check: <1s
- Total deployment: <8s

Security Features:
- HMAC authentication with SHA256
- BLAKE3 checksum validation
- IP whitelisting with CIDR support
- Constant-time signature comparison
"""

import logging
import os
from typing import Any, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from omnibase_core.models.container.model_onex_container import ModelONEXContainer
from omnibase_core.models.contracts.model_contract_effect import ModelContractEffect
from pydantic import BaseModel, Field

# Try to import the real node
try:
    from .node import NodeDeploymentReceiverEffect
except ImportError:
    NodeDeploymentReceiverEffect = None  # type: ignore

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)

# Global node instance (initialized on startup)
node_instance: Optional[NodeDeploymentReceiverEffect] = None


# ===========================
# Pydantic Request/Response Models
# ===========================


class PackageData(BaseModel):
    """Docker image package information."""

    image_tar_path: str = Field(..., description="Path to Docker image tar file")
    checksum: str = Field(..., description="BLAKE3 checksum (64 hex chars)")
    size_bytes: int = Field(..., description="Package size in bytes", gt=0)


class SenderAuth(BaseModel):
    """Sender authentication credentials."""

    sender_id: str = Field(..., description="UUID of the sender")
    auth_token: str = Field(
        ..., min_length=32, description="Authentication token (min 32 chars)"
    )
    signature: str = Field(..., description="HMAC-SHA256 signature")
    sender_ip: str = Field(..., description="Sender IP address")


class DeploymentConfig(BaseModel):
    """Container deployment configuration."""

    image_name: str = Field(..., description="Docker image name with tag")
    container_name: str = Field(..., description="Container name")
    ports: dict[str, int] = Field(
        default_factory=dict, description="Port mappings (external:internal)"
    )
    environment_vars: dict[str, str] = Field(
        default_factory=dict, description="Environment variables"
    )
    volumes: list[dict[str, str]] = Field(
        default_factory=list, description="Volume mounts"
    )
    restart_policy: str = Field(default="unless-stopped", description="Restart policy")
    resource_limits: Optional[dict[str, str]] = Field(
        None, description="CPU/memory limits"
    )


class ReceivePackageRequest(BaseModel):
    """Request to receive and validate a deployment package."""

    package_data: PackageData
    sender_auth: SenderAuth


class LoadImageRequest(BaseModel):
    """Request to load Docker image."""

    image_tar_path: str = Field(..., description="Path to Docker image tar file")


class DeployContainerRequest(BaseModel):
    """Request to deploy a container."""

    deployment_config: DeploymentConfig


class HealthCheckRequest(BaseModel):
    """Request to check container health."""

    container_name: str = Field(..., description="Container name to check")
    health_endpoint: Optional[str] = Field(
        None, description="HTTP health check endpoint"
    )


class FullDeploymentRequest(BaseModel):
    """Request for complete deployment pipeline."""

    package_data: PackageData
    sender_auth: SenderAuth
    deployment_config: DeploymentConfig


class DeploymentResponse(BaseModel):
    """Generic deployment response."""

    success: bool
    message: str
    execution_time_ms: Optional[float] = None
    data: Optional[dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    service: str
    version: str
    mode: str
    docker_available: bool


# ===========================
# FastAPI Application
# ===========================

app = FastAPI(
    title="NodeDeploymentReceiverEffect API",
    description="ONEX v2.0 Deployment Receiver - Remote container deployment with security validation",
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
    """Initialize node and Docker client on application startup."""
    global node_instance

    try:
        # AUTH_SECRET_KEY is required for HMAC signature validation
        auth_secret_key = os.getenv("AUTH_SECRET_KEY")
        if not auth_secret_key:
            raise ValueError(
                "AUTH_SECRET_KEY environment variable must be set. "
                "Generate a secure key with: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
            )

        # Create ONEX container (without config parameter for compatibility)
        container = ModelONEXContainer(enable_service_registry=True)

        # Manually set config attribute
        container.config = {
            "docker_host": os.getenv("DOCKER_HOST", "unix:///var/run/docker.sock"),
            "auth_secret_key": auth_secret_key,
            "allowed_ip_ranges": os.getenv(
                "ALLOWED_IP_RANGES", "192.168.86.0/24,10.0.0.0/8"
            ),
            "package_dir": os.getenv("PACKAGE_DIR", "/tmp/deployment_packages"),
            "kafka_bootstrap_servers": os.getenv(
                "KAFKA_BOOTSTRAP_SERVERS", "localhost:29092"
            ),
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

        logger.info("Container created successfully")

        # Initialize and register services
        from ....services.kafka_client import KafkaClient

        kafka_client = KafkaClient(
            bootstrap_servers=container.config.get(
                "kafka_bootstrap_servers", "localhost:29092"
            ),
            enable_dead_letter_queue=True,
            max_retry_attempts=3,
            timeout_seconds=30,
        )

        # Register services with container
        container.register_service("kafka_client", kafka_client)

        logger.info("Services initialized and registered with container")

        # Initialize node
        if NodeDeploymentReceiverEffect is None:
            raise ImportError(
                "NodeDeploymentReceiverEffect not available (omnibase_core not installed)"
            )
        node_instance = NodeDeploymentReceiverEffect(container)

        # Start node (publishes introspection and starts background tasks)
        if hasattr(node_instance, "startup") and callable(node_instance.startup):
            await node_instance.startup()

        logger.info("NodeDeploymentReceiverEffect initialized and ready")
        logger.info(
            f"Docker host: {os.getenv('DOCKER_HOST', 'unix:///var/run/docker.sock')}"
        )
        logger.info(
            f"Allowed IP ranges: {os.getenv('ALLOWED_IP_RANGES', '192.168.86.0/24,10.0.0.0/8')}"
        )

    except Exception as e:
        logger.error(f"Failed to initialize node: {e}", exc_info=True)
        # Continue running in degraded mode
        node_instance = None


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown node and cleanup resources."""
    global node_instance

    if node_instance:
        try:
            if hasattr(node_instance, "shutdown") and callable(node_instance.shutdown):
                await node_instance.shutdown()
            logger.info("NodeDeploymentReceiverEffect shutdown complete")
        except Exception as e:
            logger.error(f"Error during node shutdown: {e}", exc_info=True)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    docker_available = False
    if node_instance and hasattr(node_instance, "docker_client"):
        try:
            await node_instance.docker_client.ping()
            docker_available = True
        except Exception:
            docker_available = False

    return HealthResponse(
        status="healthy" if node_instance else "degraded",
        service="NodeDeploymentReceiverEffect",
        version="1.0.0",
        mode="standalone",
        docker_available=docker_available,
    )


@app.post("/deployment/receive", response_model=DeploymentResponse)
async def receive_package(request: ReceivePackageRequest, http_request: Request):
    """
    Receive and validate deployment package.

    Security: HMAC + BLAKE3 + IP whitelisting
    Target: <500ms
    """
    if not node_instance:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # Get client IP
        client_ip = http_request.client.host if http_request.client else "unknown"

        # Create ONEX contract
        contract = ModelContractEffect(
            name="deployment_receiver_receive_package",
            version="1.0.0",
            description="Receive and validate deployment package",
            node_type="EFFECT",
            input_model="ModelPackageReceiveInput",
            output_model="ModelPackageReceiveOutput",
            io_operations=["validate_package", "check_signature", "verify_checksum"],
            correlation_id=uuid4(),
            input_state={
                "operation_type": "receive_package",
                "package_data": request.package_data.model_dump(),
                "sender_auth": {
                    **request.sender_auth.model_dump(),
                    "sender_ip": client_ip,  # Override with actual client IP
                },
            },
        )

        # Execute node
        result = await node_instance.execute_effect(contract)

        if not result.success:
            raise HTTPException(
                status_code=400,
                detail=result.output_state.get("error", "Package validation failed"),
            )

        return DeploymentResponse(
            success=True,
            message="Package received and validated",
            execution_time_ms=result.output_state.get("execution_time_ms"),
            data=result.output_state,
        )

    except Exception as e:
        logger.error(f"Package reception failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/deployment/load", response_model=DeploymentResponse)
async def load_image(request: LoadImageRequest):
    """
    Load Docker image into daemon.

    Target: <3s
    """
    if not node_instance:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        contract = ModelContractEffect(
            name="deployment_receiver_load_image",
            version="1.0.0",
            description="Load Docker image into daemon",
            node_type="EFFECT",
            input_model="ModelImageLoadInput",
            output_model="ModelImageLoadOutput",
            io_operations=["load_image", "verify_image"],
            correlation_id=uuid4(),
            input_state={
                "operation_type": "load_image",
                "image_tar_path": request.image_tar_path,
            },
        )

        result = await node_instance.execute_effect(contract)

        if not result.success:
            raise HTTPException(
                status_code=500,
                detail=result.output_state.get("error", "Image load failed"),
            )

        return DeploymentResponse(
            success=True,
            message="Image loaded successfully",
            execution_time_ms=result.output_state.get("execution_time_ms"),
            data=result.output_state,
        )

    except Exception as e:
        logger.error(f"Image load failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/deployment/deploy", response_model=DeploymentResponse)
async def deploy_container(request: DeployContainerRequest):
    """
    Deploy container with configuration.

    Target: <2s
    """
    if not node_instance:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        contract = ModelContractEffect(
            name="deployment_receiver_deploy_container",
            version="1.0.0",
            description="Deploy container with configuration",
            node_type="EFFECT",
            input_model="ModelContainerDeployInput",
            output_model="ModelContainerDeployOutput",
            io_operations=["deploy_container", "configure_network", "start_container"],
            correlation_id=uuid4(),
            input_state={
                "operation_type": "deploy_container",
                "deployment_config": request.deployment_config.model_dump(),
            },
        )

        result = await node_instance.execute_effect(contract)

        if not result.success:
            raise HTTPException(
                status_code=500,
                detail=result.output_state.get("error", "Container deployment failed"),
            )

        return DeploymentResponse(
            success=True,
            message="Container deployed successfully",
            execution_time_ms=result.output_state.get("execution_time_ms"),
            data=result.output_state,
        )

    except Exception as e:
        logger.error(f"Container deployment failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/deployment/health-check", response_model=DeploymentResponse)
async def check_container_health(request: HealthCheckRequest):
    """
    Verify container health.

    Target: <1s
    """
    if not node_instance:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        contract = ModelContractEffect(
            name="deployment_receiver_health_check",
            version="1.0.0",
            description="Verify container health",
            node_type="EFFECT",
            input_model="ModelHealthCheckInput",
            output_model="ModelHealthCheckOutput",
            io_operations=["health_check", "verify_status"],
            correlation_id=uuid4(),
            input_state={
                "operation_type": "health_check",
                "container_name": request.container_name,
                "health_endpoint": request.health_endpoint,
            },
        )

        result = await node_instance.execute_effect(contract)

        if not result.success:
            raise HTTPException(
                status_code=500,
                detail=result.output_state.get("error", "Health check failed"),
            )

        return DeploymentResponse(
            success=True,
            message="Health check completed",
            execution_time_ms=result.output_state.get("execution_time_ms"),
            data=result.output_state,
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/deployment/full",
    response_model=DeploymentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def full_deployment(request: FullDeploymentRequest, http_request: Request):
    """
    Complete deployment pipeline: validate → load → deploy → health check → publish events.

    Target: <8s total
    Security: HMAC + BLAKE3 + IP whitelisting
    """
    if not node_instance:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # Get client IP
        client_ip = http_request.client.host if http_request.client else "unknown"

        # Create ONEX contract
        contract = ModelContractEffect(
            name="deployment_receiver_full_deployment",
            version="1.0.0",
            description="Complete deployment pipeline: validate → load → deploy → health check → publish events",
            node_type="EFFECT",
            input_model="ModelFullDeploymentInput",
            output_model="ModelFullDeploymentOutput",
            io_operations=[
                "validate_package",
                "load_image",
                "deploy_container",
                "health_check",
                "publish_events",
            ],
            correlation_id=uuid4(),
            input_state={
                "operation_type": "full_deployment",
                "package_data": request.package_data.model_dump(),
                "sender_auth": {
                    **request.sender_auth.model_dump(),
                    "sender_ip": client_ip,
                },
                "deployment_config": request.deployment_config.model_dump(),
            },
        )

        # Execute node
        result = await node_instance.execute_effect(contract)

        if not result.success:
            raise HTTPException(
                status_code=500,
                detail=result.output_state.get("error", "Full deployment failed"),
            )

        return DeploymentResponse(
            success=True,
            message="Full deployment completed successfully",
            execution_time_ms=result.output_state.get("total_time_ms"),
            data=result.output_state,
        )

    except Exception as e:
        logger.error(f"Full deployment failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return {
        "deployments_total": 0,
        "deployments_success": 0,
        "deployments_failed": 0,
        "mode": "standalone",
    }


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "NodeDeploymentReceiverEffect",
        "version": "1.0.0",
        "mode": "standalone",
        "status": "running",
        "message": "ONEX v2.0 Deployment Receiver - Remote container deployment with security validation",
        "endpoints": {
            "health": "/health",
            "receive_package": "/deployment/receive",
            "load_image": "/deployment/load",
            "deploy_container": "/deployment/deploy",
            "health_check": "/deployment/health-check",
            "full_deployment": "/deployment/full",
            "metrics": "/metrics",
            "docs": "/docs",
        },
        "security": {
            "hmac_auth": "enabled",
            "blake3_checksum": "enabled",
            "ip_whitelisting": "enabled",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("DEPLOYMENT_RECEIVER_PORT", "8001")),
        log_level="info",
    )
