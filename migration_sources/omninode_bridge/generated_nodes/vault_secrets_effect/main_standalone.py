"""
Standalone REST API for NodeVaultSecretsEffect - Simplified version without omnibase runtime.

This is a minimal implementation for demo/bridge environments.
Provides a REST API wrapper for HashiCorp Vault operations.
"""

import logging
import os
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Try to import the real node, fall back to None if omnibase_core is not available
try:
    from omnibase_core.models.core import ModelContainer

    from .node import NodeVaultSecretsEffect
except ImportError:
    NodeVaultSecretsEffect = None  # type: ignore
    ModelContainer = None  # type: ignore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global node instance (will be initialized on startup)
node_instance: Optional["NodeVaultSecretsEffect"] = None


# Configuration model for standalone mode
class StandaloneConfig(BaseModel):
    """Configuration for standalone Vault Effect node mode."""

    vault_addr: str
    vault_token: str
    vault_namespace: Optional[str] = None
    vault_mount_point: str = "secret"
    environment: str = "development"


# Pydantic models for API
class SecretReadRequest(BaseModel):
    """Request model for reading a secret."""

    path: str = Field(..., description="Path to the secret in Vault")
    mount_point: Optional[str] = Field(
        None, description="Vault mount point (defaults to config)"
    )


class SecretWriteRequest(BaseModel):
    """Request model for writing a secret."""

    path: str = Field(..., description="Path to the secret in Vault")
    data: dict[str, Any] = Field(..., description="Secret data to write")
    mount_point: Optional[str] = Field(
        None, description="Vault mount point (defaults to config)"
    )


class SecretListRequest(BaseModel):
    """Request model for listing secrets."""

    path: str = Field(default="", description="Path to list secrets from")
    mount_point: Optional[str] = Field(
        None, description="Vault mount point (defaults to config)"
    )


class SecretDeleteRequest(BaseModel):
    """Request model for deleting a secret."""

    path: str = Field(..., description="Path to the secret in Vault")
    mount_point: Optional[str] = Field(
        None, description="Vault mount point (defaults to config)"
    )


class SecretOperationResponse(BaseModel):
    """Response model for secret operations."""

    success: bool
    data: Optional[dict[str, Any]] = None
    error_message: Optional[str] = None
    operation: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    service: str
    version: str
    mode: str
    node_initialized: bool
    vault_connected: bool = False


# Create FastAPI application
app = FastAPI(
    title="NodeVaultSecretsEffect API (Standalone)",
    description="Standalone REST API for ONEX v2.0 Vault integration",
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
    """Initialize node on application startup."""
    global node_instance

    try:
        # Check for required environment variables
        vault_addr = os.getenv("VAULT_ADDR")
        vault_token = os.getenv("VAULT_TOKEN")

        if not vault_addr or not vault_token:
            logger.warning(
                "VAULT_ADDR or VAULT_TOKEN not set - node will run in degraded mode"
            )
            node_instance = None
            return

        # Create standalone configuration from environment
        config = StandaloneConfig(
            vault_addr=vault_addr,
            vault_token=vault_token,
            vault_namespace=os.getenv("VAULT_NAMESPACE"),
            vault_mount_point=os.getenv("VAULT_MOUNT_POINT", "secret"),
            environment=os.getenv("ENVIRONMENT", "development"),
        )

        logger.info(f"Standalone configuration loaded: vault_addr={config.vault_addr}")

        # Initialize node if omnibase_core is available
        if NodeVaultSecretsEffect and ModelContainer:
            try:
                # Create container with credentials
                container = ModelContainer(
                    value={
                        "vault_addr": config.vault_addr,
                        "vault_token": config.vault_token,
                        "vault_namespace": config.vault_namespace,
                        "vault_mount_point": config.vault_mount_point,
                    },
                    container_type="config",
                )
                node_instance = NodeVaultSecretsEffect(container)
                logger.info("NodeVaultSecretsEffect initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize node: {e}", exc_info=True)
                node_instance = None
        else:
            logger.warning("omnibase_core not available - running in degraded mode")
            node_instance = None

    except Exception as e:
        logger.error(f"Failed to initialize standalone mode: {e}", exc_info=True)
        node_instance = None


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown node and cleanup resources."""
    global node_instance

    if node_instance:
        try:
            # Node cleanup if needed
            logger.info("NodeVaultSecretsEffect shutdown complete")
        except Exception as e:
            logger.error(f"Error during node shutdown: {e}", exc_info=True)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    vault_connected = False
    if node_instance:
        try:
            # Try to check vault health
            vault_connected = node_instance.client.is_authenticated()
        except Exception:
            vault_connected = False

    return HealthResponse(
        status="healthy" if node_instance and vault_connected else "degraded",
        service="NodeVaultSecretsEffect",
        version="1.0.0",
        mode="standalone",
        node_initialized=node_instance is not None,
        vault_connected=vault_connected,
    )


@app.post(
    "/secret/read",
    response_model=SecretOperationResponse,
    status_code=status.HTTP_200_OK,
)
async def read_secret(request: SecretReadRequest):
    """
    Read a secret from Vault.

    Args:
        request: Secret read request

    Returns:
        SecretOperationResponse with secret data
    """
    if not node_instance:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vault node not initialized - check VAULT_ADDR and VAULT_TOKEN configuration",
        )

    try:
        # Import models
        from omnibase_core.models.contracts.model_contract_effect import (
            ModelContractEffect,
        )
        from omnibase_core.primitives.model_semver import ModelSemVer

        # Create contract for node execution
        contract = ModelContractEffect(
            name="vault_read_secret",
            version=ModelSemVer(major=1, minor=0, patch=0),
            description=f"Read secret from {request.path}",
            node_type="EFFECT",
            input_model="ModelVaultReadRequest",
            output_model="ModelVaultReadResponse",
            input_data={
                "path": request.path,
                "mount_point": request.mount_point or "secret",
                "operation": "read_secret",
            },
        )

        # Execute node
        result_container = await node_instance.execute(contract)

        # Extract result
        result = result_container.value

        return SecretOperationResponse(
            success=True,
            data=result.get("data"),
            error_message=None,
            operation="read_secret",
        )

    except Exception as e:
        logger.error(f"Error reading secret: {e}", exc_info=True)
        return SecretOperationResponse(
            success=False,
            data=None,
            error_message=str(e),
            operation="read_secret",
        )


@app.post(
    "/secret/write",
    response_model=SecretOperationResponse,
    status_code=status.HTTP_201_CREATED,
)
async def write_secret(request: SecretWriteRequest):
    """
    Write a secret to Vault.

    Args:
        request: Secret write request

    Returns:
        SecretOperationResponse
    """
    if not node_instance:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vault node not initialized - check VAULT_ADDR and VAULT_TOKEN configuration",
        )

    try:
        # Import models
        from omnibase_core.models.contracts.model_contract_effect import (
            ModelContractEffect,
        )
        from omnibase_core.primitives.model_semver import ModelSemVer

        # Create contract for node execution
        contract = ModelContractEffect(
            name="vault_write_secret",
            version=ModelSemVer(major=1, minor=0, patch=0),
            description=f"Write secret to {request.path}",
            node_type="EFFECT",
            input_model="ModelVaultWriteRequest",
            output_model="ModelVaultWriteResponse",
            input_data={
                "path": request.path,
                "data": request.data,
                "mount_point": request.mount_point or "secret",
                "operation": "write_secret",
            },
        )

        # Execute node
        result_container = await node_instance.execute(contract)

        # Extract result
        result = result_container.value

        return SecretOperationResponse(
            success=True,
            data={"message": "Secret written successfully"},
            error_message=None,
            operation="write_secret",
        )

    except Exception as e:
        logger.error(f"Error writing secret: {e}", exc_info=True)
        return SecretOperationResponse(
            success=False,
            data=None,
            error_message=str(e),
            operation="write_secret",
        )


@app.post(
    "/secret/list",
    response_model=SecretOperationResponse,
    status_code=status.HTTP_200_OK,
)
async def list_secrets(request: SecretListRequest):
    """
    List secrets in Vault.

    Args:
        request: Secret list request

    Returns:
        SecretOperationResponse with list of secret paths
    """
    if not node_instance:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vault node not initialized - check VAULT_ADDR and VAULT_TOKEN configuration",
        )

    try:
        # Import models
        from omnibase_core.models.contracts.model_contract_effect import (
            ModelContractEffect,
        )
        from omnibase_core.primitives.model_semver import ModelSemVer

        # Create contract for node execution
        contract = ModelContractEffect(
            name="vault_list_secrets",
            version=ModelSemVer(major=1, minor=0, patch=0),
            description=f"List secrets at {request.path}",
            node_type="EFFECT",
            input_model="ModelVaultListRequest",
            output_model="ModelVaultListResponse",
            input_data={
                "path": request.path,
                "mount_point": request.mount_point or "secret",
                "operation": "list_secrets",
            },
        )

        # Execute node
        result_container = await node_instance.execute(contract)

        # Extract result
        result = result_container.value

        return SecretOperationResponse(
            success=True,
            data=result.get("secrets"),
            error_message=None,
            operation="list_secrets",
        )

    except Exception as e:
        logger.error(f"Error listing secrets: {e}", exc_info=True)
        return SecretOperationResponse(
            success=False,
            data=None,
            error_message=str(e),
            operation="list_secrets",
        )


@app.delete(
    "/secret/delete",
    response_model=SecretOperationResponse,
    status_code=status.HTTP_200_OK,
)
async def delete_secret(request: SecretDeleteRequest):
    """
    Delete a secret from Vault.

    Args:
        request: Secret delete request

    Returns:
        SecretOperationResponse
    """
    if not node_instance:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vault node not initialized - check VAULT_ADDR and VAULT_TOKEN configuration",
        )

    try:
        # Import models
        from omnibase_core.models.contracts.model_contract_effect import (
            ModelContractEffect,
        )
        from omnibase_core.primitives.model_semver import ModelSemVer

        # Create contract for node execution
        contract = ModelContractEffect(
            name="vault_delete_secret",
            version=ModelSemVer(major=1, minor=0, patch=0),
            description=f"Delete secret at {request.path}",
            node_type="EFFECT",
            input_model="ModelVaultDeleteRequest",
            output_model="ModelVaultDeleteResponse",
            input_data={
                "path": request.path,
                "mount_point": request.mount_point or "secret",
                "operation": "delete_secret",
            },
        )

        # Execute node
        result_container = await node_instance.execute(contract)

        # Extract result
        result = result_container.value

        return SecretOperationResponse(
            success=True,
            data={"message": "Secret deleted successfully"},
            error_message=None,
            operation="delete_secret",
        )

    except Exception as e:
        logger.error(f"Error deleting secret: {e}", exc_info=True)
        return SecretOperationResponse(
            success=False,
            data=None,
            error_message=str(e),
            operation="delete_secret",
        )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return {
        "node_initialized": node_instance is not None,
        "vault_connected": (
            node_instance.client.is_authenticated() if node_instance else False
        ),
        "mode": "standalone",
    }


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "NodeVaultSecretsEffect",
        "version": "1.0.0",
        "mode": "standalone",
        "status": "running",
        "node_initialized": node_instance is not None,
        "message": "Standalone REST API for Vault operations",
        "endpoints": {
            "health": "/health",
            "read_secret": "/secret/read",
            "write_secret": "/secret/write",
            "list_secrets": "/secret/list",
            "delete_secret": "/secret/delete",
            "metrics": "/metrics",
            "docs": "/docs",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8071,
        log_level="info",
    )
