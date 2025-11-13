#!/usr/bin/env python3
"""
IO operation models for deployment receiver effect node.
ONEX v2.0 compliant input/output models for all effect operations.
"""

from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field

from .model_auth import ModelAuthCredentials
from .model_deployment import ModelDeploymentConfig, ModelHealthCheckResult

# ============================================================================
# Package Receive Operation
# ============================================================================


class ModelPackageData(BaseModel):
    """
    Docker image package data.

    Attributes:
        image_tar_path: Path to Docker image tar file
        checksum: BLAKE3 checksum for validation
        size_bytes: Package size in bytes
    """

    image_tar_path: str = Field(..., description="Path to Docker image tar file")

    checksum: str = Field(
        ..., pattern=r"^[a-f0-9]{64}$", description="BLAKE3 checksum (64 hex chars)"
    )

    size_bytes: int = Field(..., ge=0, description="Package size in bytes")


class ModelPackageReceiveInput(BaseModel):
    """
    Input for package receive operation.

    Attributes:
        package_data: Docker image package data
        sender_auth: Authentication credentials
        correlation_id: Correlation ID for tracking
    """

    package_data: ModelPackageData = Field(..., description="Docker image package data")

    sender_auth: ModelAuthCredentials = Field(
        ..., description="Sender authentication credentials"
    )

    correlation_id: UUID = Field(..., description="Correlation ID for tracking")


class ModelPackageReceiveOutput(BaseModel):
    """
    Output for package receive operation.

    Attributes:
        success: Whether package was received successfully
        package_path: Path to received package
        checksum_valid: Whether checksum validation passed
        auth_valid: Whether authentication passed
        execution_time_ms: Execution time in milliseconds
        error_message: Error message if failed
    """

    success: bool = Field(..., description="Package receive success")

    package_path: Optional[str] = Field(None, description="Path to received package")

    checksum_valid: bool = Field(
        default=False, description="Checksum validation result"
    )

    auth_valid: bool = Field(
        default=False, description="Authentication validation result"
    )

    execution_time_ms: int = Field(
        ..., ge=0, description="Execution time in milliseconds"
    )

    error_message: Optional[str] = Field(None, description="Error message if failed")


# ============================================================================
# Image Load Operation
# ============================================================================


class ModelImageLoadInput(BaseModel):
    """
    Input for Docker image load operation.

    Attributes:
        image_tar_path: Path to Docker image tar file
        correlation_id: Correlation ID for tracking
    """

    image_tar_path: str = Field(..., description="Path to Docker image tar file")

    correlation_id: UUID = Field(..., description="Correlation ID for tracking")


class ModelImageLoadOutput(BaseModel):
    """
    Output for Docker image load operation.

    Attributes:
        success: Whether image was loaded successfully
        image_id: Docker image ID (SHA256)
        image_name: Image name with tag
        execution_time_ms: Execution time in milliseconds
        error_message: Error message if failed
    """

    success: bool = Field(..., description="Image load success")

    image_id: Optional[str] = Field(None, description="Docker image ID (SHA256)")

    image_name: Optional[str] = Field(None, description="Image name with tag")

    execution_time_ms: int = Field(
        ..., ge=0, description="Execution time in milliseconds"
    )

    error_message: Optional[str] = Field(None, description="Error message if failed")


# ============================================================================
# Container Deploy Operation
# ============================================================================


class ModelContainerDeployInput(BaseModel):
    """
    Input for container deployment operation.

    Attributes:
        deployment_config: Container deployment configuration
        correlation_id: Correlation ID for tracking
    """

    deployment_config: ModelDeploymentConfig = Field(
        ..., description="Container deployment configuration"
    )

    correlation_id: UUID = Field(..., description="Correlation ID for tracking")


class ModelContainerDeployOutput(BaseModel):
    """
    Output for container deployment operation.

    Attributes:
        success: Whether container was deployed successfully
        container_id: Docker container ID (full SHA256)
        container_short_id: Docker container ID (short 12 chars)
        container_url: Container service URL
        execution_time_ms: Execution time in milliseconds
        error_message: Error message if failed
    """

    success: bool = Field(..., description="Container deployment success")

    container_id: Optional[str] = Field(
        None, pattern=r"^[a-f0-9]{64}$", description="Docker container ID (full SHA256)"
    )

    container_short_id: Optional[str] = Field(
        None,
        pattern=r"^[a-f0-9]{12}$",
        description="Docker container ID (short 12 chars)",
    )

    container_url: Optional[str] = Field(None, description="Container service URL")

    execution_time_ms: int = Field(
        ..., ge=0, description="Execution time in milliseconds"
    )

    error_message: Optional[str] = Field(None, description="Error message if failed")


# ============================================================================
# Health Check Operation
# ============================================================================


class ModelHealthCheckInput(BaseModel):
    """
    Input for container health check operation.

    Attributes:
        container_id: Docker container ID to check
        correlation_id: Correlation ID for tracking
    """

    container_id: str = Field(..., description="Docker container ID to check")

    correlation_id: UUID = Field(..., description="Correlation ID for tracking")


class ModelHealthCheckOutput(BaseModel):
    """
    Output for container health check operation.

    Attributes:
        success: Whether health check succeeded
        health_status: Health check result details
        execution_time_ms: Execution time in milliseconds
        error_message: Error message if failed
    """

    success: bool = Field(..., description="Health check operation success")

    health_status: Optional[ModelHealthCheckResult] = Field(
        None, description="Health check result details"
    )

    execution_time_ms: int = Field(
        ..., ge=0, description="Execution time in milliseconds"
    )

    error_message: Optional[str] = Field(None, description="Error message if failed")


# ============================================================================
# Deployment Event Publishing
# ============================================================================


class ModelDeploymentEventInput(BaseModel):
    """
    Input for deployment event publishing.

    Attributes:
        event_type: Type of deployment event
        deployment_config: Deployment configuration
        container_id: Docker container ID (if deployed)
        health_status: Health check status (if available)
        correlation_id: Correlation ID for tracking
    """

    event_type: str = Field(
        ...,
        pattern=r"^(DEPLOYMENT_STARTED|IMAGE_LOADED|CONTAINER_STARTED|HEALTH_CHECK_PASSED|DEPLOYMENT_COMPLETED|DEPLOYMENT_FAILED)$",
        description="Deployment event type",
    )

    deployment_config: ModelDeploymentConfig = Field(
        ..., description="Deployment configuration"
    )

    container_id: Optional[str] = Field(None, description="Docker container ID")

    health_status: Optional[ModelHealthCheckResult] = Field(
        None, description="Health check status"
    )

    correlation_id: UUID = Field(..., description="Correlation ID for tracking")


class ModelDeploymentEventOutput(BaseModel):
    """
    Output for deployment event publishing.

    Attributes:
        success: Whether event was published successfully
        event_id: Published event ID
        topic: Kafka topic published to
        execution_time_ms: Execution time in milliseconds
        error_message: Error message if failed
    """

    success: bool = Field(..., description="Event publish success")

    event_id: Optional[str] = Field(None, description="Published event ID")

    topic: Optional[str] = Field(None, description="Kafka topic published to")

    execution_time_ms: int = Field(
        ..., ge=0, description="Execution time in milliseconds"
    )

    error_message: Optional[str] = Field(None, description="Error message if failed")


# ============================================================================
# Full Deployment Operation
# ============================================================================


class ModelFullDeploymentInput(BaseModel):
    """
    Input for full deployment operation (all steps).

    Attributes:
        package_data: Docker image package data
        sender_auth: Authentication credentials
        deployment_config: Container deployment configuration
        correlation_id: Correlation ID for tracking
    """

    package_data: ModelPackageData = Field(..., description="Docker image package data")

    sender_auth: ModelAuthCredentials = Field(
        ..., description="Sender authentication credentials"
    )

    deployment_config: ModelDeploymentConfig = Field(
        ..., description="Container deployment configuration"
    )

    correlation_id: UUID = Field(..., description="Correlation ID for tracking")


class ModelFullDeploymentOutput(BaseModel):
    """
    Output for full deployment operation.

    Attributes:
        success: Overall deployment success
        deployment_success: Whether deployment succeeded
        container_id: Docker container ID (full SHA256)
        container_short_id: Docker container ID (short 12 chars)
        container_url: Container service URL
        health_status: Health check result
        image_loaded: Whether image was loaded
        image_id: Docker image ID
        kafka_events_published: List of published Kafka events
        execution_time_ms: Total execution time in milliseconds
        error_message: Error message if failed
        error_details: Detailed error information
    """

    success: bool = Field(..., description="Overall deployment success")

    deployment_success: bool = Field(
        default=False, description="Whether deployment succeeded"
    )

    container_id: Optional[str] = Field(
        None, pattern=r"^[a-f0-9]{64}$", description="Docker container ID (full SHA256)"
    )

    container_short_id: Optional[str] = Field(
        None,
        pattern=r"^[a-f0-9]{12}$",
        description="Docker container ID (short 12 chars)",
    )

    container_url: Optional[str] = Field(None, description="Container service URL")

    health_status: Optional[ModelHealthCheckResult] = Field(
        None, description="Health check result"
    )

    image_loaded: bool = Field(default=False, description="Whether image was loaded")

    image_id: Optional[str] = Field(None, description="Docker image ID")

    kafka_events_published: list[str] = Field(
        default_factory=list, description="List of published Kafka events"
    )

    execution_time_ms: int = Field(
        ..., ge=0, description="Total execution time in milliseconds"
    )

    error_message: Optional[str] = Field(None, description="Error message if failed")

    error_details: Optional[dict] = Field(
        None, description="Detailed error information"
    )


__all__ = [
    "ModelPackageData",
    "ModelPackageReceiveInput",
    "ModelPackageReceiveOutput",
    "ModelImageLoadInput",
    "ModelImageLoadOutput",
    "ModelContainerDeployInput",
    "ModelContainerDeployOutput",
    "ModelHealthCheckInput",
    "ModelHealthCheckOutput",
    "ModelDeploymentEventInput",
    "ModelDeploymentEventOutput",
    "ModelFullDeploymentInput",
    "ModelFullDeploymentOutput",
]
