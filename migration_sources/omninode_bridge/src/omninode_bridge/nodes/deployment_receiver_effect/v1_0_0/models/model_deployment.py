#!/usr/bin/env python3
"""
Deployment configuration models for deployment receiver effect node.
ONEX v2.0 compliant data models for Docker container deployment.
"""

from enum import Enum
from typing import ClassVar, Optional

from pydantic import BaseModel, Field, field_validator


class EnumRestartPolicy(str, Enum):
    """Docker container restart policy."""

    NO = "no"
    ALWAYS = "always"
    ON_FAILURE = "on-failure"
    UNLESS_STOPPED = "unless-stopped"


class EnumHealthStatus(str, Enum):
    """Container health status."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    NOT_STARTED = "not_started"


class ModelVolumeMount(BaseModel):
    """
    Docker volume mount specification.

    Attributes:
        host_path: Path on host system
        container_path: Path inside container
        mode: Mount mode (ro/rw)
    """

    host_path: str = Field(..., description="Path on host system")

    container_path: str = Field(..., description="Path inside container")

    mode: str = Field(
        default="rw",
        pattern=r"^(ro|rw)$",
        description="Mount mode: ro (read-only) or rw (read-write)",
    )

    class Config:
        json_schema_extra: ClassVar[dict] = {
            "example": {
                "host_path": "/var/run/docker.sock",
                "container_path": "/var/run/docker.sock",
                "mode": "rw",
            }
        }


class ModelResourceLimits(BaseModel):
    """
    Container resource limits.

    Attributes:
        cpu_limit: CPU limit (e.g., '1.5', '2.0')
        memory_limit: Memory limit (e.g., '512m', '1g')
    """

    cpu_limit: Optional[str] = Field(
        None, pattern=r"^\d+(\.\d+)?$", description="CPU limit as decimal string"
    )

    memory_limit: Optional[str] = Field(
        None,
        pattern=r"^\d+(k|m|g|K|M|G)$",
        description="Memory limit with unit (k/m/g)",
    )

    class Config:
        json_schema_extra: ClassVar[dict] = {
            "example": {"cpu_limit": "1.5", "memory_limit": "512m"}
        }


class ModelDeploymentConfig(BaseModel):
    """
    Complete container deployment configuration.

    Attributes:
        image_name: Docker image name with tag
        container_name: Container name for deployment
        ports: Port mappings (container:host)
        environment_vars: Environment variables
        volumes: Volume mount specifications
        networks: Docker networks to attach
        restart_policy: Container restart policy
        resource_limits: CPU/memory resource limits
    """

    image_name: str = Field(
        ...,
        pattern=r"^[a-z0-9_-]+/[a-z0-9_-]+:[a-z0-9._-]+$",
        description="Docker image name with tag",
    )

    container_name: str = Field(
        ..., pattern=r"^[a-z0-9_-]+$", description="Container name for deployment"
    )

    ports: Optional[dict[str, int]] = Field(
        default_factory=dict, description="Port mappings (container:host)"
    )

    environment_vars: Optional[dict[str, str]] = Field(
        default_factory=dict, description="Environment variables"
    )

    volumes: Optional[list[ModelVolumeMount]] = Field(
        default_factory=list, description="Volume mount specifications"
    )

    networks: Optional[list[str]] = Field(
        default_factory=list, description="Docker networks to attach"
    )

    restart_policy: EnumRestartPolicy = Field(
        default=EnumRestartPolicy.UNLESS_STOPPED, description="Container restart policy"
    )

    resource_limits: Optional[ModelResourceLimits] = Field(
        None, description="CPU/memory resource limits"
    )

    @field_validator("ports")
    @classmethod
    def validate_ports(cls, v: Optional[dict[str, int]]) -> Optional[dict[str, int]]:
        """Validate port mappings."""
        if v:
            for container_port, host_port in v.items():
                try:
                    int(container_port)
                except ValueError as e:
                    raise ValueError(f"Invalid container port: {container_port}") from e

                if not (1 <= host_port <= 65535):
                    raise ValueError(f"Host port must be 1-65535, got {host_port}")
        return v

    class Config:
        json_schema_extra: ClassVar[dict] = {
            "example": {
                "image_name": "omninode/orchestrator:v1.0.0",
                "container_name": "omninode-orchestrator",
                "ports": {"8060": 8060},
                "environment_vars": {
                    "LOG_LEVEL": "INFO",
                    "POSTGRES_HOST": "192.168.86.200",
                },
                "volumes": [
                    {
                        "host_path": "/var/run/docker.sock",
                        "container_path": "/var/run/docker.sock",
                        "mode": "rw",
                    }
                ],
                "networks": ["omninode-bridge"],
                "restart_policy": "unless-stopped",
                "resource_limits": {"cpu_limit": "1.5", "memory_limit": "512m"},
            }
        }


class ModelHealthCheckResult(BaseModel):
    """
    Container health check result.

    Attributes:
        is_healthy: Whether container is healthy
        status: Health status enum
        checks_passed: Number of health checks passed
        last_check_time: ISO timestamp of last check
        error_message: Error message if unhealthy
    """

    is_healthy: bool = Field(..., description="Whether container is healthy")

    status: EnumHealthStatus = Field(..., description="Container health status")

    checks_passed: int = Field(
        default=0, ge=0, description="Number of health checks passed"
    )

    last_check_time: Optional[str] = Field(
        None, description="ISO timestamp of last health check"
    )

    error_message: Optional[str] = Field(None, description="Error message if unhealthy")

    class Config:
        json_schema_extra: ClassVar[dict] = {
            "example": {
                "is_healthy": True,
                "status": "healthy",
                "checks_passed": 3,
                "last_check_time": "2025-10-25T17:30:00Z",
                "error_message": None,
            }
        }


__all__ = [
    "EnumRestartPolicy",
    "EnumHealthStatus",
    "ModelVolumeMount",
    "ModelResourceLimits",
    "ModelDeploymentConfig",
    "ModelHealthCheckResult",
]
