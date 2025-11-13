"""
Container package input model for Docker image building and packaging.

ONEX v2.0 Pydantic Model
"""

from typing import ClassVar, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class ModelContainerPackageInput(BaseModel):
    """
    Input model for package_container IO operation.

    Builds Docker image from Dockerfile and creates deployment package
    with compression and BLAKE3 checksum generation.
    """

    container_name: str = Field(
        ...,
        description="Name of the Docker container to build/deploy",
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
        min_length=1,
        max_length=128,
    )

    image_tag: str = Field(
        ...,
        description="Docker image tag (version)",
        pattern=r"^[a-z0-9][a-z0-9._-]*$",
        min_length=1,
        max_length=64,
    )

    dockerfile_path: str = Field(
        default="Dockerfile",
        description="Path to Dockerfile (relative to build context)",
        min_length=1,
        max_length=512,
    )

    build_context: str = Field(
        default=".",
        description="Build context directory path",
        min_length=1,
        max_length=1024,
    )

    build_args: Optional[dict[str, str]] = Field(
        default=None,
        description="Docker build arguments (ARG directives)",
    )

    compression: str = Field(
        default="gzip",
        description="Compression algorithm for package",
    )

    verify_checksum: bool = Field(
        default=True,
        description="Verify package integrity with BLAKE3 checksum",
    )

    correlation_id: Optional[UUID] = Field(
        default=None,
        description="Optional correlation ID for tracking",
    )

    deployment_metadata: Optional[dict[str, str]] = Field(
        default=None,
        description="Additional deployment metadata",
    )

    @field_validator("compression")
    @classmethod
    def validate_compression(cls, v: str) -> str:
        """Validate compression algorithm."""
        allowed = ["gzip", "zstd", "none"]
        if v not in allowed:
            raise ValueError(f"Compression must be one of {allowed}, got: {v}")
        return v

    @field_validator("build_context")
    @classmethod
    def validate_build_context(cls, v: str) -> str:
        """Validate build context path to prevent directory traversal."""
        if ".." in v:
            raise ValueError("Build context cannot contain '..' (directory traversal)")
        return v

    class Config:
        """Pydantic configuration."""

        json_schema_extra: ClassVar[dict] = {
            "example": {
                "container_name": "omninode-bridge-orchestrator",
                "image_tag": "1.0.0",
                "dockerfile_path": "deployment/Dockerfile.orchestrator",
                "build_context": "/Volumes/PRO-G40/Code/omninode_bridge",
                "build_args": {
                    "PYTHON_VERSION": "3.11",
                    "SERVICE_NAME": "orchestrator",
                },
                "compression": "gzip",
                "verify_checksum": True,
                "deployment_metadata": {
                    "environment": "production",
                    "target_host": "192.168.86.200",
                },
            }
        }
