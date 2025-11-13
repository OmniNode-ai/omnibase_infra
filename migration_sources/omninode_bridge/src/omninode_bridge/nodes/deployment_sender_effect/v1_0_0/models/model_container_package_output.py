"""
Container package output model for Docker image build results.

ONEX v2.0 Pydantic Model
"""

from typing import ClassVar, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ModelContainerPackageOutput(BaseModel):
    """
    Output model for package_container IO operation.

    Contains build results, package metadata, and checksum information.
    """

    success: bool = Field(
        ...,
        description="Whether the packaging operation succeeded",
    )

    package_id: UUID = Field(
        ...,
        description="Unique deployment package identifier",
    )

    image_id: Optional[str] = Field(
        default=None,
        description="Docker image ID (sha256:...)",
        pattern=r"^sha256:[a-f0-9]{64}$",
    )

    package_path: Optional[str] = Field(
        default=None,
        description="Local filesystem path to created package",
        min_length=1,
        max_length=1024,
    )

    package_size_mb: Optional[float] = Field(
        default=None,
        description="Deployment package size in megabytes",
        ge=0.0,
    )

    package_checksum: Optional[str] = Field(
        default=None,
        description="BLAKE3 checksum of deployment package",
        pattern=r"^[a-f0-9]{64}$",
    )

    build_duration_ms: Optional[int] = Field(
        default=None,
        description="Docker image build duration in milliseconds",
        ge=0,
    )

    compression_ratio: Optional[float] = Field(
        default=None,
        description="Compression ratio (compressed_size / original_size)",
        ge=0.0,
        le=1.0,
    )

    error_message: Optional[str] = Field(
        default=None,
        description="Error message if operation failed",
        max_length=2048,
    )

    error_code: Optional[str] = Field(
        default=None,
        description="Error code for failed operations",
        max_length=64,
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra: ClassVar[dict] = {
            "example": {
                "success": True,
                "package_id": "123e4567-e89b-12d3-a456-426614174000",
                "image_id": "sha256:a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2",
                "package_path": "/tmp/packages/orchestrator-1.0.0.tar.gz",
                "package_size_mb": 256.5,
                "package_checksum": "1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",  # pragma: allowlist secret
                "build_duration_ms": 12500,
                "compression_ratio": 0.35,
            }
        }
