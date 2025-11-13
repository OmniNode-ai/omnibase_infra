"""
Package transfer input model for remote deployment transmission.

ONEX v2.0 Pydantic Model
"""

from typing import ClassVar, Optional
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl, field_validator


class ModelPackageTransferInput(BaseModel):
    """
    Input model for transfer_package IO operation.

    Transfers deployment package to remote receiver via HTTP or rsync.
    """

    package_id: UUID = Field(
        ...,
        description="Unique deployment package identifier",
    )

    package_path: str = Field(
        ...,
        description="Local filesystem path to package file",
        min_length=1,
        max_length=1024,
    )

    package_checksum: str = Field(
        ...,
        description="BLAKE3 checksum of package for verification",
        pattern=r"^[a-f0-9]{64}$",
    )

    remote_receiver_url: HttpUrl = Field(
        ...,
        description="Remote deployment receiver endpoint",
    )

    transfer_method: str = Field(
        default="http",
        description="Transfer method for deployment package",
    )

    verify_checksum: bool = Field(
        default=True,
        description="Verify package integrity after transfer",
    )

    correlation_id: Optional[UUID] = Field(
        default=None,
        description="Optional correlation ID for tracking",
    )

    container_name: Optional[str] = Field(
        default=None,
        description="Name of the container being deployed",
        min_length=1,
        max_length=128,
    )

    image_tag: Optional[str] = Field(
        default=None,
        description="Docker image tag being deployed",
        min_length=1,
        max_length=64,
    )

    @field_validator("transfer_method")
    @classmethod
    def validate_transfer_method(cls, v: str) -> str:
        """Validate transfer method."""
        allowed = ["http", "rsync"]
        if v not in allowed:
            raise ValueError(f"Transfer method must be one of {allowed}, got: {v}")
        return v

    @field_validator("package_path")
    @classmethod
    def validate_package_path(cls, v: str) -> str:
        """Validate package path to prevent directory traversal."""
        if ".." in v:
            raise ValueError("Package path cannot contain '..' (directory traversal)")
        return v

    class Config:
        """Pydantic configuration."""

        json_schema_extra: ClassVar[dict] = {
            "example": {
                "package_id": "123e4567-e89b-12d3-a456-426614174000",
                "package_path": "/tmp/packages/orchestrator-1.0.0.tar.gz",
                "package_checksum": "1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",  # pragma: allowlist secret
                "remote_receiver_url": "http://192.168.86.200:8001/deploy",
                "transfer_method": "http",
                "verify_checksum": True,
                "container_name": "omninode-bridge-orchestrator",
                "image_tag": "1.0.0",
            }
        }
