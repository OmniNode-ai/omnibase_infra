"""
Package transfer output model for remote deployment results.

ONEX v2.0 Pydantic Model
"""

from typing import ClassVar, Optional

from pydantic import BaseModel, Field


class ModelPackageTransferOutput(BaseModel):
    """
    Output model for transfer_package IO operation.

    Contains transfer results, remote deployment ID, and performance metrics.
    """

    success: bool = Field(
        ...,
        description="Whether the transfer operation succeeded",
    )

    transfer_success: bool = Field(
        ...,
        description="Package transfer success status",
    )

    transfer_duration_ms: Optional[int] = Field(
        default=None,
        description="Package transfer duration in milliseconds",
        ge=0,
    )

    bytes_transferred: Optional[int] = Field(
        default=None,
        description="Number of bytes transferred",
        ge=0,
    )

    transfer_throughput_mbps: Optional[float] = Field(
        default=None,
        description="Transfer throughput in megabytes per second",
        ge=0.0,
    )

    remote_deployment_id: Optional[str] = Field(
        default=None,
        description="Deployment ID returned by remote receiver",
        max_length=128,
    )

    checksum_verified: bool = Field(
        default=False,
        description="Whether checksum was verified on remote receiver",
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
                "transfer_success": True,
                "transfer_duration_ms": 7500,
                "bytes_transferred": 268435456,  # 256MB
                "transfer_throughput_mbps": 34.13,
                "remote_deployment_id": "deployment-2025-10-25-001",
                "checksum_verified": True,
            }
        }
