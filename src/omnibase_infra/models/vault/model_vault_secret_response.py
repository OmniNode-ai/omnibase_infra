"""Vault secret response model."""

from pydantic import BaseModel, Field


class ModelVaultSecretResponse(BaseModel):
    """Response model for Vault secret operations."""

    success: bool = Field(
        description="Whether the operation was successful"
    )

    data: dict | None = Field(
        default=None,
        description="Secret data retrieved from Vault"
    )

    metadata: dict | None = Field(
        default=None,
        description="Secret metadata (version, created_time, etc.)"
    )

    version: int | None = Field(
        default=None,
        description="Secret version"
    )

    error: str | None = Field(
        default=None,
        description="Error message if operation failed"
    )

    lease_id: str | None = Field(
        default=None,
        description="Lease ID for dynamic secrets"
    )

    lease_duration: int | None = Field(
        default=None,
        description="Lease duration in seconds"
    )

    renewable: bool = Field(
        default=False,
        description="Whether the secret lease is renewable"
    )

    correlation_id: str = Field(
        description="Correlation ID from request"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "data": {"username": "admin", "password": "***"},
                "metadata": {"version": 1, "created_time": "2025-11-14T12:00:00Z"},
                "version": 1,
                "correlation_id": "req_123456"
            }
        }
