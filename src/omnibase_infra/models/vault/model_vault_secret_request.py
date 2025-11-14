"""Vault secret request model."""

from pydantic import BaseModel, Field


class ModelVaultSecretRequest(BaseModel):
    """Request model for Vault secret operations."""

    path: str = Field(
        description="Secret path in Vault (e.g., 'secret/data/myapp/config')"
    )

    operation: str = Field(
        description="Operation type: 'read', 'write', 'delete', 'list'"
    )

    data: dict | None = Field(
        default=None,
        description="Secret data for write operations"
    )

    version: int | None = Field(
        default=None,
        description="Secret version to retrieve (for versioned secrets)"
    )

    mount_path: str = Field(
        default="secret",
        description="Vault mount path for the secrets engine"
    )

    correlation_id: str = Field(
        description="Correlation ID for request tracking"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "path": "secret/data/myapp/db_credentials",
                "operation": "read",
                "mount_path": "secret",
                "correlation_id": "req_123456"
            }
        }
