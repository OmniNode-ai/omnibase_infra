"""Vault token request model."""

from pydantic import BaseModel, Field


class ModelVaultTokenRequest(BaseModel):
    """Request model for Vault token operations."""

    operation: str = Field(
        description="Token operation: 'create', 'renew', 'revoke', 'lookup'"
    )

    token: str | None = Field(
        default=None,
        description="Token to renew/revoke/lookup (not needed for create)"
    )

    policies: list[str] | None = Field(
        default=None,
        description="Policies to attach to new token"
    )

    ttl: str | None = Field(
        default=None,
        description="Time-to-live for token (e.g., '1h', '24h')"
    )

    renewable: bool = Field(
        default=True,
        description="Whether token should be renewable"
    )

    metadata: dict | None = Field(
        default=None,
        description="Metadata to attach to token"
    )

    correlation_id: str = Field(
        description="Correlation ID for request tracking"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "operation": "create",
                "policies": ["default", "app-readonly"],
                "ttl": "1h",
                "renewable": True,
                "correlation_id": "req_123456"
            }
        }
