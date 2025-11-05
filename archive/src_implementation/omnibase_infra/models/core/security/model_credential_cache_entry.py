"""Credential Cache Entry Model.

Strongly-typed model for credential cache entry to replace Dict[str, Any] usage.
Maintains ONEX compliance with proper field validation.
"""

from uuid import UUID

from pydantic import BaseModel, Field

from omnibase_infra.enums.enum_credential_type import EnumCredentialType
from omnibase_infra.enums.enum_deployment_environment import EnumDeploymentEnvironment


class ModelCredentialCacheEntry(BaseModel):
    """Model for credential cache entry."""

    # Credential Information
    credential_type: EnumCredentialType = Field(
        description="Type of credential",
    )

    credential_id: UUID = Field(
        description="Unique identifier for the credential",
    )

    environment: EnumDeploymentEnvironment = Field(
        description="Environment the credential is for",
    )

    # Cache Metadata
    cached_at: str = Field(
        description="ISO timestamp when credential was cached",
    )

    expires_at: str | None = Field(
        default=None,
        description="ISO timestamp when credential expires",
    )

    last_validated: str | None = Field(
        default=None,
        description="ISO timestamp of last validation",
    )

    # Usage Tracking
    access_count: int = Field(
        default=0,
        ge=0,
        description="Number of times credential was accessed",
    )

    last_accessed: str | None = Field(
        default=None,
        description="ISO timestamp of last access",
    )

    # Security Status
    is_valid: bool = Field(
        default=True,
        description="Whether credential is currently valid",
    )

    validation_failures: int = Field(
        default=0,
        ge=0,
        description="Number of validation failures",
    )

    is_revoked: bool = Field(
        default=False,
        description="Whether credential has been revoked",
    )

    # Additional Metadata
    source: str | None = Field(
        default=None,
        max_length=100,
        description="Source of the credential",
    )

    scope: list[str] | None = Field(
        default=None,
        max_items=20,
        description="Scopes associated with the credential",
    )

    class Config:
        """Pydantic model configuration."""

        validate_assignment = True
        extra = "forbid"
