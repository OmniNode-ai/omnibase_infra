"""Credential Cache Entry Model.

Strongly-typed model for credential cache entries.
Maintains ONEX compliance with proper field validation.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from .enum_credential_type import EnumCredentialType
from .enum_deployment_environment import EnumDeploymentEnvironment


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
    cached_at: datetime = Field(
        description="Timestamp when credential was cached",
    )

    expires_at: datetime | None = Field(
        default=None,
        description="Timestamp when credential expires",
    )

    last_validated: datetime | None = Field(
        default=None,
        description="Timestamp of last validation",
    )

    # Usage Tracking
    access_count: int = Field(
        default=0,
        ge=0,
        description="Number of times credential was accessed",
    )

    last_accessed: datetime | None = Field(
        default=None,
        description="Timestamp of last access",
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