"""Metadata stamp entity model.

Maps to the metadata_stamps database table for audit trail of stamp operations.
"""

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelMetadataStamp(BaseModel):
    """
    Metadata stamp entity model (maps to metadata_stamps table).

    Provides audit trail for metadata stamp operations, tracking stamped content
    with BLAKE3 hashes and complete stamp data.

    Attributes:
        id: Auto-generated primary key (UUID)
        workflow_id: Optional foreign key to parent workflow execution
        file_hash: BLAKE3 hash of stamped content (up to 128 chars)
        stamp_data: Complete stamp data as JSONB
        namespace: Multi-tenant namespace identifier
        created_at: Record creation timestamp (auto-managed)

    Example:
        ```python
        from uuid import uuid4

        stamp = ModelMetadataStamp(
            workflow_id=uuid4(),
            file_hash="abc123def456...",
            stamp_data={
                "content": "Hello World",
                "stamped_at": "2025-10-08T12:00:00Z",
                "version": "1.0"
            },
            namespace="test_app"
        )
        ```
    """

    # Auto-generated primary key
    id: Optional[UUID] = Field(
        default=None, description="Auto-generated UUID primary key"
    )

    # Optional foreign key
    workflow_id: Optional[UUID] = Field(
        default=None, description="Optional foreign key to parent workflow execution"
    )

    # Required fields
    file_hash: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="BLAKE3 hash of stamped content",
        examples=["abc123def456789..."],
    )
    stamp_data: dict[str, Any] = Field(..., description="Complete stamp data as JSONB")
    namespace: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Multi-tenant namespace identifier",
        examples=["test_app", "production_app"],
    )

    # Auto-managed timestamp
    created_at: Optional[datetime] = Field(
        default=None, description="Record creation timestamp (auto-managed by DB)"
    )

    @field_validator("file_hash")
    @classmethod
    def validate_file_hash(cls, v: str) -> str:
        """Validate file_hash is a valid hex string."""
        if not v:
            raise ValueError("file_hash cannot be empty")

        # Check if it's a valid hex string (allowing both upper and lower case)
        try:
            int(v, 16)
        except ValueError:
            raise ValueError(f"file_hash must be a valid hexadecimal string, got: {v}")

        return v.lower()  # Normalize to lowercase

    @field_validator("stamp_data")
    @classmethod
    def validate_stamp_data(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate stamp_data is not empty."""
        if not v:
            raise ValueError("stamp_data cannot be empty")
        return v

    model_config = ConfigDict(
        from_attributes=True,  # Enable ORM mode for DB row mapping
        json_schema_extra={
            "example": {
                "workflow_id": "550e8400-e29b-41d4-a716-446655440000",
                "file_hash": "abc123def456789fedcba987654321",  # pragma: allowlist secret
                "stamp_data": {
                    "content": "Hello World",
                    "stamped_at": "2025-10-08T12:00:00Z",
                    "version": "1.0",
                    "intelligence_data": {
                        "quality_score": 0.95,
                        "onex_compliance": True,
                    },
                },
                "namespace": "test_app",
            }
        },
    )
