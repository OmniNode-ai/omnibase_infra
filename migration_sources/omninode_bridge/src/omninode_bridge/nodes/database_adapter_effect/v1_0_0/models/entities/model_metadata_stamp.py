#!/usr/bin/env python3
"""
ModelMetadataStamp - Metadata Stamp Entity.

Strongly-typed Pydantic model representing metadata_stamps table.
"""

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omninode_bridge.infrastructure.validation.jsonb_validators import (
    validate_jsonb_fields,
)


class ModelMetadataStamp(BaseModel):
    """Metadata stamp entity (maps to metadata_stamps table)."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        frozen=False,
        extra="forbid",
        populate_by_name=True,
    )

    # Database-generated fields
    id: Optional[int] = Field(default=None, ge=1)

    # Workflow association (nullable)
    workflow_id: Optional[UUID] = Field(default=None)

    # File identity (BLAKE3 produces exactly 64 hex characters)
    file_hash: str = Field(..., min_length=64, max_length=64, pattern=r"^[a-f0-9]{64}$")

    # Stamp data (JSONB)
    stamp_data: dict[str, Any] = Field(..., json_schema_extra={"db_type": "jsonb"})

    # Multi-tenant isolation
    namespace: str = Field(..., min_length=1, max_length=255)

    # Database timestamps
    created_at: Optional[datetime] = Field(default=None)
    updated_at: Optional[datetime] = Field(default=None)

    @model_validator(mode="after")
    def _validate_jsonb_fields(self) -> "ModelMetadataStamp":
        """Validate that all JSONB fields have proper annotations."""
        return validate_jsonb_fields(self)
