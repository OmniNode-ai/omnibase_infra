"""Audit log entry model."""

from datetime import datetime

from pydantic import BaseModel, Field

from .model_security_event_details import ModelSecurityEventDetails
from .model_security_event_metadata import ModelSecurityEventMetadata


class ModelAuditLogEntry(BaseModel):
    """Complete audit log entry structure."""

    # Core event data
    event_details: ModelSecurityEventDetails = Field(description="Event details")
    metadata: ModelSecurityEventMetadata | None = Field(
        default=None, description="Event metadata",
    )

    # Audit trail
    created_at: datetime = Field(description="Log entry creation timestamp")
    hash_chain_value: str = Field(
        description="Hash chain value for integrity verification",
    )
    previous_hash: str | None = Field(
        default=None, description="Previous entry hash for chain verification",
    )

    # Processing status
    is_processed: bool = Field(
        default=False, description="Whether the event has been processed",
    )
    processing_notes: list[str] = Field(
        default_factory=list, description="Processing notes and actions taken",
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
