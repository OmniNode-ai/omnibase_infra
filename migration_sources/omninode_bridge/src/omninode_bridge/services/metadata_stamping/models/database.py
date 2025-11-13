"""Database models for metadata stamping service.

Database record models that map to PostgreSQL table structure.
These models ensure proper data handling and backward compatibility.
"""

from datetime import datetime
from typing import Any, ClassVar, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class MetadataStampRecord(BaseModel):
    """Database record model for metadata_stamps table.

    This model represents the complete database schema including
    all compliance fields from omnibase_3 and ai-dev patterns.
    """

    # Core fields
    id: UUID = Field(..., description="Primary key")
    file_hash: str = Field(..., description="BLAKE3 hash of the file")
    file_path: str = Field(..., description="Path to the file")
    file_size: int = Field(..., description="Size of the file in bytes")
    content_type: Optional[str] = Field(default=None, description="MIME content type")
    stamp_data: dict[str, Any] = Field(..., description="Stamp metadata")
    protocol_version: str = Field(default="1.0", description="Protocol version")

    # Compliance fields from omnibase_3 and ai-dev patterns
    intelligence_data: dict[str, Any] = Field(
        default_factory=dict, description="Intelligence data for enhanced processing"
    )
    version: int = Field(default=1, description="Schema version")
    op_id: UUID = Field(..., description="Operation ID")
    namespace: str = Field(
        default="omninode.services.metadata", description="Namespace"
    )
    metadata_version: str = Field(default="0.1", description="Metadata format version")

    # Timestamps
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    class Config:
        """Pydantic configuration."""

        from_attributes = True
        json_encoders: ClassVar[dict] = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class LegacyMetadataStampRecord(BaseModel):
    """Legacy database record model for backward compatibility.

    This model represents the pre-compliance schema structure
    to ensure existing code continues to work.
    """

    # Core fields only (pre-compliance)
    id: UUID = Field(..., description="Primary key")
    file_hash: str = Field(..., description="BLAKE3 hash of the file")
    file_path: str = Field(..., description="Path to the file")
    file_size: int = Field(..., description="Size of the file in bytes")
    content_type: Optional[str] = Field(default=None, description="MIME content type")
    stamp_data: dict[str, Any] = Field(..., description="Stamp metadata")
    protocol_version: str = Field(default="1.0", description="Protocol version")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    class Config:
        """Pydantic configuration."""

        from_attributes = True
        json_encoders: ClassVar[dict] = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }

    @classmethod
    def from_full_record(
        cls, record: MetadataStampRecord
    ) -> "LegacyMetadataStampRecord":
        """Create legacy record from full record for backward compatibility.

        Args:
            record: Full metadata stamp record with compliance fields

        Returns:
            Legacy record with only core fields
        """
        return cls(
            id=record.id,
            file_hash=record.file_hash,
            file_path=record.file_path,
            file_size=record.file_size,
            content_type=record.content_type,
            stamp_data=record.stamp_data,
            protocol_version=record.protocol_version,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )


class ProtocolHandlerRecord(BaseModel):
    """Database record model for protocol_handlers table."""

    id: UUID = Field(..., description="Primary key")
    handler_type: str = Field(..., description="Handler type identifier")
    file_extensions: list[str] = Field(..., description="Supported file extensions")
    handler_config: Optional[dict[str, Any]] = Field(
        default=None, description="Handler configuration"
    )
    is_active: bool = Field(default=True, description="Handler active status")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    class Config:
        """Pydantic configuration."""

        from_attributes = True
        json_encoders: ClassVar[dict] = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class HashMetricRecord(BaseModel):
    """Database record model for hash_metrics table."""

    id: Optional[UUID] = Field(default=None, description="Primary key")
    operation_type: str = Field(..., description="Type of operation")
    execution_time_ms: int = Field(..., description="Execution time in milliseconds")
    file_size_bytes: Optional[int] = Field(
        default=None, description="File size in bytes"
    )
    cpu_usage_percent: Optional[float] = Field(
        default=None, description="CPU usage percentage"
    )
    memory_usage_mb: Optional[int] = Field(
        default=None, description="Memory usage in MB"
    )
    timestamp: datetime = Field(..., description="Metric timestamp")

    class Config:
        """Pydantic configuration."""

        from_attributes = True
        json_encoders: ClassVar[dict] = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
