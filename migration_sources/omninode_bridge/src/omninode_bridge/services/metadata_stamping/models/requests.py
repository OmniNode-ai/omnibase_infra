"""Request models for metadata stamping API."""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class StampType(str, Enum):
    """Enumeration of stamp types."""

    LIGHTWEIGHT = "lightweight"
    RICH = "rich"


class StampOptions(BaseModel):
    """Options for stamping operations."""

    stamp_type: StampType = Field(
        default=StampType.LIGHTWEIGHT, description="Type of stamp to create"
    )
    include_metadata: bool = Field(
        default=True, description="Include metadata in stamp"
    )
    validate_integrity: bool = Field(
        default=True, description="Validate content integrity"
    )


class ValidationOptions(BaseModel):
    """Options for validation operations."""

    strict_mode: bool = Field(default=True, description="Enable strict validation")
    expected_hash: Optional[str] = Field(
        default=None, description="Expected hash for validation"
    )
    check_tampering: bool = Field(default=True, description="Check for tampering")


class StampRequest(BaseModel):
    """Request model for stamping operations."""

    content: str = Field(..., description="Content to stamp")
    file_path: Optional[str] = Field(default=None, description="Optional file path")
    options: StampOptions = Field(
        default_factory=StampOptions, description="Stamping options"
    )
    metadata: Optional[dict[str, Any]] = Field(
        default=None, description="Additional metadata"
    )
    protocol_version: str = Field(default="1.0", description="Protocol version")
    namespace: Optional[str] = Field(
        default="omninode.services.metadata", description="Namespace"
    )
    # Compliance fields from omnibase_3 and ai-dev patterns
    intelligence_data: Optional[dict[str, Any]] = Field(
        default_factory=dict, description="Intelligence data for enhanced processing"
    )
    version: int = Field(default=1, description="Schema version")
    op_id: Optional[str] = Field(
        default=None, description="Operation ID (auto-generated if None)"
    )
    metadata_version: str = Field(default="0.1", description="Metadata format version")


class ValidationRequest(BaseModel):
    """Request model for validation operations."""

    content: str = Field(..., description="Content to validate")
    options: ValidationOptions = Field(
        default_factory=ValidationOptions, description="Validation options"
    )
    namespace: Optional[str] = Field(default="default", description="Namespace")


# Batch Operation Models


class BatchStampRequest(BaseModel):
    """Request model for batch stamping operations."""

    items: list[dict[str, Any]] = Field(..., description="List of items to stamp")
    options: StampOptions = Field(
        default_factory=StampOptions, description="Default stamping options"
    )
    protocol_version: str = Field(default="1.0", description="Protocol version")
    namespace: Optional[str] = Field(default="default", description="Default namespace")


# Protocol Validation Models


class ProtocolValidationRequest(BaseModel):
    """Request model for protocol validation operations."""

    content: str = Field(..., description="Content to validate against protocol")
    target_protocol: str = Field(
        default="O.N.E.v0.1", description="Target protocol version"
    )
    validation_level: str = Field(
        default="strict", description="Validation level: strict, moderate, lenient"
    )
    namespace: Optional[str] = Field(default="default", description="Namespace")
