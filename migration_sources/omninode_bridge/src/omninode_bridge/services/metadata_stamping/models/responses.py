"""Response models for metadata stamping API."""

from datetime import datetime
from typing import Any, Optional, Union

from pydantic import BaseModel, Field


class PerformanceMetrics(BaseModel):
    """Performance metrics for operations."""

    execution_time_ms: float = Field(..., description="Execution time in milliseconds")
    file_size_bytes: Optional[int] = Field(
        default=None, description="File size in bytes"
    )
    cpu_usage_percent: Optional[float] = Field(
        default=None, description="CPU usage percentage"
    )
    performance_grade: str = Field(..., description="Performance grade (A, B, or C)")


class StampResponse(BaseModel):
    """Response model for stamping operations."""

    success: bool = Field(..., description="Operation success status")
    stamp_id: Optional[str] = Field(
        default=None, description="Created stamp identifier"
    )
    file_hash: str = Field(..., description="BLAKE3 content hash")
    stamped_content: str = Field(..., description="Content with stamps applied")
    stamp: str = Field(..., description="Generated stamp")
    stamp_type: str = Field(..., description="Type of stamp created")
    performance_metrics: PerformanceMetrics = Field(
        ..., description="Operation performance data"
    )
    created_at: Optional[datetime] = Field(
        default=None, description="Stamp creation timestamp"
    )
    # Compliance fields from omnibase_3 and ai-dev patterns
    op_id: Optional[str] = Field(default=None, description="Operation ID")
    namespace: str = Field(
        default="omninode.services.metadata", description="Namespace"
    )
    version: int = Field(default=1, description="Schema version")
    metadata_version: str = Field(default="0.1", description="Metadata format version")


class ValidationDetail(BaseModel):
    """Detailed validation result for individual stamp."""

    stamp_type: str = Field(..., description="Type of stamp validated")
    stamp_hash: str = Field(..., description="Hash from stamp")
    is_valid: bool = Field(..., description="Validation status")
    current_hash: str = Field(..., description="Current content hash")


class ValidationResponse(BaseModel):
    """Response model for validation operations."""

    success: bool = Field(..., description="Operation success status")
    is_valid: bool = Field(..., description="Overall validation status")
    stamps_found: int = Field(..., description="Number of stamps found")
    current_hash: str = Field(..., description="Current content hash")
    validation_details: list[ValidationDetail] = Field(
        ..., description="Detailed validation results"
    )
    performance_metrics: PerformanceMetrics = Field(
        ..., description="Operation performance data"
    )


class HashResponse(BaseModel):
    """Response model for hash generation operations."""

    file_hash: str = Field(..., description="BLAKE3 hash")
    execution_time_ms: float = Field(..., description="Hash generation time")
    file_size_bytes: int = Field(..., description="File size in bytes")
    performance_grade: str = Field(..., description="Performance grade")


class ComponentHealth(BaseModel):
    """Health status of individual component."""

    status: str = Field(..., description="Component health status")
    response_time_ms: Optional[float] = Field(default=None, description="Response time")
    details: Optional[dict[str, Any]] = Field(
        default=None, description="Additional details"
    )


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Overall service health")
    components: dict[str, ComponentHealth] = Field(
        ..., description="Individual component health"
    )
    uptime_seconds: Optional[float] = Field(default=None, description="Service uptime")
    version: str = Field(default="0.1.0", description="Service version")


# Unified Response Format Models


class ErrorDetail(BaseModel):
    """Detailed error information."""

    code: str = Field(..., description="Error code")
    field: Optional[str] = Field(default=None, description="Field causing error")
    message: str = Field(..., description="Human-readable error message")


class UnifiedResponse(BaseModel):
    """Unified response format for all API endpoints."""

    status: str = Field(..., description="Response status: success, error, or partial")
    data: Optional[Any] = Field(default=None, description="Response data")
    error: Optional[Union[str, list[ErrorDetail]]] = Field(
        default=None, description="Error message or details"
    )
    message: Optional[str] = Field(default=None, description="Additional message")
    metadata: Optional[dict[str, Any]] = Field(
        default=None, description="Response metadata"
    )


# Batch Operation Models


class BatchStampItem(BaseModel):
    """Individual item for batch stamping."""

    id: str = Field(..., description="Item identifier")
    content: str = Field(..., description="Content to stamp")
    file_path: Optional[str] = Field(default=None, description="Optional file path")
    namespace: Optional[str] = Field(default="default", description="Namespace")
    metadata: Optional[dict[str, Any]] = Field(
        default=None, description="Additional metadata"
    )


class BatchStampResult(BaseModel):
    """Result for individual batch stamp item."""

    id: str = Field(..., description="Item identifier")
    success: bool = Field(..., description="Operation success status")
    stamp_id: Optional[str] = Field(
        default=None, description="Created stamp identifier"
    )
    file_hash: Optional[str] = Field(default=None, description="BLAKE3 content hash")
    stamp: Optional[str] = Field(default=None, description="Generated stamp")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    performance_metrics: Optional[PerformanceMetrics] = Field(
        default=None, description="Operation performance data"
    )


class BatchStampResponse(BaseModel):
    """Response for batch stamping operations."""

    total_items: int = Field(..., description="Total items processed")
    successful_items: int = Field(..., description="Successfully processed items")
    failed_items: int = Field(..., description="Failed items")
    results: list[BatchStampResult] = Field(..., description="Individual results")
    overall_performance: PerformanceMetrics = Field(
        ..., description="Overall operation performance"
    )


# Protocol Validation Models


class ProtocolValidationResult(BaseModel):
    """Protocol validation result."""

    is_valid: bool = Field(..., description="Protocol compliance status")
    protocol_version: str = Field(..., description="Detected protocol version")
    compliance_level: str = Field(
        ..., description="Compliance level: full, partial, none"
    )
    issues: list[str] = Field(
        default_factory=list, description="Compliance issues found"
    )
    recommendations: list[str] = Field(
        default_factory=list, description="Improvement recommendations"
    )


class ProtocolValidationResponse(BaseModel):
    """Response for protocol validation operations."""

    validation_result: ProtocolValidationResult = Field(
        ..., description="Validation results"
    )
    performance_metrics: PerformanceMetrics = Field(
        ..., description="Operation performance data"
    )


# Namespace Query Models


class NamespaceStamp(BaseModel):
    """Stamp information for namespace queries."""

    stamp_id: str = Field(..., description="Stamp identifier")
    file_hash: str = Field(..., description="BLAKE3 content hash")
    file_path: str = Field(..., description="File path")
    stamp_type: str = Field(..., description="Type of stamp")
    created_at: datetime = Field(..., description="Creation timestamp")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Stamp metadata")
    # Compliance fields from omnibase_3 and ai-dev patterns
    op_id: str = Field(..., description="Operation ID")
    namespace: str = Field(
        default="omninode.services.metadata", description="Namespace"
    )
    version: int = Field(default=1, description="Schema version")
    metadata_version: str = Field(default="0.1", description="Metadata format version")
    intelligence_data: dict[str, Any] = Field(
        default_factory=dict, description="Intelligence data"
    )


class NamespaceQueryResponse(BaseModel):
    """Response for namespace queries."""

    namespace: str = Field(..., description="Queried namespace")
    total_stamps: int = Field(..., description="Total stamps in namespace")
    stamps: list[NamespaceStamp] = Field(..., description="Stamp details")
    pagination: Optional[dict[str, Any]] = Field(
        default=None, description="Pagination information"
    )
