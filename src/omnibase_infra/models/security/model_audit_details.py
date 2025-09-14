"""Audit Details Model.

Strongly-typed model for audit event details to replace Dict[str, Any] usage.
Maintains ONEX compliance with proper field validation and security measures.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Union
from datetime import datetime
from uuid import UUID


class ModelAuditDetails(BaseModel):
    """Model for audit event details with comprehensive typing."""

    # Request/Response information
    request_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Request identifier"
    )

    response_status: Optional[int] = Field(
        default=None,
        ge=100,
        le=599,
        description="HTTP response status code"
    )

    response_time_ms: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Response time in milliseconds"
    )

    # Resource information
    resource_id: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Identifier of the affected resource"
    )

    resource_type: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Type of resource being accessed"
    )

    resource_path: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Path to the resource"
    )

    # Authentication/Authorization information
    user_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="User identifier (non-sensitive)"
    )

    session_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Session identifier (hashed)"
    )

    permissions_checked: Optional[List[str]] = Field(
        default=None,
        max_items=50,
        description="List of permissions that were verified"
    )

    authentication_method: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Authentication method used"
    )

    # Operation details
    operation_name: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Name of the operation performed"
    )

    operation_parameters: Optional[List[str]] = Field(
        default=None,
        max_items=20,
        description="Operation parameters (sanitized)"
    )

    data_modified: Optional[bool] = Field(
        default=None,
        description="Whether data was modified by this operation"
    )

    records_affected: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of records affected"
    )

    # Error information
    error_code: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Error code if operation failed"
    )

    error_category: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Category of error"
    )

    error_context: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Additional error context (sanitized)"
    )

    # Security relevant information
    security_violation_type: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Type of security violation detected"
    )

    suspicious_activity: Optional[bool] = Field(
        default=None,
        description="Whether activity was flagged as suspicious"
    )

    threat_level: Optional[str] = Field(
        default=None,
        pattern="^(low|medium|high|critical)$",
        description="Assessed threat level"
    )

    # Compliance information
    compliance_requirements: Optional[List[str]] = Field(
        default=None,
        max_items=10,
        description="Applicable compliance requirements"
    )

    data_classification: Optional[str] = Field(
        default=None,
        pattern="^(public|internal|confidential|restricted)$",
        description="Classification of data accessed"
    )

    retention_period_days: Optional[int] = Field(
        default=None,
        ge=1,
        le=3650,
        description="Required retention period in days"
    )

    # Additional context
    environment: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Environment where event occurred"
    )

    service_version: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Version of the service"
    )

    correlation_id: Optional[UUID] = Field(
        default=None,
        description="Correlation ID for request tracing"
    )

    custom_fields: Optional[List[str]] = Field(
        default=None,
        max_items=10,
        description="Additional custom field names (values omitted for security)"
    )

    class Config:
        """Pydantic model configuration."""
        validate_assignment = True
        extra = "forbid"
        json_schema_extra = {
            "example": {
                "request_id": "req-123456",
                "response_status": 200,
                "response_time_ms": 42.5,
                "resource_id": "user:12345",
                "resource_type": "user_profile",
                "user_id": "user-abc123",
                "operation_name": "update_profile",
                "data_modified": True,
                "records_affected": 1,
                "environment": "production",
                "data_classification": "confidential"
            }
        }


class ModelAuditMetadata(BaseModel):
    """Model for audit event metadata with comprehensive typing."""

    # Processing information
    processing_node: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Node that processed this audit event"
    )

    processing_time: Optional[datetime] = Field(
        default=None,
        description="When audit event was processed"
    )

    batch_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Batch identifier if processed in batch"
    )

    # Storage information
    storage_location: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Where audit event is stored"
    )

    compression_used: Optional[bool] = Field(
        default=None,
        description="Whether compression was applied"
    )

    encryption_used: Optional[bool] = Field(
        default=None,
        description="Whether encryption was applied"
    )

    # Quality information
    data_quality_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Data quality score (0-1)"
    )

    completeness_percentage: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Data completeness percentage"
    )

    validation_passed: Optional[bool] = Field(
        default=None,
        description="Whether validation passed"
    )

    # Alerting information
    alert_triggered: Optional[bool] = Field(
        default=None,
        description="Whether event triggered an alert"
    )

    alert_severity: Optional[str] = Field(
        default=None,
        pattern="^(info|warning|error|critical)$",
        description="Alert severity level"
    )

    notification_sent: Optional[bool] = Field(
        default=None,
        description="Whether notification was sent"
    )

    # Archival information
    archival_required: Optional[bool] = Field(
        default=None,
        description="Whether event requires archival"
    )

    archival_date: Optional[datetime] = Field(
        default=None,
        description="When event should be archived"
    )

    retention_policy: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Applicable retention policy"
    )

    class Config:
        """Pydantic model configuration."""
        validate_assignment = True
        extra = "forbid"
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }