"""Strongly typed models for outbox event data."""

from typing import Optional, List
from pydantic import BaseModel, Field


class ModelOutboxEventData(BaseModel):
    """Strongly typed outbox event data structure."""
    
    # Core event data
    event_type: str = Field(description="Type of event being published")
    event_version: str = Field(description="Event schema version")
    entity_id: str = Field(description="ID of the entity that changed")
    entity_type: str = Field(description="Type of entity that changed")
    
    # Event payload
    payload_string: Optional[str] = Field(default=None, description="String payload data")
    payload_number: Optional[float] = Field(default=None, description="Numeric payload data")
    payload_boolean: Optional[bool] = Field(default=None, description="Boolean payload data")
    
    # Metadata
    timestamp: str = Field(description="ISO timestamp of the event")
    correlation_id: Optional[str] = Field(default=None, description="Request correlation ID")
    user_id: Optional[str] = Field(default=None, description="User who triggered the event")
    tenant_id: Optional[str] = Field(default=None, description="Tenant context")
    
    # Additional context
    tags: List[str] = Field(default_factory=list, description="Event tags for categorization")
    metadata_flags: List[str] = Field(default_factory=list, description="Metadata flags")


class ModelOutboxStatistics(BaseModel):
    """Statistics for outbox processing."""
    
    total_events: int = Field(description="Total number of events in outbox")
    pending_events: int = Field(description="Number of pending events")
    processing_events: int = Field(description="Number of events being processed")
    failed_events: int = Field(description="Number of failed events")
    completed_events: int = Field(description="Number of successfully processed events")
    
    # Performance metrics
    average_processing_time_ms: float = Field(description="Average processing time in milliseconds")
    events_per_second: float = Field(description="Current processing rate")
    last_processed_at: Optional[str] = Field(default=None, description="ISO timestamp of last processed event")
    
    # Health indicators
    oldest_pending_age_seconds: Optional[float] = Field(default=None, description="Age of oldest pending event")
    error_rate_percent: float = Field(description="Error rate percentage")
    is_healthy: bool = Field(description="Overall health status")


class ModelOutboxConfiguration(BaseModel):
    """Configuration for outbox processing."""
    
    batch_size: int = Field(default=100, description="Number of events to process per batch", ge=1, le=1000)
    processing_timeout_seconds: int = Field(default=300, description="Timeout for processing events", ge=1)
    max_retry_count: int = Field(default=3, description="Maximum retry attempts for failed events", ge=0)
    retry_delay_seconds: int = Field(default=60, description="Delay between retry attempts", ge=1)
    
    # Cleanup settings
    retention_days: int = Field(default=30, description="Days to retain completed events", ge=1)
    cleanup_batch_size: int = Field(default=1000, description="Batch size for cleanup operations", ge=1)
    
    # Performance settings
    polling_interval_seconds: int = Field(default=5, description="Polling interval for new events", ge=1)
    connection_pool_size: int = Field(default=5, description="Database connection pool size", ge=1, le=50)