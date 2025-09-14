"""Kafka Producer Pool Statistics Model.

Provides strongly typed model for Kafka producer pool statistics and health monitoring.
Used for exposing producer pool metrics through health endpoints and Prometheus integration.

Following ONEX shared model architecture for infrastructure monitoring.
"""

from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field


class ModelKafkaProducerStats(BaseModel):
    """Statistics for an individual Kafka producer."""
    
    producer_id: str = Field(description="Unique producer identifier")
    is_active: bool = Field(description="Whether producer is currently active")
    messages_sent: int = Field(ge=0, description="Total messages sent by this producer")
    messages_failed: int = Field(ge=0, description="Total failed messages")
    bytes_sent: int = Field(ge=0, description="Total bytes sent")
    average_batch_size: float = Field(ge=0, description="Average batch size in messages")
    average_response_time_ms: float = Field(ge=0, description="Average response time in milliseconds")
    last_activity: Optional[datetime] = Field(default=None, description="Last activity timestamp")
    error_count: int = Field(ge=0, description="Total error count")
    last_error: Optional[str] = Field(default=None, description="Last error message")
    connection_state: str = Field(description="Connection state: connected, connecting, disconnected")
    
    class Config:
        """Pydantic model configuration."""
        validate_assignment = True
        extra = "forbid"


class ModelKafkaTopicStats(BaseModel):
    """Statistics for Kafka topic interactions."""
    
    topic: str = Field(description="Kafka topic name")
    partition_count: int = Field(ge=0, description="Number of partitions")
    messages_sent: int = Field(ge=0, description="Total messages sent to topic")
    messages_failed: int = Field(ge=0, description="Total failed messages for topic")
    bytes_sent: int = Field(ge=0, description="Total bytes sent to topic")
    average_message_size: float = Field(ge=0, description="Average message size in bytes")
    last_activity: Optional[datetime] = Field(default=None, description="Last activity timestamp")
    
    class Config:
        """Pydantic model configuration."""
        validate_assignment = True
        extra = "forbid"


class ModelKafkaProducerPoolStats(BaseModel):
    """Kafka producer pool statistics and health information.
    
    Provides comprehensive statistics for Kafka producer pool monitoring,
    including individual producer stats, topic statistics, and overall pool health.
    Used for health endpoint exposure and Prometheus metrics integration.
    """
    
    pool_name: str = Field(description="Name of the producer pool")
    total_producers: int = Field(ge=0, description="Total number of producers in pool")
    active_producers: int = Field(ge=0, description="Number of active producers")
    idle_producers: int = Field(ge=0, description="Number of idle producers")
    failed_producers: int = Field(ge=0, description="Number of failed producers")
    
    # Pool configuration
    min_pool_size: int = Field(ge=0, description="Minimum pool size")
    max_pool_size: int = Field(ge=1, description="Maximum pool size")
    pool_utilization: float = Field(ge=0, le=100, description="Pool utilization percentage")
    
    # Aggregate statistics
    total_messages_sent: int = Field(ge=0, description="Total messages sent across all producers")
    total_messages_failed: int = Field(ge=0, description="Total failed messages across all producers")
    total_bytes_sent: int = Field(ge=0, description="Total bytes sent across all producers")
    average_throughput_mps: float = Field(ge=0, description="Average throughput in messages per second")
    average_response_time_ms: float = Field(ge=0, description="Average response time across all producers")
    
    # Health indicators
    pool_health: str = Field(description="Pool health: healthy, degraded, unhealthy")
    error_rate: float = Field(ge=0, le=100, description="Error rate percentage")
    success_rate: float = Field(ge=0, le=100, description="Success rate percentage")
    
    # Time-based metrics
    uptime_seconds: int = Field(ge=0, description="Pool uptime in seconds")
    last_activity: Optional[datetime] = Field(default=None, description="Last pool activity")
    created_at: datetime = Field(description="Pool creation timestamp")
    
    # Detailed statistics (optional for detailed monitoring)
    producer_stats: Optional[List[ModelKafkaProducerStats]] = Field(
        default=None, 
        description="Individual producer statistics"
    )
    topic_stats: Optional[List[ModelKafkaTopicStats]] = Field(
        default=None,
        description="Per-topic statistics"  
    )
    
    # Configuration snapshot - removed to eliminate Any type usage per ONEX standards
    # Use specific typed models for configuration instead of generic dictionaries
    
    def calculate_derived_metrics(self) -> None:
        """Calculate derived metrics from base statistics."""
        # Calculate pool utilization
        if self.max_pool_size > 0:
            self.pool_utilization = (self.active_producers / self.max_pool_size) * 100
        else:
            self.pool_utilization = 0.0
        
        # Calculate success and error rates
        total_messages = self.total_messages_sent + self.total_messages_failed
        if total_messages > 0:
            self.success_rate = (self.total_messages_sent / total_messages) * 100
            self.error_rate = (self.total_messages_failed / total_messages) * 100
        else:
            self.success_rate = 100.0
            self.error_rate = 0.0
    
    def determine_health_status(self) -> str:
        """Determine pool health status based on metrics."""
        if self.failed_producers == 0 and self.error_rate < 1.0:
            return "healthy"
        elif self.failed_producers < self.total_producers * 0.5 and self.error_rate < 5.0:
            return "degraded"
        else:
            return "unhealthy"
    
    @classmethod
    def create_empty_stats(cls, pool_name: str) -> "ModelKafkaProducerPoolStats":
        """Create empty statistics for a new pool."""
        return cls(
            pool_name=pool_name,
            total_producers=0,
            active_producers=0,
            idle_producers=0,
            failed_producers=0,
            min_pool_size=1,
            max_pool_size=10,
            pool_utilization=0.0,
            total_messages_sent=0,
            total_messages_failed=0,
            total_bytes_sent=0,
            average_throughput_mps=0.0,
            average_response_time_ms=0.0,
            pool_health="healthy",
            error_rate=0.0,
            success_rate=100.0,
            uptime_seconds=0,
            created_at=datetime.now()
        )
    
    class Config:
        """Pydantic model configuration."""
        validate_assignment = True
        extra = "forbid"
        schema_extra = {
            "example": {
                "pool_name": "redpanda_producer_pool",
                "total_producers": 5,
                "active_producers": 3,
                "idle_producers": 2,
                "failed_producers": 0,
                "min_pool_size": 2,
                "max_pool_size": 10,
                "pool_utilization": 30.0,
                "total_messages_sent": 10000,
                "total_messages_failed": 50,
                "total_bytes_sent": 1048576,
                "average_throughput_mps": 100.5,
                "average_response_time_ms": 25.3,
                "pool_health": "healthy",
                "error_rate": 0.5,
                "success_rate": 99.5,
                "uptime_seconds": 3600,
                "created_at": "2025-09-12T21:00:00Z"
            }
        }