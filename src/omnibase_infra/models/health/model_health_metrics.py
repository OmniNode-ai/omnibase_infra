"""Health Metrics Model.

Shared model for infrastructure health metrics and performance data.
Used across health monitoring nodes for aggregated metrics.
"""

from pydantic import BaseModel, Field
from typing import Any, Dict, Optional
from datetime import datetime


class ModelHealthMetrics(BaseModel):
    """Model for aggregated infrastructure health metrics."""
    
    timestamp: datetime = Field(
        description="Metrics collection timestamp"
    )
    
    environment: str = Field(
        description="Environment where metrics were collected"
    )
    
    # Component-specific metrics
    postgres_metrics: Dict[str, Any] = Field(
        description="PostgreSQL component metrics"
    )
    
    kafka_metrics: Dict[str, Any] = Field(
        description="Kafka component metrics"
    )
    
    circuit_breaker_metrics: Dict[str, Any] = Field(
        description="Circuit breaker component metrics"
    )
    
    consul_metrics: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Consul service discovery metrics"
    )
    
    vault_metrics: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Vault secret management metrics"
    )
    
    # Aggregate statistics
    total_connections: int = Field(
        ge=0,
        description="Total number of active connections"
    )
    
    total_messages_processed: int = Field(
        ge=0,
        description="Total number of messages processed"
    )
    
    total_events_queued: int = Field(
        ge=0,
        description="Total number of events currently queued"
    )
    
    error_rate_percent: float = Field(
        ge=0.0,
        le=100.0,
        description="Overall error rate percentage"
    )
    
    # Performance indicators
    avg_db_response_time_ms: float = Field(
        ge=0.0,
        description="Average database response time in milliseconds"
    )
    
    avg_kafka_throughput_mps: float = Field(
        ge=0.0,
        description="Average Kafka throughput in messages per second"
    )
    
    circuit_breaker_success_rate: float = Field(
        ge=0.0,
        le=100.0,
        description="Circuit breaker success rate percentage"
    )
    
    # Resource utilization
    memory_usage_percent: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Memory usage percentage"
    )
    
    cpu_usage_percent: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="CPU usage percentage"
    )
    
    disk_usage_percent: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Disk usage percentage"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }