"""Infrastructure Health Metrics Model.

Pydantic model for aggregated infrastructure health metrics, extracted from
infrastructure_health_monitor.py for shared usage across ONEX nodes.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any
from datetime import datetime


class ModelInfrastructureHealthMetrics(BaseModel):
    """Model for aggregated infrastructure health metrics."""
    
    # Overall health
    overall_status: str = Field(
        description="Overall health status: healthy, degraded, unhealthy"
    )
    
    timestamp: float = Field(
        description="Unix timestamp of health check"
    )
    
    environment: str = Field(
        description="Target environment name"
    )
    
    # Component statuses
    postgres_healthy: bool = Field(
        description="PostgreSQL connection health status"
    )
    
    kafka_healthy: bool = Field(
        description="Kafka producer health status"
    )
    
    circuit_breaker_healthy: bool = Field(
        description="Circuit breaker health status"
    )
    
    # Detailed metrics
    postgres_metrics: Dict[str, Any] = Field(
        description="Detailed PostgreSQL metrics"
    )
    
    kafka_metrics: Dict[str, Any] = Field(
        description="Detailed Kafka metrics"
    )
    
    circuit_breaker_metrics: Dict[str, Any] = Field(
        description="Detailed circuit breaker metrics"
    )
    
    # Aggregate statistics
    total_connections: int = Field(
        ge=0,
        description="Total number of active connections"
    )
    
    total_messages_processed: int = Field(
        ge=0,
        description="Total messages processed"
    )
    
    total_events_queued: int = Field(
        ge=0,
        description="Total events in queues"
    )
    
    error_rate_percent: float = Field(
        ge=0.0,
        le=100.0,
        description="Error rate percentage"
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
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }