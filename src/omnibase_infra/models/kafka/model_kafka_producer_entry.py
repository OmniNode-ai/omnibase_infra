"""
Kafka Producer Entry Model

Strongly typed model for tracking individual producers in the pool.
Replaces Dict[str, Any] usage to maintain ONEX compliance.
"""

from pydantic import BaseModel, Field
from typing import Optional


class ModelKafkaProducerEntry(BaseModel):
    """
    Entry for tracking individual producers in the pool.
    
    Replaces Dict[str, Any] for producer tracking data to maintain ONEX zero tolerance for Any types.
    """
    
    servers_key: str = Field(
        ...,
        description="Unique key identifying the server configuration",
        min_length=1
    )
    
    usage_count: int = Field(
        default=0,
        description="Number of times this producer has been used",
        ge=0
    )
    
    last_used_timestamp: float = Field(
        ...,
        description="Unix timestamp of last usage",
        gt=0
    )
    
    is_healthy: bool = Field(
        default=True,
        description="Whether the producer is currently healthy"
    )
    
    failure_count: int = Field(
        default=0,
        description="Number of consecutive failures",
        ge=0
    )
    
    last_failure_timestamp: Optional[float] = Field(
        default=None,
        description="Unix timestamp of last failure",
        ge=0
    )
    
    connection_state: str = Field(
        default="connected",
        description="Current connection state: connected, connecting, disconnected, failed"
    )
    
    class Config:
        """Pydantic model configuration."""
        validate_assignment = True
        extra = "forbid"


class ModelKafkaFailureRecord(BaseModel):
    """
    Record for tracking producer failures.
    
    Replaces Dict[str, float] for failure tracking to maintain strong typing.
    """
    
    servers_key: str = Field(
        ...,
        description="Unique key identifying the server configuration",
        min_length=1
    )
    
    failure_timestamp: float = Field(
        ...,
        description="Unix timestamp when the failure occurred",
        gt=0
    )
    
    failure_reason: Optional[str] = Field(
        default=None,
        description="Reason for the failure"
    )
    
    retry_count: int = Field(
        default=0,
        description="Number of retry attempts since failure",
        ge=0
    )
    
    class Config:
        """Pydantic model configuration."""
        validate_assignment = True
        extra = "forbid"