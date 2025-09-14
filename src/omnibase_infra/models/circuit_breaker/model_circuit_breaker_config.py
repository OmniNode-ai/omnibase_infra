"""Circuit Breaker Configuration Model.

Shared model for circuit breaker configuration settings.
Used across circuit breaker nodes for consistent configuration.
"""

from pydantic import BaseModel, Field


class ModelCircuitBreakerConfig(BaseModel):
    """Model for circuit breaker configuration."""
    
    failure_threshold: int = Field(
        default=5,
        gt=0,
        description="Number of failures before opening circuit"
    )
    
    recovery_timeout: int = Field(
        default=60,
        gt=0,
        description="Seconds before transitioning to half-open"
    )
    
    success_threshold: int = Field(
        default=3,
        gt=0,
        description="Successes needed in half-open to close circuit"
    )
    
    timeout_seconds: int = Field(
        default=30,
        gt=0,
        description="Event publishing timeout in seconds"
    )
    
    max_queue_size: int = Field(
        default=1000,
        gt=0,
        description="Maximum number of queued events when circuit is open"
    )
    
    dead_letter_enabled: bool = Field(
        default=True,
        description="Enable dead letter queue for failed events"
    )
    
    graceful_degradation: bool = Field(
        default=True,
        description="Allow operations to continue without events"
    )
    
    environment: str = Field(
        default="development",
        description="Target environment for configuration"
    )
    
    service_name: str = Field(
        default="omnibase_infrastructure",
        description="Name of the service using the circuit breaker"
    )