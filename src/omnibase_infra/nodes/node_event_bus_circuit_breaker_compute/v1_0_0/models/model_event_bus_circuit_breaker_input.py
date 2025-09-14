"""Event Bus Circuit Breaker Input Model.

Node-specific input model for the circuit breaker compute node.
"""

from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID

from omnibase_core.model.core.model_onex_event import ModelOnexEvent


class CircuitBreakerOperation(str, Enum):
    """Circuit breaker operations."""
    PUBLISH_EVENT = "publish_event"
    GET_STATE = "get_state"
    GET_METRICS = "get_metrics"
    RESET_CIRCUIT = "reset_circuit"
    GET_HEALTH_STATUS = "get_health_status"


class ModelEventBusCircuitBreakerInput(BaseModel):
    """Input model for event bus circuit breaker operations."""
    
    operation_type: CircuitBreakerOperation = Field(
        description="Type of circuit breaker operation to perform"
    )
    
    event: Optional[ModelOnexEvent] = Field(
        default=None,
        description="Event to publish (required for publish_event operation)"
    )
    
    correlation_id: UUID = Field(
        description="Correlation ID for the operation"
    )
    
    publisher_function: Optional[str] = Field(
        default=None,
        description="Name of publisher function to use for event publishing"
    )
    
    environment: str = Field(
        default="development",
        description="Environment configuration (development, staging, production)"
    )