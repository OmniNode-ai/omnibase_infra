"""
Typed models to replace Dict[str, Any] throughout the codebase.

This module provides strongly-typed TypedDict and Pydantic models for
common data structures, improving type safety and enabling better IDE support.
"""

from datetime import datetime
from typing import Any, NotRequired, Optional, TypedDict

from pydantic import BaseModel, Field

# ============================================================================
# Consul Service Registry Models
# ============================================================================


class ConsulHealthCheckResult(TypedDict):
    """
    Health check result from Consul registry client.

    Provides structured typing for health check responses, replacing
    Dict[str, Any] with specific field types.
    """

    status: str  # "healthy", "unhealthy", "unavailable", "degraded"
    consul_connected: bool
    consul_host: NotRequired[str]
    consul_port: NotRequired[int]
    service_id: NotRequired[str]
    message: NotRequired[str]
    error: NotRequired[str]


class ConsulServiceMetadata(TypedDict):
    """
    Service metadata from Consul registry.

    Used for service discovery and registration metadata.
    """

    id: str
    name: NotRequired[str]
    address: str
    port: int
    tags: list[str]
    meta: dict[str, str]  # Consul meta is always str->str


class DiscoveredServiceInstance(TypedDict):
    """
    Discovered service instance from Consul service discovery.

    Returned by discover_services() method.
    """

    id: str
    address: str
    port: int
    tags: list[str]
    meta: dict[str, str]  # Consul meta is always str->str


# ============================================================================
# FSM State Management Models
# ============================================================================


class FSMTransitionMetadata(TypedDict, total=False):
    """
    Metadata for FSM state transitions.

    Optional fields for additional context during state transitions.
    Note: Uses total=False to make all fields optional.
    """

    namespace: str
    reason: str
    triggered_by: str
    error_message: str
    execution_time_ms: float
    retry_count: int


class FSMTransitionRecord(TypedDict):
    """
    Complete FSM transition record for history tracking.

    Replaces dict[str, Any] in transition history lists.
    """

    from_state: str
    to_state: str
    trigger: str
    timestamp: datetime
    metadata: FSMTransitionMetadata


class WorkflowStateData(TypedDict):
    """
    FSM workflow state cache entry.

    Provides strong typing for workflow state data, replacing
    dict[str, Any] in FSM state cache.
    """

    current_state: str
    previous_state: Optional[str]
    transition_count: int
    metadata: FSMTransitionMetadata
    created_at: datetime
    updated_at: datetime


# ============================================================================
# Event Publishing Models
# ============================================================================


class KafkaEventPayload(TypedDict, total=False):
    """
    Base Kafka event payload structure.

    Common fields for all Kafka events. Specific event types
    can extend this with additional fields.

    Note: Uses total=False to accommodate diverse event types.
    """

    correlation_id: str
    aggregation_id: str
    event_type: str
    timestamp: str
    node_id: str
    published_at: str
    # Event-specific fields are intentionally flexible
    # to support polymorphic event types


class EventPublishingMetadata(TypedDict, total=False):
    """
    Metadata for event publishing operations.

    Provides context and categorization for published events.
    """

    event_category: str
    node_type: str
    namespace: str
    correlation_id: str
    source_node_id: str


# ============================================================================
# Database Operation Models (Pydantic for validation)
# ============================================================================


class PoolStatsResult(BaseModel):
    """
    Database connection pool statistics.

    Pydantic model for type-safe pool stats with validation.
    """

    pool_size: int = Field(ge=0, description="Total pool size")
    available: int = Field(ge=0, description="Available connections")
    in_use: int = Field(ge=0, description="Connections in use")
    utilization: float = Field(ge=0.0, le=1.0, description="Pool utilization ratio")

    model_config = {"frozen": True}  # Immutable


class CircuitBreakerMetrics(BaseModel):
    """
    Circuit breaker state and metrics.

    Provides type-safe circuit breaker metrics with validation.
    """

    state: str = Field(
        pattern="^(closed|open|half_open)$", description="Circuit breaker state"
    )
    failure_count: int = Field(ge=0, description="Current failure count")
    success_count: int = Field(ge=0, description="Current success count")
    total_failures: int = Field(ge=0, description="Total failures")
    total_successes: int = Field(ge=0, description="Total successes")
    state_transitions: int = Field(ge=0, description="Number of state transitions")
    last_failure_time: Optional[str] = Field(
        None, description="ISO timestamp of last failure"
    )
    last_state_change: Optional[str] = Field(
        None, description="ISO timestamp of last state change"
    )
    half_open_calls: int = Field(ge=0, description="Calls in half-open state")
    config: dict[str, int] = Field(description="Circuit breaker configuration")

    model_config = {"frozen": True}


class PerformanceStats(BaseModel):
    """
    Performance statistics for operations.

    Provides structured performance metrics with percentiles.
    """

    avg_execution_time_ms: float = Field(ge=0.0, description="Average execution time")
    min_execution_time_ms: float = Field(ge=0.0, description="Minimum execution time")
    max_execution_time_ms: float = Field(ge=0.0, description="Maximum execution time")
    p95_execution_time_ms: float = Field(
        ge=0.0, description="95th percentile execution time"
    )
    p99_execution_time_ms: float = Field(
        ge=0.0, description="99th percentile execution time"
    )
    sample_count: int = Field(ge=0, description="Number of samples")

    model_config = {"frozen": True}


class ErrorRateMetrics(BaseModel):
    """
    Error rate statistics.

    Provides structured error tracking metrics.
    """

    total_errors: int = Field(ge=0, description="Total error count")
    error_rate_percent: float = Field(
        ge=0.0, le=100.0, description="Error rate percentage"
    )
    errors_by_type: dict[str, int] = Field(
        default_factory=dict, description="Errors grouped by type"
    )

    model_config = {"frozen": True}


class ThroughputMetrics(BaseModel):
    """
    Throughput statistics.

    Provides operations per second metrics with sliding window.
    """

    operations_per_second: float = Field(
        ge=0.0, description="Current operations per second"
    )
    peak_operations_per_second: float = Field(
        ge=0.0, description="Peak operations per second"
    )
    window_size_seconds: int = Field(ge=0, description="Sliding window size")
    sample_count: int = Field(ge=0, description="Number of samples in window")

    model_config = {"frozen": True}


# ============================================================================
# Message Routing Models
# ============================================================================


class KafkaMessageMetadata(TypedDict):
    """
    Kafka message metadata for routing.

    Provides structured typing for Kafka message envelope data.
    """

    topic: str
    partition: int
    offset: int
    timestamp: NotRequired[int]
    key: NotRequired[str]


class KafkaMessage(TypedDict):
    """
    Complete Kafka message structure.

    Combines payload and metadata for message routing.
    """

    value: dict[str, Any]  # Payload is polymorphic by design
    topic: str
    partition: int
    offset: int
    timestamp: NotRequired[int]
    key: NotRequired[str]


# ============================================================================
# Aggregation Data Models
# ============================================================================


class AggregationStats(TypedDict, total=False):
    """
    Aggregation statistics for reducer operations.

    Provides structured typing for namespace aggregation data.

    Note: Uses total=False to accommodate incremental aggregation.
    """

    total_stamps: int
    total_size_bytes: int
    unique_file_types_count: int
    file_types: list[str]
    unique_workflows_count: int
    workflow_ids: list[str]


class HealthCheckDetails(TypedDict, total=False):
    """
    Health check component details.

    Provides additional context for health check results.
    """

    buffer_size: int
    fsm_cache_size: int
    fsm_history_size: int
    threshold: int
    error: str
    pool_size: int
    available: int
    in_use: int
    database_version: str
    uptime_seconds: int


# ============================================================================
# Intent Models (for ONEX pure function pattern)
# ============================================================================


class IntentPayload(TypedDict, total=False):
    """
    Generic intent payload structure.

    Base structure for intent payloads. Specific intent types
    can extend with additional fields.

    Note: Uses total=False to support diverse intent types.
    """

    aggregation_id: str
    workflow_id: str
    correlation_id: str
    timestamp: str
    aggregated_data: dict[str, Any]  # Aggregation results are polymorphic
    fsm_states: dict[str, str]
    # Intent-specific fields


# ============================================================================
# Utility Functions
# ============================================================================


def is_valid_health_status(status: str) -> bool:
    """
    Validate health status string.

    Args:
        status: Status string to validate

    Returns:
        True if status is valid, False otherwise
    """
    valid_statuses = {"healthy", "unhealthy", "degraded", "unavailable"}
    return status.lower() in valid_statuses


def is_valid_circuit_breaker_state(state: str) -> bool:
    """
    Validate circuit breaker state string.

    Args:
        state: State string to validate

    Returns:
        True if state is valid, False otherwise
    """
    valid_states = {"closed", "open", "half_open"}
    return state.lower() in valid_states
