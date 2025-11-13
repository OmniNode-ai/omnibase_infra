"""Pydantic settings models for bridge nodes configuration.

This module defines type-safe configuration models for orchestrator and reducer nodes,
integrating with the YAML-based configuration system.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


# Node identification models
class NodeConfig(BaseModel):
    """Base node configuration."""

    type: Literal["orchestrator", "reducer", "registry"] = Field(
        ..., description="Node type: orchestrator, reducer, or registry"
    )
    name: str = Field(..., description="Node instance name")
    version: str = Field(default="0.1.0", description="Node version")
    namespace: str = Field(
        ..., description="Node namespace (e.g., omninode.bridge.orchestrator)"
    )


# Orchestrator-specific models
class OrchestratorConfig(BaseModel):
    """Orchestrator node configuration settings."""

    # Workflow coordination
    max_concurrent_workflows: int = Field(
        default=100, ge=1, description="Maximum concurrent workflows"
    )
    workflow_timeout_seconds: int = Field(
        default=300, ge=1, description="Workflow timeout in seconds"
    )
    workflow_retry_attempts: int = Field(
        default=3, ge=0, description="Workflow retry attempts"
    )
    workflow_retry_delay_seconds: int = Field(
        default=5, ge=0, description="Workflow retry delay in seconds"
    )

    # Task management
    task_batch_size: int = Field(
        default=50, ge=1, description="Task processing batch size"
    )
    task_priority_levels: list[str] = Field(
        default=["critical", "high", "medium", "low"],
        description="Available task priority levels",
    )
    default_task_priority: str = Field(
        default="medium", description="Default task priority"
    )

    # Dependency resolution
    enable_dependency_tracking: bool = Field(
        default=True, description="Enable dependency tracking"
    )
    max_dependency_depth: int = Field(
        default=10, ge=1, description="Maximum dependency depth"
    )
    circular_dependency_detection: bool = Field(
        default=True, description="Enable circular dependency detection"
    )

    # Performance tuning
    worker_pool_size: int = Field(default=10, ge=1, description="Worker pool size")
    event_processing_buffer_size: int = Field(
        default=1000, ge=1, description="Event processing buffer size"
    )
    state_sync_interval_seconds: int = Field(
        default=30, ge=1, description="State sync interval in seconds"
    )


# Reducer-specific models
class ReducerConfig(BaseModel):
    """Reducer node configuration settings."""

    # Aggregation settings
    aggregation_window_seconds: int = Field(
        default=60, ge=1, description="Aggregation window in seconds"
    )
    aggregation_batch_size: int = Field(
        default=100, ge=1, description="Aggregation batch size"
    )
    max_aggregation_buffer_size: int = Field(
        default=10000, ge=1, description="Maximum aggregation buffer size"
    )
    enable_incremental_aggregation: bool = Field(
        default=True, description="Enable incremental aggregation"
    )

    # State management
    state_persistence_interval_seconds: int = Field(
        default=30, ge=1, description="State persistence interval in seconds"
    )
    state_snapshot_interval_seconds: int = Field(
        default=300, ge=1, description="State snapshot interval in seconds"
    )
    enable_state_compression: bool = Field(
        default=True, description="Enable state compression"
    )
    max_state_size_mb: int = Field(
        default=100, ge=1, description="Maximum state size in MB"
    )

    # Batch processing
    batch_size: int = Field(default=50, ge=1, description="Batch processing size")
    batch_timeout_seconds: int = Field(
        default=5, ge=1, description="Batch timeout in seconds"
    )
    max_batch_retry_attempts: int = Field(
        default=3, ge=0, description="Maximum batch retry attempts"
    )
    batch_retry_delay_seconds: int = Field(
        default=2, ge=0, description="Batch retry delay in seconds"
    )

    # Data retention
    retain_aggregated_data_hours: int = Field(
        default=24, ge=1, description="Retain aggregated data for hours"
    )
    retain_raw_data_hours: int = Field(
        default=6, ge=1, description="Retain raw data for hours"
    )
    enable_automatic_cleanup: bool = Field(
        default=True, description="Enable automatic cleanup"
    )

    # Performance tuning
    worker_pool_size: int = Field(default=8, ge=1, description="Worker pool size")
    aggregation_worker_threads: int = Field(
        default=4, ge=1, description="Aggregation worker threads"
    )
    io_worker_threads: int = Field(default=4, ge=1, description="I/O worker threads")
    event_processing_buffer_size: int = Field(
        default=500, ge=1, description="Event processing buffer size"
    )


# Service endpoint models
class ServiceEndpointConfig(BaseModel):
    """Service endpoint configuration."""

    host: str = Field(..., description="Service host")
    port: int = Field(..., ge=1, le=65535, description="Service port")
    base_url: str = Field(..., description="Service base URL")
    health_check_path: str = Field(
        default="/health", description="Health check endpoint path"
    )
    timeout_seconds: int = Field(
        default=30, ge=1, description="Request timeout in seconds"
    )
    retry_attempts: int = Field(default=3, ge=0, description="Retry attempts")


class ServicesConfig(BaseModel):
    """External services configuration."""

    onextree: ServiceEndpointConfig = Field(
        ..., description="OnexTree service configuration"
    )
    metadata_stamping: ServiceEndpointConfig = Field(
        ..., description="Metadata stamping service configuration"
    )


# Kafka models
class KafkaProducerConfig(BaseModel):
    """Kafka producer configuration."""

    compression_type: str = Field(default="snappy", description="Compression type")
    batch_size: int = Field(default=16384, ge=1, description="Batch size in bytes")
    linger_ms: int = Field(default=5, ge=0, description="Linger time in milliseconds")
    acks: str = Field(default="all", description="Acknowledgment mode")
    max_in_flight_requests: int = Field(
        default=5, ge=1, description="Max in-flight requests"
    )
    request_timeout_ms: int | None = Field(
        default=None, ge=1, description="Request timeout in milliseconds"
    )
    retries: int | None = Field(default=None, ge=0, description="Number of retries")


class KafkaConsumerConfig(BaseModel):
    """Kafka consumer configuration."""

    group_id: str = Field(..., description="Consumer group ID")
    auto_offset_reset: str = Field(
        default="latest", description="Auto offset reset policy"
    )
    enable_auto_commit: bool = Field(default=True, description="Enable auto commit")
    auto_commit_interval_ms: int | None = Field(
        default=None, ge=1, description="Auto commit interval in milliseconds"
    )
    max_poll_records: int = Field(default=500, ge=1, description="Max poll records")
    max_poll_interval_ms: int | None = Field(
        default=None, ge=1, description="Max poll interval in milliseconds"
    )
    session_timeout_ms: int | None = Field(
        default=None, ge=1, description="Session timeout in milliseconds"
    )
    heartbeat_interval_ms: int | None = Field(
        default=None, ge=1, description="Heartbeat interval in milliseconds"
    )
    fetch_min_bytes: int | None = Field(
        default=None, ge=1, description="Minimum fetch bytes"
    )
    fetch_max_wait_ms: int | None = Field(
        default=None, ge=1, description="Maximum fetch wait in milliseconds"
    )


class KafkaTopicsConfig(BaseModel):
    """Kafka topics configuration."""

    workflow_commands: str | None = Field(
        default=None, description="Workflow commands topic"
    )
    workflow_events: str = Field(..., description="Workflow events topic")
    task_events: str = Field(..., description="Task events topic")
    state_sync: str | None = Field(default=None, description="State sync topic")
    aggregated_metrics: str | None = Field(
        default=None, description="Aggregated metrics topic"
    )
    state_snapshots: str | None = Field(
        default=None, description="State snapshots topic"
    )


class KafkaNodeConfig(BaseModel):
    """Kafka configuration for bridge nodes."""

    bootstrap_servers: str = Field(..., description="Kafka bootstrap servers")
    producer: KafkaProducerConfig = Field(..., description="Producer configuration")
    consumer: KafkaConsumerConfig = Field(..., description="Consumer configuration")
    topics: KafkaTopicsConfig = Field(..., description="Topics configuration")


# Database models
class DatabaseNodeConfig(BaseModel):
    """Database configuration for bridge nodes."""

    host: str = Field(..., description="Database host")
    port: int = Field(..., ge=1, le=65535, description="Database port")
    database: str = Field(..., description="Database name")
    user: str = Field(..., description="Database user")
    password: str | None = Field(default=None, description="Database password")

    # Connection pooling
    pool_min_size: int = Field(default=5, ge=1, description="Minimum pool size")
    pool_max_size: int = Field(default=20, ge=1, description="Maximum pool size")
    pool_timeout_seconds: int = Field(
        default=10, ge=1, description="Pool timeout in seconds"
    )

    # Performance
    query_timeout_seconds: int = Field(
        default=30, ge=1, description="Query timeout in seconds"
    )
    command_timeout_seconds: int = Field(
        default=60, ge=1, description="Command timeout in seconds"
    )
    statement_cache_size: int | None = Field(
        default=None, ge=0, description="Statement cache size"
    )
    batch_insert_size: int | None = Field(
        default=None, ge=1, description="Batch insert size"
    )


# Logging models
class LogOutputConfig(BaseModel):
    """Log output configuration."""

    type: Literal["console", "file"] = Field(..., description="Output type")
    enabled: bool = Field(default=True, description="Enable this output")
    path: str | None = Field(default=None, description="File path (for file output)")
    max_size_mb: int | None = Field(
        default=None, ge=1, description="Max file size in MB"
    )
    backup_count: int | None = Field(
        default=None, ge=0, description="Number of backup files"
    )
    level: str | None = Field(default=None, description="Log level for this output")


class LoggingNodeConfig(BaseModel):
    """Logging configuration for bridge nodes."""

    level: str = Field(default="INFO", description="Log level")
    format: Literal["json", "pretty", "text"] = Field(
        default="json", description="Log format"
    )
    enable_structured_logging: bool = Field(
        default=True, description="Enable structured logging"
    )
    log_requests: bool = Field(default=True, description="Log requests")
    log_responses: bool = Field(default=False, description="Log responses")
    log_aggregations: bool | None = Field(
        default=None, description="Log aggregations (reducer only)"
    )
    log_state_changes: bool | None = Field(
        default=None, description="Log state changes (reducer only)"
    )
    outputs: list[LogOutputConfig] = Field(default=[], description="Log outputs")


# Monitoring models
class MonitoringNodeConfig(BaseModel):
    """Monitoring configuration for bridge nodes."""

    enable_prometheus: bool = Field(
        default=True, description="Enable Prometheus metrics"
    )
    prometheus_port: int = Field(
        default=9090, ge=1, le=65535, description="Prometheus port"
    )
    metrics_interval_seconds: int = Field(
        default=15, ge=1, description="Metrics interval in seconds"
    )
    health_check_interval_seconds: int = Field(
        default=30, ge=1, description="Health check interval in seconds"
    )
    service_health_timeout_seconds: int | None = Field(
        default=None, ge=1, description="Service health timeout in seconds"
    )

    # Tracking flags
    track_workflow_latency: bool | None = Field(
        default=None, description="Track workflow latency"
    )
    track_task_latency: bool | None = Field(
        default=None, description="Track task latency"
    )
    track_dependency_resolution_time: bool | None = Field(
        default=None, description="Track dependency resolution time"
    )
    track_aggregation_latency: bool | None = Field(
        default=None, description="Track aggregation latency (reducer only)"
    )
    track_batch_processing_latency: bool | None = Field(
        default=None, description="Track batch processing latency (reducer only)"
    )
    track_state_persistence_latency: bool | None = Field(
        default=None, description="Track state persistence latency (reducer only)"
    )
    track_buffer_utilization: bool | None = Field(
        default=None, description="Track buffer utilization (reducer only)"
    )


# Circuit breaker models
class CircuitBreakerNodeConfig(BaseModel):
    """Circuit breaker configuration for bridge nodes."""

    enabled: bool = Field(default=True, description="Enable circuit breaker")
    failure_threshold: int = Field(default=5, ge=1, description="Failure threshold")
    recovery_timeout_seconds: int = Field(
        default=60, ge=1, description="Recovery timeout in seconds"
    )
    half_open_max_requests: int = Field(
        default=3, ge=1, description="Max requests in half-open state"
    )


# Cache models
class CacheNodeConfig(BaseModel):
    """Cache configuration for bridge nodes."""

    enabled: bool = Field(default=True, description="Enable caching")
    workflow_state_ttl_seconds: int | None = Field(
        default=None, ge=1, description="Workflow state TTL in seconds"
    )
    task_result_ttl_seconds: int | None = Field(
        default=None, ge=1, description="Task result TTL in seconds"
    )
    dependency_graph_ttl_seconds: int | None = Field(
        default=None, ge=1, description="Dependency graph TTL in seconds"
    )
    aggregation_state_ttl_seconds: int | None = Field(
        default=None,
        ge=1,
        description="Aggregation state TTL in seconds (reducer only)",
    )
    dimension_cache_ttl_seconds: int | None = Field(
        default=None, ge=1, description="Dimension cache TTL in seconds (reducer only)"
    )
    max_cache_size_mb: int = Field(
        default=256, ge=1, description="Max cache size in MB"
    )


# Aggregation models
class MetricsAggregationConfig(BaseModel):
    """Metrics aggregation configuration."""

    enabled: bool = Field(default=True, description="Enable metrics aggregation")
    window_seconds: int = Field(
        default=60, ge=1, description="Aggregation window in seconds"
    )
    functions: list[str] = Field(
        default=["count", "sum", "avg", "min", "max"],
        description="Aggregation functions",
    )
    dimensions: list[str] = Field(
        default=["node_type", "operation", "status"],
        description="Aggregation dimensions",
    )


class EventsAggregationConfig(BaseModel):
    """Events aggregation configuration."""

    enabled: bool = Field(default=True, description="Enable events aggregation")
    window_seconds: int = Field(
        default=30, ge=1, description="Aggregation window in seconds"
    )
    group_by: list[str] = Field(
        default=["event_type", "source"],
        description="Group by fields",
    )
    count_threshold: int = Field(
        default=100, ge=1, description="Count threshold for aggregation"
    )


class WorkflowStateAggregationConfig(BaseModel):
    """Workflow state aggregation configuration."""

    enabled: bool = Field(default=True, description="Enable workflow state aggregation")
    window_seconds: int = Field(
        default=120, ge=1, description="Aggregation window in seconds"
    )
    track_transitions: bool = Field(default=True, description="Track state transitions")
    track_duration: bool = Field(default=True, description="Track workflow duration")
    track_errors: bool | None = Field(default=None, description="Track workflow errors")


class AggregationConfig(BaseModel):
    """Aggregation configuration (reducer only)."""

    metrics: MetricsAggregationConfig = Field(
        ..., description="Metrics aggregation configuration"
    )
    events: EventsAggregationConfig = Field(
        ..., description="Events aggregation configuration"
    )
    workflow_state: WorkflowStateAggregationConfig = Field(
        ..., description="Workflow state aggregation configuration"
    )


# Windowing models
class TumblingWindowConfig(BaseModel):
    """Tumbling window configuration."""

    enabled: bool = Field(default=True, description="Enable tumbling windows")
    sizes: list[int] = Field(
        default=[60, 300, 3600],
        description="Window sizes in seconds",
    )


class SlidingWindowConfig(BaseModel):
    """Sliding window configuration."""

    enabled: bool = Field(default=False, description="Enable sliding windows")
    size_seconds: int = Field(default=300, ge=1, description="Window size in seconds")
    slide_seconds: int = Field(
        default=60, ge=1, description="Slide interval in seconds"
    )


class SessionWindowConfig(BaseModel):
    """Session window configuration."""

    enabled: bool = Field(default=False, description="Enable session windows")
    gap_seconds: int = Field(default=300, ge=1, description="Session gap in seconds")


class WindowingConfig(BaseModel):
    """Windowing configuration (reducer only)."""

    tumbling: TumblingWindowConfig = Field(
        ..., description="Tumbling window configuration"
    )
    sliding: SlidingWindowConfig = Field(
        ..., description="Sliding window configuration"
    )
    session: SessionWindowConfig = Field(
        ..., description="Session window configuration"
    )


# ============================================================================
# Comprehensive Batch Size Configuration
# ============================================================================


class BatchSizeConfig(BaseModel):
    """Comprehensive batch size configuration for all operations."""

    # Database operations
    database_batch_size: int = Field(
        default=50, ge=1, le=1000, description="Database batch insert/update size"
    )
    database_query_limit: int = Field(
        default=1000, ge=1, le=10000, description="Maximum rows per query"
    )
    database_statement_cache_size: int = Field(
        default=100, ge=1, le=1000, description="Prepared statement cache size"
    )

    # Kafka operations
    kafka_producer_batch_size: int = Field(
        default=16384,
        ge=1,
        le=1048576,
        description="Kafka producer batch size in bytes",
    )
    kafka_consumer_max_poll_records: int = Field(
        default=500, ge=1, le=5000, description="Maximum records per Kafka poll"
    )
    kafka_consumer_fetch_min_bytes: int = Field(
        default=1, ge=1, le=1024, description="Minimum bytes per Kafka fetch"
    )
    kafka_consumer_fetch_max_bytes: int = Field(
        default=1048576, ge=1, le=10485760, description="Maximum bytes per Kafka fetch"
    )

    # Redis operations
    redis_batch_size: int = Field(
        default=100, ge=1, le=1000, description="Redis operation batch size"
    )
    redis_pipeline_size: int = Field(
        default=100, ge=1, le=1000, description="Redis pipeline size"
    )

    # Node operations
    orchestrator_batch_size: int = Field(
        default=50, ge=1, le=500, description="Orchestrator task batch size"
    )
    reducer_batch_size: int = Field(
        default=100, ge=1, le=1000, description="Reducer aggregation batch size"
    )
    registry_batch_size: int = Field(
        default=50, ge=1, le=500, description="Registry operation batch size"
    )

    # Performance optimization
    performance_task_batch_size: int = Field(
        default=50, ge=1, le=200, description="Performance task batch size"
    )
    processing_buffer_size: int = Field(
        default=1000, ge=1, le=10000, description="Event processing buffer size"
    )

    # Cleanup operations
    cleanup_batch_size: int = Field(
        default=100, ge=1, le=1000, description="Cleanup operation batch size"
    )
    retention_cleanup_batch_size: int = Field(
        default=1000, ge=1, le=10000, description="Retention cleanup batch size"
    )

    # File operations
    file_processing_batch_size: int = Field(
        default=50, ge=1, le=500, description="File processing batch size"
    )
    metadata_extraction_batch_size: int = Field(
        default=20, ge=1, le=200, description="Metadata extraction batch size"
    )


# Environment-specific batch size configurations
class EnvironmentBatchSizeConfig(BaseModel):
    """Environment-specific batch size configurations."""

    development: BatchSizeConfig = Field(
        default_factory=lambda: BatchSizeConfig(
            database_batch_size=10,
            kafka_producer_batch_size=8192,
            redis_batch_size=50,
            orchestrator_batch_size=5,
            reducer_batch_size=25,
            file_processing_batch_size=10,
        ),
        description="Development environment batch sizes",
    )

    test: BatchSizeConfig = Field(
        default_factory=lambda: BatchSizeConfig(
            database_batch_size=20,
            kafka_producer_batch_size=16384,
            redis_batch_size=50,
            orchestrator_batch_size=10,
            reducer_batch_size=50,
            file_processing_batch_size=20,
        ),
        description="Test environment batch sizes",
    )

    staging: BatchSizeConfig = Field(
        default_factory=lambda: BatchSizeConfig(
            database_batch_size=100,
            kafka_producer_batch_size=32768,
            redis_batch_size=200,
            orchestrator_batch_size=50,
            reducer_batch_size=200,
            file_processing_batch_size=50,
        ),
        description="Staging environment batch sizes",
    )

    production: BatchSizeConfig = Field(
        default_factory=lambda: BatchSizeConfig(
            database_batch_size=200,
            kafka_producer_batch_size=65536,
            redis_batch_size=500,
            orchestrator_batch_size=100,
            reducer_batch_size=500,
            file_processing_batch_size=100,
        ),
        description="Production environment batch sizes",
    )


# Main configuration models
class OrchestratorSettings(BaseModel):
    """Complete orchestrator node settings."""

    environment: str = Field(default="development", description="Environment name")
    node: NodeConfig = Field(..., description="Node identification")
    orchestrator: OrchestratorConfig = Field(
        ..., description="Orchestrator configuration"
    )
    services: ServicesConfig = Field(..., description="External services configuration")
    kafka: KafkaNodeConfig = Field(..., description="Kafka configuration")
    database: DatabaseNodeConfig = Field(..., description="Database configuration")
    consul: ConsulNodeConfig = Field(
        ..., description="Consul service discovery configuration"
    )
    logging: LoggingNodeConfig = Field(..., description="Logging configuration")
    monitoring: MonitoringNodeConfig = Field(
        ..., description="Monitoring configuration"
    )
    circuit_breaker: CircuitBreakerNodeConfig = Field(
        ..., description="Circuit breaker configuration"
    )
    cache: CacheNodeConfig = Field(..., description="Cache configuration")
    batch_sizes: BatchSizeConfig = Field(
        default_factory=BatchSizeConfig, description="Batch size configuration"
    )

    @field_validator("environment")
    @classmethod
    def _validate_environment(cls, v: str) -> str:
        """Validate environment name."""
        valid_environments = {"development", "staging", "production", "test"}
        if v.lower() not in valid_environments:
            raise ValueError(
                f"Invalid environment: {v}. Must be one of {valid_environments}"
            )
        return v.lower()


class ReducerSettings(BaseModel):
    """Complete reducer node settings."""

    environment: str = Field(default="development", description="Environment name")
    node: NodeConfig = Field(..., description="Node identification")
    reducer: ReducerConfig = Field(..., description="Reducer configuration")
    database: DatabaseNodeConfig = Field(..., description="Database configuration")
    kafka: KafkaNodeConfig = Field(..., description="Kafka configuration")
    consul: ConsulNodeConfig = Field(
        ..., description="Consul service discovery configuration"
    )
    aggregation: AggregationConfig = Field(..., description="Aggregation configuration")
    logging: LoggingNodeConfig = Field(..., description="Logging configuration")
    monitoring: MonitoringNodeConfig = Field(
        ..., description="Monitoring configuration"
    )
    circuit_breaker: CircuitBreakerNodeConfig = Field(
        ..., description="Circuit breaker configuration"
    )
    cache: CacheNodeConfig = Field(..., description="Cache configuration")
    windowing: WindowingConfig = Field(..., description="Windowing configuration")
    batch_sizes: BatchSizeConfig = Field(
        default_factory=BatchSizeConfig, description="Batch size configuration"
    )

    @field_validator("environment")
    @classmethod
    def _validate_environment(cls, v: str) -> str:
        """Validate environment name."""
        valid_environments = {"development", "staging", "production", "test"}
        if v.lower() not in valid_environments:
            raise ValueError(
                f"Invalid environment: {v}. Must be one of {valid_environments}"
            )
        return v.lower()


# Consul service discovery models
class ConsulNodeConfig(BaseModel):
    """Consul service discovery configuration."""

    host: str = Field(..., description="Consul host")
    port: int = Field(..., ge=1, le=65535, description="Consul port")
    enable_registration: bool = Field(
        default=True, description="Enable service registration with Consul"
    )
    registration_timeout_seconds: int = Field(
        default=10, ge=1, description="Registration timeout in seconds"
    )
    health_check_interval_seconds: int = Field(
        default=30, ge=1, description="Health check interval in seconds"
    )
    health_check_timeout_seconds: int = Field(
        default=5, ge=1, description="Health check timeout in seconds"
    )


# Registry-specific models
class RegistryConfig(BaseModel):
    """Registry node configuration settings."""

    # Node registration
    enable_auto_registration: bool = Field(
        default=True, description="Enable automatic node registration"
    )
    registration_interval_seconds: int = Field(
        default=30, ge=1, description="Node registration interval in seconds"
    )
    heartbeat_interval_seconds: int = Field(
        default=15, ge=1, description="Node heartbeat interval in seconds"
    )
    heartbeat_timeout_seconds: int = Field(
        default=60, ge=1, description="Node heartbeat timeout in seconds"
    )

    # Service discovery
    enable_service_discovery: bool = Field(
        default=True, description="Enable service discovery"
    )
    discovery_refresh_interval_seconds: int = Field(
        default=60, ge=1, description="Service discovery refresh interval in seconds"
    )

    # Health checking
    enable_health_checks: bool = Field(default=True, description="Enable health checks")
    health_check_interval_seconds: int = Field(
        default=30, ge=1, description="Health check interval in seconds"
    )
    health_check_timeout_seconds: int = Field(
        default=5, ge=1, description="Health check timeout in seconds"
    )

    # Node lifecycle
    node_ttl_seconds: int = Field(default=300, ge=1, description="Node TTL in seconds")
    enable_automatic_deregistration: bool = Field(
        default=True, description="Enable automatic node deregistration"
    )

    # Performance tuning
    max_registered_nodes: int = Field(
        default=1000, ge=1, description="Maximum registered nodes"
    )
    node_cache_ttl_seconds: int = Field(
        default=60, ge=1, description="Node cache TTL in seconds"
    )
    event_processing_buffer_size: int = Field(
        default=500, ge=1, description="Event processing buffer size"
    )


class RegistrySettings(BaseModel):
    """Complete registry node settings."""

    environment: str = Field(default="development", description="Environment name")
    node: NodeConfig = Field(..., description="Node identification")
    registry: RegistryConfig = Field(..., description="Registry configuration")
    database: DatabaseNodeConfig = Field(..., description="Database configuration")
    consul: ConsulNodeConfig = Field(
        ..., description="Consul service discovery configuration"
    )
    kafka: KafkaNodeConfig = Field(..., description="Kafka configuration")
    logging: LoggingNodeConfig = Field(..., description="Logging configuration")
    monitoring: MonitoringNodeConfig = Field(
        ..., description="Monitoring configuration"
    )
    circuit_breaker: CircuitBreakerNodeConfig = Field(
        ..., description="Circuit breaker configuration"
    )
    cache: CacheNodeConfig = Field(..., description="Cache configuration")
    batch_sizes: BatchSizeConfig = Field(
        default_factory=BatchSizeConfig, description="Batch size configuration"
    )

    @field_validator("environment")
    @classmethod
    def _validate_environment(cls, v: str) -> str:
        """Validate environment name."""
        valid_environments = {"development", "staging", "production", "test"}
        if v.lower() not in valid_environments:
            raise ValueError(
                f"Invalid environment: {v}. Must be one of {valid_environments}"
            )
        return v.lower()
