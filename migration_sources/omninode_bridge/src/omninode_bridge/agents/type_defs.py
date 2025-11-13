"""
TypedDict definitions for complex return types across the codebase.

This module provides strongly-typed dictionary definitions to improve
type safety and IDE autocomplete for function return values.

Organization:
- Coordination types: Workflow coordination and orchestration results
- Workflow types: Code generation workflow results
- Metrics types: Performance and health monitoring
- Infrastructure types: Kafka, database, and service health
"""

from typing import Any, TypedDict

# ============================================================
# Coordination Orchestration Types
# ============================================================


class CoordinationResultDict(TypedDict, total=False):
    """Result from CoordinationOrchestrator.coordinate_workflow().

    Attributes:
        workflow_id: Workflow identifier
        coordination_id: Generated coordination ID
        duration_ms: Total coordination time in milliseconds
        contexts_distributed: Number of contexts distributed
        dependencies_resolved: Number of dependencies resolved
        signals_sent: Number of signals sent
        routing_decisions: Number of routing decisions made
        agent_contexts: Distributed agent contexts (dict of agent_id -> context dict)
        error: Error message if coordination failed
        dependency_details: Details about dependency resolution
        routing_details: Details about routing decisions
    """

    workflow_id: str
    coordination_id: str
    duration_ms: float
    contexts_distributed: int
    dependencies_resolved: int
    signals_sent: int
    routing_decisions: int
    agent_contexts: dict[str, dict[str, Any]]
    error: str  # Optional, only present on failure
    dependency_details: list[dict[str, Any]]  # Optional
    routing_details: list[dict[str, Any]]  # Optional


class SignalMetricsDict(TypedDict):
    """Signal coordination metrics.

    Attributes:
        total_signals_sent: Total number of signals sent
        successful_signals: Number of successful signals
        failed_signals: Number of failed signals
    """

    total_signals_sent: int
    successful_signals: int
    failed_signals: int


class ContextMetricsDict(TypedDict):
    """Context distribution metrics.

    Attributes:
        total_contexts: Total contexts distributed
        successful_distributions: Number of successful distributions
        failed_distributions: Number of failed distributions
    """

    total_contexts: int
    successful_distributions: int
    failed_distributions: int


class RoutingMetricsDict(TypedDict):
    """Routing decision metrics.

    Attributes:
        total_decisions: Total routing decisions made
        successful_decisions: Number of successful decisions
        failed_decisions: Number of failed decisions
    """

    total_decisions: int
    successful_decisions: int
    failed_decisions: int


class CoordinationMetricsDict(TypedDict):
    """Comprehensive coordination metrics from get_coordination_metrics().

    Attributes:
        signal_metrics: Signal coordination metrics
        context_metrics: Context distribution metrics
        routing_metrics: Routing decision metrics
    """

    signal_metrics: SignalMetricsDict
    context_metrics: ContextMetricsDict
    routing_metrics: RoutingMetricsDict


class DependencyResolutionDict(TypedDict):
    """Dependency resolution result.

    Attributes:
        resolved_count: Number of dependencies resolved
        details: List of dependency resolution details
    """

    resolved_count: int
    details: list[dict[str, Any]]


class RoutingResultDict(TypedDict):
    """Routing result for workflow tasks.

    Attributes:
        decision_count: Number of routing decisions made
        details: List of routing decision details
    """

    decision_count: int
    details: list[dict[str, Any]]


# ============================================================
# Code Generation Workflow Types
# ============================================================


class ParsedContractDict(TypedDict):
    """Result from _parse_contract_executor().

    Attributes:
        contract_path: Path to contract file
        node_type: Type of node (effect, compute, etc.)
        node_name: Name of the node
        version: Node version
    """

    contract_path: str
    node_type: str
    node_name: str
    version: str


class GeneratedCodeDict(TypedDict):
    """Result from code generation executors (model, validator, test).

    Attributes:
        template_id: Template identifier used
        generated_code: Generated code content
        size_bytes: Size of generated code in bytes
    """

    template_id: str
    generated_code: str
    size_bytes: int


class ValidationResultDict(TypedDict, total=False):
    """Result from _validate_code_executor().

    Attributes:
        pipeline_passed: Whether validation pipeline passed
        pipeline_score: Overall validation score
        pipeline_duration_ms: Pipeline duration in milliseconds
        quorum_passed: Whether AI quorum validation passed (optional)
        quorum_score: AI quorum consensus score (optional)
        quorum_duration_ms: AI quorum duration in milliseconds (optional)
    """

    pipeline_passed: bool
    pipeline_score: float
    pipeline_duration_ms: float
    quorum_passed: bool | None
    quorum_score: float | None
    quorum_duration_ms: float | None


class PackageInfoDict(TypedDict, total=False):
    """Result from _package_node_executor().

    Attributes:
        package_path: Path to generated package
        package_size_bytes: Size of package in bytes
        packaged_at: Timestamp when packaged
    """

    package_path: str | None
    package_size_bytes: int
    packaged_at: float


class ErrorRecoveryStatsDict(TypedDict):
    """Error recovery statistics from error recovery orchestrator.

    Attributes:
        total_attempts: Total recovery attempts
        successful_recoveries: Number of successful recoveries
        success_rate: Recovery success rate (0.0-1.0)
    """

    total_attempts: int
    successful_recoveries: int
    success_rate: float


class WorkflowStatisticsDict(TypedDict, total=False):
    """Workflow statistics from get_statistics().

    Attributes:
        generation_count: Number of code generations performed
        total_duration_ms: Total duration of all generations
        avg_duration_ms: Average duration per generation
        template_hit_rate: Template cache hit rate
        template_cache_size: Current template cache size
        template_avg_load_ms: Average template load time
        quorum_validations: Total AI quorum validations (optional)
        quorum_pass_rate: AI quorum pass rate (optional)
        optimization_enabled: Whether optimization is enabled (optional)
        error_recovery: Error recovery statistics (optional)
        performance_optimization: Performance optimization summary (optional)
        monitoring: Monitoring status (optional)
    """

    generation_count: int
    total_duration_ms: float
    avg_duration_ms: float
    template_hit_rate: float
    template_cache_size: int
    template_avg_load_ms: float
    quorum_validations: int
    quorum_pass_rate: float
    optimization_enabled: bool
    error_recovery: dict[str, Any]
    performance_optimization: dict[str, Any]
    monitoring: dict[str, Any]


class RecoveryStatsDict(TypedDict):
    """Detailed error recovery statistics from get_recovery_stats().

    Attributes:
        enabled: Whether error recovery is enabled
        total_attempts: Total recovery attempts
        successful_recoveries: Number of successful recoveries
        failed_recoveries: Number of failed recoveries
        success_rate: Recovery success rate (0.0-1.0)
        strategies_used: Count of each strategy used
        error_types_seen: Count of each error type encountered
    """

    enabled: bool
    total_attempts: int
    successful_recoveries: int
    failed_recoveries: int
    success_rate: float
    strategies_used: dict[str, int]
    error_types_seen: dict[str, int]


class SLAComplianceDict(TypedDict, total=False):
    """SLA compliance report from get_sla_compliance().

    Attributes:
        enabled: Whether monitoring is enabled
        overall_compliant: Overall SLA compliance status
        violations: List of SLA violations
        metrics: Current metrics being monitored
    """

    enabled: bool
    overall_compliant: bool
    violations: list[dict[str, Any]]
    metrics: dict[str, float]


class OptimizationSummaryDict(TypedDict):
    """Performance optimization summary from get_optimization_summary().

    Attributes:
        enabled: Whether optimization is enabled
        total_optimizations: Total optimizations applied
        avg_speedup: Average speedup achieved
    """

    enabled: bool
    total_optimizations: int
    avg_speedup: float


# ============================================================
# Kafka Client Types
# ============================================================


class KafkaHealthCheckDict(TypedDict, total=False):
    """Kafka health check result from health_check().

    Attributes:
        status: Health status (healthy/unhealthy)
        connected: Whether connected to Kafka
        error: Error message if unhealthy (optional)
        bootstrap_servers: Kafka bootstrap servers (optional)
        producer_active: Whether producer is active (optional)
    """

    status: str
    connected: bool
    error: str
    bootstrap_servers: str
    producer_active: bool


class ConnectionStatusDict(TypedDict):
    """Kafka connection status information.

    Attributes:
        connected: Whether connected
        bootstrap_servers: Kafka bootstrap servers
        connection_timeout: Connection timeout in seconds
    """

    connected: bool
    bootstrap_servers: str
    connection_timeout: int


class ResilienceConfigDict(TypedDict):
    """Kafka resilience configuration.

    Attributes:
        dead_letter_queue_enabled: Whether DLQ is enabled
        max_retry_attempts: Maximum retry attempts
        retry_backoff_base: Retry backoff base delay
        circuit_breaker_enabled: Whether circuit breaker is enabled
    """

    dead_letter_queue_enabled: bool
    max_retry_attempts: int
    retry_backoff_base: float
    circuit_breaker_enabled: bool


class FailureStatisticsDict(TypedDict):
    """Kafka failure statistics.

    Attributes:
        total_failures: Total number of failures
        recent_failures_1h: Failures in last hour
        active_retries: Number of active retries
        failed_messages_in_dlq: Messages in dead letter queue
    """

    total_failures: int
    recent_failures_1h: int
    active_retries: int
    failed_messages_in_dlq: int


class PerformanceMetricsDict(TypedDict):
    """Kafka performance metrics.

    Attributes:
        retry_counts: Retry counts per topic/key
        last_failure_time: Timestamp of last failure (ISO format)
    """

    retry_counts: dict[str, int]
    last_failure_time: str | None


class KafkaResilienceMetricsDict(TypedDict):
    """Kafka resilience metrics from get_resilience_metrics().

    Attributes:
        connection_status: Connection status information
        resilience_config: Resilience configuration
        failure_statistics: Failure statistics
        performance_metrics: Performance metrics
    """

    connection_status: ConnectionStatusDict
    resilience_config: ResilienceConfigDict
    failure_statistics: FailureStatisticsDict
    performance_metrics: PerformanceMetricsDict


class EnvelopePublishingDict(TypedDict):
    """Envelope publishing statistics.

    Attributes:
        total_events_published: Total events published
        total_events_failed: Total events failed
        success_rate: Publishing success rate
    """

    total_events_published: int
    total_events_failed: int
    success_rate: float


class LatencyMetricsDict(TypedDict):
    """Envelope publishing latency metrics.

    Attributes:
        average: Average latency in milliseconds
        p50: 50th percentile latency
        p95: 95th percentile latency
        p99: 99th percentile latency
        min: Minimum latency
        max: Maximum latency
        sample_count: Number of samples
    """

    average: float
    p50: float
    p95: float
    p99: float
    min: float
    max: float
    sample_count: int


class PerformanceSummaryDict(TypedDict):
    """Performance summary for envelope publishing.

    Attributes:
        meets_target_latency: Whether average latency meets target (<100ms)
        meets_success_rate: Whether success rate meets target (≥95%)
    """

    meets_target_latency: bool
    meets_success_rate: bool


class KafkaEnvelopeMetricsDict(TypedDict):
    """Kafka envelope metrics from get_envelope_metrics().

    Attributes:
        envelope_publishing: Publishing statistics
        latency_metrics_ms: Latency metrics in milliseconds
        performance_summary: Performance summary
    """

    envelope_publishing: EnvelopePublishingDict
    latency_metrics_ms: LatencyMetricsDict
    performance_summary: PerformanceSummaryDict


class KafkaRetryResultDict(TypedDict):
    """Result from retry_failed_messages().

    Attributes:
        total: Total messages attempted to retry
        successful: Number of successful retries
        failed: Number of failed retries
    """

    total: int
    successful: int
    failed: int


# ============================================================
# LLM Client Types
# ============================================================


class LLMResponseHeadersDict(TypedDict):
    """HTTP headers for LLM API requests.

    Attributes:
        Content-Type: Content type header
        Authorization: Authorization header (optional)
    """

    # Using string keys for HTTP headers
    pass  # Will be defined with string keys: dict[str, str]


# Note: LLM client methods return Tuple[bool, float, str] not dict,
# but _make_request_with_retry returns dict[str, Any] for API responses
