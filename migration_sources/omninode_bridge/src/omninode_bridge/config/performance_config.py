"""Performance configuration for omninode bridge nodes.

This module provides environment-based configuration for timeouts, thresholds,
and performance-related settings across orchestrator and reducer nodes.

All values have sensible defaults and can be overridden via environment variables.
"""

import os


def get_int(env_var: str, default: int) -> int:
    """Get integer value from environment variable with fallback to default.

    Args:
        env_var: Environment variable name
        default: Default value if env var not set

    Returns:
        Integer value from environment or default
    """
    value = os.getenv(env_var)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def get_float(env_var: str, default: float) -> float:
    """Get float value from environment variable with fallback to default.

    Args:
        env_var: Environment variable name
        default: Default value if env var not set

    Returns:
        Float value from environment or default
    """
    value = os.getenv(env_var)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


# ============================================================================
# Orchestrator Timeout Configuration
# ============================================================================

# Event-driven workflow coordination timeout
# Default: 30.0 seconds
# Used by: NodeBridgeOrchestrator._execute_event_driven_workflow()
# Context: Maximum time to wait for reducer completion event
WORKFLOW_COMPLETION_TIMEOUT_SECONDS = get_float(
    "WORKFLOW_COMPLETION_TIMEOUT_SECONDS", 30.0
)

# OnexTree intelligence service timeout
# Default: 0.5 seconds (500ms for fast graceful degradation)
# Used by: NodeBridgeOrchestrator (onextree_timeout_ms container config)
# Context: Maximum time for OnexTree intelligence analysis
ONEXTREE_INTELLIGENCE_TIMEOUT_SECONDS = get_float(
    "ONEXTREE_INTELLIGENCE_TIMEOUT_SECONDS", 0.5
)

# Metadata stamping client timeout
# Default: 30.0 seconds
# Used by: AsyncMetadataStampingClient initialization
# Context: HTTP client timeout for stamping service calls
METADATA_STAMPING_CLIENT_TIMEOUT_SECONDS = get_float(
    "METADATA_STAMPING_CLIENT_TIMEOUT_SECONDS", 30.0
)

# Kafka client timeout
# Default: 30 seconds
# Used by: KafkaClient initialization
# Context: Kafka connection and operation timeout
KAFKA_CLIENT_TIMEOUT_SECONDS = get_int("KAFKA_CLIENT_TIMEOUT_SECONDS", 30)


# ============================================================================
# Reducer Aggregation Configuration
# ============================================================================

# Maximum cardinality samples for bounded memory tracking
# Default: 100 samples
# Used by: NodeBridgeReducer.execute_reduction()
# Context: Prevents unbounded memory growth for file_types and workflow_ids sets
MAX_CARDINALITY_SAMPLES = get_int("MAX_CARDINALITY_SAMPLES", 100)


# ============================================================================
# FSM Cache and Buffer Thresholds
# ============================================================================

# FSM cache warning threshold
# Default: 10,000 workflows
# Used by: NodeBridgeReducer._check_aggregation_buffer_health()
# Context: Triggers health status degradation when FSM cache grows too large
FSM_CACHE_WARNING_THRESHOLD = get_int("FSM_CACHE_WARNING_THRESHOLD", 10000)

# Aggregation buffer warning threshold
# Default: 10,000 items
# Used by: NodeBridgeReducer._check_aggregation_buffer_health()
# Context: Triggers health status degradation when aggregation buffer grows too large
AGGREGATION_BUFFER_WARNING_THRESHOLD = get_int(
    "AGGREGATION_BUFFER_WARNING_THRESHOLD", 10000
)


# ============================================================================
# Health Check Timeout Configuration
# ============================================================================

# Metadata stamping service health check timeout
# Default: 3.0 seconds
# Used by: NodeBridgeOrchestrator._check_metadata_stamping_health()
# Context: HTTP timeout for health check endpoint
METADATA_STAMPING_HEALTH_CHECK_TIMEOUT_SECONDS = get_float(
    "METADATA_STAMPING_HEALTH_CHECK_TIMEOUT_SECONDS", 3.0
)

# OnexTree service health check timeout
# Default: 3.0 seconds
# Used by: NodeBridgeOrchestrator._check_onextree_health()
# Context: HTTP timeout for health check endpoint
ONEXTREE_HEALTH_CHECK_TIMEOUT_SECONDS = get_float(
    "ONEXTREE_HEALTH_CHECK_TIMEOUT_SECONDS", 3.0
)

# Kafka health check timeout
# Default: 3.0 seconds
# Used by: NodeBridgeOrchestrator._check_kafka_health()
# Context: Kafka client health check operation timeout
KAFKA_HEALTH_CHECK_TIMEOUT_SECONDS = get_float(
    "KAFKA_HEALTH_CHECK_TIMEOUT_SECONDS", 3.0
)

# EventBus health check timeout
# Default: 3.0 seconds
# Used by: NodeBridgeOrchestrator._check_event_bus_health()
# Context: EventBus service health check timeout
EVENT_BUS_HEALTH_CHECK_TIMEOUT_SECONDS = get_float(
    "EVENT_BUS_HEALTH_CHECK_TIMEOUT_SECONDS", 3.0
)

# Aggregation buffer health check timeout
# Default: 1.0 second
# Used by: NodeBridgeReducer._register_component_checks()
# Context: Internal memory check timeout (fast in-memory operation)
AGGREGATION_BUFFER_HEALTH_CHECK_TIMEOUT_SECONDS = get_float(
    "AGGREGATION_BUFFER_HEALTH_CHECK_TIMEOUT_SECONDS", 1.0
)


# ============================================================================
# Performance Tuning Guidelines
# ============================================================================

"""
Environment-Specific Recommendations:

Development:
- WORKFLOW_COMPLETION_TIMEOUT_SECONDS=15.0  # Faster failure for dev iteration
- MAX_CARDINALITY_SAMPLES=50                # Smaller memory footprint
- FSM_CACHE_WARNING_THRESHOLD=1000          # Earlier warnings

Staging:
- WORKFLOW_COMPLETION_TIMEOUT_SECONDS=30.0  # Default settings
- MAX_CARDINALITY_SAMPLES=100
- FSM_CACHE_WARNING_THRESHOLD=10000

Production:
- WORKFLOW_COMPLETION_TIMEOUT_SECONDS=60.0  # More tolerance for high load
- MAX_CARDINALITY_SAMPLES=200               # Better cardinality tracking
- FSM_CACHE_WARNING_THRESHOLD=50000         # Higher threshold for scale

Load Testing:
- WORKFLOW_COMPLETION_TIMEOUT_SECONDS=5.0   # Quick failure detection
- MAX_CARDINALITY_SAMPLES=1000              # Stress test memory management
- FSM_CACHE_WARNING_THRESHOLD=100000        # Test high concurrency
"""
