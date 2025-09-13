# Observability Enhancement Specifications

## Target: agent-production-monitor

### 1. Detailed Performance Tracking for Database Operations

**File:** `src/omnibase_infra/infrastructure/postgres_connection_manager.py`
**Enhancements needed:**

```python
@dataclass
class DatabaseOperationMetrics:
    """Enhanced database operation metrics."""
    operation_type: str
    duration_ms: float
    query_complexity_score: int
    connection_pool_usage: float
    cache_hit_ratio: float
    rows_affected: int
    bytes_transferred: int
    execution_plan_cost: Optional[float] = None
    connection_acquire_time_ms: float = 0.0
    query_parse_time_ms: float = 0.0
    query_execution_time_ms: float = 0.0
```

**Metrics to add:**
- Query execution time breakdown (parse, plan, execute)
- Connection acquisition timing
- Connection pool utilization percentages
- Cache hit/miss ratios for prepared statements
- Query complexity scoring
- Bytes transferred per operation
- Connection lifecycle metrics

### 2. Enhanced Circuit Breaker Metrics with Latency Tracking

**File:** `src/omnibase_infra/infrastructure/event_bus_circuit_breaker.py`
**Enhancements needed:**

```python
@dataclass
class CircuitBreakerLatencyMetrics:
    """Enhanced circuit breaker metrics with latency tracking."""
    operation_latency_p50: float
    operation_latency_p95: float
    operation_latency_p99: float
    state_transition_duration_ms: float
    queue_processing_latency_ms: float
    dead_letter_processing_latency_ms: float
    recovery_test_latency_ms: float
    circuit_open_duration_ms: float
```

**Metrics to add:**
- Latency percentiles (P50, P95, P99) for all operations
- State transition timing
- Queue processing latency
- Dead letter queue processing time
- Recovery test execution time
- Circuit open/close duration tracking

### 3. Operation Type-Specific Metrics

**Files:** All infrastructure components
**Enhancements needed:**

```python
@dataclass
class OperationTypeMetrics:
    """Operation type-specific performance metrics."""
    read_operations: Dict[str, float]
    write_operations: Dict[str, float]
    admin_operations: Dict[str, float]
    health_check_operations: Dict[str, float]
    connection_operations: Dict[str, float]
```

**Metrics to add:**
- Separate tracking for read vs write operations
- Administrative operation timing
- Health check operation performance
- Connection management operation timing
- Operation success/failure rates by type

### 4. Producer Pool Performance Metrics

**File:** `src/omnibase_infra/infrastructure/kafka_producer_pool.py`
**Enhancements needed:**

```python
@dataclass
class ProducerPoolPerformanceMetrics:
    """Enhanced producer pool performance metrics."""
    pool_scaling_events: int
    producer_creation_time_ms: float
    producer_destruction_time_ms: float
    message_throughput_per_second: float
    producer_utilization_percentage: float
    pool_efficiency_score: float
    connection_establishment_time_ms: float
    message_serialization_time_ms: float
```

**Metrics to add:**
- Pool scaling event tracking
- Producer lifecycle timing
- Message throughput calculations  
- Producer utilization efficiency
- Pool efficiency scoring
- Connection establishment timing
- Message serialization performance

### 5. Enhanced Connection Lifecycle Tracking

**File:** `src/omnibase_infra/infrastructure/postgres_connection_manager.py`
**Enhancements needed:**

```python
@dataclass
class ConnectionLifecycleMetrics:
    """Enhanced connection lifecycle tracking."""
    connection_creation_time_ms: float
    connection_authentication_time_ms: float
    connection_idle_time_ms: float
    connection_active_time_ms: float
    connection_close_time_ms: float
    connection_error_recovery_time_ms: float
    connection_pool_wait_time_ms: float
```

**Metrics to add:**
- Connection creation and authentication timing
- Connection idle vs active time tracking
- Connection close and cleanup timing
- Error recovery timing
- Pool wait time tracking
- Connection reuse efficiency

## Observability Integration Requirements

### Prometheus Metrics Export
- All new metrics must be exportable to Prometheus
- Metric names must follow Prometheus naming conventions
- Metrics must include appropriate labels for filtering

### Structured Logging Enhancement
- All performance metrics logged as structured data
- Log levels appropriate for production monitoring
- Correlation IDs for tracing across operations

### Dashboard Integration
- Metrics must be compatible with existing dashboard infrastructure
- Time-series data for trend analysis
- Alerting thresholds configurable

## Success Metrics

- 100% observability coverage for all infrastructure operations
- <1% performance overhead from metrics collection
- Real-time visibility into all component performance
- Actionable alerts for performance degradation
- Trend analysis capability for capacity planning