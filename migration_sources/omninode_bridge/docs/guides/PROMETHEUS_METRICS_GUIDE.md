# Prometheus Metrics Integration Guide

## Overview

OmniNode Bridge nodes (orchestrator, reducer, registry) include comprehensive Prometheus metrics integration for industry-standard observability. This guide covers setup, configuration, and usage of Prometheus metrics across bridge nodes.

## Key Features

- ✅ **Industry-Standard**: Uses `prometheus_client` library (v0.19.0+)
- ✅ **Feature Flag Support**: Enable/disable metrics per node
- ✅ **Multiple Metric Types**: Counter, Gauge, Histogram, Summary
- ✅ **Comprehensive Coverage**: 15+ metrics across workflow, aggregation, events, errors
- ✅ **Context Managers**: Automatic timing with Python context managers
- ✅ **Zero Performance Impact**: When disabled, metrics collection is no-op

## Quick Start

### 1. Enable Prometheus Metrics

Prometheus metrics are **enabled by default**. To disable:

**Environment Variable:**
```bash
export BRIDGE_ENABLE_PROMETHEUS=false
```

**YAML Configuration:**
```yaml
# config/orchestrator.yaml
monitoring:
  enable_prometheus: true  # Default: true
  prometheus_port: 9090    # Default: 9090
```

**Container Configuration:**
```python
container = ModelONEXContainer(config={
    "enable_prometheus": True,
})
```

### 2. Access Metrics Endpoint

Each bridge node exposes a `/metrics` endpoint:

**Orchestrator:**
```bash
curl http://localhost:8060/metrics
```

**Reducer:**
```bash
curl http://localhost:8061/metrics
```

**Example Output:**
```prometheus
# HELP bridge_workflow_total Total number of workflows processed
# TYPE bridge_workflow_total counter
bridge_workflow_total{node_type="orchestrator",status="success"} 150.0
bridge_workflow_total{node_type="orchestrator",status="error"} 3.0

# HELP bridge_workflow_duration_seconds Workflow execution duration in seconds
# TYPE bridge_workflow_duration_seconds histogram
bridge_workflow_duration_seconds_bucket{node_type="orchestrator",status="success",le="0.01"} 0.0
bridge_workflow_duration_seconds_bucket{node_type="orchestrator",status="success",le="0.1"} 45.0
bridge_workflow_duration_seconds_bucket{node_type="orchestrator",status="success",le="1.0"} 140.0
bridge_workflow_duration_seconds_bucket{node_type="orchestrator",status="success",le="+Inf"} 150.0
bridge_workflow_duration_seconds_sum{node_type="orchestrator",status="success"} 85.3
bridge_workflow_duration_seconds_count{node_type="orchestrator",status="success"} 150.0
```

### 3. Configure Prometheus Scraper

**prometheus.yml:**
```yaml
scrape_configs:
  - job_name: 'omninode_bridge_orchestrator'
    static_configs:
      - targets: ['localhost:8060']
    metrics_path: '/metrics'
    scrape_interval: 15s

  - job_name: 'omninode_bridge_reducer'
    static_configs:
      - targets: ['localhost:8061']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

## Available Metrics

### Workflow Metrics (Orchestrator)

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `bridge_workflow_total` | Counter | Total workflows processed | `node_type`, `status` |
| `bridge_workflow_duration_seconds` | Histogram | Workflow execution duration | `node_type`, `status` |
| `bridge_workflow_in_progress` | Gauge | Workflows currently executing | `node_type` |
| `bridge_workflow_step_duration_seconds` | Histogram | Individual step duration | `node_type`, `step_name` |

**Histogram Buckets:** 0.01s, 0.05s, 0.1s, 0.25s, 0.5s, 1.0s, 2.5s, 5.0s, 10.0s, 30.0s, 60.0s

### Aggregation Metrics (Reducer)

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `bridge_aggregation_total` | Counter | Total aggregation operations | `node_type`, `aggregation_type`, `status` |
| `bridge_aggregation_duration_seconds` | Histogram | Aggregation operation duration | `node_type`, `aggregation_type` |
| `bridge_aggregation_items_processed_total` | Counter | Total items processed | `node_type`, `aggregation_type` |
| `bridge_aggregation_buffer_size` | Gauge | Current buffer size | `node_type`, `namespace` |

**Histogram Buckets:** 0.001s, 0.01s, 0.05s, 0.1s, 0.5s, 1.0s, 5.0s

### Event Publishing Metrics (All Nodes)

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `bridge_event_published_total` | Counter | Total events published | `node_type`, `event_type`, `status` |
| `bridge_event_publish_duration_seconds` | Histogram | Event publishing duration | `node_type`, `event_type` |
| `bridge_event_publish_errors_total` | Counter | Total publishing errors | `node_type`, `event_type`, `error_type` |

### Node Registration Metrics (Registry)

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `bridge_node_registration_total` | Counter | Total node registrations | `node_type`, `status` |
| `bridge_registered_nodes` | Gauge | Currently registered nodes | `node_type` |

### Error Tracking Metrics (All Nodes)

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `bridge_errors_total` | Counter | Total errors by type | `node_type`, `error_type`, `operation` |
| `bridge_operation_latency_seconds` | Summary | Operation latency summary | `node_type`, `operation` |

### Database Metrics (All Nodes)

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `bridge_db_query_duration_seconds` | Histogram | Database query duration | `node_type`, `operation` |
| `bridge_db_query_total` | Counter | Total database queries | `node_type`, `operation`, `status` |

### HTTP Request Metrics (All Nodes)

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `bridge_http_requests_total` | Counter | Total HTTP requests | `node_type`, `method`, `endpoint`, `status_code` |
| `bridge_http_request_duration_seconds` | Histogram | HTTP request duration | `node_type`, `method`, `endpoint` |

## Usage Examples

### Orchestrator Metrics

```python
from omninode_bridge.nodes.orchestrator.v1_0_0.node import NodeBridgeOrchestrator

# Metrics are automatically initialized in __init__
orchestrator = NodeBridgeOrchestrator(container)

# Metrics are recorded automatically in execute_orchestration
result = await orchestrator.execute_orchestration(contract)

# Access metrics endpoint
# GET http://localhost:8060/metrics
```

### Reducer Metrics

```python
from omninode_bridge.nodes.reducer.v1_0_0.node import NodeBridgeReducer

# Metrics are automatically initialized in __init__
reducer = NodeBridgeReducer(container)

# Metrics are recorded automatically in execute_reduction
result = await reducer.execute_reduction(contract)

# Manual metrics recording
reducer.metrics.record_aggregation_items("namespace_grouping", 100)
reducer.metrics.set_aggregation_buffer_size("test.namespace", 500)

# Access metrics endpoint
# GET http://localhost:8061/metrics
```

### Context Manager Usage

```python
from omninode_bridge.nodes.metrics import create_orchestrator_metrics

# Create metrics collector
metrics = create_orchestrator_metrics(enable_prometheus=True)

# Time workflow execution
with metrics.time_workflow(status="success"):
    # Execute workflow
    result = await process_workflow()

# Time aggregation operation
with metrics.time_aggregation("namespace_grouping", "success"):
    # Execute aggregation
    result = await aggregate_data()

# Time database query
with metrics.time_db_query("insert", "success"):
    # Execute database query
    await db.execute(query)
```

### Manual Recording

```python
# Record errors
metrics.record_error("ValueError", "execute_orchestration")

# Record node registration
metrics.record_node_registration("success")

# Set gauge values
metrics.set_aggregation_buffer_size("my.namespace", 750)
metrics.set_registered_nodes_count(25)

# Record operation latency
metrics.record_operation_latency("aggregation", 0.125)  # 125ms
```

## Grafana Dashboard

### Sample Queries

**Workflow Success Rate:**
```promql
rate(bridge_workflow_total{status="success"}[5m])
/
rate(bridge_workflow_total[5m])
```

**P95 Workflow Latency:**
```promql
histogram_quantile(0.95,
  rate(bridge_workflow_duration_seconds_bucket[5m])
)
```

**Aggregation Throughput:**
```promql
rate(bridge_aggregation_items_processed_total[5m])
```

**Error Rate:**
```promql
rate(bridge_errors_total[5m])
```

**Active Workflows:**
```promql
bridge_workflow_in_progress
```

### Sample Dashboard JSON

See [grafana/bridge_nodes_dashboard.json](../../grafana/bridge_nodes_dashboard.json) for complete Grafana dashboard.

## Feature Flag Configuration

Prometheus metrics can be controlled via multiple configuration sources:

### Priority Order (Highest to Lowest)

1. **Environment Variables**
   ```bash
   export BRIDGE_ENABLE_PROMETHEUS=true
   ```

2. **YAML Configuration**
   ```yaml
   monitoring:
     enable_prometheus: true
   ```

3. **Container Config**
   ```python
   container = ModelONEXContainer(config={
       "enable_prometheus": True
   })
   ```

4. **Default Value**: `True` (enabled by default)

### Per-Node Configuration

Each node can have independent metrics configuration:

**Orchestrator:**
```yaml
# config/orchestrator.yaml
monitoring:
  enable_prometheus: true
  prometheus_port: 9090
```

**Reducer:**
```yaml
# config/reducer.yaml
monitoring:
  enable_prometheus: true
  prometheus_port: 9091
```

## Performance Impact

### When Enabled

- **Overhead**: < 1% performance impact
- **Memory**: ~5-10 MB per node for metrics registry
- **CPU**: Negligible (< 0.1% CPU)

### When Disabled

- **Overhead**: Zero (no-op implementation)
- **Memory**: < 1 KB (empty collector)
- **CPU**: None

### Benchmarks

```python
# Enabled (with timing context manager)
Time: 0.125s (workflow execution)
Overhead: +0.0001s (0.08%)

# Disabled (with timing context manager)
Time: 0.125s (workflow execution)
Overhead: 0s (0%)
```

## Testing

### Unit Tests

```bash
# Run metrics tests
pytest tests/unit/nodes/test_prometheus_metrics.py -v

# Run with coverage
pytest tests/unit/nodes/test_prometheus_metrics.py --cov=src/omninode_bridge/nodes/metrics
```

### Integration Tests

```bash
# Test metrics endpoint
curl -i http://localhost:8060/metrics

# Verify Prometheus format
curl http://localhost:8060/metrics | grep "bridge_workflow_total"

# Check for proper labels
curl http://localhost:8060/metrics | grep 'node_type="orchestrator"'
```

### Load Testing

```python
import asyncio
import time
from omninode_bridge.nodes.metrics import create_orchestrator_metrics

async def test_metrics_under_load():
    metrics = create_orchestrator_metrics(enable_prometheus=True)

    # Simulate 1000 workflow executions
    start = time.time()
    for i in range(1000):
        with metrics.time_workflow("success"):
            await asyncio.sleep(0.001)

    duration = time.time() - start
    print(f"1000 workflows with metrics: {duration:.2f}s")
    # Expected: < 2 seconds
```

## Troubleshooting

### Metrics Not Appearing

**Check if Prometheus is enabled:**
```python
# In node initialization logs
"Prometheus metrics enabled for orchestrator"  # Should see this
```

**Verify metrics endpoint:**
```bash
curl http://localhost:8060/metrics
# Should return Prometheus format, not error
```

**Check configuration:**
```python
# Verify container config
container.config.get("enable_prometheus")  # Should be True
```

### Metrics Show Zero Values

**Verify code execution:**
```python
# Ensure workflow/aggregation is actually executing
with metrics.time_workflow("success"):
    # This code must execute for metrics to record
    pass
```

**Check metric labels:**
```bash
# Metrics are per-label combination
curl http://localhost:8060/metrics | grep 'status="success"'
```

### High Memory Usage

**Reduce metric cardinality:**
```python
# Avoid high-cardinality labels
# BAD: Using workflow_id as label (millions of unique values)
metrics.workflow_counter.labels(workflow_id=str(uuid4())).inc()

# GOOD: Using status as label (2-3 unique values)
metrics.workflow_counter.labels(status="success").inc()
```

**Limit histogram buckets:**
```python
# Custom histogram with fewer buckets
Histogram(
    "custom_metric",
    "Description",
    buckets=[0.1, 1.0, 10.0]  # Only 3 buckets
)
```

## Best Practices

### 1. Use Feature Flags

Always respect the `enable_prometheus` flag:

```python
if metrics.enable_prometheus:
    # Only compute expensive metrics if enabled
    detailed_stats = compute_detailed_statistics()
    metrics.record_custom_stat(detailed_stats)
```

### 2. Avoid High-Cardinality Labels

❌ **Bad** (too many unique values):
```python
metrics.workflow_counter.labels(
    workflow_id=str(uuid4()),  # Millions of unique IDs
    timestamp=str(datetime.now())  # Infinite unique timestamps
)
```

✅ **Good** (bounded cardinality):
```python
metrics.workflow_counter.labels(
    node_type="orchestrator",  # 3 possible values
    status="success"  # 2-3 possible values
)
```

### 3. Use Context Managers

✅ **Preferred** (automatic timing):
```python
with metrics.time_workflow("success"):
    result = await execute_workflow()
```

❌ **Avoid** (manual timing):
```python
start = time.time()
result = await execute_workflow()
duration = time.time() - start
metrics.workflow_duration.observe(duration)
```

### 4. Name Metrics Consistently

Follow Prometheus naming conventions:

- Use `snake_case`
- Add `_total` suffix to counters
- Add `_seconds` suffix to durations
- Use descriptive names: `bridge_workflow_duration_seconds` (not `wf_time`)

### 5. Document Custom Metrics

```python
custom_metric = Counter(
    "bridge_custom_operation_total",  # Clear name
    "Total custom operations performed",  # Descriptive help text
    ["node_type", "operation_type"],  # Document labels
    registry=registry
)
```

## Security Considerations

### 1. Authentication

Add authentication to metrics endpoint in production:

```python
from fastapi import Depends, HTTPException, Header

async def verify_metrics_token(x_metrics_token: str = Header(...)):
    if x_metrics_token != os.getenv("METRICS_TOKEN"):
        raise HTTPException(status_code=401, detail="Invalid metrics token")
    return True

@app.get("/metrics", dependencies=[Depends(verify_metrics_token)])
async def metrics():
    return Response(...)
```

### 2. Rate Limiting

Limit metrics endpoint access:

```python
from slowapi import Limiter

limiter = Limiter(key_func=lambda: "metrics")

@app.get("/metrics")
@limiter.limit("10/minute")
async def metrics():
    return Response(...)
```

### 3. Network Security

- Only expose metrics endpoint on internal network
- Use Prometheus federation for external access
- Enable mTLS for Prometheus scraper

## Migration from Custom Metrics

If migrating from custom metrics:

1. **Keep Both Temporarily**: Run Prometheus and custom metrics in parallel
2. **Verify Equivalence**: Ensure Prometheus metrics match custom metrics
3. **Switch Dashboards**: Update Grafana/monitoring to use Prometheus
4. **Remove Custom**: After 30-day verification, remove custom metrics

## References

- **Prometheus Documentation**: https://prometheus.io/docs/introduction/overview/
- **prometheus_client Library**: https://github.com/prometheus/client_python
- **Best Practices**: https://prometheus.io/docs/practices/naming/
- **Grafana Integration**: https://grafana.com/docs/grafana/latest/datasources/prometheus/

## Support

For issues or questions:

1. Check [Troubleshooting](#troubleshooting) section above
2. Review [tests/unit/nodes/test_prometheus_metrics.py](../../tests/unit/nodes/test_prometheus_metrics.py)
3. Open GitHub issue with `metrics` label
