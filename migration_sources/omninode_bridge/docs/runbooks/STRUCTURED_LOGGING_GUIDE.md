# Structured Logging Guide - Pure Reducer Architecture

**Version**: 1.0.0
**Last Updated**: 2025-10-21
**Wave**: 6C - Metrics & Observability

## Overview

All Pure Reducer services implement structured logging with correlation IDs and context propagation. This guide documents the existing logging patterns and best practices.

## Logging Architecture

### Standard Python Logging with Extra Context

All services use Python's `logging` module with structured `extra` parameters:

```python
import logging

logger = logging.getLogger(__name__)

logger.info(
    "State committed successfully",
    extra={
        "workflow_key": workflow_key,
        "version": new_version,
        "correlation_id": correlation_id,
        "latency_ms": latency_ms,
    }
)
```

### Log Levels

- **DEBUG**: Detailed diagnostic information (disabled in production)
- **INFO**: General informational messages (successful operations)
- **WARNING**: Warning messages (retries, non-critical issues)
- **ERROR**: Error messages (failures, exceptions)

## Service-Specific Logging Patterns

### CanonicalStoreService

**State Commit Success**:
```python
logger.info(
    f"Successfully committed state for '{workflow_key}': v{expected_version} → v{new_version}",
    extra={
        "workflow_key": workflow_key,
        "expected_version": expected_version,
        "new_version": new_version,
        "correlation_id": event.correlation_id,
        "latency_ms": round(latency_ms, 2),
    },
)
```

**State Conflict**:
```python
logger.warning(
    f"Version conflict for '{workflow_key}': expected v{expected_version}, actual v{actual_version}",
    extra={
        "workflow_key": workflow_key,
        "expected_version": expected_version,
        "actual_version": actual_version,
        "correlation_id": event.correlation_id,
    },
)
```

**Get State Error**:
```python
logger.error(
    f"Failed to get workflow state for '{workflow_key}': {e}",
    exc_info=True,
    extra={"workflow_key": workflow_key},
)
```

### ReducerService

**Action Success**:
```python
logger.info(
    f"Successfully committed state for {action.workflow_key}: "
    f"v{state_record.version} → v{commit_result.new_version} "
    f"(attempt {attempt}/{self.max_attempts})"
)
```

**Conflict Retry**:
```python
logger.warning(
    f"Version conflict on {action.workflow_key}: "
    f"expected v{commit_result.expected_version}, "
    f"actual v{commit_result.actual_version}. "
    f"Attempt {attempt}/{self.max_attempts}, "
    f"backoff {backoff_ms}ms"
)
```

**Reducer Gave Up**:
```python
logger.error(
    f"Reducer gave up on {action.workflow_key} after {self.max_attempts} attempts"
)
```

**Duplicate Action Skipped**:
```python
logger.info(
    f"Skipping duplicate action {action.action_id} for workflow {action.workflow_key}"
)
```

### ProjectionMaterializerService

**Projection Materialized**:
```python
logger.debug(
    "Projection materialized successfully",
    extra={
        "workflow_key": event.workflow_key,
        "version": event.version,
        "partition_id": partition_id,
        "offset": offset,
        "lag_ms": lag_ms,
    },
)
```

**Watermark Regression**:
```python
logger.warning(
    "Watermark regression detected",
    extra={
        "partition_id": partition_id,
        "current_offset": current_offset,
        "new_offset": offset,
    },
)
```

**Processing Error**:
```python
logger.error(
    f"Failed to process StateCommitted event: {e}",
    exc_info=True,
    extra={
        "offset": msg.get("offset"),
        "partition": msg.get("partition"),
    },
)
```

**Metrics Logging** (every 60 seconds):
```python
logger.info(
    "Projection Materializer Metrics",
    extra={
        "projections_materialized": self._metrics.projections_materialized_total,
        "projections_failed": self._metrics.projections_failed_total,
        "watermark_updates": self._metrics.watermark_updates_total,
        "wm_regressions": self._metrics.wm_regressions_total,
        "lag_ms": round(self._metrics.projection_wm_lag_ms, 2),
        "max_lag_ms": round(self._metrics.max_lag_ms, 2),
        "events_per_second": round(self._metrics.events_processed_per_second, 2),
        "duplicates_skipped": self._metrics.duplicate_events_skipped,
    },
)
```

### ActionDedupService

**Duplicate Detected**:
```python
logger.info(
    f"Duplicate action detected: workflow_key={workflow_key}, action_id={action_id}"
)
```

**Action Recorded**:
```python
logger.debug(
    f"Recorded processed action: workflow_key={workflow_key}, "
    f"action_id={action_id}, expires_at={expires_at}"
)
```

**Cleanup Completed**:
```python
if deleted_count > 0:
    logger.info(f"Cleaned up {deleted_count} expired dedup entries")
```

### EventBusService

**Event Published**:
```python
logger.info(
    f"Published Action event (correlation_id={correlation_id}, action_type={action_type})"
)
```

**Event Received**:
```python
logger.info(
    f"Received event for correlation_id={correlation_id_str}: {event.get('event_type')}"
)
```

**Event Timeout**:
```python
logger.warning(
    f"Timeout waiting for event (correlation_id={correlation_id_str}, timeout={timeout_seconds}s)"
)
```

**Event Routed**:
```python
logger.debug(
    f"Routed event to listener (correlation_id={correlation_id_str}, event_type={event_type})"
)
```

## Correlation ID Patterns

### Generation

Correlation IDs are generated using `uuid4()` and propagated throughout the system:

```python
from uuid import uuid4

correlation_id = str(uuid4())
```

### Propagation

**In Events**:
```python
event = EventStateCommitted(
    workflow_key=workflow_key,
    new_version=new_version,
    correlation_id=correlation_id,  # Preserved from incoming action
)
```

**In Logs**:
```python
logger.info(
    "Operation completed",
    extra={
        "correlation_id": correlation_id,
        "workflow_key": workflow_key,
    }
)
```

**In Kafka Messages**:
```python
await kafka_client.publish_event(
    topic=topic,
    event=event_dict,
    key=workflow_key,  # Partition key
)
```

## Log Aggregation & Analysis

### Grep for Correlation ID

```bash
# Find all logs for a specific correlation ID
docker logs reducer-service 2>&1 | grep "correlation_id.*abc-123-def"

# Find all logs for a specific workflow
docker logs canonical-store 2>&1 | grep "workflow_key.*workflow-123"
```

### JSON Log Parsing (if using structlog)

If structlog is enabled, logs are in JSON format:

```bash
# Parse JSON logs with jq
docker logs reducer-service 2>&1 | jq 'select(.correlation_id == "abc-123-def")'

# Filter by log level
docker logs reducer-service 2>&1 | jq 'select(.level == "ERROR")'

# Extract specific fields
docker logs reducer-service 2>&1 | jq '{time: .timestamp, level: .level, msg: .event, workflow: .workflow_key}'
```

### Centralized Logging (ELK Stack)

For production deployments, use centralized logging:

**Elasticsearch Query**:
```json
{
  "query": {
    "bool": {
      "must": [
        {"match": {"correlation_id": "abc-123-def"}},
        {"range": {"@timestamp": {"gte": "now-1h"}}}
      ]
    }
  },
  "sort": [{"@timestamp": "asc"}]
}
```

**Kibana Query**:
```
correlation_id:"abc-123-def" AND level:ERROR
```

## Best Practices

### 1. Always Include Correlation ID

**✅ Good**:
```python
logger.info(
    "State committed",
    extra={
        "correlation_id": correlation_id,
        "workflow_key": workflow_key,
        "version": version,
    }
)
```

**❌ Bad**:
```python
logger.info("State committed")  # No correlation ID
```

### 2. Use Structured Extra Parameters

**✅ Good**:
```python
logger.error(
    "Failed to commit state",
    exc_info=True,
    extra={
        "workflow_key": workflow_key,
        "expected_version": expected_version,
        "error_type": type(e).__name__,
    }
)
```

**❌ Bad**:
```python
logger.error(f"Failed to commit state for {workflow_key}: {e}")  # String interpolation only
```

### 3. Include Performance Metrics in Logs

**✅ Good**:
```python
logger.info(
    "State committed successfully",
    extra={
        "workflow_key": workflow_key,
        "latency_ms": round(latency_ms, 2),
        "retry_count": attempt - 1,
    }
)
```

### 4. Use Appropriate Log Levels

| Level | When to Use |
|-------|-------------|
| DEBUG | Detailed diagnostic info (disabled in production) |
| INFO | Normal operations, successful completions |
| WARNING | Retries, non-critical issues, degraded performance |
| ERROR | Failures, exceptions, critical issues |

### 5. Include Exception Info for Errors

**✅ Good**:
```python
logger.error(
    "Database error",
    exc_info=True,  # Includes full stack trace
    extra={"workflow_key": workflow_key}
)
```

**❌ Bad**:
```python
logger.error(f"Database error: {e}")  # No stack trace
```

## Tracing End-to-End Workflows

### Example: Trace Complete Action Processing

**1. Action Received**:
```
[2025-10-21 12:00:00] INFO [reducer-service] Handling action
  correlation_id: abc-123-def
  workflow_key: workflow-123
  action_id: action-456
```

**2. Dedup Check**:
```
[2025-10-21 12:00:00] INFO [action-dedup] Checking dedup
  correlation_id: abc-123-def
  workflow_key: workflow-123
  action_id: action-456
  result: should_process
```

**3. State Read**:
```
[2025-10-21 12:00:00] DEBUG [canonical-store] Reading state
  correlation_id: abc-123-def
  workflow_key: workflow-123
  version: 5
```

**4. State Commit**:
```
[2025-10-21 12:00:00] INFO [canonical-store] State committed
  correlation_id: abc-123-def
  workflow_key: workflow-123
  expected_version: 5
  new_version: 6
  latency_ms: 8.3
```

**5. Event Published**:
```
[2025-10-21 12:00:00] DEBUG [canonical-store] Published event
  correlation_id: abc-123-def
  topic: omninode_bridge_state_committed_v1
  workflow_key: workflow-123
```

**6. Projection Updated**:
```
[2025-10-21 12:00:00] DEBUG [projection-materializer] Projection materialized
  correlation_id: abc-123-def
  workflow_key: workflow-123
  version: 6
  lag_ms: 45.2
```

### Query for Complete Trace

```bash
# Grep all logs for correlation ID
docker logs reducer-service 2>&1 | grep "abc-123-def" > trace.log
docker logs canonical-store 2>&1 | grep "abc-123-def" >> trace.log
docker logs projection-materializer 2>&1 | grep "abc-123-def" >> trace.log

# Sort by timestamp
sort trace.log > trace_sorted.log
```

## Future Enhancements (Optional)

### Structlog Integration

For production deployments, consider migrating to `structlog` for native JSON logging:

```python
import structlog

logger = structlog.get_logger()

logger.info(
    "state_committed",
    correlation_id=correlation_id,
    workflow_key=workflow_key,
    version=version,
    latency_ms=latency_ms,
)
```

**Output** (JSON):
```json
{
  "event": "state_committed",
  "correlation_id": "abc-123-def",
  "workflow_key": "workflow-123",
  "version": 6,
  "latency_ms": 8.3,
  "timestamp": "2025-10-21T12:00:00.123Z",
  "level": "info"
}
```

### OpenTelemetry Tracing

For distributed tracing, consider OpenTelemetry integration:

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("commit_state") as span:
    span.set_attribute("workflow_key", workflow_key)
    span.set_attribute("version", version)
    result = await canonical_store.try_commit(...)
    span.set_attribute("latency_ms", latency_ms)
```

## Contact

For questions or issues with logging:
- Platform Team: platform@omninode.ai
- Documentation: `docs/runbooks/PURE_REDUCER_OPERATIONS.md`
