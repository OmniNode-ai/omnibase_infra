# Troubleshooting Guide - Pure Reducer Architecture

**Status**: Production Ready (Wave 7 - Documentation Phase)
**Last Updated**: October 21, 2025
**Architecture Version**: ONEX v2.0

## Table of Contents

- [Overview](#overview)
- [Common Issues](#common-issues)
- [Performance Debugging](#performance-debugging)
- [Event Flow Debugging](#event-flow-debugging)
- [State Management Issues](#state-management-issues)
- [FSM Troubleshooting](#fsm-troubleshooting)
- [Projection Lag Issues](#projection-lag-issues)
- [Operational Procedures](#operational-procedures)
- [Diagnostic Queries](#diagnostic-queries)

---

## Overview

This guide provides solutions to common issues in the Pure Reducer architecture, including performance debugging, event tracing, and operational procedures.

### Quick Troubleshooting Checklist

1. ✅ Check service health endpoints (`/health`)
2. ✅ Review Prometheus metrics for anomalies
3. ✅ Trace events using `correlation_id`
4. ✅ Check Kafka consumer lag
5. ✅ Review PostgreSQL slow query logs
6. ✅ Verify FSM state consistency
7. ✅ Check watermark progression

---

## Common Issues

### 1. High Conflict Rate

**Symptom**: Frequent `STATE_CONFLICT` events, high retry counts

**Root Causes**:
- Hot key contention (many reducers updating same workflow)
- Insufficient partitioning
- Long-running transactions
- High concurrent write load

**Diagnosis**:
```bash
# Check conflict rate per workflow
SELECT workflow_key, COUNT(*) as conflict_count
FROM workflow_state_history
WHERE event_type = 'STATE_CONFLICT'
  AND timestamp > NOW() - INTERVAL '1 hour'
GROUP BY workflow_key
ORDER BY conflict_count DESC
LIMIT 10;

# Check Prometheus metrics
conflict_attempts_total{workflow_key="workflow-123"}
```

**Solutions**:

1. **Increase Kafka partitions** - Distribute load across more partitions:
```bash
kafka-topics --alter --topic dev.omninode_bridge.onex.evt.state-committed.v1 \
  --partitions 12 --bootstrap-server localhost:9092
```

2. **Optimize hot key distribution**:
```python
# Use sub-keys for hot workflows
partition_key = f"{workflow_key}:{hash(action_id) % 10}"
```

3. **Reduce transaction duration**:
```python
# Minimize time between read and commit
async with self.db.transaction():
    # Quick read
    state = await self.db.fetch_state(workflow_key)
    # Quick commit (no expensive operations)
    await self.db.commit_state(workflow_key, state_prime)
```

4. **Increase max retry attempts** (if conflicts are transient):
```python
# In ReducerService configuration
self.max_attempts = 5  # Increase from 3
self.backoff_cap_ms = 500  # Increase backoff cap
```

**Monitoring**:
```python
# Alert if conflict rate > 0.5% (p99)
conflict_rate = conflict_attempts_total / state_commits_total
if conflict_rate > 0.005:
    alert("High conflict rate detected")
```

### 2. Projection Lag

**Symptom**: Projection query returns stale data, high `projection_wm_lag_ms`

**Root Causes**:
- Slow projection materializer
- Database bottleneck (CPU, I/O)
- Large batch backlog
- Consumer group rebalancing

**Diagnosis**:
```bash
# Check projection lag per workflow
SELECT workflow_key,
       version as canonical_version,
       (SELECT version FROM workflow_projection WHERE workflow_key = ws.workflow_key) as projection_version,
       (ws.version - COALESCE(p.version, 0)) as version_lag
FROM workflow_state ws
LEFT JOIN workflow_projection p ON ws.workflow_key = p.workflow_key
WHERE ws.version - COALESCE(p.version, 0) > 5
ORDER BY version_lag DESC
LIMIT 10;

# Check watermark lag
SELECT partition_id,
       offset as current_watermark,
       (SELECT MAX(offset) FROM kafka_partition_offsets WHERE partition_id = w.partition_id) as latest_offset,
       ((SELECT MAX(offset) FROM kafka_partition_offsets WHERE partition_id = w.partition_id) - offset) as lag
FROM projection_watermarks w
ORDER BY lag DESC;
```

**Solutions**:

1. **Scale projection materializer** - Add more consumer instances:
```yaml
# docker-compose.yml
projection_materializer:
  replicas: 3  # Scale to 3 instances
```

2. **Optimize projection query**:
```sql
-- Add covering index for common queries
CREATE INDEX idx_projection_namespace_version
ON workflow_projection(namespace, version DESC)
INCLUDE (workflow_key, tag, last_action);
```

3. **Batch projection updates**:
```python
# Process events in batches
async def process_batch(self, events: list[StateCommitted]):
    async with self.db.transaction():
        for event in events:
            await self.upsert_projection(event)
        await self.advance_watermark(max(e.offset for e in events))
```

4. **Use fallback to canonical** (temporary):
```python
# In ProjectionStoreService
async def get_state(self, workflow_key, required_version=None, max_wait_ms=100):
    # Wait for projection with timeout
    projection = await self._wait_for_projection(workflow_key, required_version, max_wait_ms)
    if projection:
        return projection

    # Fallback to canonical (always up-to-date)
    canonical = await self.canonical_store.get_state(workflow_key)
    return self._to_projection(canonical)
```

**Monitoring**:
```python
# Alert if projection lag > 250ms (p99)
projection_lag_ms = metrics.histogram("projection_wm_lag_ms")
if projection_lag_ms.p99 > 250:
    alert("Projection lag exceeds target")
```

### 3. Reducer Gave Up

**Symptom**: `REDUCER_GAVE_UP` events published, workflows stuck

**Root Causes**:
- Persistent version conflicts (hot key)
- Database unavailability
- Network timeouts
- Bug in reducer logic

**Diagnosis**:
```bash
# Find workflows that gave up
SELECT workflow_key, action_id, attempts, last_error
FROM reducer_gaveup_log
WHERE timestamp > NOW() - INTERVAL '1 hour'
ORDER BY timestamp DESC;

# Check correlation_id for full event trace
SELECT event_type, timestamp, payload
FROM event_log
WHERE correlation_id = '550e8400-e29b-41d4-a716-446655440000'
ORDER BY timestamp;
```

**Solutions**:

1. **Manual retry** - Replay action event:
```bash
# Republish action event with same correlation_id
kafka-console-producer --topic dev.omninode_bridge.onex.evt.action.v1 \
  --bootstrap-server localhost:9092 \
  --property "key=workflow-123" \
  --property "correlation-id=550e8400-e29b-41d4-a716-446655440000"
```

2. **Escalate to orchestrator** - Implement compensation policy:
```python
# In Orchestrator
async def handle_reducer_gaveup(self, event: ReducerGaveUp):
    # Option 1: Retry with backoff
    await asyncio.sleep(60)  # Wait 1 minute
    await self.retry_workflow(event.workflow_key)

    # Option 2: Mark as failed and alert
    await self.mark_workflow_failed(event.workflow_key)
    await self.alert_ops_team(event)

    # Option 3: Compensate (undo partial work)
    await self.compensate_workflow(event.workflow_key)
```

3. **Investigate hot key** - Check if specific workflow is hot:
```python
# If workflow-123 repeatedly gives up, split it
if workflow_key == "workflow-123":
    # Use sub-keys to distribute load
    sub_key = f"{workflow_key}:{hash(action_id) % 10}"
    await self.publish_action(sub_key, action)
```

**Prevention**:
```python
# Increase max attempts for critical workflows
if workflow_priority == "HIGH":
    self.max_attempts = 10
    self.backoff_cap_ms = 1000
```

### 4. Memory Growth (Aggregation Buffer)

**Symptom**: Reducer memory usage increasing, `HealthStatus.DEGRADED` from buffer check

**Root Causes**:
- Aggregation buffer not flushed
- FSM state cache growing unbounded
- Memory leak in aggregation logic

**Diagnosis**:
```bash
# Check health endpoint
curl http://localhost:8080/health | jq '.components.aggregation_buffer'

# Example output:
{
  "status": "DEGRADED",
  "message": "Aggregation buffer large, may need flushing",
  "details": {
    "buffer_size": 15000,
    "fsm_cache_size": 12000,
    "fsm_history_size": 50000
  }
}
```

**Solutions**:

1. **Implement periodic flushing**:
```python
# In NodeBridgeReducer
async def _flush_aggregation_buffer(self):
    if len(self._aggregation_buffer) > 10000:
        # Persist buffer to database
        await self._persist_buffer()
        # Clear buffer
        self._aggregation_buffer.clear()
        logger.info("Flushed aggregation buffer")
```

2. **Implement FSM cache eviction**:
```python
# In FSMStateManager
async def _cleanup_terminal_states(self):
    # Remove completed/failed workflows after 1 hour
    cutoff = datetime.now(UTC) - timedelta(hours=1)
    for workflow_id, state_data in list(self._state_cache.items()):
        if state_data["current_state"] in self._terminal_states:
            if state_data["updated_at"] < cutoff:
                del self._state_cache[workflow_id]
                del self._transition_history[workflow_id]
```

3. **Monitor memory usage**:
```python
# Add memory metrics
import psutil
process = psutil.Process()
memory_mb = process.memory_info().rss / 1024 / 1024
metrics.gauge("reducer_memory_mb", memory_mb)
```

**Prevention**:
```python
# Set buffer size limits
MAX_BUFFER_SIZE = 10000
MAX_FSM_CACHE_SIZE = 10000

if len(self._aggregation_buffer) > MAX_BUFFER_SIZE:
    await self._flush_aggregation_buffer()

if len(self._fsm_manager._state_cache) > MAX_FSM_CACHE_SIZE:
    await self._fsm_manager._cleanup_terminal_states()
```

### 5. FSM State Validation Failures

**Symptom**: Invalid FSM transition errors, workflows stuck in unexpected states

**Root Causes**:
- FSM subcontract mismatch
- Case sensitivity issues (PENDING vs pending)
- Missing transition definitions
- Out-of-order event processing

**Diagnosis**:
```bash
# Check FSM transition history
SELECT workflow_id, from_state, to_state, trigger, timestamp
FROM fsm_transition_history
WHERE workflow_id = 'workflow-123'
ORDER BY timestamp;

# Check for invalid transitions
SELECT workflow_id, from_state, to_state, error_message
FROM fsm_validation_errors
WHERE timestamp > NOW() - INTERVAL '1 hour';
```

**Solutions**:

1. **Verify FSM subcontract**:
```yaml
# contract.yaml - Ensure all transitions are defined
state_transitions:
  states:
    - state_name: "PENDING"
    - state_name: "PROCESSING"
    - state_name: "COMPLETED"
    - state_name: "FAILED"

  transitions:
    # Define ALL allowed transitions
    - from_state: "PENDING"
      to_state: "PROCESSING"
    - from_state: "PROCESSING"
      to_state: "COMPLETED"
    - from_state: "PROCESSING"
      to_state: "FAILED"
    - from_state: "PENDING"
      to_state: "FAILED"
```

2. **Normalize state names** (already implemented):
```python
# FSMStateManager normalizes to uppercase
normalized_state = initial_state.upper()
```

3. **Manual state reset** (emergency):
```python
# Reset workflow to valid state
async def reset_workflow_state(workflow_id: UUID, new_state: str):
    # Bypass validation (use with caution)
    self._fsm_manager._state_cache[workflow_id]["current_state"] = new_state.upper()
    await self._fsm_manager._persist_state_transition(workflow_id, {...})
```

**Prevention**:
```python
# Add FSM validation tests
def test_all_transitions_defined():
    fsm_config = load_fsm_subcontract()
    for state in fsm_config.states:
        # Ensure all states have transitions or are terminal
        if state.state_name not in fsm_config.terminal_states:
            assert state.state_name in fsm_config.transitions
```

### 6. Event Loss / Missing Events

**Symptom**: Events published but not consumed, projection missing updates

**Root Causes**:
- Consumer group offset reset
- Kafka retention expired
- Consumer crash during processing
- Network partition

**Diagnosis**:
```bash
# Check Kafka consumer lag
kafka-consumer-groups --bootstrap-server localhost:9092 \
  --group projection-materializer \
  --describe

# Example output:
GROUP                    TOPIC                           PARTITION  CURRENT-OFFSET  LOG-END-OFFSET  LAG
projection-materializer  state-committed.v1              0          1000            1500            500
projection-materializer  state-committed.v1              1          1200            1200            0

# Check for missing events
SELECT correlation_id, COUNT(DISTINCT event_type) as event_count
FROM event_log
WHERE timestamp > NOW() - INTERVAL '1 hour'
GROUP BY correlation_id
HAVING COUNT(DISTINCT event_type) < 5;  -- Expected: STARTED, PROCESSED, COMPLETED, PERSISTED, COMMITTED
```

**Solutions**:

1. **Replay from Kafka** - Reset consumer offset:
```bash
# Reset to specific offset
kafka-consumer-groups --bootstrap-server localhost:9092 \
  --group projection-materializer \
  --topic state-committed.v1 \
  --reset-offsets --to-offset 1000 \
  --execute

# Reset to timestamp
kafka-consumer-groups --bootstrap-server localhost:9092 \
  --group projection-materializer \
  --reset-offsets --to-datetime 2025-10-21T12:00:00.000 \
  --execute
```

2. **Rebuild projection** - Full resync from canonical:
```python
async def rebuild_projection(workflow_key: Optional[str] = None):
    if workflow_key:
        # Rebuild single workflow
        canonical = await self.canonical_store.get_state(workflow_key)
        await self.projection_store.upsert(self._to_projection(canonical))
    else:
        # Rebuild all workflows
        async for canonical in self.canonical_store.scan_all():
            await self.projection_store.upsert(self._to_projection(canonical))
```

3. **Implement event reconciliation**:
```python
# Periodic reconciliation job
async def reconcile_projections():
    # Find workflows where projection version < canonical version
    async for workflow_key in self._find_stale_projections():
        # Trigger projection rebuild
        canonical = await self.canonical_store.get_state(workflow_key)
        await self.projection_store.upsert(self._to_projection(canonical))
```

**Prevention**:
```python
# Increase Kafka retention
kafka-topics --alter --topic state-committed.v1 \
  --config retention.ms=2592000000  # 30 days

# Enable compaction for critical topics
kafka-topics --alter --topic state-committed.v1 \
  --config cleanup.policy=compact
```

### 7. Slow Aggregation Performance

**Symptom**: High `aggregation_duration_ms`, low `items_per_second`

**Root Causes**:
- Large batch sizes
- Inefficient aggregation logic
- Memory allocation overhead
- Blocking I/O in pure function

**Diagnosis**:
```bash
# Check aggregation metrics
aggregation_duration_ms{quantile="0.99"}  # Should be < 100ms for 1000 items
items_per_second{quantile="0.50"}  # Should be > 1000

# Profile reducer execution
python -m cProfile -o reducer.prof src/omninode_bridge/nodes/reducer/v1_0_0/node.py
python -m pstats reducer.prof
```

**Solutions**:

1. **Optimize batch size**:
```python
# Tune batch size based on item complexity
if metadata.complexity == "HIGH":
    batch_size = 50  # Smaller batches for complex items
else:
    batch_size = 200  # Larger batches for simple items
```

2. **Use defaultdict for aggregations**:
```python
# Efficient aggregation structure
aggregated_data = defaultdict(lambda: {
    "total_stamps": 0,
    "total_size_bytes": 0,
    "file_types": set(),  # Use set for O(1) deduplication
    "workflow_ids": set(),
})
```

3. **Avoid blocking operations** - Ensure reducer is pure:
```python
# ❌ BAD - Blocking I/O in reducer
async def execute_reduction(self, contract):
    await self.db.write(...)  # BLOCKS!

# ✅ GOOD - Return intents instead
async def execute_reduction(self, contract):
    return ModelReducerOutputState(
        intents=[ModelIntent(intent_type="PersistState", ...)]
    )
```

4. **Pre-allocate collections**:
```python
# Pre-allocate lists for better memory performance
batch: list[ModelReducerInputState] = [None] * batch_size
```

**Monitoring**:
```python
# Track aggregation performance
start = time.perf_counter()
result = await self.execute_reduction(contract)
duration_ms = (time.perf_counter() - start) * 1000
metrics.histogram("aggregation_duration_ms", duration_ms)
```

---

## Performance Debugging

### Metrics to Monitor

| Metric | Target | Alert Threshold | Action |
|--------|--------|-----------------|--------|
| `aggregation_duration_ms` (p99) | <100ms | >150ms | Profile reducer, check batch size |
| `items_per_second` (p50) | >1000 | <500 | Optimize aggregation logic |
| `conflict_attempts_total` | <0.5% | >1% | Increase partitions, optimize hot keys |
| `projection_wm_lag_ms` (p95) | <100ms | >250ms | Scale materializer, optimize DB |
| `state_commits_total` | - | Sudden drop | Check DB health, network |
| `reducer_gaveup_total` | 0 | >0 | Investigate conflicts, DB issues |

### Performance Profiling

**1. Python Profiling**:
```bash
# Profile reducer execution
python -m cProfile -o reducer.prof -m pytest tests/unit/nodes/reducer/test_reducer.py -k test_aggregation_performance

# Analyze profile
python -c "import pstats; p = pstats.Stats('reducer.prof'); p.sort_stats('cumulative').print_stats(20)"
```

**2. Memory Profiling**:
```python
# Install memory_profiler
pip install memory_profiler

# Profile memory usage
from memory_profiler import profile

@profile
async def execute_reduction(self, contract):
    # ... reducer logic ...
```

**3. Async Profiling**:
```python
# Track async operation times
import time

async def execute_reduction(self, contract):
    timings = {}

    start = time.perf_counter()
    async for batch in self._stream_metadata(contract):
        timings["stream"] = time.perf_counter() - start

        start = time.perf_counter()
        # Process batch
        timings["process"] = time.perf_counter() - start

    logger.info(f"Reducer timings: {timings}")
```

### Database Performance

**Slow Query Analysis**:
```sql
-- Enable slow query logging
ALTER SYSTEM SET log_min_duration_statement = 100;  -- Log queries > 100ms

-- Check slow queries
SELECT query, calls, mean_exec_time, max_exec_time
FROM pg_stat_statements
WHERE mean_exec_time > 100
ORDER BY mean_exec_time DESC
LIMIT 10;
```

**Index Optimization**:
```sql
-- Find missing indexes
SELECT schemaname, tablename, attname, n_distinct, correlation
FROM pg_stats
WHERE schemaname = 'public'
  AND tablename IN ('workflow_state', 'workflow_projection')
  AND correlation < 0.1;

-- Add covering indexes
CREATE INDEX idx_workflow_state_key_version
ON workflow_state(workflow_key, version DESC)
INCLUDE (state, updated_at);
```

---

## Event Flow Debugging

### Tracing Events with correlation_id

**1. Find all events for a workflow**:
```sql
-- PostgreSQL event log
SELECT event_type, timestamp, payload
FROM event_log
WHERE correlation_id = '550e8400-e29b-41d4-a716-446655440000'
ORDER BY timestamp;

-- Expected sequence:
-- 1. AGGREGATION_STARTED
-- 2. BATCH_PROCESSED (x N)
-- 3. AGGREGATION_COMPLETED
-- 4. STATE_PERSISTED
-- 5. STATE_COMMITTED
```

**2. Check Kafka events**:
```bash
# Search Kafka for correlation_id
kafka-console-consumer --bootstrap-server localhost:9092 \
  --topic dev.omninode_bridge.onex.evt.aggregation-started.v1 \
  --from-beginning \
  | jq 'select(.correlation_id == "550e8400-e29b-41d4-a716-446655440000")'
```

**3. Visualize event timeline**:
```python
# Generate event timeline
events = await get_events_by_correlation_id(correlation_id)
for event in events:
    print(f"{event['timestamp']} - {event['event_type']}")
    if event['event_type'] == 'STATE_CONFLICT':
        print(f"  ↳ Conflict: expected={event['payload']['expected_version']}, actual={event['payload']['actual_version']}")
```

### Event Validation

**Check event schema**:
```python
from pydantic import ValidationError

def validate_event(event_data: dict):
    try:
        envelope = ModelEventEnvelope(**event_data)
        print(f"✅ Valid event: {envelope.event_type}")
    except ValidationError as e:
        print(f"❌ Invalid event: {e}")
```

**Check event ordering**:
```python
# Ensure events are ordered correctly
def check_event_ordering(events: list[dict]):
    expected_sequence = [
        "AGGREGATION_STARTED",
        "BATCH_PROCESSED",
        "AGGREGATION_COMPLETED",
        "STATE_PERSISTED",
        "STATE_COMMITTED"
    ]

    actual_sequence = [e['event_type'] for e in events]

    # Check if actual follows expected pattern
    for expected in expected_sequence:
        if expected not in actual_sequence:
            print(f"❌ Missing event: {expected}")
```

---

## State Management Issues

### Version Conflict Debugging

**1. Check version history**:
```sql
-- View all version changes for workflow
SELECT version, updated_at, provenance->>'action_id' as action_id
FROM workflow_state_history
WHERE workflow_key = 'workflow-123'
ORDER BY version;
```

**2. Identify concurrent writers**:
```sql
-- Find overlapping commits
SELECT ws1.workflow_key,
       ws1.version as version1,
       ws1.provenance->>'effect_id' as effect_id1,
       ws2.version as version2,
       ws2.provenance->>'effect_id' as effect_id2,
       ws1.updated_at,
       ws2.updated_at
FROM workflow_state_history ws1
JOIN workflow_state_history ws2 ON ws1.workflow_key = ws2.workflow_key
WHERE ws1.version = ws2.version - 1
  AND ws2.updated_at - ws1.updated_at < INTERVAL '100 milliseconds'
ORDER BY ws1.updated_at DESC;
```

**3. Simulate conflict**:
```python
# Test conflict handling
async def test_conflict():
    # Read state
    state = await canonical_store.get_state("workflow-123")

    # Simulate concurrent update (bump version)
    await canonical_store.try_commit("workflow-123", state.version + 1, {...})

    # Try to commit with stale version
    result = await canonical_store.try_commit("workflow-123", state.version, {...})

    assert isinstance(result, StateConflict)
    assert result.expected_version == state.version
    assert result.actual_version == state.version + 1
```

### Canonical vs Projection Consistency

**Check consistency**:
```sql
-- Find workflows where projection is behind canonical
SELECT c.workflow_key,
       c.version as canonical_version,
       p.version as projection_version,
       (c.version - p.version) as version_lag,
       c.updated_at as canonical_updated,
       p.updated_at as projection_updated
FROM workflow_state c
LEFT JOIN workflow_projection p ON c.workflow_key = p.workflow_key
WHERE c.version > COALESCE(p.version, 0)
ORDER BY version_lag DESC
LIMIT 20;
```

**Reconcile state**:
```python
async def reconcile_workflow(workflow_key: str):
    # Read canonical (source of truth)
    canonical = await canonical_store.get_state(workflow_key)

    # Read projection
    projection = await projection_store.get_state(workflow_key)

    # If projection is behind, rebuild
    if not projection or projection.version < canonical.version:
        await projection_store.upsert(
            workflow_key=workflow_key,
            version=canonical.version,
            state=canonical.state
        )
        logger.info(f"Reconciled projection for {workflow_key}")
```

---

## FSM Troubleshooting

### FSM State Debugging

**View FSM state cache**:
```python
# In NodeBridgeReducer
def get_fsm_state_summary(self) -> dict:
    return {
        "total_workflows": len(self._fsm_manager._state_cache),
        "state_distribution": {
            state: len([w for w in self._fsm_manager._state_cache.values() if w["current_state"] == state])
            for state in self._fsm_manager._valid_states
        },
        "terminal_workflows": len([
            w for w in self._fsm_manager._state_cache.values()
            if w["current_state"] in self._fsm_manager._terminal_states
        ])
    }
```

**Check transition history**:
```python
# Get transition history for workflow
history = self._fsm_manager.get_transition_history(workflow_id)
for transition in history:
    print(f"{transition['timestamp']}: {transition['from_state']} → {transition['to_state']} (trigger: {transition['trigger']})")
```

### FSM Validation Errors

**Common validation errors**:

1. **Invalid transition**:
```
ERROR: Invalid transition: PROCESSING -> PENDING for workflow workflow-123
```
**Solution**: Check FSM subcontract, ensure transition is defined

2. **State mismatch**:
```
ERROR: Current state mismatch for workflow-123: expected PENDING, got PROCESSING
```
**Solution**: Verify state before transition, handle concurrent updates

3. **Unknown state**:
```
ERROR: Invalid from_state: UNKNOWN
```
**Solution**: Ensure FSM subcontract includes all states, check case sensitivity

---

## Projection Lag Issues

### Watermark Debugging

**Check watermark progression**:
```sql
-- View watermark history
SELECT partition_id, offset, updated_at
FROM projection_watermark_history
WHERE partition_id = 'kafka-partition-0'
ORDER BY updated_at DESC
LIMIT 20;

-- Check for watermark regressions
SELECT partition_id, offset, updated_at
FROM projection_watermark_history
WHERE partition_id = 'kafka-partition-0'
  AND offset < (
      SELECT offset FROM projection_watermark_history
      WHERE partition_id = 'kafka-partition-0'
        AND updated_at < projection_watermark_history.updated_at
      LIMIT 1
  );
```

**Reset watermark** (emergency):
```sql
-- Reset to specific offset
UPDATE projection_watermarks
SET offset = 10000, updated_at = NOW()
WHERE partition_id = 'kafka-partition-0';
```

### Projection Materializer Health

**Check materializer status**:
```bash
# Health endpoint
curl http://localhost:8081/health | jq '.components.projection_materializer'

# Check consumer lag
kafka-consumer-groups --bootstrap-server localhost:9092 \
  --group projection-materializer \
  --describe
```

**Monitor metrics**:
```python
# Projection lag metrics
projection_wm_lag_ms{workflow_key="workflow-123", quantile="0.95"}

# Watermark progression rate
rate(projection_watermark_offset[5m])

# Projection update rate
rate(projection_upsert_total[5m])
```

---

## Operational Procedures

### Emergency Procedures

**1. Circuit Breaker - Stop Processing**:
```bash
# Stop all reducer instances
kubectl scale deployment reducer --replicas=0

# Stop projection materializer
kubectl scale deployment projection-materializer --replicas=0
```

**2. Reset Consumer Offsets**:
```bash
# Reset all consumer groups to latest
for group in reducer-service projection-materializer; do
  kafka-consumer-groups --bootstrap-server localhost:9092 \
    --group $group \
    --reset-offsets --to-latest --all-topics \
    --execute
done
```

**3. Drain Kafka Topic**:
```bash
# Increase retention to preserve events
kafka-topics --alter --topic state-committed.v1 \
  --config retention.ms=86400000  # 24 hours

# Wait for consumers to catch up
# Then reduce retention
kafka-topics --alter --topic state-committed.v1 \
  --config retention.ms=604800000  # 7 days
```

### Routine Maintenance

**1. Clean Up Expired FSM States**:
```python
# Run daily
async def cleanup_terminal_states():
    cutoff = datetime.now(UTC) - timedelta(days=7)
    deleted = await db.execute("""
        DELETE FROM fsm_workflow_states
        WHERE current_state IN ('COMPLETED', 'FAILED')
          AND updated_at < $1
    """, cutoff)
    logger.info(f"Cleaned up {deleted} terminal FSM states")
```

**2. Vacuum Projection Tables**:
```sql
-- Run weekly
VACUUM ANALYZE workflow_projection;
VACUUM ANALYZE projection_watermarks;
```

**3. Archive Old Events**:
```python
# Archive events older than 30 days
async def archive_old_events():
    cutoff = datetime.now(UTC) - timedelta(days=30)
    await db.execute("""
        INSERT INTO event_log_archive
        SELECT * FROM event_log
        WHERE timestamp < $1
    """, cutoff)
    await db.execute("DELETE FROM event_log WHERE timestamp < $1", cutoff)
```

---

## Diagnostic Queries

### Health Check Queries

**1. Overall System Health**:
```sql
-- Check reducer health
SELECT 'reducer' as component,
       COUNT(*) as active_workflows,
       COUNT(*) FILTER (WHERE current_state = 'PROCESSING') as processing,
       COUNT(*) FILTER (WHERE current_state IN ('COMPLETED', 'FAILED')) as terminal
FROM fsm_workflow_states;

-- Check projection health
SELECT 'projection' as component,
       COUNT(*) as total_projections,
       AVG(c.version - COALESCE(p.version, 0)) as avg_version_lag
FROM workflow_state c
LEFT JOIN workflow_projection p ON c.workflow_key = p.workflow_key;
```

**2. Performance Metrics**:
```sql
-- Aggregation performance (last hour)
SELECT AVG(duration_ms) as avg_duration,
       PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms) as p95_duration,
       PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY duration_ms) as p99_duration,
       AVG(items_per_second) as avg_throughput
FROM aggregation_metrics
WHERE timestamp > NOW() - INTERVAL '1 hour';
```

**3. Error Rate**:
```sql
-- Error rates (last hour)
SELECT event_type,
       COUNT(*) as count,
       COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () as percentage
FROM event_log
WHERE timestamp > NOW() - INTERVAL '1 hour'
  AND event_type IN ('AGGREGATION_FAILED', 'STATE_CONFLICT', 'REDUCER_GAVE_UP')
GROUP BY event_type
ORDER BY count DESC;
```

---

## References

- **Pure Reducer Architecture**: [PURE_REDUCER_ARCHITECTURE.md](./PURE_REDUCER_ARCHITECTURE.md)
- **Event Contracts**: [EVENT_CONTRACTS.md](./EVENT_CONTRACTS.md)
- **Prometheus Metrics**: See deployment configuration
- **Grafana Dashboards**: `dashboards/pure_reducer_monitoring.json`

---

**Document Version**: 1.0.0
**Last Review**: October 21, 2025
**Next Review**: Post Wave 6 completion
