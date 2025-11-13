# Pure Reducer Operations Runbook

**Version**: 1.0.0
**Last Updated**: 2025-10-21
**Owner**: Platform Team
**Wave**: 6C - Metrics & Observability

## Table of Contents

1. [Overview](#overview)
2. [Architecture Quick Reference](#architecture-quick-reference)
3. [Key Metrics](#key-metrics)
4. [Common Operational Tasks](#common-operational-tasks)
5. [Troubleshooting Guide](#troubleshooting-guide)
6. [Alert Response Procedures](#alert-response-procedures)
7. [Performance Tuning](#performance-tuning)
8. [Disaster Recovery](#disaster-recovery)

---

## Overview

This runbook provides operational procedures for the Pure Reducer architecture implementing Wave 6C metrics and observability. The system consists of five core services:

1. **CanonicalStoreService** - Version-controlled state management with optimistic concurrency
2. **ReducerService** - Service wrapper with conflict resolution and retry logic
3. **ProjectionMaterializerService** - Eventual consistency projection updates
4. **ActionDedupService** - Idempotent action processing
5. **EventBusService** - Event-driven coordination via Kafka

### SLA Targets

| Metric | Target | Critical Threshold |
|--------|--------|-------------------|
| Commit Latency (p95) | < 10ms | > 50ms |
| Projection Lag | < 100ms | > 500ms |
| Conflict Rate | < 5% | > 10% |
| Success Rate | > 95% | < 90% |
| Event Timeout Rate | < 1% | > 5% |

---

## Architecture Quick Reference

```
┌─────────────────────────────────────────────────────────────┐
│                    Pure Reducer Flow                         │
├─────────────────────────────────────────────────────────────┤
│  1. ReducerService.handle_action(action)                    │
│     ├─> ActionDedupService.should_process() [<5ms]          │
│     ├─> CanonicalStoreService.get_state() [<5ms]            │
│     ├─> Pure Reducer (compute state_prime) [<50ms]          │
│     └─> CanonicalStoreService.try_commit() [<10ms]          │
│          ├─> [SUCCESS] → EventBus.publish() → Kafka         │
│          └─> [CONFLICT] → Retry with backoff (10-250ms)     │
│                                                               │
│  2. ProjectionMaterializerService (async consumer)          │
│     ├─> Kafka.consume(state-committed events)               │
│     ├─> ProjectionStore.upsert_projection() [<10ms]         │
│     └─> Watermark.advance() [atomic with projection]        │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Metrics

### Canonical Store Metrics

```promql
# Commits per second by workflow
rate(canonical_store_state_commits_total[5m])

# Conflicts per second by workflow
rate(canonical_store_state_conflicts_total[5m])

# Commit latency percentiles
histogram_quantile(0.95, rate(canonical_store_commit_latency_ms_bucket[5m]))
histogram_quantile(0.99, rate(canonical_store_commit_latency_ms_bucket[5m]))

# Get state operations
rate(canonical_store_get_state_total[5m])
rate(canonical_store_get_state_errors[5m])
```

### Reducer Service Metrics

```promql
# Successful actions per second
rate(reducer_successful_actions_total[5m])

# Failed actions (max retries exceeded)
rate(reducer_failed_actions_total[5m])

# Conflict attempts (retries)
rate(reducer_conflict_attempts_total[5m])

# Backoff delay distribution
histogram_quantile(0.50, rate(reducer_backoff_ms_bucket[5m]))
histogram_quantile(0.95, rate(reducer_backoff_ms_bucket[5m]))

# Gave up count (critical)
reducer_gave_up_total
```

### Projection Materializer Metrics

```promql
# Projections materialized per second
rate(projection_materializer_projections_materialized_total[5m])

# Failed projections
rate(projection_materializer_projections_failed_total[5m])

# Watermark lag (should be < 100ms)
projection_materializer_wm_lag_ms

# Max lag observed
projection_materializer_max_lag_ms

# Watermark regressions (should be 0)
projection_materializer_wm_regressions_total

# Duplicate events skipped
rate(projection_materializer_duplicate_events_skipped[5m])

# Processing rate
projection_materializer_events_processed_per_second
```

### Action Dedup Metrics

```promql
# Dedup checks per second
rate(action_dedup_checks_total[5m])

# Dedup hit rate (percentage)
rate(action_dedup_hits_total[5m]) / rate(action_dedup_checks_total[5m])

# Records created per second
rate(action_dedup_records_total[5m])

# Cleanup operations
rate(action_dedup_cleanup_deleted_total[5m])
```

### Event Bus Metrics

```promql
# Events published per second by type
rate(event_bus_published_total[5m])

# Events consumed per second by type
rate(event_bus_consumed_total[5m])

# Timeout rate (should be < 1%)
rate(event_bus_timeout_total[5m]) / (rate(event_bus_consumed_total[5m]) + rate(event_bus_timeout_total[5m]))

# Wait time distribution
histogram_quantile(0.95, rate(event_bus_wait_time_ms_bucket[5m]))
```

---

## Common Operational Tasks

### 1. Checking System Health

**Quick Health Check**:
```bash
# Check all services are running
docker ps | grep -E "(canonical|reducer|projection|event-bus)"

# Check Prometheus metrics endpoint
curl http://localhost:8000/metrics | grep -E "canonical|reducer|projection"

# Check Grafana dashboard
open http://localhost:3000/d/pure-reducer-6c
```

**Detailed Health Check**:
```bash
# Check PostgreSQL connection pool status
psql -U postgres -d omninode_bridge -c "SELECT count(*) FROM pg_stat_activity WHERE datname = 'omninode_bridge';"

# Check Kafka consumer lag
kafka-consumer-groups --bootstrap-server localhost:29092 --group projection-materializer --describe

# Check recent error logs
docker logs --tail 100 reducer-service 2>&1 | grep -i error
docker logs --tail 100 projection-materializer 2>&1 | grep -i error
```

### 2. Scaling Services

**Horizontal Scaling - ReducerService**:
```bash
# Scale to 3 replicas
docker-compose up -d --scale reducer-service=3

# Verify all replicas are healthy
docker ps | grep reducer-service
```

**Horizontal Scaling - ProjectionMaterializer**:
```bash
# Scale to 2 replicas (max = Kafka partitions)
docker-compose up -d --scale projection-materializer=2

# Note: Number of replicas should not exceed number of Kafka partitions
# Check partition count: kafka-topics --bootstrap-server localhost:29092 --describe --topic state-committed
```

### 3. Monitoring Projection Lag

**Check Current Lag**:
```bash
# Query Prometheus for current lag
curl -s 'http://localhost:9090/api/v1/query?query=projection_materializer_wm_lag_ms' | jq '.data.result'

# Check watermark positions
psql -U postgres -d omninode_bridge -c "SELECT * FROM projection_watermarks ORDER BY updated_at DESC LIMIT 10;"
```

**If Lag is High (> 500ms)**:
1. Check Kafka consumer lag: `kafka-consumer-groups --bootstrap-server localhost:29092 --group projection-materializer --describe`
2. Scale up ProjectionMaterializer replicas
3. Check for database connection pool exhaustion
4. Review slow queries in PostgreSQL logs

### 4. Handling High Conflict Rates

**Check Conflict Rate**:
```promql
# Conflict rate per workflow
rate(canonical_store_state_conflicts_total[5m]) /
(rate(canonical_store_state_commits_total[5m]) + rate(canonical_store_state_conflicts_total[5m]))
```

**If Conflict Rate > 10%**:
1. Identify hot workflows: `SELECT workflow_key FROM workflow_state ORDER BY version DESC LIMIT 10;`
2. Review reducer backoff configuration (increase `backoff_cap_ms`)
3. Consider sharding hot workflows across multiple workflow_keys
4. Check for concurrent client operations causing contention

### 5. Deduplication Cleanup

**Manual Cleanup**:
```python
# Connect to service and trigger cleanup
from omninode_bridge.services.action_dedup import ActionDedupService
service = ActionDedupService(postgres_client)
deleted = await service.cleanup_expired()
print(f"Cleaned up {deleted} expired entries")
```

**Scheduled Cleanup** (cron):
```bash
# Add to crontab for hourly cleanup
0 * * * * docker exec reducer-service python -c "from omninode_bridge.services.action_dedup import ActionDedupService; import asyncio; asyncio.run(ActionDedupService(...).cleanup_expired())"
```

### 6. Kafka Topic Management

**Check Topic Health**:
```bash
# List topics
kafka-topics --bootstrap-server localhost:29092 --list

# Describe state-committed topic
kafka-topics --bootstrap-server localhost:29092 --describe --topic omninode_bridge.onex.evt.state-committed.v1

# Check consumer group lag
kafka-consumer-groups --bootstrap-server localhost:29092 --group projection-materializer --describe
```

**Reset Consumer Group** (if stuck):
```bash
# Stop projection materializer
docker-compose stop projection-materializer

# Reset offsets to latest
kafka-consumer-groups --bootstrap-server localhost:29092 --group projection-materializer --reset-offsets --to-latest --execute --all-topics

# Restart projection materializer
docker-compose start projection-materializer
```

---

## Troubleshooting Guide

### Issue: High Commit Latency (> 50ms p95)

**Symptoms**:
- `canonical_store_commit_latency_ms` p95 > 50ms
- Slow end-to-end action processing

**Diagnosis**:
```bash
# Check PostgreSQL query performance
psql -U postgres -d omninode_bridge -c "SELECT query, mean_exec_time, calls FROM pg_stat_statements WHERE query LIKE '%workflow_state%' ORDER BY mean_exec_time DESC LIMIT 10;"

# Check connection pool status
psql -U postgres -d omninode_bridge -c "SELECT count(*) as active, state FROM pg_stat_activity GROUP BY state;"
```

**Resolution**:
1. Increase PostgreSQL connection pool size (check `POSTGRES_POOL_SIZE`)
2. Add indexes if missing: `CREATE INDEX IF NOT EXISTS idx_workflow_state_workflow_key ON workflow_state(workflow_key);`
3. Tune PostgreSQL `shared_buffers` and `effective_cache_size`
4. Check for long-running transactions blocking updates

---

### Issue: Reducer Gave Up (Max Retries Exceeded)

**Symptoms**:
- `reducer_gave_up_total` increasing
- Actions failing after max retries

**Diagnosis**:
```bash
# Check gave up events
docker logs reducer-service 2>&1 | grep "Reducer gave up"

# Check conflict rate
curl -s 'http://localhost:9090/api/v1/query?query=rate(reducer_conflict_attempts_total[5m])' | jq
```

**Resolution**:
1. Increase `max_attempts` in ReducerService configuration (default: 3)
2. Increase `backoff_cap_ms` for longer backoff delays
3. Identify hot workflows causing high contention
4. Implement workflow sharding or partitioning strategy

---

### Issue: Projection Watermark Regressions

**Symptoms**:
- `projection_materializer_wm_regressions_total` > 0
- Duplicate projection updates detected

**Diagnosis**:
```bash
# Check watermark regression logs
docker logs projection-materializer 2>&1 | grep "Watermark regression"

# Query watermark table
psql -U postgres -d omninode_bridge -c "SELECT * FROM projection_watermarks ORDER BY updated_at DESC LIMIT 20;"
```

**Resolution**:
1. Investigate Kafka consumer group rebalancing
2. Check for multiple projection materializer instances consuming same partition
3. Verify Kafka consumer offset management
4. Review projection materializer restart/crash logs

---

### Issue: High Event Timeout Rate

**Symptoms**:
- `event_bus_timeout_total` increasing
- EventBus `wait_for_completion()` timing out

**Diagnosis**:
```bash
# Check event timeout logs
docker logs orchestrator 2>&1 | grep "Timeout waiting for event"

# Check Kafka consumer lag
kafka-consumer-groups --bootstrap-server localhost:29092 --describe --all-groups
```

**Resolution**:
1. Increase `timeout_seconds` in EventBus configuration
2. Scale up reducer service to process actions faster
3. Check Kafka broker health and partition leadership
4. Review reducer service processing delays

---

### Issue: Database Connection Pool Exhaustion

**Symptoms**:
- `canonical_store_get_state_errors` increasing
- Errors like "connection pool exhausted"

**Diagnosis**:
```bash
# Check active connections
psql -U postgres -d omninode_bridge -c "SELECT count(*) FROM pg_stat_activity WHERE datname = 'omninode_bridge';"

# Check max connections
psql -U postgres -d omninode_bridge -c "SHOW max_connections;"
```

**Resolution**:
1. Increase PostgreSQL `max_connections` (default: 100)
2. Increase application connection pool size (`POSTGRES_POOL_MAX`)
3. Implement connection pooling with pgBouncer
4. Check for connection leaks in application code

---

## Alert Response Procedures

### Alert: High Conflict Rate (> 10%)

**Severity**: Warning
**SLA Impact**: Medium

**Immediate Actions**:
1. Check Grafana dashboard for affected workflows
2. Query conflict rate: `rate(canonical_store_state_conflicts_total[5m]) / (rate(canonical_store_state_commits_total[5m]) + rate(canonical_store_state_conflicts_total[5m]))`
3. Identify top workflows by conflict count

**Investigation**:
1. Review recent changes to reducer logic
2. Check for increased concurrent load on specific workflows
3. Analyze reducer backoff histogram for retry patterns

**Resolution**:
1. Increase backoff delays to reduce contention
2. Implement workflow partitioning for hot workflows
3. Review and optimize reducer execution time

---

### Alert: Projection Lag High (> 500ms)

**Severity**: Critical
**SLA Impact**: High

**Immediate Actions**:
1. Check projection materializer service health: `docker ps | grep projection-materializer`
2. Query current lag: `projection_materializer_wm_lag_ms`
3. Check Kafka consumer group lag

**Investigation**:
1. Review projection materializer logs for errors
2. Check database query performance on `workflow_projection` table
3. Verify Kafka broker health and partition availability

**Resolution**:
1. Scale up projection materializer replicas
2. Restart stuck consumers if needed
3. Increase database connection pool size
4. Add indexes on projection table if missing

---

### Alert: Reducer Gave Up (Count > 0)

**Severity**: Critical
**SLA Impact**: High

**Immediate Actions**:
1. Query gave up count: `reducer_gave_up_total`
2. Check reducer service logs for details
3. Identify affected workflows

**Investigation**:
1. Review conflict rate for affected workflows
2. Check max retry configuration
3. Analyze backoff delays and retry patterns

**Resolution**:
1. Increase `max_attempts` in reducer configuration
2. Increase `backoff_cap_ms` for longer retry windows
3. Manually retry failed actions if needed
4. Implement circuit breaker for persistent failures

---

### Alert: Event Timeout Rate High (> 5%)

**Severity**: Warning
**SLA Impact**: Medium

**Immediate Actions**:
1. Query timeout rate: `rate(event_bus_timeout_total[5m])`
2. Check event bus service health
3. Review Kafka consumer lag

**Investigation**:
1. Check reducer service processing time
2. Review Kafka broker health
3. Analyze event wait time histogram

**Resolution**:
1. Increase EventBus `timeout_seconds` configuration
2. Scale up reducer service for faster processing
3. Investigate Kafka broker performance issues

---

## Performance Tuning

### ReducerService Tuning

**Configuration Parameters**:
```python
ReducerService(
    max_attempts=3,         # Increase for high-conflict scenarios (default: 3)
    backoff_base_ms=10,     # Base backoff delay (default: 10ms)
    backoff_cap_ms=250,     # Max backoff delay (default: 250ms)
)
```

**Recommendations**:
- **High Conflict Rate**: Increase `max_attempts` to 5, `backoff_cap_ms` to 500ms
- **Low Latency**: Decrease `backoff_base_ms` to 5ms
- **High Throughput**: Increase concurrency, monitor conflict rate

---

### ProjectionMaterializerService Tuning

**Configuration Parameters**:
```python
ProjectionMaterializerService(
    enable_idempotence=True,  # Enable duplicate detection (recommended: true)
)
```

**Kafka Consumer Tuning**:
```python
KafkaConsumerWrapper(
    batch_timeout_ms=1000,  # Polling interval (default: 1000ms)
    max_records=500,        # Max records per batch (default: 500)
)
```

**Recommendations**:
- **Low Lag**: Decrease `batch_timeout_ms` to 500ms
- **High Throughput**: Increase `max_records` to 1000
- **Memory Constrained**: Decrease `max_records` to 100

---

### CanonicalStoreService Tuning

**PostgreSQL Connection Pool**:
```python
PostgresClient(
    min_pool_size=10,   # Min connections (default: 10)
    max_pool_size=50,   # Max connections (default: 50)
)
```

**Recommendations**:
- **High Load**: Increase `max_pool_size` to 100
- **Low Memory**: Decrease `max_pool_size` to 20
- **Connection Pooling**: Use pgBouncer for connection multiplexing

---

## Disaster Recovery

### Scenario 1: PostgreSQL Database Failure

**Impact**: Complete service outage

**Recovery Steps**:
1. Restore from latest backup
2. Replay Kafka events from last checkpoint
3. Verify projection watermarks are consistent
4. Resume service operations

**Prevention**:
- Regular database backups (hourly snapshots + WAL archiving)
- PostgreSQL replication (streaming replication or logical replication)
- Point-in-time recovery (PITR) capability

---

### Scenario 2: Kafka Cluster Failure

**Impact**: Event publishing and consumption halted

**Recovery Steps**:
1. Restore Kafka cluster from backups
2. Verify topic partitions and offsets
3. Resume projection materializer consumption
4. Monitor projection lag for catch-up

**Prevention**:
- Kafka cluster replication (min 3 brokers)
- Topic replication factor ≥ 2
- Regular Kafka backups with MirrorMaker

---

### Scenario 3: Projection Watermark Corruption

**Impact**: Duplicate projections or projection skips

**Recovery Steps**:
1. Stop projection materializer service
2. Query current watermarks: `SELECT * FROM projection_watermarks;`
3. Compare with Kafka consumer group offsets
4. Reset watermarks to correct offsets
5. Resume projection materializer

**Prevention**:
- Atomic watermark + projection updates (transaction)
- Regular watermark validation checks
- Idempotence enabled in projection materializer

---

## Appendix

### Useful Commands

**Prometheus Queries**:
```bash
# Query Prometheus metrics
curl -s 'http://localhost:9090/api/v1/query?query=canonical_store_state_commits_total' | jq

# Query range (last 5 minutes)
curl -s 'http://localhost:9090/api/v1/query_range?query=rate(reducer_successful_actions_total[5m])&start=...' | jq
```

**PostgreSQL Queries**:
```sql
-- Check workflow state versions
SELECT workflow_key, version, updated_at FROM workflow_state ORDER BY updated_at DESC LIMIT 10;

-- Check projection watermarks
SELECT * FROM projection_watermarks ORDER BY updated_at DESC;

-- Check action dedup log
SELECT workflow_key, count(*) FROM action_dedup_log GROUP BY workflow_key;
```

**Docker Commands**:
```bash
# View logs with timestamps
docker logs -f --timestamps reducer-service

# Follow logs for specific service
docker-compose logs -f projection-materializer

# Restart service
docker-compose restart canonical-store
```

---

## Contact & Escalation

**Primary Contacts**:
- Platform Team: platform@omninode.ai
- On-Call Engineer: Pagerduty escalation

**Escalation Path**:
1. Platform Engineer (15 min response)
2. Senior Platform Engineer (30 min response)
3. Principal Engineer (1 hour response)

**References**:
- Grafana Dashboard: http://localhost:3000/d/pure-reducer-6c
- Prometheus: http://localhost:9090
- Architecture Docs: `docs/planning/PURE_REDUCER_REFACTOR_PLAN.md`
