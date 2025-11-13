# Event Logs Table - Design Rationale

**Migration**: `005_20251015_event_logs_table.py`
**Created**: 2025-10-15
**Related PR**: #26 (Event Infrastructure MVP)

## Overview

The `event_logs` table provides persistent storage for autonomous code generation events flowing through 13 Kafka topics with 9 distinct event types. This design supports comprehensive event tracing, performance metrics calculation, and debugging for the omniclaude ↔ omniarchon workflow.

## Schema Design

### Core Columns

| Column | Type | Nullable | Purpose | Design Rationale |
|--------|------|----------|---------|------------------|
| `event_id` | UUID | NOT NULL | Primary key | Auto-generated UUID for unique event identification |
| `session_id` | UUID | NOT NULL | Session tracking | Links events to code generation sessions |
| `correlation_id` | UUID | NULL | Request/response matching | Nullable for status events that don't have correlations |
| `event_type` | VARCHAR(50) | NOT NULL | Event classification | CHECK constraint ensures valid types: request, response, status, error |
| `topic` | VARCHAR(255) | NOT NULL | Kafka topic name | Enables topic-based analysis and bottleneck identification |
| `timestamp` | TIMESTAMPTZ | NOT NULL | Temporal ordering | UTC timestamps with timezone awareness |
| `status` | VARCHAR(50) | NOT NULL | Event status | CHECK constraint: sent, received, failed, processing, completed |
| `processing_time_ms` | INTEGER | NULL | Performance metrics | Optional, only present for response events |
| `payload` | JSONB | NOT NULL | Event data | Full event schema stored for flexibility |
| `metadata` | JSONB | NOT NULL | Additional context | Source, destination, retries, etc. |

### Key Design Decisions

1. **UUID for Primary Key**
   - Globally unique across distributed systems
   - Supports future sharding/partitioning
   - No sequence coordination needed

2. **Nullable `correlation_id`**
   - Status events (CodegenStatusEvent) have session_id but no correlation_id
   - Allows flexible event modeling without artificial correlation generation

3. **JSONB for `payload` and `metadata`**
   - Schema evolution without migrations
   - Supports all 9 event types (request/response pairs + status)
   - GIN indexes enable fast JSON queries
   - Trade-off: Storage overhead vs. query flexibility

4. **CHECK Constraints**
   - Enforces data integrity at database level
   - Prevents invalid event_type and status values
   - Self-documenting schema

5. **TIMESTAMPTZ (not TIMESTAMP)**
   - UTC timezone awareness prevents timezone bugs
   - Supports distributed systems across timezones
   - Essential for accurate performance metrics

## Index Strategy

### Performance-Optimized Indexes

| Index Name | Type | Columns | Purpose | Query Pattern |
|------------|------|---------|---------|---------------|
| `idx_event_logs_session_id` | B-tree | session_id | Session-based queries | `trace_session_events()` |
| `idx_event_logs_correlation_id` | B-tree | correlation_id | Correlation matching | `find_correlated_events()` |
| `idx_event_logs_timestamp` | B-tree | timestamp DESC | Time-range queries | Time-based filtering |
| `idx_event_logs_event_type` | B-tree | event_type | Event type filtering | Metrics grouping |
| `idx_event_logs_topic` | B-tree | topic | Topic grouping | Bottleneck analysis |
| `idx_event_logs_session_timestamp` | B-tree | (session_id, timestamp DESC) | **Optimal composite** | `trace_session_events()` with time range |
| `idx_event_logs_payload_gin` | GIN | payload | JSONB queries | Payload field lookups |
| `idx_event_logs_metadata_gin` | GIN | metadata | JSONB queries | Metadata field lookups |

### Index Selection Rationale

1. **Composite Index `(session_id, timestamp DESC)`**
   - **Critical optimization** for `trace_session_events()` which filters by session_id and sorts by timestamp
   - Avoids separate index scans + sort operation
   - Expected query: `WHERE session_id = $1 AND timestamp >= $2 ORDER BY timestamp ASC`
   - PostgreSQL query planner can use this index for both filtering and sorting

2. **GIN Indexes on JSONB Columns**
   - Enables fast JSON field queries: `payload @> '{"analysis_type": "full"}'`
   - Supports containment and existence operators
   - Trade-off: Slower writes, faster reads (acceptable for event log use case)

3. **Separate vs. Composite Indexes**
   - Individual indexes on session_id, correlation_id, timestamp support different query patterns
   - Composite (session_id, timestamp) specifically optimizes most common query

## Query Performance Analysis

### Query 1: `trace_session_events(session_id, time_range_hours)`

```sql
SELECT event_id, event_type, topic, timestamp, correlation_id,
       status, processing_time_ms, payload, metadata
FROM event_logs
WHERE session_id = $1
  AND timestamp >= $2
ORDER BY timestamp ASC
```

**Index Used**: `idx_event_logs_session_timestamp`
**Expected Performance**: < 50ms for 1000 events
**Optimization**: Composite index eliminates sort operation

### Query 2: `get_session_metrics(session_id)`

```sql
SELECT event_type, status, processing_time_ms, topic, timestamp
FROM event_logs
WHERE session_id = $1
ORDER BY timestamp ASC
```

**Index Used**: `idx_event_logs_session_timestamp`
**Expected Performance**: < 100ms for metrics calculation
**Notes**: In-memory aggregation for percentiles, grouping

### Query 3: `find_correlated_events(correlation_id)`

```sql
SELECT event_id, session_id, correlation_id, event_type,
       topic, timestamp, status, processing_time_ms,
       payload, metadata
FROM event_logs
WHERE correlation_id = $1
ORDER BY timestamp ASC
```

**Index Used**: `idx_event_logs_correlation_id`
**Expected Performance**: < 20ms for correlation lookup
**Notes**: Typically 2-4 events per correlation (request → response chain)

## Data Model Alignment

### Event Schemas (9 Types)

The table design accommodates all event types from `codegen_schemas.py`:

1. **Request Events** (4 types)
   - `CodegenAnalysisRequest` → topic: `omninode_codegen_request_analyze_v1`
   - `CodegenValidationRequest` → topic: `omninode_codegen_request_validate_v1`
   - `CodegenPatternRequest` → topic: `omninode_codegen_request_pattern_v1`
   - `CodegenMixinRequest` → topic: `omninode_codegen_request_mixin_v1`

2. **Response Events** (4 types)
   - `CodegenAnalysisResponse` → topic: `omninode_codegen_response_analyze_v1`
   - `CodegenValidationResponse` → topic: `omninode_codegen_response_validate_v1`
   - `CodegenPatternResponse` → topic: `omninode_codegen_response_pattern_v1`
   - `CodegenMixinResponse` → topic: `omninode_codegen_response_mixin_v1`

3. **Status Events** (1 type)
   - `CodegenStatusEvent` → topic: `omninode_codegen_status_session_v1`

### Common Field Mapping

| Event Schema Field | Table Column | Notes |
|--------------------|--------------|-------|
| `correlation_id` | `correlation_id` | UUID |
| `session_id` | `session_id` | UUID |
| `timestamp` | `timestamp` | TIMESTAMPTZ |
| `schema_version` | `payload->'schema_version'` | Stored in JSONB payload |
| `processing_time_ms` | `processing_time_ms` | Direct mapping for responses |
| (entire event) | `payload` | Full event stored as JSONB |

## Dashboard Integration

### CodegenEventTracer Methods

The table design directly supports all tracer methods:

1. **`trace_session_events(session_id, time_range_hours)`**
   - Uses `idx_event_logs_session_timestamp` composite index
   - Returns complete event chain ordered chronologically
   - Calculates session_duration_ms from min/max timestamps

2. **`get_session_metrics(session_id)`**
   - Aggregates events by type and topic
   - Calculates percentiles (p50, p95, p99) from processing_time_ms
   - Identifies bottlenecks (topics with avg_response_time_ms > 5000ms)

3. **`find_correlated_events(correlation_id)`**
   - Uses `idx_event_logs_correlation_id` index
   - Returns request/response chain for debugging
   - Enables multi-hop event tracing

## Performance Targets

| Metric | Target | Justification |
|--------|--------|---------------|
| `trace_session_events()` | < 50ms | Index-optimized query for 1000 events |
| `get_session_metrics()` | < 100ms | In-memory aggregation acceptable |
| `find_correlated_events()` | < 20ms | Small result set (2-4 events) |
| Insert throughput | > 1000 events/sec | Async batch inserts from Kafka consumer |
| Index overhead | < 30% | 8 indexes, acceptable for read-heavy workload |

## Scalability Considerations

### Current Design (Phase 1)

- **Single table**: No partitioning
- **All indexes**: Full coverage for query patterns
- **JSONB storage**: Flexible schema evolution

### Future Optimizations (Phase 2+)

1. **Table Partitioning by `timestamp`**
   - Monthly partitions similar to `hash_metrics` table
   - Enables efficient time-range queries and old data cleanup
   - Prune partitions older than 90 days

2. **Materialized Views**
   - Pre-computed session metrics for faster dashboard loads
   - Refresh on event insert trigger or periodic refresh

3. **Connection Pooling**
   - Use existing PostgresConnectionManager
   - Pool size: 20-50 connections
   - Circuit breaker for resilience

4. **Query Optimization**
   - Add `EXPLAIN ANALYZE` monitoring
   - Identify slow queries with `pg_stat_statements`
   - Add covering indexes if needed

## Trade-offs and Alternatives

### Decision: JSONB vs. Separate Tables

**Chosen: JSONB `payload` column**

**Pros:**
- Schema flexibility (9 event types with different fields)
- No migrations needed for event schema changes
- Single table simplifies queries
- GIN indexes enable fast JSON queries

**Cons:**
- Storage overhead (~30% larger than normalized)
- No strong typing at database level
- Requires application-level validation

**Alternative: Separate tables per event type**
- Rejected: 9 tables + views complexity
- Rejected: JOIN overhead for session queries
- Rejected: Schema migrations for each event change

### Decision: Composite Index `(session_id, timestamp)`

**Chosen: Composite index for most common query**

**Pros:**
- Eliminates sort operation for `trace_session_events()`
- PostgreSQL query planner optimizes for both filter + sort
- Significant performance gain (2-3x faster)

**Cons:**
- Additional index storage (~15% of table size)
- Slower writes (marginal impact for event logs)

**Alternative: Separate indexes only**
- Rejected: PostgreSQL would scan session_id index, fetch rows, then sort
- Performance penalty for most frequent query

## Maintenance and Monitoring

### Cleanup Strategy

```sql
-- Delete events older than 90 days (retention policy)
DELETE FROM event_logs
WHERE timestamp < NOW() - INTERVAL '90 days';
```

### Monitoring Queries

```sql
-- Check table size
SELECT pg_size_pretty(pg_total_relation_size('event_logs'));

-- Check index usage
SELECT indexrelname, idx_scan, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes
WHERE schemaname = 'public' AND relname = 'event_logs';

-- Identify slow queries
SELECT query, mean_exec_time, calls
FROM pg_stat_statements
WHERE query LIKE '%event_logs%'
ORDER BY mean_exec_time DESC
LIMIT 10;
```

### Vacuum and Analyze

```sql
-- Regular maintenance
VACUUM ANALYZE event_logs;

-- Check bloat
SELECT schemaname, tablename,
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE tablename = 'event_logs';
```

## Helper Views

Two convenience views for common dashboard queries:

1. **`event_logs_session_summary`**
   - Session-level aggregates (last 24 hours)
   - Total events, unique correlations, session duration
   - Success/failure counts, avg processing time

2. **`event_logs_topic_performance`**
   - Topic-level performance metrics (last 24 hours)
   - Event counts, avg/max/p95 processing times
   - Failed vs. success counts for reliability tracking

## Testing Recommendations

### Integration Tests

1. **Event Insertion**
   - Insert all 9 event types
   - Verify payload and metadata JSONB storage
   - Check CHECK constraints enforcement

2. **Query Performance**
   - Benchmark `trace_session_events()` with 1000 events
   - Measure `get_session_metrics()` calculation time
   - Verify `find_correlated_events()` correctness

3. **Index Usage**
   - Run `EXPLAIN ANALYZE` on all tracer methods
   - Verify composite index is used for session queries
   - Check GIN index usage for JSONB queries

### Load Testing

```python
# Insert 10,000 events and measure performance
import asyncio
from uuid import uuid4

async def load_test():
    session_id = uuid4()
    events = []
    for i in range(10000):
        events.append({
            'session_id': session_id,
            'event_type': 'request',
            'topic': 'omninode_codegen_request_analyze_v1',
            'status': 'sent',
            'payload': {'test': True},
        })

    # Batch insert
    await db.batch_insert('event_logs', events)

    # Measure query performance
    start = time.time()
    trace = await tracer.trace_session_events(session_id)
    print(f"Query time: {(time.time() - start) * 1000:.2f}ms")
```

## Related Files

- **Migration**: `alembic/versions/005_20251015_event_logs_table.py`
- **Schema Documentation**: `migrations/schema.sql`
- **Event Schemas**: `src/omninode_bridge/events/codegen_schemas.py`
- **Tracer Implementation**: `src/omninode_bridge/dashboard/codegen_event_tracer.py`
- **Dashboard Models**: `src/omninode_bridge/dashboard/models/`

## Changelog

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-10-15 | 1.0 | Initial design for PR #25 | Agent A1 |

## Conclusion

This event_logs table design provides a robust foundation for autonomous code generation event tracing with:

✅ Comprehensive event storage (all 9 types)
✅ Optimized query performance (composite indexes)
✅ Flexible schema evolution (JSONB payloads)
✅ Strong data integrity (CHECK constraints)
✅ Dashboard integration (3 tracer methods)
✅ Scalability path (partitioning, materialized views)

The design prioritizes read performance for dashboard queries while maintaining acceptable write performance for Kafka event ingestion.
