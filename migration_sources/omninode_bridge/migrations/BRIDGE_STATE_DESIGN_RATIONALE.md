# Bridge State Tables - Design Rationale

**Created**: October 15, 2025
**Migrations**: 009, 010
**Purpose**: Enhance existing tables for orchestrator and reducer bridge nodes

## Overview

This document explains the design decisions for the bridge node persistence layer, which enhances existing generic tables (`workflow_executions` and `bridge_states`) with orchestrator and reducer-specific fields.

## Design Decision: Enhancement vs. New Tables

### Options Considered

**Option A: Enhance existing tables (SELECTED)**
- Add orchestrator-specific columns to `workflow_executions`
- Add reducer-specific columns to `bridge_states`
- Use JSONB for flexible metadata
- Maintain unified workflow tracking

**Option B: Create new dedicated tables**
- Create `orchestrator_workflows` table
- Create `reducer_aggregations` table
- Duplicate some common fields
- Separate tracking systems

### Decision: Option A - Enhance Existing Tables

**Rationale:**
1. **Avoid Data Duplication**: Orchestrator workflows are workflows - no need to duplicate base workflow tracking
2. **Unified Tracking**: Single source of truth for workflow execution across all node types
3. **Backwards Compatibility**: New columns are nullable or have defaults, maintaining compatibility
4. **Query Simplicity**: Single table queries vs. cross-table joins
5. **Database Efficiency**: Fewer tables = fewer indexes = better performance
6. **Multi-Tenant Design**: Existing namespace columns support multi-tenancy without duplication

## Migration 009: workflow_executions Enhancement

### Purpose
Add orchestrator-specific fields to track stamping workflow results and performance.

### New Columns

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `stamp_id` | VARCHAR(255) | YES | Unique stamp identifier (set when workflow completes) |
| `file_hash` | VARCHAR(64) | YES | BLAKE3 hash of stamped file content |
| `workflow_steps` | JSONB | YES | Array of workflow step execution details |
| `intelligence_data` | JSONB | YES | OnexTree intelligence analysis results |
| `hash_generation_time_ms` | INTEGER | YES | BLAKE3 hash generation time |
| `workflow_steps_executed` | INTEGER | YES | Total number of workflow steps executed (default: 0) |
| `session_id` | UUID | YES | Session identifier for multi-session grouping |

### Design Considerations

**1. Nullable Columns**
- All new columns are nullable to maintain backwards compatibility
- Existing workflows don't break when schema is updated
- Allows progressive migration of data

**2. JSONB for Flexibility**
- `workflow_steps`: Array of step objects with variable structure
  ```json
  [
    {"step_type": "validation", "status": "success", "duration_ms": 5},
    {"step_type": "hash_generation", "status": "success", "duration_ms": 1.5, "file_hash": "abc123..."},
    {"step_type": "stamp_creation", "status": "success", "duration_ms": 2, "stamp_id": "uuid..."}
  ]
  ```
- `intelligence_data`: Optional OnexTree analysis (graceful degradation when unavailable)
  ```json
  {
    "analysis_type": "content_analysis",
    "confidence_score": "0.85",
    "recommendations": "Consider optimizing file structure",
    "analyzed_at": "2025-10-15T10:30:00Z"
  }
  ```

**3. Performance Indexes**
- Partial indexes on nullable columns (only index non-NULL values)
- GIN indexes for JSONB columns (fast JSON queries)
- Index on `session_id` for session-based queries

**4. Integration with Orchestrator**

The orchestrator node (`NodeBridgeOrchestrator`) will:
1. Create workflow record at start (correlation_id, workflow_type, namespace)
2. Update workflow_steps array as each step completes
3. Set stamp_id and file_hash on successful completion
4. Store intelligence_data if OnexTree analysis succeeded
5. Update execution_time_ms and workflow_steps_executed on completion
6. Set current_state to COMPLETED or FAILED

**Example Query Patterns:**
```sql
-- Get workflow with stamp details
SELECT correlation_id, stamp_id, file_hash, current_state,
       workflow_steps, intelligence_data, execution_time_ms
FROM workflow_executions
WHERE correlation_id = $1;

-- Find workflows by stamp
SELECT * FROM workflow_executions
WHERE stamp_id = $1;

-- Query workflows with intelligence
SELECT * FROM workflow_executions
WHERE intelligence_data IS NOT NULL
  AND intelligence_data @> '{"analysis_type": "content_analysis"}';

-- Session-based queries
SELECT * FROM workflow_executions
WHERE session_id = $1
ORDER BY started_at DESC;
```

## Migration 010: bridge_states Enhancement

### Purpose
Add reducer-specific fields to track detailed aggregation statistics and performance.

### New Columns

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| `total_size_bytes` | BIGINT | NO | 0 | Total size of all stamped files in bytes |
| `unique_file_types` | TEXT[] | NO | '{}' | Array of unique MIME content types |
| `unique_workflows` | UUID[] | NO | '{}' | Array of unique workflow UUIDs processed |
| `aggregation_type` | VARCHAR(50) | NO | 'namespace_grouping' | Type of aggregation strategy |
| `window_start` | TIMESTAMPTZ | YES | NULL | Start timestamp of aggregation window |
| `window_end` | TIMESTAMPTZ | YES | NULL | End timestamp of aggregation window |
| `aggregation_duration_ms` | INTEGER | YES | NULL | Duration of aggregation operation |
| `items_per_second` | NUMERIC(10,2) | YES | NULL | Aggregation throughput |
| `version` | INTEGER | NO | 1 | Version for optimistic locking |
| `configuration` | JSONB | NO | '{}' | Aggregation configuration |

### Design Considerations

**1. PostgreSQL Arrays vs. JSONB**
- Used native PostgreSQL arrays (`TEXT[]`, `UUID[]`) for collections
- Arrays are more efficient than JSONB arrays for simple containment queries
- GIN indexes on arrays enable fast `@>` (contains) and `&&` (overlaps) operations

**2. Aggregation Strategies**
- `namespace_grouping`: Group stamps by namespace (primary strategy)
- `time_window`: Group stamps by time windows
- `file_type_grouping`: Group stamps by MIME type
- `size_buckets`: Group stamps by size ranges
- `workflow_grouping`: Group stamps by workflow type
- `custom`: User-defined aggregation logic

**3. Windowed Aggregations**
- `window_start` and `window_end` track time-based aggregation windows
- NULL when using non-windowed aggregation (e.g., namespace_grouping only)
- Indexed for efficient time-range queries

**4. Optimistic Locking**
- `version` column enables conflict detection during concurrent updates
- Application-level optimistic locking pattern:
  ```python
  # Read current version
  current = await db.fetchrow("SELECT * FROM bridge_states WHERE bridge_id = $1", id)

  # Update with version check
  result = await db.execute(
      "UPDATE bridge_states SET total_items_aggregated = $1, version = version + 1 "
      "WHERE bridge_id = $2 AND version = $3",
      new_count, id, current['version']
  )

  if result == "UPDATE 0":
      raise ConflictError("Concurrent modification detected")
  ```

**5. Configuration Storage**
- `configuration` JSONB stores aggregation parameters
  ```json
  {
    "batch_size": 100,
    "window_size_ms": 5000,
    "aggregation_strategy": "namespace_grouping",
    "flush_interval_seconds": 30
  }
  ```

**6. Integration with Reducer**

The reducer node (`NodeBridgeReducer`) will:
1. Create or update bridge_state record per namespace
2. Increment counters (total_items_aggregated, total_size_bytes)
3. Update arrays (unique_file_types, unique_workflows)
4. Track window boundaries for time-based aggregations
5. Record performance metrics (aggregation_duration_ms, items_per_second)
6. Use optimistic locking for concurrent aggregation safety

**Example Query Patterns:**
```sql
-- Get aggregation state by namespace
SELECT * FROM bridge_states
WHERE namespace = $1;

-- Find states with specific file types
SELECT * FROM bridge_states
WHERE unique_file_types @> ARRAY['application/pdf'];

-- Time-window queries
SELECT * FROM bridge_states
WHERE window_start >= $1 AND window_end <= $2
ORDER BY window_start DESC;

-- Aggregation type statistics
SELECT aggregation_type,
       COUNT(*) as state_count,
       SUM(total_items_aggregated) as total_items,
       AVG(items_per_second) as avg_throughput
FROM bridge_states
GROUP BY aggregation_type;

-- Performance analysis
SELECT namespace,
       total_items_aggregated,
       total_size_bytes,
       items_per_second,
       aggregation_duration_ms
FROM bridge_states
WHERE last_aggregation_timestamp > NOW() - INTERVAL '1 hour'
ORDER BY items_per_second DESC;
```

## Performance Optimization Strategy

### Index Design Principles

**1. Partial Indexes**
- Index only non-NULL values for optional columns
- Example: `WHERE stamp_id IS NOT NULL`
- Reduces index size and maintenance overhead

**2. GIN Indexes for Complex Types**
- JSONB columns: Fast JSON containment and path queries
- Array columns: Fast containment (`@>`) and overlap (`&&`) queries
- Trade-off: Slower writes for faster reads (appropriate for read-heavy workloads)

**3. Compound Indexes**
- `idx_bridge_states_namespace_aggregation_type`: Common filter combination
- `idx_bridge_states_namespace_window`: Time-range queries per namespace
- Query optimizer uses these for multiple conditions

**4. Index Selectivity**
- High-selectivity columns indexed first in compound indexes
- Example: `namespace` (high selectivity) before `aggregation_type` (low selectivity)

### Query Optimization

**1. Workflow Execution Queries**
- Correlation ID lookup: O(1) via unique index
- Stamp ID lookup: O(1) via partial index
- Session queries: O(log n) via B-tree index

**2. Bridge State Queries**
- Namespace lookup: O(1) via B-tree index
- Array containment: O(log n) via GIN index
- Time-range queries: O(log n) via compound index

**3. Aggregation Queries**
- COUNT, SUM, AVG: Optimized by covering indexes
- GROUP BY namespace: Uses namespace index
- Window functions: Uses temporal indexes

## Multi-Tenant Design

### Namespace Isolation

Both enhanced tables use the `namespace` column for multi-tenant isolation:
- Orchestrator workflows: Isolated by namespace
- Reducer aggregations: Per-namespace state tracking
- Query patterns: Always filter by namespace first

### Row-Level Security (Future Enhancement)

For production multi-tenant deployments, consider adding RLS policies:

```sql
-- Enable RLS
ALTER TABLE workflow_executions ENABLE ROW LEVEL SECURITY;
ALTER TABLE bridge_states ENABLE ROW LEVEL SECURITY;

-- Create policies (example)
CREATE POLICY namespace_isolation ON workflow_executions
    USING (namespace = current_setting('app.current_namespace')::text);

CREATE POLICY namespace_isolation ON bridge_states
    USING (namespace = current_setting('app.current_namespace')::text);
```

## Backwards Compatibility

### Schema Evolution Strategy

**1. Additive Changes Only**
- New columns are nullable or have defaults
- Existing queries continue to work
- No breaking changes to application code

**2. Progressive Migration**
- Existing workflows remain valid after schema upgrade
- New workflows populate new columns
- Optional: Backfill historical data

**3. Application-Level Compatibility**
- Python models use Pydantic defaults for new fields
- Database queries use explicit column lists (not SELECT *)
- Version checks in application code for feature gating

## Testing Strategy

### Migration Testing

**1. Forward Migration Test**
```bash
# Apply migrations
psql -f migrations/009_enhance_workflow_executions.sql
psql -f migrations/010_enhance_bridge_states.sql

# Verify schema
\d workflow_executions
\d bridge_states

# Test queries
SELECT * FROM workflow_executions LIMIT 1;
SELECT * FROM bridge_states LIMIT 1;
```

**2. Rollback Migration Test**
```bash
# Rollback
psql -f migrations/010_rollback_bridge_states.sql
psql -f migrations/009_rollback_workflow_executions.sql

# Verify rollback
\d workflow_executions
\d bridge_states
```

**3. Idempotency Test**
```bash
# Run migrations twice (should not error)
psql -f migrations/009_enhance_workflow_executions.sql
psql -f migrations/009_enhance_workflow_executions.sql
```

### Integration Testing

**1. Orchestrator Integration**
- Create workflow with all orchestrator fields
- Query workflow by correlation_id
- Query workflow by stamp_id
- Query workflows by session_id
- Test JSONB queries on workflow_steps and intelligence_data

**2. Reducer Integration**
- Create/update bridge_state with all reducer fields
- Test array containment queries (unique_file_types, unique_workflows)
- Test time-window queries (window_start, window_end)
- Test optimistic locking (version conflict detection)
- Test aggregation statistics queries

**3. Performance Testing**
- Insert 10,000 workflows
- Measure query performance (correlation_id, stamp_id, session_id)
- Insert 1,000 bridge_states
- Measure aggregation query performance
- Test index usage with EXPLAIN ANALYZE

## Migration Execution Order

### Forward Migration
```
001-008: Existing migrations (already applied)
009: enhance_workflow_executions
010: enhance_bridge_states
```

### Rollback Order (Reverse)
```
010: rollback_bridge_states
009: rollback_workflow_executions
```

## Performance Targets

### Orchestrator Queries
- Workflow lookup by correlation_id: < 5ms (p95)
- Workflow lookup by stamp_id: < 10ms (p95)
- Session-based queries: < 50ms for 100 workflows (p95)
- JSONB containment queries: < 20ms (p95)

### Reducer Queries
- Bridge state lookup by namespace: < 5ms (p95)
- Array containment queries: < 15ms (p95)
- Time-window queries: < 30ms for 1000 states (p95)
- Aggregation statistics: < 100ms (p95)

## Future Enhancements

### Potential Improvements

**1. Partitioning**
- Partition workflow_executions by created_at (monthly)
- Partition bridge_states by namespace (for large multi-tenant deployments)

**2. Materialized Views**
- Pre-computed aggregation statistics
- Session summary views
- Namespace performance dashboards

**3. Time-Series Optimization**
- TimescaleDB hypertables for workflow_executions
- Continuous aggregates for bridge_states
- Automatic data retention policies

**4. Advanced Indexing**
- BRIN indexes for time-series columns
- Full-text search on metadata JSONB
- PostGIS for spatial data (if needed)

## References

- **Bridge Orchestrator Implementation**: `src/omninode_bridge/nodes/orchestrator/v1_0_0/node.py`
- **Bridge Reducer Implementation**: `src/omninode_bridge/nodes/reducer/v1_0_0/node.py`
- **ModelBridgeState**: `src/omninode_bridge/nodes/reducer/v1_0_0/models/model_bridge_state.py`
- **ModelStampResponseOutput**: `src/omninode_bridge/nodes/orchestrator/v1_0_0/models/model_stamp_response_output.py`
- **Existing Migrations**: `migrations/001-008_*.sql`
- **PostgreSQL Documentation**: https://www.postgresql.org/docs/current/

## Change Log

### v1.0.0 (October 15, 2025)
- ✅ Migration 009: workflow_executions enhancement (7 new columns, 5 indexes)
- ✅ Migration 010: bridge_states enhancement (10 new columns, 9 indexes)
- ✅ Backwards compatible schema evolution
- ✅ Comprehensive design rationale documentation
