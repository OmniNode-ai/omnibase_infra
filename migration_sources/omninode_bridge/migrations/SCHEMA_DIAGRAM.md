# Database Schema Diagram

## Entity Relationship Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     WORKFLOW TRACKING                            │
└─────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────┐
    │  workflow_executions     │
    │  (001)                   │
    ├──────────────────────────┤
    │ • id (PK)               │◄─────┐
    │ • correlation_id (UK)    │      │
    │ • workflow_type          │      │ FK: ON DELETE CASCADE
    │ • current_state          │      │
    │ • namespace              │      │
    │ • started_at             │      │
    │ • completed_at           │      │
    │ • execution_time_ms      │      │
    │ • error_message          │      │
    │ • metadata (JSONB)       │      │
    └──────────────────────────┘      │
                │                      │
                │ FK: ON DELETE        │
                │ CASCADE              │
                ▼                      │
    ┌──────────────────────────┐      │
    │   workflow_steps         │      │
    │   (002)                  │      │
    ├──────────────────────────┤      │
    │ • id (PK)               │      │
    │ • workflow_id (FK)       │──────┘
    │ • step_name              │
    │ • step_order             │
    │ • status                 │
    │ • execution_time_ms      │
    │ • step_data (JSONB)      │
    │ • error_message          │
    └──────────────────────────┘


    ┌──────────────────────────┐
    │   metadata_stamps        │
    │   (006)                  │
    ├──────────────────────────┤
    │ • id (PK)               │
    │ • workflow_id (FK)       │──────┐
    │ • file_hash              │      │
    │ • stamp_data (JSONB)     │      │ FK: ON DELETE SET NULL
    │ • namespace              │      │ (nullable, audit trail)
    └──────────────────────────┘      │
                                      │
                                      └──► workflow_executions.id


┌─────────────────────────────────────────────────────────────────┐
│                    STATE MANAGEMENT                              │
└─────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────┐
    │   fsm_transitions        │
    │   (003)                  │
    ├──────────────────────────┤
    │ • id (PK)               │
    │ • entity_id (UUID)       │  ◄── Generic reference
    │ • entity_type            │      (workflow, bridge, etc)
    │ • from_state             │
    │ • to_state               │
    │ • transition_event       │
    │ • transition_data (JSON) │
    └──────────────────────────┘


    ┌──────────────────────────┐
    │    bridge_states         │
    │    (004)                 │
    ├──────────────────────────┤
    │ • bridge_id (PK)        │
    │ • namespace              │
    │ • total_workflows_proc   │
    │ • total_items_agg        │
    │ • aggregation_metadata   │
    │ • current_fsm_state      │
    │ • last_agg_timestamp     │
    └──────────────────────────┘


┌─────────────────────────────────────────────────────────────────┐
│                  SERVICE DISCOVERY                               │
└─────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────┐
    │  node_registrations      │
    │  (005)                   │
    ├──────────────────────────┤
    │ • node_id (PK)          │
    │ • node_type              │
    │ • node_version           │
    │ • capabilities (JSONB)   │
    │ • endpoints (JSONB)      │
    │ • metadata (JSONB)       │
    │ • health_status          │
    │ • last_heartbeat         │
    └──────────────────────────┘
```

## Table Dependencies

### Migration Order (Forward)

```
Level 1: Independent Tables (No Dependencies)
├── 001_create_workflow_executions.sql
├── 003_create_fsm_transitions.sql
├── 004_create_bridge_states.sql
└── 005_create_node_registrations.sql

Level 2: Dependent Tables (References Level 1)
├── 002_create_workflow_steps.sql (→ workflow_executions)
└── 006_create_metadata_stamps.sql (→ workflow_executions)
```

### Rollback Order (Reverse)

```
Level 2: Drop Dependent Tables First
├── 006_rollback_metadata_stamps.sql
└── 002_rollback_workflow_steps.sql

Level 1: Drop Independent Tables Last
├── 005_rollback_node_registrations.sql
├── 004_rollback_bridge_states.sql
├── 003_rollback_fsm_transitions.sql
└── 001_rollback_workflow_executions.sql
```

## Index Strategy

### Single Column Indexes (17 total)

**workflow_executions:**
- `correlation_id` - Unique correlation tracking
- `namespace` - Multi-tenant filtering
- `current_state` - FSM state queries
- `started_at DESC` - Time-based queries

**workflow_steps:**
- `workflow_id` - Parent relationship
- `status` - Status filtering
- `created_at DESC` - Chronological order

**fsm_transitions:**
- `(entity_id, entity_type)` - Composite entity lookup
- `entity_type` - Type filtering
- `created_at DESC` - Audit trail queries

**bridge_states:**
- `namespace` - Multi-tenant filtering
- `current_fsm_state` - State queries
- `last_aggregation_timestamp DESC` - Recent activity

**node_registrations:**
- `health_status` - Health monitoring
- `node_type` - Type filtering
- `last_heartbeat DESC` - Staleness detection

**metadata_stamps:**
- `workflow_id` - Workflow relationship
- `file_hash` - Hash-based lookup
- `namespace` - Multi-tenant filtering
- `created_at DESC` - Audit trail

### Compound Indexes (8 total)

**workflow_steps:**
- `(workflow_id, step_order)` - Sequential step retrieval

**fsm_transitions:**
- `(from_state, to_state)` - Transition analysis

**bridge_states:**
- `(namespace, current_fsm_state)` - Tenant + state queries

**node_registrations:**
- `(node_type, health_status)` - Type + health filtering

**metadata_stamps:**
- `(namespace, created_at DESC)` - Tenant audit queries

## Data Flow

```
┌──────────────────┐         ┌──────────────────┐
│  Orchestrator    │─────────▶│ workflow_exec.   │
│  (Bridge Node)   │ Events   │ (INSERT/UPDATE)  │
└──────────────────┘         └──────────────────┘
        │                            │
        │ Step Events                │ FK Reference
        ▼                            ▼
┌──────────────────┐         ┌──────────────────┐
│ workflow_steps   │         │ metadata_stamps  │
│ (INSERT)         │         │ (INSERT)         │
└──────────────────┘         └──────────────────┘


┌──────────────────┐         ┌──────────────────┐
│  Reducer         │─────────▶│ bridge_states    │
│  (Bridge Node)   │ Events   │ (UPSERT)         │
└──────────────────┘         └──────────────────┘


┌──────────────────┐         ┌──────────────────┐
│  All Nodes       │─────────▶│ fsm_transitions  │
│  (FSM Events)    │ Events   │ (INSERT)         │
└──────────────────┘         └──────────────────┘


┌──────────────────┐         ┌──────────────────┐
│  Registry        │─────────▶│ node_regist.     │
│  (Bridge Node)   │ Events   │ (INSERT/UPDATE)  │
└──────────────────┘         └──────────────────┘
```

## Query Patterns

### Common Query 1: Get Workflow Status with Steps

```sql
SELECT
    we.correlation_id,
    we.workflow_type,
    we.current_state,
    we.execution_time_ms,
    json_agg(
        json_build_object(
            'step_name', ws.step_name,
            'step_order', ws.step_order,
            'status', ws.status,
            'execution_time_ms', ws.execution_time_ms
        ) ORDER BY ws.step_order
    ) AS steps
FROM workflow_executions we
LEFT JOIN workflow_steps ws ON ws.workflow_id = we.id
WHERE we.correlation_id = $1
GROUP BY we.id;
```

### Common Query 2: Get Recent Failed Workflows by Namespace

```sql
SELECT
    we.correlation_id,
    we.workflow_type,
    we.error_message,
    we.started_at,
    COUNT(ws.id) AS step_count
FROM workflow_executions we
LEFT JOIN workflow_steps ws ON ws.workflow_id = we.id
WHERE we.namespace = $1
  AND we.current_state = 'FAILED'
  AND we.started_at > NOW() - INTERVAL '1 hour'
GROUP BY we.id
ORDER BY we.started_at DESC
LIMIT 20;
```

### Common Query 3: Get FSM Transition History

```sql
SELECT
    entity_id,
    entity_type,
    from_state,
    to_state,
    transition_event,
    created_at
FROM fsm_transitions
WHERE entity_id = $1
  AND entity_type = $2
ORDER BY created_at ASC;
```

### Common Query 4: Get Bridge Aggregation Statistics

```sql
SELECT
    namespace,
    SUM(total_workflows_processed) AS total_workflows,
    SUM(total_items_aggregated) AS total_items,
    COUNT(*) AS bridge_count,
    MAX(last_aggregation_timestamp) AS latest_activity
FROM bridge_states
WHERE namespace = $1
GROUP BY namespace;
```

### Common Query 5: Get Healthy Nodes by Type

```sql
SELECT
    node_id,
    node_version,
    capabilities,
    endpoints,
    last_heartbeat
FROM node_registrations
WHERE node_type = $1
  AND health_status = 'HEALTHY'
  AND last_heartbeat > NOW() - INTERVAL '5 minutes'
ORDER BY last_heartbeat DESC;
```

## Performance Considerations

### Index Usage Verification

```sql
-- Check index usage after deployment
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan AS times_used,
    idx_tup_read AS tuples_read,
    idx_tup_fetch AS tuples_fetched
FROM pg_stat_user_indexes
WHERE schemaname = 'public'
ORDER BY idx_scan DESC;
```

### Missing Indexes Detection

```sql
-- Identify tables doing sequential scans
SELECT
    schemaname,
    tablename,
    seq_scan,
    seq_tup_read,
    idx_scan,
    seq_tup_read / seq_scan AS avg_seq_read
FROM pg_stat_user_tables
WHERE seq_scan > 0
ORDER BY seq_tup_read DESC
LIMIT 10;
```

### JSONB Performance

```sql
-- Create GIN indexes for JSONB columns if needed
CREATE INDEX IF NOT EXISTS idx_workflow_exec_metadata_gin
    ON workflow_executions USING GIN(metadata);

CREATE INDEX IF NOT EXISTS idx_bridge_states_agg_metadata_gin
    ON bridge_states USING GIN(aggregation_metadata);
```

## Multi-Tenant Isolation

### Namespace-Based Queries

All queries should include namespace filtering for multi-tenant isolation:

```sql
-- Always filter by namespace
SELECT * FROM workflow_executions
WHERE namespace = $1 AND correlation_id = $2;

-- Aggregate by namespace
SELECT namespace, COUNT(*) AS workflow_count
FROM workflow_executions
GROUP BY namespace;

-- Cross-namespace summary (admin only)
SELECT
    we.namespace,
    COUNT(DISTINCT we.id) AS total_workflows,
    AVG(we.execution_time_ms) AS avg_execution_time,
    COUNT(DISTINCT bs.bridge_id) AS bridge_count
FROM workflow_executions we
LEFT JOIN bridge_states bs ON bs.namespace = we.namespace
GROUP BY we.namespace
ORDER BY total_workflows DESC;
```

## Monitoring Queries

### System Health Dashboard

```sql
-- Overall system health metrics
SELECT
    'workflows' AS metric,
    COUNT(*) AS total,
    COUNT(CASE WHEN current_state = 'COMPLETED' THEN 1 END) AS completed,
    COUNT(CASE WHEN current_state = 'FAILED' THEN 1 END) AS failed,
    COUNT(CASE WHEN current_state = 'PROCESSING' THEN 1 END) AS in_progress
FROM workflow_executions
WHERE started_at > NOW() - INTERVAL '1 hour'

UNION ALL

SELECT
    'nodes' AS metric,
    COUNT(*) AS total,
    COUNT(CASE WHEN health_status = 'HEALTHY' THEN 1 END) AS healthy,
    COUNT(CASE WHEN health_status = 'UNHEALTHY' THEN 1 END) AS unhealthy,
    COUNT(CASE WHEN last_heartbeat < NOW() - INTERVAL '5 minutes' THEN 1 END) AS stale
FROM node_registrations;
```

## Migration Version Tracking

To track which migrations have been applied, consider creating a version table:

```sql
-- Optional: Migration version tracking
CREATE TABLE IF NOT EXISTS schema_migrations (
    version VARCHAR(50) PRIMARY KEY,
    applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    description TEXT
);

-- Record migrations as they're applied
INSERT INTO schema_migrations (version, description) VALUES
    ('001', 'Create workflow_executions table'),
    ('002', 'Create workflow_steps table'),
    ('003', 'Create fsm_transitions table'),
    ('004', 'Create bridge_states table'),
    ('005', 'Create node_registrations table'),
    ('006', 'Create metadata_stamps table');
```
