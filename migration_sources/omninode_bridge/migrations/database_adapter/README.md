# Database Adapter Migrations

This directory contains PostgreSQL migration scripts for the Bridge Database Adapter Effect Node.

## Migration Files

### UP Migrations (Create Tables)
1. `001_create_workflow_executions.sql` - Workflow execution tracking table
2. `002_create_workflow_steps.sql` - Workflow step history table
3. `003_create_fsm_transitions.sql` - FSM state transition history table
4. `004_create_bridge_states.sql` - Bridge aggregation state table
5. `005_create_node_registrations.sql` - Node registration and health tracking table
6. `006_create_metadata_stamps.sql` - Metadata stamp audit trail table

### DOWN Migrations (Drop Tables)
1. `001_drop_workflow_executions.sql` - Rollback workflow_executions table
2. `002_drop_workflow_steps.sql` - Rollback workflow_steps table
3. `003_drop_fsm_transitions.sql` - Rollback fsm_transitions table
4. `004_drop_bridge_states.sql` - Rollback bridge_states table
5. `005_drop_node_registrations.sql` - Rollback node_registrations table
6. `006_drop_metadata_stamps.sql` - Rollback metadata_stamps table

## Database Schema Overview

### workflow_executions
- **Purpose**: Track orchestrator workflow execution state
- **Key Fields**: correlation_id, workflow_type, current_state, namespace
- **Relationships**: Parent to workflow_steps and metadata_stamps

### workflow_steps
- **Purpose**: Track individual steps within workflow executions
- **Key Fields**: workflow_id (FK), step_name, step_order, status
- **Relationships**: Child of workflow_executions

### fsm_transitions
- **Purpose**: Track finite state machine state transitions
- **Key Fields**: entity_id, entity_type, from_state, to_state
- **Relationships**: None (audit trail)

### bridge_states
- **Purpose**: Track reducer aggregation state
- **Key Fields**: bridge_id, namespace, total_workflows_processed
- **Relationships**: None (standalone state)

### node_registrations
- **Purpose**: Service discovery and health tracking
- **Key Fields**: node_id, node_type, health_status, last_heartbeat
- **Relationships**: None (registry)

### metadata_stamps
- **Purpose**: Audit trail for metadata stamp operations
- **Key Fields**: workflow_id (FK), file_hash, stamp_data, namespace
- **Relationships**: Child of workflow_executions (optional)

## Running Migrations

### Using psql

#### Apply All UP Migrations
```bash
# Apply migrations in order
for i in {1..6}; do
    psql -U postgres -d omninode_bridge -f "00${i}_create_*.sql"
done
```

#### Verify Tables
```bash
psql -U postgres -d omninode_bridge -c "\dt"
psql -U postgres -d omninode_bridge -c "\d workflow_executions"
```

#### Apply All DOWN Migrations (Rollback)
```bash
# Apply rollbacks in reverse order
for i in {6..1}; do
    psql -U postgres -d omninode_bridge -f "00${i}_drop_*.sql"
done
```

### Using the Helper Script

```bash
# Apply all UP migrations
./apply_migrations.sh up

# Rollback all migrations
./apply_migrations.sh down

# Show migration status
./apply_migrations.sh status
```

## Migration Order

**IMPORTANT**: Migrations must be applied in the correct order due to foreign key dependencies:

**UP Migrations** (1 → 6):
1. workflow_executions (no dependencies)
2. workflow_steps (depends on workflow_executions)
3. fsm_transitions (no dependencies)
4. bridge_states (no dependencies)
5. node_registrations (no dependencies)
6. metadata_stamps (depends on workflow_executions)

**DOWN Migrations** (6 → 1):
1. metadata_stamps (has FK to workflow_executions)
2. node_registrations (no dependencies)
3. bridge_states (no dependencies)
4. fsm_transitions (no dependencies)
5. workflow_steps (has FK to workflow_executions)
6. workflow_executions (parent table)

## Environment Variables

```bash
# PostgreSQL connection settings
export PGHOST=localhost
export PGPORT=5432
export PGDATABASE=omninode_bridge
export PGUSER=postgres
export PGPASSWORD=your_password
```

## Testing Migrations

```bash
# 1. Start PostgreSQL (Docker)
docker-compose up -d postgres

# 2. Apply all migrations
./apply_migrations.sh up

# 3. Verify tables exist
psql -U postgres -d omninode_bridge -c "\dt"

# 4. Check table structure
psql -U postgres -d omninode_bridge -c "\d+ workflow_executions"

# 5. Test data insertion
psql -U postgres -d omninode_bridge -c "
    INSERT INTO workflow_executions (correlation_id, workflow_type, current_state, namespace, started_at)
    VALUES (uuid_generate_v4(), 'test_workflow', 'PENDING', 'test_namespace', NOW());
"

# 6. Verify data
psql -U postgres -d omninode_bridge -c "SELECT * FROM workflow_executions;"

# 7. Rollback migrations
./apply_migrations.sh down

# 8. Verify clean rollback
psql -U postgres -d omninode_bridge -c "\dt"
```

## Performance Indexes

All tables include performance-optimized indexes:
- Primary key indexes (automatic)
- Foreign key indexes
- Query optimization indexes (namespace, state, timestamps)
- Multi-column composite indexes where appropriate

## ONEX v2.0 Compliance

These migrations support the ONEX v2.0 event-driven architecture:
- UUID-based correlation tracking
- Multi-tenant namespace isolation
- JSONB for flexible metadata storage
- Timestamp tracking for audit trails
- FSM state transition history

## Next Steps

After applying migrations:
1. Implement Database Adapter Effect Node
2. Create event consumer for Kafka events
3. Integrate with bridge nodes (Orchestrator, Reducer, Registry)
4. Add monitoring and health checks
5. Implement data retention policies
