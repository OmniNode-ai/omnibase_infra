# PostgreSQL Database Migrations

This directory contains PostgreSQL migration scripts for the OmniNode Bridge database schema.

## Overview

The migration system provides version-controlled database schema management for the bridge nodes PostgreSQL database. Each migration is numbered sequentially and includes both forward migration and rollback scripts.

## Migration Structure

```
migrations/
├── README.md                                    # This file
├── 001_create_workflow_executions.sql          # Workflow execution tracking
├── 001_rollback_workflow_executions.sql
├── 002_create_workflow_steps.sql               # Workflow step history
├── 002_rollback_workflow_steps.sql
├── 003_create_fsm_transitions.sql              # FSM state transitions
├── 003_rollback_fsm_transitions.sql
├── 004_create_bridge_states.sql                # Bridge aggregation state
├── 004_rollback_bridge_states.sql
├── 005_create_node_registrations.sql           # Node service discovery
├── 005_rollback_node_registrations.sql
├── 006_create_metadata_stamps.sql              # Metadata stamp audit trail
└── 006_rollback_metadata_stamps.sql
```

## Database Schema

### Tables Created

1. **workflow_executions** - Tracks workflow execution lifecycle and state
   - Primary key: `id` (UUID)
   - Unique: `correlation_id` (UUID)
   - Indexes: correlation_id, namespace, current_state, started_at

2. **workflow_steps** - Tracks individual step execution within workflows
   - Primary key: `id` (UUID)
   - Foreign key: `workflow_id` → workflow_executions(id) ON DELETE CASCADE
   - Indexes: workflow_id, status, created_at, (workflow_id, step_order)

3. **fsm_transitions** - Tracks all state machine transitions across entities
   - Primary key: `id` (UUID)
   - Indexes: (entity_id, entity_type), entity_type, created_at, (from_state, to_state)

4. **bridge_states** - Tracks aggregation state for bridge reducer nodes
   - Primary key: `bridge_id` (UUID)
   - Indexes: namespace, current_fsm_state, last_aggregation_timestamp, (namespace, current_fsm_state)

5. **node_registrations** - Service discovery and health tracking
   - Primary key: `node_id` (VARCHAR)
   - Indexes: health_status, node_type, last_heartbeat, (node_type, health_status)

6. **metadata_stamps** - Audit trail for metadata stamping operations
   - Primary key: `id` (UUID)
   - Foreign key: `workflow_id` → workflow_executions(id) ON DELETE SET NULL (optional)
   - Indexes: workflow_id, file_hash, namespace, created_at, (namespace, created_at)

### Dependencies

- **PostgreSQL Extensions Required**:
  - `uuid-ossp`: UUID generation functions
  - `pg_stat_statements`: Performance monitoring

- **Foreign Key Dependencies**:
  - `workflow_steps.workflow_id` → `workflow_executions.id`
  - `metadata_stamps.workflow_id` → `workflow_executions.id` (nullable)

## Running Migrations

### Prerequisites

1. PostgreSQL 13+ installed and running
2. Database created: `omninode_bridge`
3. Database user with CREATE TABLE and CREATE INDEX privileges
4. **PostgreSQL extensions created** (one-time setup):
   ```bash
   # Automated setup (recommended)
   bash deployment/scripts/setup_postgres_extensions.sh

   # OR manual setup
   psql -U postgres -d omninode_bridge -c "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";"
   psql -U postgres -d omninode_bridge -c "CREATE EXTENSION IF NOT EXISTS \"pg_stat_statements\";"
   ```

### Method 1: Manual Execution (Development)

```bash
# Connect to database
psql -h localhost -U postgres -d omninode_bridge

# Run migrations in order
\i migrations/001_create_workflow_executions.sql
\i migrations/002_create_workflow_steps.sql
\i migrations/003_create_fsm_transitions.sql
\i migrations/004_create_bridge_states.sql
\i migrations/005_create_node_registrations.sql
\i migrations/006_create_metadata_stamps.sql

# Verify tables created
\dt
\d+ workflow_executions
```

### Method 2: Script Execution

```bash
# Run all migrations
for migration in migrations/00*.sql; do
    if [[ ! $migration =~ rollback ]]; then
        echo "Running: $migration"
        psql -h localhost -U postgres -d omninode_bridge -f "$migration"
    fi
done
```

### Method 3: Docker Container

```bash
# Assuming PostgreSQL running in Docker
docker exec -i postgres_container psql -U postgres -d omninode_bridge < migrations/001_create_workflow_executions.sql
docker exec -i postgres_container psql -U postgres -d omninode_bridge < migrations/002_create_workflow_steps.sql
# ... continue for all migrations
```

## Rolling Back Migrations

### Rollback Order

**IMPORTANT**: Always rollback in reverse order due to foreign key constraints!

```bash
# Connect to database
psql -h localhost -U postgres -d omninode_bridge

# Rollback migrations in REVERSE order
\i migrations/006_rollback_metadata_stamps.sql
\i migrations/005_rollback_node_registrations.sql
\i migrations/004_rollback_bridge_states.sql
\i migrations/003_rollback_fsm_transitions.sql
\i migrations/002_rollback_workflow_steps.sql
\i migrations/001_rollback_workflow_executions.sql
```

### Script-Based Rollback

```bash
# Rollback all migrations in reverse order
for migration in $(ls -r migrations/*_rollback_*.sql); do
    echo "Rolling back: $migration"
    psql -h localhost -U postgres -d omninode_bridge -f "$migration"
done
```

## Testing Migrations

### 1. Test Forward Migration

```bash
# Run all migrations
./scripts/run_migrations.sh

# Verify schema
psql -h localhost -U postgres -d omninode_bridge -c "\dt"
psql -h localhost -U postgres -d omninode_bridge -c "\di"

# Check foreign keys
psql -h localhost -U postgres -d omninode_bridge -c "
SELECT
    tc.table_name,
    kcu.column_name,
    ccu.table_name AS foreign_table_name,
    ccu.column_name AS foreign_column_name
FROM information_schema.table_constraints AS tc
JOIN information_schema.key_column_usage AS kcu
  ON tc.constraint_name = kcu.constraint_name
JOIN information_schema.constraint_column_usage AS ccu
  ON ccu.constraint_name = tc.constraint_name
WHERE tc.constraint_type = 'FOREIGN KEY';
"
```

### 2. Test Rollback

```bash
# Rollback all migrations
./scripts/rollback_migrations.sh

# Verify tables dropped
psql -h localhost -U postgres -d omninode_bridge -c "\dt"
```

### 3. Test Idempotency

```bash
# Run migrations twice (should not error due to IF NOT EXISTS)
./scripts/run_migrations.sh
./scripts/run_migrations.sh

# Should see no errors
```

## Migration Guidelines

### Creating New Migrations

1. **Naming Convention**: `XXX_description.sql` and `XXX_rollback_description.sql`
   - Use sequential numbers (001, 002, 003...)
   - Use descriptive names (create_table_name, add_index_name, etc.)

2. **Migration Content**:
   - Always use `IF NOT EXISTS` for CREATE statements
   - Include comments explaining purpose and dependencies
   - Add COMMENT ON statements for documentation
   - Create appropriate indexes for performance

3. **Rollback Content**:
   - Always use `IF EXISTS` for DROP statements
   - Drop indexes before dropping tables
   - Use CASCADE for tables with foreign key references
   - Reverse order of operations from forward migration

4. **Dependencies**:
   - Document dependencies in migration header
   - Respect foreign key order (referenced tables first)
   - Consider data migration needs

### Best Practices

- ✅ **DO**: Test migrations on development database first
- ✅ **DO**: Include rollback scripts for every migration
- ✅ **DO**: Use transactions for data migrations
- ✅ **DO**: Document breaking changes and manual steps
- ✅ **DO**: Version control all migration scripts

- ❌ **DON'T**: Modify existing migration files after deployment
- ❌ **DON'T**: Skip migration numbers
- ❌ **DON'T**: Delete old migration files
- ❌ **DON'T**: Run migrations directly on production without testing

## Environment-Specific Considerations

### Development

```bash
# Quick reset for development
psql -h localhost -U postgres -d omninode_bridge -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
./scripts/run_migrations.sh
```

### Staging

```bash
# Run migrations with logging
./scripts/run_migrations.sh 2>&1 | tee migration_$(date +%Y%m%d_%H%M%S).log
```

### Production

```bash
# ALWAYS backup first!
pg_dump -h production_host -U postgres omninode_bridge > backup_$(date +%Y%m%d_%H%M%S).sql

# Run migrations with transaction
psql -h production_host -U postgres -d omninode_bridge << EOF
BEGIN;
\i migrations/XXX_new_migration.sql
-- Verify results
SELECT COUNT(*) FROM new_table;
-- If all looks good:
COMMIT;
-- If issues:
-- ROLLBACK;
EOF
```

## Troubleshooting

### Error: "extension uuid-ossp does not exist"

```sql
-- Connect as superuser
psql -h localhost -U postgres -d omninode_bridge
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
```

### Error: "relation already exists"

This is expected if migrations are idempotent. Verify with:
```sql
SELECT tablename FROM pg_tables WHERE schemaname = 'public';
```

### Error: "foreign key constraint violation"

Ensure migrations run in correct order:
1. workflow_executions (001)
2. workflow_steps (002) - depends on 001
3. fsm_transitions (003)
4. bridge_states (004)
5. node_registrations (005)
6. metadata_stamps (006) - depends on 001

### Error: "permission denied"

Grant necessary privileges:
```sql
GRANT ALL PRIVILEGES ON DATABASE omninode_bridge TO your_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO your_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO your_user;
```

## Monitoring

### Check Migration Status

```sql
-- List all tables and their row counts
SELECT
    schemaname,
    tablename,
    n_live_tup AS row_count
FROM pg_stat_user_tables
ORDER BY tablename;

-- Check index usage
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan AS index_scans
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;
```

### Performance Analysis

```sql
-- Enable pg_stat_statements
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- View query performance
SELECT
    substring(query, 1, 50) AS query,
    calls,
    total_time,
    mean_time,
    max_time
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 20;
```

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review migration comments and documentation
3. Consult DATABASE_ADAPTER_EFFECT_NODE_PLAN.md
4. Check PostgreSQL logs: `tail -f /var/log/postgresql/postgresql-*.log`

## Version History

- **v1.0.0** (2025-10-07): Initial schema with 6 core tables
  - workflow_executions
  - workflow_steps
  - fsm_transitions
  - bridge_states
  - node_registrations
  - metadata_stamps
