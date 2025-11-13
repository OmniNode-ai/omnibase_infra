# Database Migrations Guide - OmniNode Bridge

**Comprehensive PostgreSQL Migration Management Guide**
**Version**: 2.0 (Consolidated)
**Last Updated**: October 29, 2025
**Target Audience**: Developers, DevOps Engineers, SREs

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [PostgreSQL Extensions](#postgresql-extensions)
4. [Migration Inventory](#migration-inventory)
5. [Deployment Procedures](#deployment-procedures)
6. [Migration Management](#migration-management)
7. [Verification & Testing](#verification--testing)
8. [Rollback Procedures](#rollback-procedures)
9. [Troubleshooting](#troubleshooting)
10. [CI/CD Integration](#cicd-integration)
11. [Best Practices](#best-practices)

---

## Overview

This guide covers complete database migration management for the OmniNode Bridge project, including PostgreSQL extension requirements, deployment procedures for all environments, and troubleshooting.

**Migration Tool**: SQL scripts (manual execution)
**Location**: `/migrations/`
**Total Migrations**: 11 migrations (001-011) + schema.sql

**Related Documentation**:
- [Database Guide](./DATABASE_GUIDE.md) - Complete database architecture
- [Pre-Deployment Checklist](../deployment/PRE_DEPLOYMENT_CHECKLIST.md) - Full deployment checklist
- [Technical Migration Details](../../migrations/README.md) - Developer-focused details

---

## Quick Start

### For Developers (Local Setup)

```bash
# 1. Setup PostgreSQL extensions (one-time)
bash deployment/scripts/setup_postgres_extensions.sh

# 2. Apply all migrations
for migration in migrations/00*.sql migrations/01*.sql; do
    if [[ ! $migration =~ rollback ]]; then
        echo "Applying: $migration"
        psql -h localhost -U postgres -d omninode_bridge -f "$migration"
    fi
done

# 3. Verify tables created
psql -h localhost -U postgres -d omninode_bridge -c "\dt"
```

### For DevOps (Production Deployment)

See [Deployment Procedures → Production Environment](#production-environment)

---

## PostgreSQL Extensions

### Required Extensions

This project requires the following PostgreSQL extensions:

1. **uuid-ossp** (CRITICAL) - UUID generation functions
   - **Used for**: Auto-generating UUIDs for primary keys
   - **Requires**: Superuser privileges to create
   - **Status**: Required for all migrations

2. **pg_stat_statements** (RECOMMENDED) - Query performance tracking
   - **Used for**: Database performance monitoring and query analysis
   - **Requires**: Superuser privileges to create
   - **Status**: Optional in development, recommended for production

### Why Superuser Privileges Are Required

PostgreSQL extensions modify the database system catalog and can add new functions, operators, and types. For security reasons, only superuser accounts can install extensions to prevent unauthorized modifications to the database system.

**Reference**: [PostgreSQL CREATE EXTENSION Documentation](https://www.postgresql.org/docs/current/sql-createextension.html)

### Extension Setup Options

#### Option 1: Automated Setup Script (Recommended)

Use the provided setup script for all environments:

```bash
# Set environment variables
export POSTGRES_USER=postgres
export POSTGRES_DB=omninode_bridge
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432

# Run setup script
bash deployment/scripts/setup_postgres_extensions.sh
```

**When to use**:
- ✅ Automated deployment pipelines
- ✅ Infrastructure-as-Code setups
- ✅ Reproducible environment creation
- ✅ First-time environment setup

#### Option 2: Manual Extension Creation (Production)

This is the **recommended approach** for production deployments where the application database user does not have superuser privileges.

```bash
# Connect as superuser
psql -U postgres -d omninode_bridge

# Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

# Grant usage to application user
GRANT USAGE ON SCHEMA public TO omninode_app;

# Verify extensions
SELECT extname, extversion FROM pg_extension
WHERE extname IN ('uuid-ossp', 'pg_stat_statements');
```

Expected output:
```
      extname       | extversion
--------------------+------------
 uuid-ossp          | 1.1
 pg_stat_statements | 1.9
```

**When to use**:
- ✅ Production environments
- ✅ Managed database services (RDS, Cloud SQL, etc.)
- ✅ Environments with strict security policies
- ✅ When DBA team manages database infrastructure

#### Option 3: Run Migrations as Superuser (Development Only)

This approach is suitable for local development environments where you have full control.

```bash
# Run migrations with superuser credentials
POSTGRES_USER=postgres POSTGRES_PASSWORD=<password> bash migrations/run_migrations.sh
```

**When to use**:
- ✅ Local development environments
- ✅ Docker-based development
- ✅ Testing environments
- ✅ POC/demo environments

**⚠️ Warning**: Never use this approach in production. Always use a dedicated application user with minimal privileges.

#### Option 4: Skip Optional Extensions (Minimal Setup)

If `pg_stat_statements` is not needed for your use case:

1. **Comment out in migration files**:
```sql
-- CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
```

2. **Disable performance monitoring features** in application configuration:
```python
DATABASE_CONFIG = {
    "enable_pg_stat_statements": False,
    "enable_performance_monitoring": False,
}
```

**When to use**:
- Minimal development setups
- CI/CD test environments
- Environments without performance monitoring requirements

**⚠️ Note**: `uuid-ossp` is **required** and cannot be skipped.

---

## Migration Inventory

### Current Migrations (v1.0.0)

| Migration | Description | Dependencies | Critical For | Typical Duration |
|-----------|-------------|--------------|--------------|------------------|
| 001 | workflow_executions table | None | Orchestrator workflows | <1s |
| 002 | workflow_steps table | 001 (FK) | Workflow step tracking | <1s |
| 003 | fsm_transitions table | None | State transition history | <1s |
| 004 | bridge_states table | None | Reducer aggregation state | <1s |
| **005** | **node_registrations table** | **None** | **Dual registration system** | **<1s** |
| 006 | metadata_stamps table | 001 (FK, nullable) | Stamping audit trail | <1s |
| 007 | workflow indexes | 001, 002 | Query performance | <2s |
| 008 | composite indexes | Various | Advanced queries | <2s |
| 009 | workflow enhancements | 001 | Extended workflow features | <1s |
| 010 | bridge state enhancements | 004 | Enhanced reducer state | <1s |
| 011 | canonical state + dedup | Various | Event processing | <3s |

**Total Migrations**: 11 (plus rollbacks)
**Total Migration Time**: ~10-15 seconds for clean database

### Critical Migration: 005 - Node Registrations

**Status**: ⚠️ **REQUIRED for dual registration system**

**What It Creates**:
- `node_registrations` table - Node registry with capabilities and health tracking
- Performance indexes (health_status, node_type, last_heartbeat)
- Composite index (node_type + health_status)

**Required Before**:
- Starting NodeBridgeRegistry service
- Using service discovery features
- Health monitoring and heartbeat tracking

**Table Schema**:
```sql
CREATE TABLE node_registrations (
    node_id VARCHAR(255) PRIMARY KEY,
    node_type VARCHAR(100) NOT NULL,
    node_version VARCHAR(50) NOT NULL,
    capabilities JSONB DEFAULT '{}',
    endpoints JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    health_status VARCHAR(50) NOT NULL DEFAULT 'UNKNOWN',
    last_heartbeat TIMESTAMP WITH TIME ZONE,
    registered_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

**Expected Indexes**:
- `node_registrations_pkey` (PRIMARY KEY on node_id)
- `idx_node_registrations_health` (health_status)
- `idx_node_registrations_type` (node_type)
- `idx_node_registrations_last_heartbeat` (last_heartbeat)
- `idx_node_registrations_type_health` (node_type, health_status)

---

## Deployment Procedures

### Development Environment

#### Quick Start

```bash
# Navigate to project root
cd /Volumes/PRO-G40/Code/omninode_bridge

# Apply all migrations in order
for migration in migrations/00*.sql migrations/01*.sql; do
    if [[ ! $migration =~ rollback ]]; then
        echo "Applying: $migration"
        psql -h localhost -U postgres -d omninode_bridge -f "$migration"
    fi
done

# Verify tables created
psql -h localhost -U postgres -d omninode_bridge -c "\dt"
```

#### Using Docker Container

```bash
# If PostgreSQL is running in Docker
for migration in migrations/00*.sql migrations/01*.sql; do
    if [[ ! $migration =~ rollback ]]; then
        echo "Applying: $migration"
        docker exec -i postgres psql -U postgres -d omninode_bridge < "$migration"
    fi
done
```

#### Individual Migration Application

```bash
# Apply specific migration (e.g., migration 005)
psql -h localhost -U postgres -d omninode_bridge -f migrations/005_create_node_registrations.sql

# Verify table created
psql -h localhost -U postgres -d omninode_bridge -c "\d node_registrations"
```

### Staging Environment

#### Pre-Deployment

```bash
# 1. Create backup
pg_dump -h staging-db-host -U postgres omninode_bridge > \
    backup_staging_$(date +%Y%m%d_%H%M%S).sql

# 2. Verify backup
ls -lh backup_staging_*.sql

# 3. Test connection
psql -h staging-db-host -U postgres -d omninode_bridge -c "SELECT version();"
```

#### Migration Execution

```bash
# Run migrations with logging
for migration in migrations/00*.sql migrations/01*.sql; do
    if [[ ! $migration =~ rollback ]]; then
        echo "Applying: $migration"
        psql -h staging-db-host -U postgres -d omninode_bridge -f "$migration" \
            2>&1 | tee -a migration_staging_$(date +%Y%m%d_%H%M%S).log
    fi
done
```

#### Post-Deployment Verification

```bash
# Verify all tables exist
psql -h staging-db-host -U postgres -d omninode_bridge << 'EOF'
SELECT tablename
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY tablename;
EOF

# Expected tables:
# - bridge_states
# - fsm_transitions
# - metadata_stamps
# - node_registrations
# - workflow_executions
# - workflow_steps
```

### Production Environment

#### ⚠️ Critical Production Deployment Steps

**MANDATORY CHECKLIST**:

1. **Backup Database** (CRITICAL)
   ```bash
   # Create full backup with compression
   pg_dump -h production-db-host -U postgres -Fc omninode_bridge > \
       backup_production_$(date +%Y%m%d_%H%M%S).dump

   # Verify backup size (should be >0 bytes)
   ls -lh backup_production_*.dump

   # Store backup in secure location
   aws s3 cp backup_production_*.dump s3://backups/omninode-bridge/
   ```

2. **Schedule Maintenance Window**
   - Notify stakeholders of migration window
   - Estimate downtime (typically <5 minutes for schema-only migrations)
   - Prepare rollback plan

3. **Dry-Run on Staging**
   - Apply migrations to staging environment first
   - Run full integration test suite
   - Verify application functionality

4. **Production Migration with Transaction**

   ```bash
   # Execute migrations within transaction for safety
   psql -h production-db-host -U postgres -d omninode_bridge << 'EOF'
   BEGIN;

   -- Apply migrations
   \i migrations/001_create_workflow_executions.sql
   \i migrations/002_create_workflow_steps.sql
   \i migrations/003_create_fsm_transitions.sql
   \i migrations/004_create_bridge_states.sql
   \i migrations/005_create_node_registrations.sql
   \i migrations/006_create_metadata_stamps.sql
   \i migrations/007_add_missing_workflow_indexes.sql
   \i migrations/008_add_composite_indexes.sql
   \i migrations/009_enhance_workflow_executions.sql
   \i migrations/010_enhance_bridge_states.sql
   \i migrations/011_canonical_workflow_state.sql
   \i migrations/011_create_action_dedup.sql
   \i migrations/011_projection_and_watermarks.sql

   -- Verify critical tables
   SELECT COUNT(*) FROM pg_tables WHERE schemaname = 'public';

   -- Check for errors in output above
   -- If any errors occurred: ROLLBACK;
   -- If all successful: COMMIT;

   COMMIT;
   EOF
   ```

5. **Post-Migration Verification**

   ```bash
   # Verify table creation
   psql -h production-db-host -U postgres -d omninode_bridge -c "
   SELECT tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
   FROM pg_tables
   WHERE schemaname = 'public'
   ORDER BY tablename;
   "

   # Verify indexes
   psql -h production-db-host -U postgres -d omninode_bridge -c "
   SELECT schemaname, tablename, indexname
   FROM pg_indexes
   WHERE schemaname = 'public'
   ORDER BY tablename, indexname;
   "

   # Check foreign key constraints
   psql -h production-db-host -U postgres -d omninode_bridge -c "
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

6. **Application Deployment**
   - Deploy application code that depends on new schema
   - Monitor error logs for database-related issues
   - Verify health endpoints

---

## Migration Management

### Running Migrations

**Full schema setup (recommended)**:
```bash
# As superuser or after pre-creating extensions
psql -U omninode_app -d omninode_bridge -f migrations/schema.sql
```

**Individual migrations**:
```bash
# Run migrations in order
for migration in migrations/00*.sql; do
    echo "Running $migration..."
    psql -U omninode_app -d omninode_bridge -f "$migration"
done
```

**Using the migration script**:
```bash
# Set database connection
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=omninode_bridge
export POSTGRES_USER=omninode_app
export POSTGRES_PASSWORD=<password>

# Run migrations
bash migrations/run_migrations.sh
```

---

## Verification & Testing

### Quick Verification Script

```bash
#!/bin/bash
# verify-migrations.sh

DB_HOST="${POSTGRES_HOST:-localhost}"
DB_USER="${POSTGRES_USER:-postgres}"
DB_NAME="${POSTGRES_DATABASE:-omninode_bridge}"

echo "Verifying migrations on $DB_HOST..."

# Check table count
TABLE_COUNT=$(psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -t -c "
SELECT COUNT(*) FROM pg_tables WHERE schemaname = 'public';
")

echo "Tables found: $TABLE_COUNT"

if [ "$TABLE_COUNT" -lt 6 ]; then
    echo "❌ ERROR: Expected at least 6 tables, found $TABLE_COUNT"
    exit 1
fi

# Check critical tables
CRITICAL_TABLES=(
    "workflow_executions"
    "workflow_steps"
    "fsm_transitions"
    "bridge_states"
    "node_registrations"
    "metadata_stamps"
)

for table in "${CRITICAL_TABLES[@]}"; do
    EXISTS=$(psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -t -c "
    SELECT EXISTS (
        SELECT FROM pg_tables
        WHERE schemaname = 'public' AND tablename = '$table'
    );
    ")

    if [[ "$EXISTS" == *"t"* ]]; then
        echo "✅ $table exists"
    else
        echo "❌ ERROR: $table does not exist"
        exit 1
    fi
done

echo "✅ All migrations verified successfully"
```

### Manual Verification

```sql
-- Connect to database
psql -h <host> -U postgres -d omninode_bridge

-- List all tables
\dt

-- Describe critical table (node_registrations)
\d+ node_registrations

-- Check row counts (should be 0 for new deployment)
SELECT
    schemaname,
    tablename,
    n_live_tup AS row_count
FROM pg_stat_user_tables
ORDER BY tablename;

-- Verify indexes exist
SELECT
    schemaname,
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE schemaname = 'public' AND tablename = 'node_registrations'
ORDER BY indexname;
```

---

## Rollback Procedures

### Development Rollback

```bash
# Rollback all migrations in reverse order
for migration in $(ls -r migrations/*_rollback_*.sql); do
    echo "Rolling back: $migration"
    psql -h localhost -U postgres -d omninode_bridge -f "$migration"
done
```

### Production Rollback

**⚠️ ONLY USE IF MIGRATION FAILED**

```bash
# Rollback specific migration (e.g., migration 005)
psql -h production-db-host -U postgres -d omninode_bridge << 'EOF'
BEGIN;

\i migrations/005_rollback_node_registrations.sql

-- Verify table dropped
SELECT tablename FROM pg_tables WHERE tablename = 'node_registrations';

COMMIT;
EOF
```

### Full Rollback (Nuclear Option)

```bash
# DANGER: This drops all tables!
# Only use in development or emergency recovery

psql -h <host> -U postgres -d omninode_bridge << 'EOF'
BEGIN;

-- Rollback in reverse order (respect foreign keys)
\i migrations/011_rollback_projection_and_watermarks.sql
\i migrations/011_rollback_action_dedup.sql
\i migrations/011_rollback_canonical_workflow_state.sql
\i migrations/010_rollback_bridge_states.sql
\i migrations/009_rollback_workflow_executions.sql
\i migrations/008_rollback_composite_indexes.sql
\i migrations/007_rollback_workflow_indexes.sql
\i migrations/006_rollback_metadata_stamps.sql
\i migrations/005_rollback_node_registrations.sql
\i migrations/004_rollback_bridge_states.sql
\i migrations/003_rollback_fsm_transitions.sql
\i migrations/002_rollback_workflow_steps.sql
\i migrations/001_rollback_workflow_executions.sql

COMMIT;
EOF
```

---

## Troubleshooting

### Error: "permission denied to create extension"

**Cause**: Your database user lacks superuser privileges.

**Solutions**:

1. **Pre-create extensions as superuser** (Recommended):
   ```sql
   -- Connect as superuser
   psql -U postgres -d omninode_bridge

   -- Create extensions
   CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
   CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
   ```

2. **Grant superuser temporarily** (Development only):
   ```sql
   -- As superuser
   ALTER USER omninode_app WITH SUPERUSER;

   -- Run migrations
   -- (execute migration scripts)

   -- Revoke superuser
   ALTER USER omninode_app WITH NOSUPERUSER;
   ```

3. **Request DBA to create extensions** (Production):
   - Submit change request to DBA team
   - Provide extension names: `uuid-ossp`, `pg_stat_statements`
   - Include purpose and security justification

### Error: "extension does not exist"

**Cause**: PostgreSQL extension packages not installed on the server.

**Solutions**:

1. **Ubuntu/Debian**:
   ```bash
   sudo apt-get update
   sudo apt-get install postgresql-contrib
   ```

2. **RHEL/CentOS**:
   ```bash
   sudo yum install postgresql-contrib
   ```

3. **Docker**:
   ```dockerfile
   # Already included in official postgres image
   FROM postgres:15
   # Extensions are pre-installed
   ```

4. **Managed Database Services**:
   - **AWS RDS**: Extensions pre-installed, just CREATE EXTENSION
   - **Google Cloud SQL**: Extensions pre-installed, just CREATE EXTENSION
   - **Azure Database**: Extensions pre-installed, just CREATE EXTENSION

### Error: "function uuid_generate_v4() does not exist"

**Cause**: `uuid-ossp` extension not created or not loaded.

**Solution**:
```sql
-- Create extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Verify it works
SELECT uuid_generate_v4();
```

### Error: "relation already exists"

**Cause**: Attempting to run migrations on a database that already has tables.

**Solutions**:

1. **Skip to next migration**:
   ```bash
   # Check which migrations have already run
   psql -U omninode_app -d omninode_bridge -c "\dt"

   # Run only missing migrations
   psql -U omninode_app -d omninode_bridge -f migrations/007_*.sql
   ```

2. **Start fresh** (Development only):
   ```bash
   # Drop database
   dropdb -U postgres omninode_bridge

   # Recreate database
   createdb -U postgres omninode_bridge

   # Run all migrations
   bash migrations/run_migrations.sh
   ```

### Migration Performance Issues

**Symptoms**: Migrations taking too long, database locks, timeouts.

**Solutions**:

1. **Run during maintenance window**: Schedule migrations during low-traffic periods
2. **Add indexes offline**: Use `CREATE INDEX CONCURRENTLY` for large tables
3. **Batch large data migrations**: Split into smaller chunks
4. **Increase statement timeout**:
   ```sql
   SET statement_timeout = '30min';
   ```

### Migration 005 Not Applied

**Detection**:
```bash
docker exec postgres psql -U postgres -d omninode_bridge -c "\d node_registrations"
# Output: "Did not find any relation named 'node_registrations'."
```

**Solution**:
```bash
# Apply migration 005
docker exec -i postgres psql -U postgres -d omninode_bridge < \
    migrations/005_create_node_registrations.sql

# Verify
docker exec postgres psql -U postgres -d omninode_bridge -c "\d node_registrations"
```

### Foreign Key Constraint Errors

**Cause**: Migrations applied out of order

**Solution**:
```bash
# Check migration dependencies
# Correct order:
# 001 - workflow_executions (no dependencies)
# 002 - workflow_steps (depends on 001)
# 003 - fsm_transitions (no dependencies)
# 004 - bridge_states (no dependencies)
# 005 - node_registrations (no dependencies)
# 006 - metadata_stamps (depends on 001, nullable)

# Rollback and reapply in correct order
```

### Rollback Failed

**Symptoms**: Rollback script fails, database in inconsistent state.

**Solutions**:

1. **Restore from backup**:
   ```bash
   # Stop application
   systemctl stop omninode-bridge

   # Restore database
   psql -U postgres -d omninode_bridge < backup_20251025.sql

   # Restart application
   systemctl start omninode-bridge
   ```

2. **Manual cleanup**:
   ```sql
   -- Drop problematic objects
   DROP TABLE IF EXISTS <table_name> CASCADE;

   -- Recreate from migration
   -- (run specific migration file)
   ```

---

## CI/CD Integration

### GitHub Actions Workflow

```yaml
name: Deploy Database Migrations

on:
  push:
    branches: [main]
    paths:
      - 'migrations/*.sql'

jobs:
  deploy-migrations:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Install PostgreSQL Client
        run: sudo apt-get install -y postgresql-client

      - name: Create Backup
        env:
          PGPASSWORD: ${{ secrets.POSTGRES_PASSWORD }}
        run: |
          pg_dump -h ${{ secrets.POSTGRES_HOST }} \
                  -U postgres \
                  -d omninode_bridge \
                  > backup_$(date +%Y%m%d_%H%M%S).sql

      - name: Apply Migrations
        env:
          PGPASSWORD: ${{ secrets.POSTGRES_PASSWORD }}
        run: |
          for migration in migrations/00*.sql migrations/01*.sql; do
              if [[ ! $migration =~ rollback ]]; then
                  echo "Applying: $migration"
                  psql -h ${{ secrets.POSTGRES_HOST }} \
                       -U postgres \
                       -d omninode_bridge \
                       -f "$migration"
              fi
          done

      - name: Verify Migrations
        env:
          PGPASSWORD: ${{ secrets.POSTGRES_PASSWORD }}
        run: |
          bash verify-migrations.sh
```

### Docker Compose Integration

```yaml
# deployment/docker-compose.yml

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: omninode_bridge
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./migrations:/docker-entrypoint-initdb.d:ro  # Auto-apply on first run
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5

  migration-runner:
    image: postgres:15
    depends_on:
      postgres:
        condition: service_healthy
    environment:
      POSTGRES_HOST: postgres
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - ./migrations:/migrations:ro
    command: >
      bash -c "
        for migration in /migrations/00*.sql /migrations/01*.sql; do
          if [[ ! $$migration =~ rollback ]]; then
            echo 'Applying: '$$migration
            psql -h postgres -U postgres -d omninode_bridge -f $$migration
          fi
        done
      "

volumes:
  postgres_data:
```

---

## Best Practices

### Pre-Migration Checklist

- ✅ **Always backup production database before migrations**
- ✅ **Test migrations on staging environment first**
- ✅ **Verify migrations are idempotent (can run multiple times)**
- ✅ **Review migration dependencies and order**
- ✅ **Schedule maintenance window for production**
- ✅ **Verify extensions are created by DBA or superuser**

### Deployment Workflow

1. **Pre-deployment**:
   - Backup production database
   - Verify backup integrity
   - Test on staging first

2. **Extension setup** (if not already done):
   ```bash
   # Run as superuser
   bash deployment/scripts/setup_postgres_extensions.sh
   ```

3. **Run migrations**:
   ```bash
   # Run as application user
   bash migrations/run_migrations.sh
   ```

4. **Verify migration success**:
   ```bash
   # Check tables exist
   psql -U omninode_app -d omninode_bridge -c "\dt"

   # Check indexes
   psql -U omninode_app -d omninode_bridge -c "\di"

   # Verify extensions
   psql -U omninode_app -d omninode_bridge -c "SELECT extname, extversion FROM pg_extension WHERE extname IN ('uuid-ossp', 'pg_stat_statements');"
   ```

5. **Application smoke test**:
   ```bash
   # Test basic database connectivity
   python -c "from omninode_bridge.infrastructure.postgres_client import PostgresConnectionManager; import asyncio; asyncio.run(PostgresConnectionManager(config).initialize())"
   ```

### Migration Safety Guidelines

1. **Always use transactions**: All migration scripts should be wrapped in transactions for rollback capability
2. **Test on staging first**: Never run untested migrations on production
3. **Document changes**: Include comments in migration files explaining the purpose
4. **Monitor performance**: Watch for slow queries or table locks during migration
5. **Plan maintenance windows**: Run migrations during low-traffic periods
6. **Keep rollback scripts updated**: Ensure rollback scripts match forward migrations

### Never Do This

- ❌ **Don't modify existing migration files after deployment**
- ❌ **Don't skip migration numbers**
- ❌ **Don't run migrations on production without testing**
- ❌ **Don't delete old migration files**
- ❌ **Don't run migrations without backups**

---

## Performance Monitoring

### Index Usage Monitoring

```sql
-- Enable pg_stat_statements
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Monitor index usage (run after application is deployed)
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan AS index_scans,
    idx_tup_read AS tuples_read,
    idx_tup_fetch AS tuples_fetched
FROM pg_stat_user_indexes
WHERE schemaname = 'public'
ORDER BY idx_scan DESC;
```

### Table Growth Monitoring

```sql
SELECT
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS total_size,
    pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) AS table_size,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename) - pg_relation_size(schemaname||'.'||tablename)) AS index_size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

---

## Related Documentation

- **[Database Guide](./DATABASE_GUIDE.md)** - Complete database schema and operations guide
- **[Pre-Deployment Checklist](../deployment/PRE_DEPLOYMENT_CHECKLIST.md)** - Production deployment checklist
- **[Setup Guide](../SETUP.md)** - Development environment setup
- **[Operations Guide](../operations/OPERATIONS_GUIDE.md)** - Production operations and monitoring
- **[Technical Migration Details](../../migrations/README.md)** - Developer-focused migration documentation

---

## Version History

- **v2.0.0** (2025-10-29): Consolidated deployment and technical guides
  - Merged docs/database and docs/deployment versions
  - Unified audience (developers + DevOps/SREs)
  - Added comprehensive troubleshooting
  - Enhanced CI/CD integration examples
  - Improved navigation and structure

- **v1.0.0** (2025-10-25): Initial separate guides
  - Technical guide (database/)
  - Operational guide (deployment/)

---

**Document Version**: 2.0 (Consolidated)
**Maintained By**: omninode_bridge team
**Last Review**: October 29, 2025
**Next Review**: November 29, 2025

**Note**: This document consolidates the previous separate technical and operational migration guides. For migration-specific implementation details, see `/migrations/README.md`.
