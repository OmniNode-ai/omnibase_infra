# Operations Runbook

**Version**: 1.0
**Last Updated**: 2025-11-07
**Status**: ✅ Complete

---

## Table of Contents

- [Overview](#overview)
- [Partition Lifecycle Management](#partition-lifecycle-management)
  - [Understanding Partitions](#understanding-partitions)
  - [Partition Strategy](#partition-strategy)
  - [Automated Management](#automated-management)
  - [Manual Operations](#manual-operations)
  - [Monitoring and Alerts](#monitoring-and-alerts)
  - [Troubleshooting](#troubleshooting)
- [Database Operations](#database-operations)
- [Service Management](#service-management)
- [Incident Response](#incident-response)
- [Backup and Recovery](#backup-and-recovery)

---

## Overview

This runbook provides operational procedures for managing OmniNode Bridge infrastructure, with emphasis on partition lifecycle management for high-volume metrics tables.

**Target Audience**: DevOps engineers, SREs, database administrators

**Prerequisites**:
- Access to production PostgreSQL database
- Permissions to create/drop tables
- Familiarity with PostgreSQL partitioning
- Understanding of cron job scheduling

---

## Partition Lifecycle Management

### Understanding Partitions

OmniNode Bridge uses **table partitioning** to manage high-volume agent metrics data efficiently. Partitioning provides:

**Benefits**:
- **Query Performance**: Queries scan only relevant partitions (partition pruning)
- **Maintenance Efficiency**: Drop entire partitions instead of DELETE operations
- **Storage Management**: Easier to move old partitions to cheaper storage
- **Backup Optimization**: Backup recent partitions more frequently

**Partitioned Tables** (5 tables total):

| Table | Purpose | Partition Key | Expected Volume |
|-------|---------|---------------|-----------------|
| `agent_routing_metrics` | Agent routing decisions and timing | `created_at` | High (1000+/day) |
| `agent_state_metrics` | Agent state transitions | `created_at` | Medium (500+/day) |
| `agent_coordination_metrics` | Multi-agent coordination | `created_at` | Medium (500+/day) |
| `agent_workflow_metrics` | Workflow execution metrics | `created_at` | High (2000+/day) |
| `agent_quorum_metrics` | AI quorum validation metrics | `created_at` | Low (100+/day) |

**Partition Scheme**:
- **Type**: RANGE partitioning
- **Interval**: Monthly (1st day of each month)
- **Naming**: `{table_name}_YYYY_MM` (e.g., `agent_routing_metrics_2025_11`)
- **Retention**: 90 days (configurable)

**Migration Reference**: See [Migration 014](../../migrations/014_create_agent_metrics_tables.sql) for schema definition.

---

### Partition Strategy

#### Initial Setup

During migration 014, three partitions are created automatically:
```sql
-- Example for agent_routing_metrics
CREATE TABLE agent_routing_metrics_2025_11 PARTITION OF agent_routing_metrics
    FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');

CREATE TABLE agent_routing_metrics_2025_12 PARTITION OF agent_routing_metrics
    FOR VALUES FROM ('2025-12-01') TO ('2026-01-01');

CREATE TABLE agent_routing_metrics_2026_01 PARTITION OF agent_routing_metrics
    FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
```

#### Ongoing Management

**Monthly Operations**:
1. **Create Future Partitions** (1st of each month)
   - Create partitions 3 months ahead
   - Ensures buffer for uninterrupted data ingestion
   - Run time: ~5 seconds per table

2. **Drop Expired Partitions** (1st of each month)
   - Drop partitions older than 90 days
   - Frees disk space
   - Run time: ~2 seconds per partition

**Recommended Schedule**:
```bash
# Cron job (runs monthly at 2 AM on 1st day)
0 2 1 * * /path/to/manage_metric_partitions.sh create && /path/to/manage_metric_partitions.sh drop
```

**Why This Works**:
- Runs during low-traffic hours (2 AM)
- Creates partitions well before needed (3 months buffer)
- Drops old data after retention period (90 days)
- Single atomic operation (create && drop)

---

### Automated Management

#### Partition Management Script

**Location**: `deployment/scripts/manage_metric_partitions.sh`

**Features**:
- ✅ Create future partitions (default: 3 months ahead)
- ✅ Drop expired partitions (default: >90 days old)
- ✅ Verify partition health and coverage
- ✅ List partitions with statistics
- ✅ Dry-run mode for testing
- ✅ Verbose logging for debugging

#### Usage Examples

**1. Create Future Partitions**

```bash
# Create partitions for next 3 months (default)
bash deployment/scripts/manage_metric_partitions.sh create

# Create partitions for next 6 months
bash deployment/scripts/manage_metric_partitions.sh create --months=6

# Dry-run (test without changes)
bash deployment/scripts/manage_metric_partitions.sh create --dry-run --verbose
```

**Example Output**:
```
================================================================================
[INFO] Creating partitions for next 3 months...
================================================================================
[INFO] Processing table: agent_routing_metrics
[INFO]   Creating partition: agent_routing_metrics_2026_02
[SUCCESS]   ✓ Partition agent_routing_metrics_2026_02 created successfully
[INFO]   Creating partition: agent_routing_metrics_2026_03
[SUCCESS]   ✓ Partition agent_routing_metrics_2026_03 created successfully
[INFO]   Creating partition: agent_routing_metrics_2026_04
[SUCCESS]   ✓ Partition agent_routing_metrics_2026_04 created successfully

================================================================================
[INFO] Summary:
[SUCCESS]   Created: 15 partitions
[INFO]   Skipped (already exist): 0 partitions
================================================================================
```

**2. Drop Expired Partitions**

```bash
# Drop partitions older than 90 days (default)
bash deployment/scripts/manage_metric_partitions.sh drop

# Drop partitions older than 60 days
bash deployment/scripts/manage_metric_partitions.sh drop --retention=60

# Dry-run to see what would be dropped
bash deployment/scripts/manage_metric_partitions.sh drop --dry-run
```

**Example Output**:
```
================================================================================
[INFO] Dropping partitions older than 90 days...
================================================================================
[INFO] Cutoff date: 2025-08-01 (partitions before this will be dropped)

[INFO] Processing table: agent_routing_metrics
[WARNING]   Dropping expired partition: agent_routing_metrics_2025_07
[SUCCESS]   ✓ Partition agent_routing_metrics_2025_07 dropped successfully
[WARNING]   Dropping expired partition: agent_routing_metrics_2025_06
[SUCCESS]   ✓ Partition agent_routing_metrics_2025_06 dropped successfully

================================================================================
[INFO] Summary:
[WARNING]   Dropped: 10 partitions
[INFO]   Kept: 15 partitions
================================================================================
```

**3. Verify Partition Health**

```bash
# Verify all partitions and check for gaps
bash deployment/scripts/manage_metric_partitions.sh verify
```

**Example Output**:
```
================================================================================
[INFO] Verifying partition health...
================================================================================
[INFO] Table: agent_routing_metrics
[INFO]   Total partitions: 5
[SUCCESS]   ✓ No gaps detected in partition coverage

[INFO] Table: agent_state_metrics
[INFO]   Total partitions: 5
[WARNING]   ⚠ Gap detected between 2025-11-01 and 2026-01-01

================================================================================
[INFO] Verification Summary:
[INFO]   Total partitions: 25
[SUCCESS]   Healthy: 24
[WARNING]   Issues found: 1
================================================================================
```

**4. List All Partitions**

```bash
# Show all partitions with row counts and sizes
bash deployment/scripts/manage_metric_partitions.sh list
```

**Example Output**:
```
================================================================================
[INFO] Listing all partitions with statistics...
================================================================================
[INFO] Table: agent_routing_metrics

      partition_name       | partition_size | row_count | index_size
---------------------------+----------------+-----------+------------
 agent_routing_metrics_2025_11 | 2048 kB        |      5423 | 1024 kB
 agent_routing_metrics_2025_12 | 3584 kB        |      8932 | 1536 kB
 agent_routing_metrics_2026_01 | 1024 kB        |      2105 | 512 kB
```

**5. Show Statistics**

```bash
# Display overall partition statistics
bash deployment/scripts/manage_metric_partitions.sh stats
```

**Example Output**:
```
================================================================================
[INFO] Partition Statistics and Health Metrics
================================================================================
[INFO] Overall Statistics:
[INFO]   Total partitions: 25
[INFO]   Total size: 45 MB
[INFO]   Total rows: 45,682

[INFO] Per-Table Statistics:
  agent_routing_metrics:
    Partitions: 5
    Size: 12 MB
    Rows: 15,432

[INFO] Coverage Analysis:
[INFO]   Oldest partition: agent_routing_metrics_2025_11
[INFO]   Newest partition: agent_workflow_metrics_2026_01
================================================================================
```

#### Script Options

| Option | Description | Default |
|--------|-------------|---------|
| `--months=N` | Number of months ahead to create | 3 |
| `--retention=N` | Retention period in days | 90 |
| `--dry-run` | Show actions without executing | false |
| `--verbose` | Enable debug logging | false |

#### Environment Variables

```bash
# Database connection (set in .env or export manually)
export POSTGRES_HOST=192.168.86.200
export POSTGRES_PORT=5436
export POSTGRES_DB=omninode_bridge
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=your_password

# Run script
bash deployment/scripts/manage_metric_partitions.sh create
```

**Security Note**: For production, use `.pgpass` file or secrets manager instead of exporting password.

---

### Manual Operations

#### Creating Partitions Manually

**When to use**: Emergency partition creation outside normal schedule

```sql
-- Create partition for specific month
CREATE TABLE IF NOT EXISTS agent_routing_metrics_2026_05
PARTITION OF agent_routing_metrics
FOR VALUES FROM ('2026-05-01') TO ('2026-06-01');

-- Verify creation
SELECT tablename, pg_size_pretty(pg_total_relation_size(tablename::regclass))
FROM pg_tables
WHERE tablename = 'agent_routing_metrics_2026_05';
```

**Best Practice**: Use the script instead of manual SQL to ensure consistency across all 5 tables.

#### Dropping Partitions Manually

**When to use**: Emergency space recovery or data deletion

```sql
-- Check partition size and row count before dropping
SELECT
    c.relname AS partition_name,
    pg_size_pretty(pg_total_relation_size(c.oid)) AS size,
    s.n_live_tup AS row_count
FROM pg_class c
LEFT JOIN pg_stat_user_tables s ON s.relname = c.relname
WHERE c.relname = 'agent_routing_metrics_2025_07';

-- Drop partition (data is lost permanently!)
DROP TABLE IF EXISTS agent_routing_metrics_2025_07;
```

**⚠️ WARNING**: Dropping partitions is **irreversible**. Always:
1. Backup data first (pg_dump)
2. Verify correct partition name
3. Use `--dry-run` with script to preview
4. Test on staging environment first

#### Checking Partition Coverage

```sql
-- List all partitions for a table
SELECT
    c.relname AS partition_name,
    pg_get_expr(c.relpartbound, c.oid) AS partition_range,
    pg_size_pretty(pg_total_relation_size(c.oid)) AS size,
    COALESCE(s.n_live_tup, 0) AS row_count
FROM pg_class c
LEFT JOIN pg_stat_user_tables s ON s.relname = c.relname
WHERE c.relispartition
    AND c.relname LIKE 'agent_routing_metrics_%'
ORDER BY c.relname;
```

#### Identifying Missing Partitions

```sql
-- Find gaps in partition coverage
WITH partition_dates AS (
    SELECT
        relname,
        substring(relname from '\d{4}_\d{2}$') AS date_part
    FROM pg_class
    WHERE relispartition
        AND relname LIKE 'agent_routing_metrics_%'
)
SELECT
    to_char(date_trunc('month', generate_series), 'YYYY_MM') AS expected_partition,
    CASE
        WHEN pd.relname IS NOT NULL THEN '✓ Exists'
        ELSE '✗ Missing'
    END AS status
FROM generate_series(
    '2025-11-01'::date,
    CURRENT_DATE + interval '3 months',
    interval '1 month'
) AS generate_series
LEFT JOIN partition_dates pd
    ON to_char(generate_series, 'YYYY_MM') = pd.date_part
ORDER BY generate_series;
```

---

### Monitoring and Alerts

#### Key Metrics to Monitor

**1. Partition Count Per Table**

```sql
-- Alert if partition count < 3 (risk of data loss)
SELECT
    parent.relname AS table_name,
    COUNT(*) AS partition_count,
    CASE
        WHEN COUNT(*) < 3 THEN 'CRITICAL: Need more partitions'
        WHEN COUNT(*) < 6 THEN 'WARNING: Low partition buffer'
        ELSE 'OK'
    END AS status
FROM pg_class parent
JOIN pg_inherits ON inhparent = parent.oid
JOIN pg_class child ON inhrelid = child.oid
WHERE parent.relname IN (
    'agent_routing_metrics',
    'agent_state_metrics',
    'agent_coordination_metrics',
    'agent_workflow_metrics',
    'agent_quorum_metrics'
)
GROUP BY parent.relname;
```

**Alert Threshold**: `partition_count < 3` → **CRITICAL**

**2. Partition Size Growth**

```sql
-- Monitor partition size growth rate
SELECT
    c.relname AS partition_name,
    pg_size_pretty(pg_total_relation_size(c.oid)) AS size,
    COALESCE(s.n_live_tup, 0) AS row_count,
    pg_size_pretty(pg_total_relation_size(c.oid)::numeric / NULLIF(s.n_live_tup, 0)) AS avg_row_size
FROM pg_class c
LEFT JOIN pg_stat_user_tables s ON s.relname = c.relname
WHERE c.relispartition
    AND c.relname LIKE 'agent_%_metrics_%'
ORDER BY pg_total_relation_size(c.oid) DESC
LIMIT 10;
```

**Alert Threshold**: `partition_size > 5 GB` → **WARNING** (consider reducing retention)

**3. Disk Space Available**

```sql
-- Check database size and available space
SELECT
    pg_database.datname,
    pg_size_pretty(pg_database_size(pg_database.datname)) AS size
FROM pg_database
WHERE datname = 'omninode_bridge';
```

**Alert Threshold**: `disk_usage > 85%` → **WARNING**

**4. Partition Gaps**

**Script-based check**:
```bash
# Run daily to detect gaps
bash deployment/scripts/manage_metric_partitions.sh verify | grep "WARNING"

# Alert if output contains "Gap detected"
```

**Alert Threshold**: Any gap detected → **WARNING**

#### Recommended Alerts

**Prometheus Alert Rules** (if using Prometheus):

```yaml
# monitoring/prometheus/rules/partition_alerts.yml
groups:
- name: partition_alerts
  rules:
  - alert: LowPartitionCount
    expr: partition_count < 3
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "Low partition count for {{ $labels.table_name }}"
      description: "Only {{ $value }} partitions remaining. Risk of data loss."

  - alert: PartitionSizeLarge
    expr: partition_size_bytes > 5368709120  # 5 GB
    for: 30m
    labels:
      severity: warning
    annotations:
      summary: "Large partition size for {{ $labels.partition_name }}"
      description: "Partition size is {{ $value | humanize }}. Consider reducing retention."

  - alert: DiskSpaceHigh
    expr: disk_usage_percent > 85
    for: 15m
    labels:
      severity: warning
    annotations:
      summary: "High disk usage on database server"
      description: "Disk usage is {{ $value }}%. Consider dropping old partitions."
```

**Grafana Dashboard**:
- Total partitions per table (gauge)
- Partition size over time (graph)
- Row count per partition (table)
- Disk space available (gauge)
- Partition creation/drop history (graph)

#### Automated Monitoring Script

**Daily Health Check** (add to cron):

```bash
#!/bin/bash
# Check partition health daily and alert if issues found

SCRIPT_PATH="/path/to/manage_metric_partitions.sh"
EMAIL="ops@example.com"

# Run verification
OUTPUT=$(bash "$SCRIPT_PATH" verify 2>&1)

# Check for issues
if echo "$OUTPUT" | grep -q "Issues found: [^0]"; then
    echo "Partition health check failed!"
    echo "$OUTPUT"
    echo "$OUTPUT" | mail -s "OmniNode Bridge: Partition Issues Detected" "$EMAIL"
    exit 1
else
    echo "Partition health check passed"
    exit 0
fi
```

**Cron Setup**:
```bash
# Daily at 6 AM
0 6 * * * /path/to/partition_health_check.sh
```

---

### Troubleshooting

#### Problem 1: Partition Creation Failed

**Symptom**: Script reports "Failed to create partition"

**Possible Causes**:
1. **Database connection issue**
   ```bash
   # Test connection
   psql -h $POSTGRES_HOST -p $POSTGRES_PORT -U $POSTGRES_USER -d $POSTGRES_DB -c "SELECT 1;"
   ```

2. **Insufficient permissions**
   ```sql
   -- Check user permissions
   SELECT has_table_privilege('postgres', 'agent_routing_metrics', 'CREATE');

   -- Grant permissions if needed (as superuser)
   GRANT CREATE ON DATABASE omninode_bridge TO postgres;
   ```

3. **Partition already exists**
   ```sql
   -- Check existing partitions
   SELECT tablename FROM pg_tables
   WHERE tablename LIKE 'agent_routing_metrics_%'
   ORDER BY tablename;
   ```

4. **Invalid date range**
   - Verify partition date doesn't overlap existing partitions
   - Check for gaps in coverage

**Resolution**:
```bash
# Verbose mode for debugging
bash deployment/scripts/manage_metric_partitions.sh create --verbose

# Manual creation with explicit dates
psql -c "CREATE TABLE agent_routing_metrics_2026_05 PARTITION OF agent_routing_metrics FOR VALUES FROM ('2026-05-01') TO ('2026-06-01');"
```

#### Problem 2: Partition Drop Failed

**Symptom**: Script reports "Failed to drop partition"

**Possible Causes**:
1. **Active queries on partition**
   ```sql
   -- Check for active queries
   SELECT pid, query, state
   FROM pg_stat_activity
   WHERE query LIKE '%agent_routing_metrics_%';

   -- Terminate blocking queries (if safe)
   SELECT pg_terminate_backend(pid)
   FROM pg_stat_activity
   WHERE query LIKE '%agent_routing_metrics_2025_07%';
   ```

2. **Foreign key constraints** (unlikely with metrics tables)
   ```sql
   -- Check for foreign keys
   SELECT conname, conrelid::regclass
   FROM pg_constraint
   WHERE confrelid = 'agent_routing_metrics_2025_07'::regclass;
   ```

3. **Insufficient permissions**
   ```sql
   -- Check drop permissions
   SELECT has_table_privilege('postgres', 'agent_routing_metrics_2025_07', 'DROP');
   ```

**Resolution**:
```bash
# Use verbose and dry-run to debug
bash deployment/scripts/manage_metric_partitions.sh drop --dry-run --verbose

# Manual drop with CASCADE (use with caution)
psql -c "DROP TABLE IF EXISTS agent_routing_metrics_2025_07 CASCADE;"
```

#### Problem 3: Partition Gaps Detected

**Symptom**: Verification reports "Gap detected"

**Impact**: Data inserted into missing partition month will be **rejected** by PostgreSQL.

**Example Error**:
```
ERROR: no partition of relation "agent_routing_metrics" found for row
DETAIL: Partition key of the failing row contains (created_at) = (2025-12-15 10:30:00+00)
```

**Resolution**:
```bash
# Create missing partitions
bash deployment/scripts/manage_metric_partitions.sh create --months=12

# Verify fix
bash deployment/scripts/manage_metric_partitions.sh verify
```

**Prevention**:
- Run creation script monthly (automated cron)
- Monitor partition count (alert if < 3)
- Create partitions 3+ months ahead

#### Problem 4: Disk Space Running Out

**Symptom**: Disk usage > 85%, database writes failing

**Immediate Actions**:
1. **Check partition sizes**
   ```bash
   bash deployment/scripts/manage_metric_partitions.sh list
   ```

2. **Drop oldest partitions** (emergency only)
   ```bash
   # Reduce retention to 60 days
   bash deployment/scripts/manage_metric_partitions.sh drop --retention=60

   # Or drop specific old partition
   psql -c "DROP TABLE IF EXISTS agent_routing_metrics_2025_06;"
   ```

3. **Backup and compress old data** (before dropping)
   ```bash
   # Backup specific partition
   pg_dump -h $POSTGRES_HOST -p $POSTGRES_PORT -U $POSTGRES_USER \
       -d $POSTGRES_DB -t agent_routing_metrics_2025_06 \
       -Fc -f agent_routing_metrics_2025_06_backup.pgdump

   # Compress backup
   gzip agent_routing_metrics_2025_06_backup.pgdump
   ```

**Long-term Solutions**:
- Reduce retention period (e.g., 90 days → 60 days)
- Archive old partitions to S3/object storage
- Increase disk capacity
- Implement data aggregation/rollup for old data

#### Problem 5: High Query Latency on Partitioned Tables

**Symptom**: Slow queries on metrics tables

**Diagnosis**:
```sql
-- Check if partition pruning is working
EXPLAIN (ANALYZE, BUFFERS)
SELECT * FROM agent_routing_metrics
WHERE created_at >= '2025-11-01'
  AND created_at < '2025-12-01';

-- Look for "Partitions scanned" in output
-- Should only scan 1 partition, not all
```

**Common Causes**:
1. **Missing WHERE clause on partition key**
   ```sql
   -- BAD: Scans all partitions
   SELECT * FROM agent_routing_metrics WHERE agent_id = 'agent-123';

   -- GOOD: Prunes to specific partition
   SELECT * FROM agent_routing_metrics
   WHERE agent_id = 'agent-123'
     AND created_at >= '2025-11-01'
     AND created_at < '2025-12-01';
   ```

2. **Too many partitions**
   - Each partition adds planning overhead
   - Recommended: < 100 partitions per table
   - Current strategy (monthly, 90-day retention): ~3-4 partitions ✅

3. **Outdated statistics**
   ```sql
   -- Analyze specific partition
   ANALYZE agent_routing_metrics_2025_11;

   -- Analyze all partitions
   ANALYZE agent_routing_metrics;
   ```

**Resolution**:
- Always include `created_at` filter in queries
- Keep partition count reasonable
- Run ANALYZE after creating partitions
- Consider pg_cron for automated ANALYZE

---

### Advanced Topics

#### Alternative: pg_partman Extension

**What is pg_partman?**
- PostgreSQL extension for automated partition management
- Handles creation, maintenance, and retention automatically
- Supports time-based, serial, and custom partitioning schemes

**Pros**:
- ✅ Fully automated (no cron jobs needed)
- ✅ Built-in retention management
- ✅ Automatic partition creation before needed
- ✅ Battle-tested in production environments
- ✅ Supports native partitioning (PostgreSQL 10+)

**Cons**:
- ❌ Requires superuser to install extension
- ❌ Additional dependency to manage
- ❌ Learning curve for configuration
- ❌ Less control over exact timing

**When to Consider**:
- Managing > 10 partitioned tables
- Need for dynamic partition intervals
- Complex retention policies
- Want zero-touch partition management

**Migration Path** (if needed):
```sql
-- 1. Install extension (requires superuser)
CREATE EXTENSION pg_partman;

-- 2. Configure partition management
SELECT partman.create_parent(
    p_parent_table := 'public.agent_routing_metrics',
    p_control := 'created_at',
    p_interval := '1 month',
    p_premake := 3,
    p_start_partition := '2025-11-01'
);

-- 3. Set retention policy
UPDATE partman.part_config
SET retention = '90 days',
    retention_keep_table = false
WHERE parent_table = 'public.agent_routing_metrics';

-- 4. Enable automatic maintenance (add to cron)
SELECT partman.run_maintenance();
```

**Current Recommendation**: Stick with script-based approach unless:
- You need to manage > 10 partitioned tables
- You want zero-touch automation
- You have superuser access to production database

**Resources**:
- [pg_partman GitHub](https://github.com/pgpartman/pg_partman)
- [pg_partman Documentation](https://github.com/pgpartman/pg_partman/blob/master/doc/pg_partman.md)

#### Partition Archival Strategy

**Goal**: Move old partitions to cheaper storage instead of dropping

**Approach**:
1. **Dump partition to file**
   ```bash
   pg_dump -h $POSTGRES_HOST -p $POSTGRES_PORT -U $POSTGRES_USER \
       -d $POSTGRES_DB -t agent_routing_metrics_2025_06 \
       -Fc -f archive/agent_routing_metrics_2025_06.pgdump
   ```

2. **Upload to object storage**
   ```bash
   # AWS S3
   aws s3 cp archive/agent_routing_metrics_2025_06.pgdump \
       s3://omninode-archives/metrics/2025/06/

   # Or use rclone for any provider
   rclone copy archive/ remote:omninode-archives/metrics/
   ```

3. **Detach partition** (optional - keeps data without dropping)
   ```sql
   -- Detach partition from parent (data still exists)
   ALTER TABLE agent_routing_metrics
   DETACH PARTITION agent_routing_metrics_2025_06;

   -- Rename for clarity
   ALTER TABLE agent_routing_metrics_2025_06
   RENAME TO agent_routing_metrics_2025_06_archived;
   ```

4. **Drop partition**
   ```sql
   DROP TABLE IF EXISTS agent_routing_metrics_2025_06;
   ```

**Benefits**:
- Compliance with data retention requirements
- Historical analysis without database overhead
- Cost-effective long-term storage

**Automation**:
```bash
#!/bin/bash
# Archive and drop old partitions

RETENTION_DAYS=90
ARCHIVE_DIR="/backups/metrics/archives"

# Find partitions older than retention
CUTOFF_DATE=$(date -u -d "now - $RETENTION_DAYS days" +%Y-%m-01)

# Archive each partition
psql -tAc "SELECT tablename FROM pg_tables WHERE tablename LIKE 'agent_%_metrics_%' ORDER BY tablename;" | while read partition; do
    # Extract date and compare
    # ... (implement date comparison logic)

    # Dump partition
    pg_dump -t "$partition" -Fc -f "$ARCHIVE_DIR/${partition}.pgdump"

    # Upload to S3
    aws s3 cp "$ARCHIVE_DIR/${partition}.pgdump" "s3://omninode-archives/metrics/"

    # Drop partition
    psql -c "DROP TABLE IF EXISTS $partition;"
done
```

---

## Database Operations

(Additional sections to be added)

- Backup and restore procedures
- Connection pool management
- Index maintenance
- Vacuum and analyze schedules
- Extension management
- Migration procedures

---

## Service Management

(Additional sections to be added)

- Container orchestration
- Health checks
- Rolling updates
- Scaling procedures
- Configuration management

---

## Incident Response

(Additional sections to be added)

- Incident classification
- Escalation procedures
- Runbooks for common incidents
- Post-incident review template

---

## Backup and Recovery

(Additional sections to be added)

- Backup schedules
- Recovery procedures
- DR testing
- RTO/RPO targets

---

## Quick Reference

### Common Commands

```bash
# Partition Management
bash deployment/scripts/manage_metric_partitions.sh create        # Create future partitions
bash deployment/scripts/manage_metric_partitions.sh drop          # Drop old partitions
bash deployment/scripts/manage_metric_partitions.sh verify        # Check health
bash deployment/scripts/manage_metric_partitions.sh list          # List all partitions
bash deployment/scripts/manage_metric_partitions.sh stats         # Show statistics

# Database Operations
psql -h $POSTGRES_HOST -p $POSTGRES_PORT -U $POSTGRES_USER -d $POSTGRES_DB   # Connect
\dt                                                               # List tables
\d+ agent_routing_metrics                                         # Describe table
\di                                                               # List indexes

# Health Checks
curl http://192.168.86.200:8001/health                           # Hook receiver health
docker ps | grep omninode                                        # Container status
```

### Key Contacts

- **Database Team**: dba@omninode.ai
- **DevOps Team**: devops@omninode.ai
- **On-Call**: [PagerDuty rotation]

### Related Documentation

- **[Database Guide](../database/DATABASE_GUIDE.md)** - Database architecture and schema
- **[Database Migrations](../../migrations/README.md)** - Migration procedures
- **[Monitoring Guide](./MONITORING.md)** - Monitoring and alerting
- **[Deployment Guide](./DEPLOYMENT.md)** - Deployment procedures
- **[Pre-Deployment Checklist](./PRE_DEPLOYMENT_CHECKLIST.md)** - Production deployment checklist

---

**Document Version**: 1.0
**Last Updated**: 2025-11-07
**Next Review**: 2025-12-07
**Maintained By**: DevOps Team

For questions or updates to this runbook, please file an issue or submit a pull request.
