# Migration Versioning Convention

This document describes the versioning scheme for database migrations in the ONEX infrastructure.

## File Naming Pattern

Migrations follow the format: `NNN[a-z]_description[_modifier].sql`

| Component | Format | Example | Description |
|-----------|--------|---------|-------------|
| **Sequence** | `NNN` | `001`, `002`, `003` | Three-digit main migration number |
| **Sub-sequence** | `[a-z]` | `003a`, `003b` | Optional letter suffix for related migrations |
| **Description** | `_snake_case` | `_capability_fields` | Descriptive name of the change |
| **Modifier** | `_concurrent` | `_concurrent` | Optional suffix indicating special execution mode |

## Migration Types

### Main Migrations (`NNN_description.sql`)

Main migrations represent a logical unit of schema change. They are numbered sequentially:

- `001_registration_projection.sql` - Initial projection schema
- `002_updated_at_audit_index.sql` - Audit query indexes
- `003_capability_fields.sql` - Capability discovery columns and indexes

**When to increment the main number:**
- New tables or major schema changes
- Independent features that don't depend on a companion script
- Changes that represent a distinct logical unit

### Sub-Migrations (`NNNa_description.sql`)

Sub-migrations are companion scripts related to a main migration. They share the same numeric prefix with a letter suffix:

- `003a_capability_fields_concurrent.sql` - Concurrent version of 003

**When to use letter suffixes:**
- Production-optimized variants of a migration (e.g., `CONCURRENTLY` indexes)
- Split migrations that must run in a specific order
- Rollback or cleanup scripts for a main migration
- Alternative implementations for different environments

**Ordering:**
- Sub-migrations (`003a`, `003b`) always relate to their parent (`003`)
- Execute in alphabetical order: `003` -> `003a` -> `003b` (when applicable)
- Check migration header comments for specific ordering requirements

## Modifier Suffixes

### `_concurrent` Suffix

Indicates a migration designed for **production environments with live traffic**:

```
003_capability_fields.sql           # Standard version (may lock tables)
003a_capability_fields_concurrent.sql  # Production version (non-blocking)
```

**Characteristics of `_concurrent` migrations:**
- Uses `CREATE INDEX CONCURRENTLY` instead of `CREATE INDEX`
- Cannot run inside a transaction block (requires autocommit)
- Longer execution time but no table locks
- May leave indexes in INVALID state if interrupted (must be dropped and recreated)

**When to use which:**

| Environment | Migration Type | Notes |
|-------------|----------------|-------|
| Development | Standard (`003`) | Faster, can run in transactions |
| CI/CD | Standard (`003`) | Transactions required for rollback |
| Staging | Either | Depends on table size and requirements |
| Production (<100K rows) | Standard (`003`) | Brief locks acceptable |
| Production (>100K rows) | Concurrent (`003a`) | Avoid blocking writes |
| Production (>1M rows) | Concurrent (`003a`) | **Required** for zero-downtime |

## Determining the Next Migration Number

**For a new independent feature:**
- Use the next sequential number: `004_new_feature.sql`

**For a companion to an existing migration:**
- Use the next available letter: `003b_capability_fields_rollback.sql`

**Current state:**
```
001_registration_projection.sql     # Main: initial schema
002_updated_at_audit_index.sql      # Main: audit indexes
003_capability_fields.sql           # Main: capability columns
003a_capability_fields_concurrent.sql  # Sub: production indexes
```

**Next available:**
- Next main migration: `004_*.sql`
- Next sub-migration for 003: `003b_*.sql`

## Migration Header Requirements

Every migration should include a header comment with:

```sql
-- =============================================================================
-- MIGRATION: Brief Description
-- =============================================================================
-- Ticket: OMN-XXXX (Feature Name)
-- Version: 1.0.0
-- Companion to: NNN_parent_migration.sql (if sub-migration)
--
-- PURPOSE:
--   What this migration does and why
--
-- PRODUCTION DATABASE SAFETY:
--   1. BACKUP your database before running
--   2. TEST on staging environment first
--   3. MONITOR execution time and locks
--
-- ROLLBACK:
--   SQL statements to reverse this migration
--
-- =============================================================================
```

## Execution Guidelines

### Standard Migrations

```bash
# Can run in a transaction (for CI/CD with rollback support)
psql -h $HOST -d $DB -f 003_capability_fields.sql
```

### Concurrent Migrations

```bash
# MUST run outside transaction block (autocommit mode)
psql -h $HOST -d $DB -f 003a_capability_fields_concurrent.sql

# DO NOT wrap in BEGIN/COMMIT - will error:
# "CREATE INDEX CONCURRENTLY cannot run inside a transaction block"
```

### Monitoring Concurrent Index Creation

```sql
-- Check progress (PostgreSQL 12+)
SELECT
    p.phase,
    ROUND(100.0 * p.blocks_done / NULLIF(p.blocks_total, 0), 1) AS pct_complete,
    i.relname AS index_name
FROM pg_stat_progress_create_index p
JOIN pg_class i ON i.oid = p.index_relid;

-- Check for invalid indexes after failure
SELECT indexrelid::regclass AS index_name, indisvalid
FROM pg_index
WHERE NOT indisvalid;
```

## Summary

| Pattern | Example | Use Case |
|---------|---------|----------|
| `NNN_desc.sql` | `004_new_table.sql` | New independent feature |
| `NNNa_desc.sql` | `003a_concurrent.sql` | Companion to NNN |
| `*_concurrent.sql` | `003a_*_concurrent.sql` | Non-blocking production variant |

**Decision tree for next migration:**

1. Is this related to an existing migration? -> Use letter suffix (`003b`)
2. Is this a production-optimized variant? -> Add `_concurrent` suffix
3. Otherwise -> Use next main number (`004`)
