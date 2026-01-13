# Migration Upgrade Guide: 003a to 004

This document explains the migration renumbering from `003a_capability_fields_concurrent.sql` to `004_capability_fields_concurrent.sql` and provides upgrade instructions for existing deployments.

## Summary of Change

| Before | After | Reason |
|--------|-------|--------|
| `003a_capability_fields_concurrent.sql` | `004_capability_fields_concurrent.sql` | Standardize on main sequence numbering |

## Why the Renumbering?

The original naming (`003a`) followed a sub-migration convention where letter suffixes indicated companion scripts to a parent migration. However, this created confusion:

1. **Ambiguous execution order**: Sub-migrations (`003a`, `003b`) imply optional/alternative paths, but `003a` was actually the **recommended** production migration for large tables.

2. **Documentation consistency**: The production concurrent variant should be a first-class migration, not a sub-variant.

3. **Tooling compatibility**: Some migration tracking tools don't handle letter suffixes well.

The renumbering to `004` makes it explicit that this is a standalone migration that can be applied instead of (or in addition to) `003`.

## Migration Relationship

```
003_capability_fields.sql
    |
    |-- Adds columns: contract_type, intent_types, protocols, capability_tags, contract_version
    |-- Creates indexes with standard CREATE INDEX (may lock tables briefly)
    |
004_capability_fields_concurrent.sql (formerly 003a)
    |
    |-- Creates the SAME indexes as 003, but using CREATE INDEX CONCURRENTLY
    |-- Does NOT add columns (assumes 003 was already applied)
    |-- Use for production deployments with >100K rows
```

## Upgrade Scenarios

### Scenario 1: Fresh Deployment (No Migrations Applied)

Apply migrations in order:

```bash
# Apply base schema
psql -h $HOST -d $DB -f 001_registration_projection.sql
psql -h $HOST -d $DB -f 002_updated_at_audit_index.sql

# For development/staging (standard indexes)
psql -h $HOST -d $DB -f 003_capability_fields.sql

# OR for production with large tables (concurrent indexes)
# First, apply 003 with indexes commented out, then run 004:
psql -h $HOST -d $DB -f 003_capability_fields.sql  # Column creation only
psql -h $HOST -d $DB -f 004_capability_fields_concurrent.sql  # CONCURRENTLY indexes
```

### Scenario 2: Production Database with 003a Already Applied

If your database already has `003a_capability_fields_concurrent.sql` applied:

**Good news**: No action required. The migration `004_capability_fields_concurrent.sql` is functionally identical to the old `003a`. Your indexes are already in place.

**Verification steps**:

```sql
-- Verify capability columns exist
SELECT column_name
FROM information_schema.columns
WHERE table_name = 'registration_projections'
  AND column_name IN ('contract_type', 'intent_types', 'protocols', 'capability_tags', 'contract_version');

-- Expected: 5 rows

-- Verify GIN indexes exist
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename = 'registration_projections'
  AND indexname IN (
    'idx_registration_capability_tags',
    'idx_registration_intent_types',
    'idx_registration_protocols',
    'idx_registration_contract_type_state'
  );

-- Expected: 4 rows (all indexes valid)

-- Verify all indexes are valid (not in failed state)
SELECT indexrelid::regclass AS index_name, indisvalid
FROM pg_index
WHERE indrelid = 'registration_projections'::regclass
  AND NOT indisvalid;

-- Expected: 0 rows (no invalid indexes)
```

### Scenario 3: Production Database with Only 003 Applied (Standard Indexes)

If you applied `003_capability_fields.sql` and want to upgrade to concurrent indexes:

```bash
# Drop existing standard indexes
psql -h $HOST -d $DB -c "DROP INDEX IF EXISTS idx_registration_capability_tags;"
psql -h $HOST -d $DB -c "DROP INDEX IF EXISTS idx_registration_intent_types;"
psql -h $HOST -d $DB -c "DROP INDEX IF EXISTS idx_registration_protocols;"
psql -h $HOST -d $DB -c "DROP INDEX IF EXISTS idx_registration_contract_type_state;"

# Apply concurrent indexes (cannot run in transaction block)
psql -h $HOST -d $DB -f 004_capability_fields_concurrent.sql
```

### Scenario 4: Migration Tracking System Records "003a"

If your deployment tracking (manual logs, wiki, etc.) shows "003a" as applied:

1. **Update your records**: Note that `003a` is now `004`
2. **No database changes needed**: The underlying SQL is identical
3. **Document the alias**: In your tracking system, note:
   ```
   003a_capability_fields_concurrent.sql (RENAMED to 004)
   Status: Applied
   Notes: Renamed in PR #143. Same SQL content as original 003a.
   ```

## Migration File Checksums

To verify you have the correct version of each migration, compare SHA-256 checksums:

```bash
# Generate checksums for your migration files
cd docker/migrations/
sha256sum *.sql
```

The index creation SQL in `003_capability_fields.sql` and `004_capability_fields_concurrent.sql` should produce identical indexes (just using different creation methods).

## Version Compatibility Matrix

| Migration Version | Creates Columns | Creates Indexes | Index Method | Suitable For |
|-------------------|-----------------|-----------------|--------------|--------------|
| 003 only | Yes | Yes | Standard | Dev/Staging, <100K rows |
| 003 + 004 | Yes (003) | Yes (004) | Concurrent | Production, >100K rows |
| 003a (legacy) | No | Yes | Concurrent | **DEPRECATED** - use 004 |

## Troubleshooting

### "Index already exists" Error

If `004` fails because indexes already exist from `003`:

```sql
-- Drop existing indexes first
DROP INDEX IF EXISTS idx_registration_capability_tags;
DROP INDEX IF EXISTS idx_registration_intent_types;
DROP INDEX IF EXISTS idx_registration_protocols;
DROP INDEX IF EXISTS idx_registration_contract_type_state;
```

### Invalid Index State

If a concurrent index creation was interrupted, it may be left in an invalid state:

```sql
-- Check for invalid indexes
SELECT indexrelid::regclass AS index_name
FROM pg_index
WHERE NOT indisvalid
  AND indrelid = 'registration_projections'::regclass;

-- Drop invalid index and recreate
DROP INDEX <invalid_index_name>;
-- Then re-run 004_capability_fields_concurrent.sql
```

### Missing Columns

If `004` fails because columns don't exist:

```
ERROR: Missing columns: contract_type, intent_types, protocols, capability_tags
```

You need to apply `003_capability_fields.sql` first to create the columns.

## Related Documentation

- `README.md` - Migration versioning conventions
- `003_capability_fields.sql` - Column and standard index creation
- `004_capability_fields_concurrent.sql` - Concurrent index creation
- `ADR-005-denormalize-capability-fields-in-registration-projections.md` - Design rationale

## Changelog

| Date | Change |
|------|--------|
| 2025-01-13 | Renamed `003a` to `004` per PR #143 review feedback |
| 2025-01-XX | Initial `003a_capability_fields_concurrent.sql` created |
