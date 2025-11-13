# Migration 004 - Node Registrations Table

## Overview

Migration 004 adds the `node_registrations` table for dynamic node discovery and registration in the OmniNode Bridge system. This migration includes comprehensive enum validation and idempotency guarantees.

## Changes

### Database Objects Created

1. **Enum Type: `node_type_enum`**
   - Values: `effect`, `compute`, `reducer`, `orchestrator`
   - Validated for consistency on every migration run

2. **Table: `node_registrations`**
   - Stores dynamic node registration data
   - Columns: id, node_id, node_type, capabilities, endpoints, metadata, health_endpoint, last_heartbeat, registered_at, updated_at

3. **Indexes** (7 total):
   - B-tree indexes: node_id, node_type, last_heartbeat, updated_at
   - GIN indexes: capabilities, endpoints, metadata (for JSONB queries)

## Idempotency Guarantees

### Enum Validation

This migration implements **strict enum validation** to prevent silent failures:

- ✅ **Safe**: Enum doesn't exist → Creates enum with correct values
- ✅ **Safe**: Enum exists with correct values → Migration succeeds (idempotent)
- ❌ **FAILS**: Enum exists with wrong values → Migration fails with detailed error

### Error Messages

When enum values don't match, the migration fails with a clear error message:

```
MIGRATION VALIDATION FAILED: Enum 'node_type_enum' exists with incompatible values!
  Expected values: ['compute', 'effect', 'orchestrator', 'reducer']
  Existing values: ['compute', 'effect', 'wrong_value']
  Missing values: ['orchestrator', 'reducer']
  Extra values: ['wrong_value']

To fix this issue:
  1. Manually drop the enum: DROP TYPE node_type_enum CASCADE;
  2. Re-run the migration
  OR
  3. Manually alter the enum to match expected values
```

### Migration Safety Features

1. **Validation Before Creation**: Checks existing enum values before attempting creation
2. **Fail Fast**: Immediately fails if inconsistency detected
3. **Detailed Logging**: Clear messages for all operations
4. **Dependency Checking**: Downgrade validates no external dependencies exist
5. **Data Loss Warning**: Warns when downgrading with existing data

## Running the Migration

### Upgrade

```bash
# Standard upgrade
alembic upgrade 004

# Or upgrade to latest
alembic upgrade head
```

### Downgrade

```bash
# Downgrade to previous migration
alembic downgrade 003

# With data (requires environment variable)
ALLOW_DESTRUCTIVE_MIGRATION=true alembic downgrade 003
```

## Testing

Comprehensive tests are available in `tests/integration/test_migration_004_idempotency.py`:

```bash
# Run all migration tests
pytest tests/integration/test_migration_004_idempotency.py -v

# Run specific test class
pytest tests/integration/test_migration_004_idempotency.py::TestMigration004FreshInstall -v

# Run specific test
pytest tests/integration/test_migration_004_idempotency.py::TestMigration004EnumValidation::test_migration_fails_with_wrong_enum_values -v
```

### Test Coverage

- ✅ Fresh installation (clean database)
- ✅ Idempotent re-runs (existing enum and table)
- ✅ Enum validation (wrong/missing/extra values)
- ✅ Downgrade and cleanup
- ✅ Full migration cycle (upgrade → downgrade → upgrade)
- ✅ Edge cases (concurrent usage, order independence)

## Troubleshooting

### Migration Fails with "incompatible values"

**Cause**: Enum exists with different values than expected.

**Solution**:
1. Check what enum values currently exist:
   ```sql
   SELECT enumlabel FROM pg_enum WHERE enumtypid = (
       SELECT oid FROM pg_type WHERE typname = 'node_type_enum'
   );
   ```

2. Option A - Drop and recreate (safe if no data):
   ```sql
   DROP TYPE node_type_enum CASCADE;
   ```
   Then re-run migration.

3. Option B - Fix existing enum (preserves data):
   ```sql
   -- Add missing values
   ALTER TYPE node_type_enum ADD VALUE 'orchestrator';

   -- Remove extra values (requires dropping constraints first)
   -- This is complex - usually easier to drop and recreate
   ```

### Migration Fails on Downgrade

**Cause**: Enum has external dependencies (other tables using it).

**Solution**:
1. Find dependencies:
   ```sql
   SELECT n.nspname, c.relname, a.attname
   FROM pg_type t
   JOIN pg_enum e ON t.oid = e.enumtypid
   JOIN pg_attribute a ON a.atttypid = t.oid
   JOIN pg_class c ON a.attrelid = c.oid
   JOIN pg_namespace n ON c.relnamespace = n.oid
   WHERE t.typname = 'node_type_enum';
   ```

2. Drop dependent objects first, then retry downgrade.

### Downgrade Blocked by Data Warning

**Cause**: Protection against accidental data loss.

**Solution**:
Set environment variable to confirm:
```bash
ALLOW_DESTRUCTIVE_MIGRATION=true alembic downgrade 003
```

## Implementation Details

### Validation Function

```python
def _validate_enum_values(conn, enum_name: str, expected_values: list[str]) -> bool:
    """
    Validate that an existing enum type has the expected values.
    Returns True if enum exists with correct values.
    Raises RuntimeError if enum exists with wrong values.
    Returns False if enum doesn't exist.
    """
```

### Safe Enum Creation

```python
def _create_enum_safe(conn, enum_name: str, values: list[str]):
    """
    Create enum type with validation for idempotency.
    Validates existing enum values before attempting creation.
    """
```

### Safe Enum Deletion

```python
def _drop_enum_safe(conn, enum_name: str, cascade: bool = False):
    """
    Drop enum type safely with dependency checking.
    Fails if dependencies exist and cascade is False.
    """
```

## Best Practices

1. **Always test migrations** in development environment first
2. **Backup production database** before running migrations
3. **Review migration logs** for warnings or errors
4. **Validate database state** after migration completes
5. **Test rollback procedure** in development before production use

## Version History

- **004** (2025-10-03): Initial version with comprehensive enum validation and idempotency
- Enhanced with strict validation, fail-fast behavior, and dependency checking

## Related Documentation

- [Alembic Migration Guide](../../docs/ALEMBIC_MIGRATIONS.md)
- [Database Schema Documentation](../../docs/DATABASE_SCHEMA.md)
- [Node Registration System](../../docs/NODE_REGISTRATION.md)
