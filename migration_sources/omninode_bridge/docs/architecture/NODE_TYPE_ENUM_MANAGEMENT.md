# Node Type ENUM Management Guide

## Overview

This guide documents the process for managing ONEX node types in the PostgreSQL ENUM-based system implemented in migration `004_20251003_node_registrations.py`.

## Current Node Types

The system currently supports four ONEX-compliant node types:

```python
class EnumNodeType(str, Enum):
    """ONEX-compliant node type enumeration following naming conventions."""

    EFFECT = "effect"
    COMPUTE = "compute"
    REDUCER = "reducer"
    ORCHESTRATOR = "orchestrator"
```

## PostgreSQL ENUM Type

The database uses a PostgreSQL ENUM type `node_type_enum` with the same values:

```sql
CREATE TYPE node_type_enum AS ENUM ('effect', 'compute', 'reducer', 'orchestrator');
```

## Adding New Node Types

When adding a new node type to the system, follow these steps in order:

### Step 1: Update Python Enum

First, add the new node type to the Python enum in `src/omninode_bridge/models/database_models.py`:

```python
class EnumNodeType(str, Enum):
    """ONEX-compliant node type enumeration following naming conventions."""

    EFFECT = "effect"
    COMPUTE = "compute"
    REDUCER = "reducer"
    ORCHESTRATOR = "orchestrator"
    NEW_NODE_TYPE = "new_type"  # Add new type here
```

**Naming Convention**: Follow ONEX v2.0 compliance with `UPPER_SNAKE_CASE` naming and lowercase string values.

### Step 2: Create Database Migration

Create a new Alembic migration to extend the PostgreSQL ENUM type:

```bash
poetry run alembic revision -m "Add new_node_type to node_type_enum"
```

In the generated migration file, add the following:

```python
def upgrade():
    """Add new_node_type to node_type_enum."""

    # PostgreSQL requires adding new enum values at the end
    op.execute("ALTER TYPE node_type_enum ADD VALUE 'new_type'")

def downgrade():
    """Remove new_node_type from node_type_enum."""

    # Note: PostgreSQL doesn't support removing enum values directly
    # This would require recreating the type, which is complex
    # For now, document that enum value removal requires manual intervention
    logger.warning(
        "PostgreSQL does not support removing enum values directly. "
        "Manual database intervention required to remove 'new_type' from node_type_enum."
    )
```

**Important**: PostgreSQL only allows adding enum values at the end. You cannot insert values in the middle or remove existing values.

### Step 3: Update Application Code

Update any application code that references node types:

- Update type hints and validation logic
- Update documentation and examples
- Update test cases to include the new node type
- Update service configurations that may reference node types

### Step 4: Update Tests

Add test coverage for the new node type:

```python
def test_new_node_type_registration(self):
    """Test registration of new node type."""
    registration = ModelNodeRegistrationCreate(
        node_id="test-new-node",
        node_type=EnumNodeType.NEW_NODE_TYPE,  # Use the enum
        capabilities={"test": True},
        endpoints={"test": "http://test:8080"}
    )

    # Test the registration works correctly
    result = self.repository.create_registration(registration)
    assert result.node_type == EnumNodeType.NEW_NODE_TYPE
```

### Step 5: Update Documentation

Update relevant documentation:

- API documentation with new node type examples
- Architecture documentation
- Developer onboarding guides
- Configuration examples

## Removing Node Types (Advanced)

**Warning**: PostgreSQL does not support removing individual values from ENUM types. To remove a node type:

1. **Create a new ENUM type** without the unwanted value
2. **Create a new column** using the new ENUM type
3. **Migrate data** from old column to new column
4. **Drop old column and ENUM type**
5. **Rename new column** to original name

This process requires careful planning and should be done during maintenance windows.

## Best Practices

### 1. Use Enums in Application Code

Always use the `EnumNodeType` class in application code:

```python
# Good - Use the enum
node_type = EnumNodeType.EFFECT

# Avoid - Hardcoded strings
node_type = "effect"
```

### 2. Case-Insensitive Validation

The Pydantic models automatically convert lowercase strings to the appropriate enum:

```python
# These all work and convert to EnumNodeType.EFFECT
registration = ModelNodeRegistrationCreate(node_type="effect")
registration = ModelNodeRegistrationCreate(node_type=EnumNodeType.EFFECT)
```

### 3. Database Performance

PostgreSQL ENUMs provide better performance than CHECK constraints:

- **Storage**: ENUMs store values as integers (more efficient)
- **Indexing**: Faster index operations on ENUM columns
- **Validation**: Built-in type checking at the database level

### 4. Migration Safety

Always test migrations in development before applying to production:

```bash
# Test migration
poetry run alembic upgrade head +1
poetry run alembic downgrade -1

# Verify no data loss or corruption
```

## Troubleshooting

### Common Issues

1. **Migration Error: "enum value already exists"**
   - The enum value was already added
   - Check if the migration was already applied

2. **Type Validation Errors**
   - Ensure the string value matches the enum value exactly
   - Check for case sensitivity issues

3. **Database Connection Issues**
   - Ensure the database user has permissions to alter types
   - Check for active connections using the enum type

### Debugging Tips

Use these SQL queries to inspect the enum type:

```sql
-- Check enum values
SELECT unnest(enum_range(NULL::node_type_enum));

-- Check current usage
SELECT DISTINCT node_type FROM node_registrations;

-- Check table definition
\d node_registrations
```

## Migration History

- **Migration 004** (2025-10-03): Initial implementation with 4 node types using PostgreSQL ENUM
  - Replaced CHECK constraint with ENUM type for better performance and type safety
  - Updated both SQLAlchemy models and Pydantic validation models

## References

- [PostgreSQL ENUM Documentation](https://www.postgresql.org/docs/current/datatype-enum.html)
- [Alembic Migration Guide](https://alembic.sqlalchemy.org/en/latest/)
- [ONEX v2.0 Naming Conventions](../ONEX_NODE_IMPLEMENTATION_PLAN.md)
- [Node Registration System](../docs/BRIDGE_NODES_GUIDE.md)
