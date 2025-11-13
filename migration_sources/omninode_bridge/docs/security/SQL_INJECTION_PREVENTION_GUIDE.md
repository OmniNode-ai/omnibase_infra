# SQL Injection Prevention Guide

**Quick Reference for OmniNode Bridge Developers**

---

## 🔒 Security Status: ALL CLEAR ✅

All CRUD operations in the OmniNode Bridge codebase are **SQL injection safe**. This guide documents our security patterns and best practices to maintain this security posture.

---

## Core Security Principles

### 1. Always Use Parameterized Queries

**✅ CORRECT - Use positional parameters**:
```python
# PostgreSQL positional parameters ($1, $2, ...)
query = "SELECT * FROM users WHERE email = $1 AND status = $2"
params = [user_email, user_status]
result = await execute_query(query, *params)
```

**❌ WRONG - Never concatenate user input**:
```python
# VULNERABLE - Do not do this!
query = f"SELECT * FROM users WHERE email = '{user_email}'"
query = "SELECT * FROM users WHERE email = '" + user_email + "'"
```

---

### 2. Validate All SQL Identifiers

**SQL identifiers** are table names, column names, and field names that cannot be parameterized.

**✅ CORRECT - Validate identifiers**:
```python
from omninode_bridge.security.validation import InputSanitizer

# Validate table/column names before using in queries
validated_table = InputSanitizer.validate_sql_identifier(table_name)
validated_column = InputSanitizer.validate_sql_identifier(column_name)

query = f"SELECT * FROM {validated_table} WHERE {validated_column} = $1"
```

**❌ WRONG - Never use raw identifiers**:
```python
# VULNERABLE - Missing validation
query = f"SELECT * FROM {table_name} WHERE {column_name} = $1"
```

---

### 3. Use Operator Whitelisting

**✅ CORRECT - Hardcoded operators**:
```python
ALLOWED_OPERATORS = {"eq", "gt", "gte", "lt", "lte", "ne", "in"}

if operator not in ALLOWED_OPERATORS:
    raise ValueError(f"Unsupported operator: {operator}")

if operator == "eq":
    condition = f"{validated_field} = ${param_counter}"
elif operator == "gt":
    condition = f"{validated_field} > ${param_counter}"
# ... etc
```

**❌ WRONG - User-controlled operators**:
```python
# VULNERABLE - Never do this
condition = f"{field} {user_operator} ${param_counter}"
```

---

### 4. Validate Integer Parameters

For LIMIT, OFFSET, and other integer values that can be safely concatenated:

**✅ CORRECT - Type and bounds checking**:
```python
# Validate type
if not isinstance(limit, int) or limit < 0:
    raise ValueError("LIMIT must be a non-negative integer")

# Validate bounds
if limit > MAX_LIMIT:
    raise ValueError(f"LIMIT exceeds maximum: {MAX_LIMIT}")

# Safe to concatenate after validation
query += f" LIMIT {limit}"
```

**❌ WRONG - No validation**:
```python
# VULNERABLE - Missing validation
query += f" LIMIT {limit}"
```

---

## Practical Examples

### Example 1: Simple SELECT Query

```python
from omninode_bridge.security.validation import InputSanitizer

async def get_users_by_status(status: str, limit: int) -> list:
    """Get users by status with pagination."""

    # Step 1: Validate table name (if dynamic)
    validated_table = InputSanitizer.validate_sql_identifier("users")

    # Step 2: Validate limit (integer bounds checking)
    if not isinstance(limit, int) or limit < 0:
        raise ValueError("Invalid limit")
    if limit > 1000:
        raise ValueError("Limit too large")

    # Step 3: Build parameterized query
    query = f"""
        SELECT * FROM {validated_table}
        WHERE status = $1
        ORDER BY created_at DESC
        LIMIT {limit}
    """

    # Step 4: Execute with parameters
    params = [status]
    return await execute_query(query, *params)
```

---

### Example 2: Dynamic WHERE Clause

```python
async def build_where_clause(filters: dict) -> tuple[str, list]:
    """Build parameterized WHERE clause from filters."""

    conditions = []
    params = []
    param_counter = 1

    for key, value in filters.items():
        # Parse field and operator
        if "__" in key:
            field, operator = key.rsplit("__", 1)
        else:
            field, operator = key, "eq"

        # CRITICAL: Validate field name
        validated_field = InputSanitizer.validate_sql_identifier(field)

        # Build condition with parameterized value
        if operator == "eq":
            conditions.append(f"{validated_field} = ${param_counter}")
            params.append(value)
            param_counter += 1
        elif operator == "gt":
            conditions.append(f"{validated_field} > ${param_counter}")
            params.append(value)
            param_counter += 1
        elif operator == "in":
            if not isinstance(value, list):
                raise ValueError("IN operator requires list")
            placeholders = [f"${param_counter + i}" for i in range(len(value))]
            conditions.append(f"{validated_field} IN ({', '.join(placeholders)})")
            params.extend(value)
            param_counter += len(value)
        else:
            raise ValueError(f"Unsupported operator: {operator}")

    where_clause = " AND ".join(conditions) if conditions else "TRUE"
    return where_clause, params
```

---

### Example 3: Batch INSERT

```python
async def batch_insert_users(users: list[dict]) -> list[int]:
    """Batch insert users with parameterized query."""

    # Step 1: Validate table and column names
    validated_table = InputSanitizer.validate_sql_identifier("users")
    columns = ["name", "email", "status"]
    validated_columns = [
        InputSanitizer.validate_sql_identifier(col) for col in columns
    ]

    # Step 2: Build parameterized batch query
    values_list = []
    params = []
    param_counter = 1

    for user in users:
        row_placeholders = []
        for col in columns:
            row_placeholders.append(f"${param_counter}")
            params.append(user[col])
            param_counter += 1
        values_list.append(f"({', '.join(row_placeholders)})")

    # Step 3: Build query
    query = f"""
        INSERT INTO {validated_table} ({', '.join(validated_columns)})
        VALUES {', '.join(values_list)}
        RETURNING id
    """

    # Step 4: Execute in transaction
    async with connection_manager.transaction() as conn:
        result = await conn.fetch(query, *params)
        return [row["id"] for row in result]
```

---

## What NOT to Do

### ❌ Anti-Pattern 1: String Concatenation

```python
# NEVER DO THIS
query = f"SELECT * FROM users WHERE email = '{email}'"
query = "SELECT * FROM users WHERE email = '" + email + "'"
query = "SELECT * FROM users WHERE email = '%s'" % email
query = "SELECT * FROM users WHERE email = {}".format(email)
```

**Why it's vulnerable**:
```python
email = "test@example.com' OR '1'='1"
# Resulting query: SELECT * FROM users WHERE email = 'test@example.com' OR '1'='1'
# Result: Returns ALL users (SQL injection successful)
```

---

### ❌ Anti-Pattern 2: Unvalidated Identifiers

```python
# NEVER DO THIS
table_name = user_input  # e.g., "users; DROP TABLE users--"
query = f"SELECT * FROM {table_name}"
```

**Why it's vulnerable**:
```python
table_name = "users; DROP TABLE users--"
# Resulting query: SELECT * FROM users; DROP TABLE users--
# Result: Executes multiple statements (stacked queries)
```

---

### ❌ Anti-Pattern 3: User-Controlled Operators

```python
# NEVER DO THIS
operator = user_input  # e.g., "= 1 OR 1=1"
query = f"SELECT * FROM users WHERE id {operator}"
```

**Why it's vulnerable**:
```python
operator = "= 1 OR 1=1"
# Resulting query: SELECT * FROM users WHERE id = 1 OR 1=1
# Result: Returns ALL users
```

---

## InputSanitizer Reference

### validate_sql_identifier()

**Purpose**: Validate SQL identifiers (table names, column names, field names)

**Validation Rules**:
- Must be non-empty string
- Maximum length: 63 characters (PostgreSQL limit)
- Must match regex: `^[a-zA-Z_][a-zA-Z0-9_]*$`
  - Starts with letter or underscore
  - Contains only letters, digits, underscores

**Usage**:
```python
from omninode_bridge.security.validation import InputSanitizer

# Validate identifier
try:
    validated_table = InputSanitizer.validate_sql_identifier(table_name)
except ValueError as e:
    # Handle validation error
    raise OnexError(
        error_code=EnumCoreErrorCode.VALIDATION_ERROR,
        message=str(e)
    )
```

**Examples**:
```python
# ✅ VALID
InputSanitizer.validate_sql_identifier("users")           # → "users"
InputSanitizer.validate_sql_identifier("bridge_state")    # → "bridge_state"
InputSanitizer.validate_sql_identifier("_internal")       # → "_internal"
InputSanitizer.validate_sql_identifier("table123")        # → "table123"

# ❌ INVALID
InputSanitizer.validate_sql_identifier("users; DROP")     # → ValueError
InputSanitizer.validate_sql_identifier("users--")         # → ValueError
InputSanitizer.validate_sql_identifier("users'")          # → ValueError
InputSanitizer.validate_sql_identifier("users OR 1=1")    # → ValueError
InputSanitizer.validate_sql_identifier("123table")        # → ValueError
```

---

## Security Checklist for Pull Requests

When adding new database operations, verify:

- [ ] All user values are passed as parameters (`$1, $2, ...`), not concatenated
- [ ] All table/column names are validated via `InputSanitizer.validate_sql_identifier()`
- [ ] All SQL operators are hardcoded, not user-controlled
- [ ] LIMIT/OFFSET values are validated as integers with bounds checking
- [ ] Batch operations validate batch size limits
- [ ] Complex queries are reviewed by security team
- [ ] Tests include SQL injection attempts (should be safely handled)

---

## Testing for SQL Injection

### Manual Testing

Test each endpoint with injection payloads:

```python
# Test parameterization
malicious_value = "'; DROP TABLE users--"
result = await get_users_by_status(status=malicious_value)
# Should execute safely with no SQL injection

# Test identifier validation
malicious_field = "email; DROP TABLE users--"
try:
    InputSanitizer.validate_sql_identifier(malicious_field)
    assert False, "Should have raised ValueError"
except ValueError:
    pass  # Expected
```

### Automated Testing

```python
import pytest
from omninode_bridge.security.validation import InputSanitizer

def test_sql_injection_payloads():
    """Test common SQL injection payloads."""

    injection_payloads = [
        "' OR '1'='1",
        "'; DROP TABLE users--",
        "1' UNION SELECT * FROM users--",
        "'; WAITFOR DELAY '00:00:10'--",
        "admin'--",
        "' OR 1=1--",
    ]

    for payload in injection_payloads:
        # Test identifier validation rejects payload
        with pytest.raises(ValueError):
            InputSanitizer.validate_sql_identifier(payload)

        # Test parameterized query safely handles payload
        result = await get_user_by_email(email=payload)
        # Should return no results (or specific user with that exact email)
```

---

## Integration with Generic CRUD Handlers

The `GenericCRUDHandlers` class in `_generic_crud_handlers.py` implements all security best practices:

**Usage**:
```python
from omninode_bridge.nodes.database_adapter_effect.v1_0_0.models.inputs import (
    ModelDatabaseOperationInput
)
from omninode_bridge.infrastructure.enum_entity_type import EnumEntityType
from omninode_bridge.nodes.database_adapter_effect.v1_0_0.enums import (
    EnumDatabaseOperationType
)

# Create operation input
operation_input = ModelDatabaseOperationInput(
    operation_type=EnumDatabaseOperationType.QUERY,
    entity_type=EnumEntityType.WORKFLOW_EXECUTION,
    correlation_id=uuid4(),
    query_filters={"namespace": "production", "status__in": ["active", "pending"]},
    limit=50,
    offset=0,
    sort_by="created_at",
    sort_order="desc"
)

# Execute through generic handler (automatically secured)
result = await node.process(operation_input)
```

**Security Guarantees**:
- ✅ All identifiers validated
- ✅ All values parameterized
- ✅ Operators whitelisted
- ✅ Limits/offsets validated
- ✅ Circuit breaker protection
- ✅ Transaction support

---

## Additional Resources

### Internal Documentation
- **[Complete Security Audit Report](./SQL_INJECTION_AUDIT_REPORT.md)** - Full vulnerability assessment
- **[Generic CRUD Handlers](../../src/omninode_bridge/nodes/database_adapter_effect/v1_0_0/_generic_crud_handlers.py)** - Reference implementation
- **[Security Validator](../../src/omninode_bridge/nodes/database_adapter_effect/v1_0_0/security_validator.py)** - Additional security layer

### External Resources
- [OWASP SQL Injection Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html)
- [CWE-89: SQL Injection](https://cwe.mitre.org/data/definitions/89.html)
- [PostgreSQL Security Best Practices](https://www.postgresql.org/docs/current/sql-prepare.html)
- [asyncpg Parameterized Queries](https://magicstack.github.io/asyncpg/current/usage.html#prepared-statements)

---

## Contact

**Security Questions**: Contact the security team
**Code Reviews**: Request review from database team lead
**Vulnerabilities**: Report immediately via security incident process

---

**Last Updated**: October 18, 2025
**Version**: 1.0
**Next Review**: Quarterly (January 2026)
