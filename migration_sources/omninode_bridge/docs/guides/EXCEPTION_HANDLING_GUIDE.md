# Exception Handling Guide for ONEX Nodes

**Version**: 1.0
**Last Updated**: 2025-11-03
**Status**: ✅ Production Ready

## Overview

This guide defines exception handling best practices for ONEX v2.0 nodes. Proper exception handling improves debugging, system reliability, and operational observability.

## Core Principles

### 1. Catch Specific Exceptions First

**❌ ANTI-PATTERN** (Overly broad):
```python
try:
    result = await database.query(sql)
except Exception as e:
    logger.error(f"Query failed: {e}")
    raise
```

**✅ CORRECT PATTERN** (Specific exceptions):
```python
try:
    result = await database.query(sql)
except asyncpg.PostgresError as e:
    # Handle database-specific errors
    logger.error(f"Database error: {e}", extra={"error_type": "postgres"})
    raise ModelOnexError(
        error_code=EnumCoreErrorCode.DATABASE_ERROR,
        message=f"Database operation failed: {e!s}",
        details={"original_error": str(e), "error_type": type(e).__name__}
    ) from e
except (ConnectionError, TimeoutError) as e:
    # Handle network errors
    logger.error(f"Network error: {e}", extra={"error_type": "network"})
    raise ModelOnexError(
        error_code=EnumCoreErrorCode.NETWORK_ERROR,
        message=f"Network error: {e!s}",
        details={"original_error": str(e)}
    ) from e
except Exception as e:
    # Log unexpected errors with full traceback and re-raise
    logger.critical(f"Unexpected error: {e}", exc_info=True)
    raise ModelOnexError(
        error_code=EnumCoreErrorCode.EXECUTION_ERROR,
        message=f"Unexpected error: {e!s}",
        details={"original_error": str(e), "error_type": type(e).__name__}
    ) from e
```

### 2. Always Use Exception Chaining (`from e`)

Exception chaining preserves the original traceback and makes debugging easier:

```python
except ValueError as e:
    raise ModelOnexError(
        error_code=EnumCoreErrorCode.INVALID_INPUT,
        message=f"Invalid input: {e!s}",
        details={"original_error": str(e)}
    ) from e  # ← CRITICAL: Preserves original traceback
```

### 3. Log at Appropriate Levels

- **ERROR**: Expected errors that are handled (database failures, network timeouts)
- **CRITICAL**: Unexpected errors that indicate bugs (unhandled exceptions, assertion failures)
- **WARNING**: Degraded operation (circuit breaker opened, fallback triggered)

```python
except asyncpg.PostgresError as e:
    # Expected error - log at ERROR level
    emit_log_event(LogLevel.ERROR, f"Database error: {e!s}", {...})

except Exception as e:
    # Unexpected error - log at CRITICAL level with full traceback
    emit_log_event(LogLevel.CRITICAL, f"Unexpected error: {e!s}", {...})
```

## Exception Types by Node Type

### Effect Nodes (External I/O)

Effect nodes interact with external systems and should handle:

**Database Operations**:
```python
import asyncpg
import psycopg2

try:
    result = await pool.execute(query)
except asyncpg.PostgresError as e:
    # Handle PostgreSQL errors
    raise ModelOnexError(
        error_code=EnumCoreErrorCode.DATABASE_ERROR,
        message=f"Database error: {e!s}",
        details={"original_error": str(e)}
    ) from e
except psycopg2.Error as e:
    # Handle psycopg2 errors
    raise ModelOnexError(
        error_code=EnumCoreErrorCode.DATABASE_ERROR,
        message=f"Database error: {e!s}",
        details={"original_error": str(e)}
    ) from e
```

**HTTP/Network Operations**:
```python
import httpx

try:
    response = await client.get(url)
except httpx.HTTPError as e:
    # Handle HTTP-specific errors
    raise ModelOnexError(
        error_code=EnumCoreErrorCode.NETWORK_ERROR,
        message=f"HTTP error: {e!s}",
        details={"original_error": str(e), "url": url}
    ) from e
except (ConnectionError, TimeoutError) as e:
    # Handle network errors
    raise ModelOnexError(
        error_code=EnumCoreErrorCode.NETWORK_ERROR,
        message=f"Network error: {e!s}",
        details={"original_error": str(e)}
    ) from e
```

**File Operations**:
```python
try:
    content = await read_file(path)
except FileNotFoundError as e:
    raise ModelOnexError(
        error_code=EnumCoreErrorCode.NOT_FOUND,
        message=f"File not found: {path}",
        details={"path": str(path)}
    ) from e
except PermissionError as e:
    raise ModelOnexError(
        error_code=EnumCoreErrorCode.PERMISSION_DENIED,
        message=f"Permission denied: {path}",
        details={"path": str(path)}
    ) from e
except OSError as e:
    raise ModelOnexError(
        error_code=EnumCoreErrorCode.SYSTEM_ERROR,
        message=f"OS error: {e!s}",
        details={"original_error": str(e)}
    ) from e
```

### Compute Nodes (Pure Functions)

Compute nodes perform transformations and should handle:

**Data Validation**:
```python
try:
    validated_data = transform(input_data)
except (ValueError, TypeError, KeyError) as e:
    # Handle data validation errors
    raise ModelOnexError(
        error_code=EnumCoreErrorCode.INVALID_INPUT,
        message=f"Data validation error: {e!s}",
        details={"original_error": str(e)}
    ) from e
```

**Mathematical Operations**:
```python
try:
    result = compute_value(x, y)
except (ArithmeticError, OverflowError, ZeroDivisionError) as e:
    # Handle mathematical errors
    raise ModelOnexError(
        error_code=EnumCoreErrorCode.EXECUTION_ERROR,
        message=f"Mathematical error: {e!s}",
        details={"original_error": str(e)}
    ) from e
```

### Reducer Nodes (Aggregation)

Reducer nodes aggregate data and should handle:

**Memory/Resource Limits**:
```python
try:
    aggregated = reduce_data(stream)
except (MemoryError, OverflowError) as e:
    # Handle resource exhaustion
    raise ModelOnexError(
        error_code=EnumCoreErrorCode.RESOURCE_EXHAUSTED,
        message=f"Resource exhaustion: {e!s}",
        details={"original_error": str(e)}
    ) from e
```

**Data Access**:
```python
try:
    value = accumulated_state[key]
except KeyError as e:
    # Handle missing keys
    raise ModelOnexError(
        error_code=EnumCoreErrorCode.INVALID_INPUT,
        message=f"Key not found: {key}",
        details={"key": key}
    ) from e
```

### Orchestrator Nodes (Workflow Coordination)

Orchestrator nodes coordinate workflows and should handle:

**Downstream Service Failures**:
```python
try:
    result = await call_downstream_service(url)
except (ConnectionError, TimeoutError) as e:
    # Handle network errors
    raise ModelOnexError(
        error_code=EnumCoreErrorCode.NETWORK_ERROR,
        message=f"Downstream service error: {e!s}",
        details={"original_error": str(e), "service": url}
    ) from e
```

**Workflow Configuration**:
```python
try:
    workflow = parse_workflow_config(config)
except ValueError as e:
    # Handle invalid configuration
    raise ModelOnexError(
        error_code=EnumCoreErrorCode.INVALID_INPUT,
        message=f"Invalid workflow configuration: {e!s}",
        details={"original_error": str(e)}
    ) from e
```

## Complete Exception Handling Template

### Effect Node Template

```python
async def execute_effect(self, contract: ModelContractEffect) -> Any:
    """Execute effect operation."""
    emit_log_event(
        LogLevel.INFO,
        "Executing effect",
        {"node_id": str(self.node_id), "correlation_id": str(contract.correlation_id)}
    )

    try:
        # IMPLEMENTATION: Add effect-specific logic here
        result = await perform_io_operation()

        emit_log_event(
            LogLevel.INFO,
            "Effect executed successfully",
            {"node_id": str(self.node_id), "correlation_id": str(contract.correlation_id)}
        )

        return result

    except (ConnectionError, TimeoutError) as e:
        # Network/connection failures
        emit_log_event(
            LogLevel.ERROR,
            f"Network error: {e!s}",
            {
                "node_id": str(self.node_id),
                "correlation_id": str(contract.correlation_id),
                "error_type": type(e).__name__
            }
        )
        raise ModelOnexError(
            error_code=EnumCoreErrorCode.NETWORK_ERROR,
            message=f"Network error: {e!s}",
            details={"original_error": str(e), "error_type": type(e).__name__}
        ) from e

    except ValueError as e:
        # Invalid input/configuration
        emit_log_event(
            LogLevel.ERROR,
            f"Invalid input: {e!s}",
            {"node_id": str(self.node_id), "correlation_id": str(contract.correlation_id)}
        )
        raise ModelOnexError(
            error_code=EnumCoreErrorCode.INVALID_INPUT,
            message=f"Invalid input: {e!s}",
            details={"original_error": str(e)}
        ) from e

    except Exception as e:
        # Unexpected errors - log with full traceback
        emit_log_event(
            LogLevel.CRITICAL,
            f"Unexpected error: {e!s}",
            {
                "node_id": str(self.node_id),
                "correlation_id": str(contract.correlation_id),
                "error": str(e),
                "error_type": type(e).__name__
            }
        )
        raise ModelOnexError(
            error_code=EnumCoreErrorCode.EXECUTION_ERROR,
            message=f"Unexpected error: {e!s}",
            details={"original_error": str(e), "error_type": type(e).__name__}
        ) from e
```

## Common Exception Types Reference

### Standard Library Exceptions

| Exception | When to Use | Example |
|-----------|-------------|---------|
| `ValueError` | Invalid input data, bad configuration | `int("not a number")` |
| `TypeError` | Wrong type passed to function | `len(123)` |
| `KeyError` | Missing dictionary key | `dict["nonexistent"]` |
| `FileNotFoundError` | File doesn't exist | `open("missing.txt")` |
| `PermissionError` | Insufficient permissions | File access denied |
| `ConnectionError` | Network connection failed | Socket errors |
| `TimeoutError` | Operation timed out | Slow API calls |
| `MemoryError` | Out of memory | Large data processing |
| `OverflowError` | Numeric overflow | Math operations |
| `ZeroDivisionError` | Division by zero | `x / 0` |

### Database Exceptions

**PostgreSQL (asyncpg)**:
- `asyncpg.PostgresError` - Base exception for all PostgreSQL errors
- `asyncpg.InterfaceError` - Client-side errors
- `asyncpg.DataError` - Data-related errors

**MySQL (mysql.connector)**:
- `mysql.connector.Error` - Base exception
- `mysql.connector.IntegrityError` - Constraint violations
- `mysql.connector.OperationalError` - Connection errors

### HTTP Exceptions

**httpx**:
- `httpx.HTTPError` - Base exception for all HTTP errors
- `httpx.ConnectError` - Connection failures
- `httpx.TimeoutException` - Request timeout

**aiohttp**:
- `aiohttp.ClientError` - Base exception
- `aiohttp.ClientConnectorError` - Connection errors
- `aiohttp.ClientTimeout` - Timeout errors

## Testing Exception Handling

Always test exception handling paths:

```python
@pytest.mark.asyncio
async def test_network_error_handling(node):
    """Test network error handling."""
    with patch('httpx.AsyncClient.get', side_effect=ConnectionError("Network down")):
        with pytest.raises(ModelOnexError) as exc_info:
            await node.execute_effect(contract)

        assert exc_info.value.error_code == EnumCoreErrorCode.NETWORK_ERROR
        assert "Network down" in str(exc_info.value.message)

@pytest.mark.asyncio
async def test_unexpected_error_logging(node, caplog):
    """Test unexpected errors are logged at CRITICAL level."""
    with patch('some_function', side_effect=RuntimeError("Unexpected!")):
        with pytest.raises(ModelOnexError):
            await node.execute_effect(contract)

        # Verify CRITICAL log level
        assert any(
            record.levelname == "CRITICAL" and "Unexpected error" in record.message
            for record in caplog.records
        )
```

## Migration Guide

For existing nodes with broad exception handlers:

### Before (Anti-pattern)
```python
try:
    result = await database.query(sql)
except Exception as e:
    logger.error(f"Failed: {e}")
    raise
```

### After (Best practice)
```python
try:
    result = await database.query(sql)
except asyncpg.PostgresError as e:
    emit_log_event(LogLevel.ERROR, f"Database error: {e!s}", {...})
    raise ModelOnexError(
        error_code=EnumCoreErrorCode.DATABASE_ERROR,
        message=f"Database error: {e!s}",
        details={"original_error": str(e)}
    ) from e
except (ConnectionError, TimeoutError) as e:
    emit_log_event(LogLevel.ERROR, f"Network error: {e!s}", {...})
    raise ModelOnexError(
        error_code=EnumCoreErrorCode.NETWORK_ERROR,
        message=f"Network error: {e!s}",
        details={"original_error": str(e)}
    ) from e
except Exception as e:
    emit_log_event(LogLevel.CRITICAL, f"Unexpected error: {e!s}", {...})
    raise ModelOnexError(
        error_code=EnumCoreErrorCode.EXECUTION_ERROR,
        message=f"Unexpected error: {e!s}",
        details={"original_error": str(e), "error_type": type(e).__name__}
    ) from e
```

## EnumCoreErrorCode Reference

Common error codes from omnibase_core:

- `EXECUTION_ERROR` - General execution failure
- `NETWORK_ERROR` - Network/connection errors
- `DATABASE_ERROR` - Database operation errors
- `INVALID_INPUT` - Invalid input data
- `NOT_FOUND` - Resource not found
- `PERMISSION_DENIED` - Permission errors
- `RESOURCE_EXHAUSTED` - Out of resources (memory, disk, etc.)
- `SYSTEM_ERROR` - OS-level errors
- `TIMEOUT_ERROR` - Operation timeout

## Summary

**Key Takeaways**:

1. ✅ **Always catch specific exceptions first** before generic `Exception`
2. ✅ **Use exception chaining** (`from e`) to preserve tracebacks
3. ✅ **Log at appropriate levels** (ERROR for expected, CRITICAL for unexpected)
4. ✅ **Include error details** in ModelOnexError for debugging
5. ✅ **Test exception paths** to ensure proper handling

**Anti-Patterns to Avoid**:

- ❌ Catching `Exception` as the only handler
- ❌ Swallowing exceptions without re-raising
- ❌ Not using exception chaining (`from e`)
- ❌ Insufficient logging context
- ❌ Not testing exception handling paths

## Related Documentation

- [ONEX v2.0 Architecture Guide](../architecture/ARCHITECTURE.md)
- [Code Generation Guide](./CODE_GENERATION_GUIDE.md)
- [Testing Guide](./TESTING_GUIDE.md)
- [Logging Best Practices](./LOGGING_GUIDE.md)

---

**Version History**:
- 1.0 (2025-11-03): Initial version with comprehensive exception handling patterns
