# Exception Handling Refactoring Guide

**Purpose**: Refactor overly broad `except Exception as e` patterns to catch specific exceptions for better error visibility and debugging.

**PR Context**: PR #36 review identified broad exception handling that masks unexpected errors.

## Completed Refactorings

### ✅ 1. orchestrator/v1_0_0/node.py (3 critical handlers)

#### ConfigLoader Exception Handling (Line ~417)
**Before**:
```python
except Exception as e:
    emit_log_event(LogLevel.WARNING, "ConfigLoader failed")
    return {}
```

**After**:
```python
except ImportError as e:
    # ConfigLoader not available
    emit_log_event(LogLevel.WARNING, "ConfigLoader not available")
    return {}
except FileNotFoundError as e:
    # Configuration file not found
    emit_log_event(LogLevel.WARNING, "Configuration file not found")
    return {}
except (OSError, IOError) as e:
    # File system errors
    emit_log_event(LogLevel.WARNING, "File system error")
    return {}
except (ValueError, KeyError) as e:
    # Validation errors
    emit_log_event(LogLevel.WARNING, "Configuration validation failed")
    return {}
except Exception as e:
    # Unexpected - log as ERROR for visibility
    emit_log_event(LogLevel.ERROR, "Unexpected error")
    return {}
```

#### Hash Generation Exception Handling (Line ~1353)
**Before**:
```python
except Exception as e:
    raise OnexError(code=CoreErrorCode.OPERATION_FAILED, message=f"Hash generation failed: {e}")
```

**After**:
```python
except OnexError:
    # Re-raise OnexError to preserve error context
    raise
except ConnectionError as e:
    # Network connectivity errors
    raise OnexError(code=CoreErrorCode.CONNECTION_ERROR, message=f"Network connection failed: {e}")
except (TimeoutError, asyncio.TimeoutError) as e:
    # Request timeout errors
    raise OnexError(code=CoreErrorCode.TIMEOUT, message=f"Request timed out: {e}")
except (ValueError, KeyError, AttributeError) as e:
    # Data validation/parsing errors
    raise OnexError(code=CoreErrorCode.VALIDATION_ERROR, message=f"Invalid data: {e}")
except Exception as e:
    # Unexpected errors - wrap and re-raise with full context
    raise OnexError(code=CoreErrorCode.OPERATION_FAILED, message=f"Unexpected error: {e}")
```

#### Stamp Creation Exception Handling (Line ~1480)
**Before**:
```python
except Exception as e:
    raise OnexError(code=CoreErrorCode.OPERATION_FAILED, message=f"Stamp creation failed: {e}")
```

**After**:
```python
except (TypeError, ValueError) as e:
    # Data type or value errors
    raise OnexError(code=CoreErrorCode.VALIDATION_ERROR, message=f"Invalid data type: {e}")
except Exception as e:
    # Unexpected errors - log and wrap
    emit_log_event(LogLevel.ERROR, f"Unexpected error: {type(e).__name__}")
    raise OnexError(code=CoreErrorCode.OPERATION_FAILED, message=f"Unexpected error: {e}")
```

### ✅ 2. database_adapter_effect/v1_0_0/node.py (2 critical handlers)

#### Database Health Check (Line ~254)
**Before**:
```python
except Exception as e:
    return (HealthStatus.UNHEALTHY, f"Database health check failed: {e}")
```

**After**:
```python
except ConnectionError as e:
    return (HealthStatus.UNHEALTHY, f"Database connection failed: {e}")
except (TimeoutError, asyncio.TimeoutError) as e:
    return (HealthStatus.DEGRADED, f"Database health check timed out: {e}")
except Exception as e:
    # Unexpected errors - log for debugging
    logger.error(f"Unexpected error: {type(e).__name__}", exc_info=True)
    return (HealthStatus.UNHEALTHY, f"Database health check failed unexpectedly: {e}")
```

#### Database Connectivity Test (Line ~393)
**Before**:
```python
except Exception as e:
    raise OnexError(code=CoreErrorCode.DATABASE_ERROR, message="Database connectivity test failed")
```

**After**:
```python
except OnexError:
    # Re-raise OnexError from circuit breaker
    raise
except ConnectionError as e:
    raise OnexError(code=CoreErrorCode.CONNECTION_ERROR, message="Database connection failed")
except (TimeoutError, asyncio.TimeoutError) as e:
    raise OnexError(code=CoreErrorCode.TIMEOUT, message="Database test timed out")
except Exception as e:
    # Unexpected - log and wrap
    logger.error(f"Unexpected error: {type(e).__name__}", exc_info=True)
    raise OnexError(code=CoreErrorCode.DATABASE_ERROR, message=f"Unexpected error: {e}")
```

### ✅ 3. reducer/v1_0_0/node.py (1 critical handler)

#### ConfigLoader Exception Handling (Line ~626)
Same pattern as orchestrator ConfigLoader refactoring above.

## Refactoring Patterns

### Pattern 1: ConfigLoader / File Operations
**Use for**: Configuration loading, file I/O operations

```python
try:
    # Load configuration
    config = load_config()
except ImportError as e:
    # Module not available
    handle_fallback()
except FileNotFoundError as e:
    # File not found
    handle_fallback()
except (OSError, IOError) as e:
    # File system errors
    handle_fallback()
except (ValueError, KeyError) as e:
    # Validation/parsing errors
    handle_fallback()
except Exception as e:
    # Unexpected - log as ERROR
    logger.error(f"Unexpected: {type(e).__name__}")
    handle_fallback()
```

### Pattern 2: Network / HTTP Operations
**Use for**: HTTP requests, network calls, service routing

```python
try:
    # Make HTTP request
    response = await client.request()
except OnexError:
    # Re-raise existing OnexError
    raise
except ConnectionError as e:
    # Network connectivity
    raise OnexError(code=CoreErrorCode.CONNECTION_ERROR, ...)
except (TimeoutError, asyncio.TimeoutError) as e:
    # Request timeout
    raise OnexError(code=CoreErrorCode.TIMEOUT, ...)
except (ValueError, KeyError, AttributeError) as e:
    # Data validation
    raise OnexError(code=CoreErrorCode.VALIDATION_ERROR, ...)
except Exception as e:
    # Unexpected - wrap and re-raise
    logger.error(f"Unexpected: {type(e).__name__}")
    raise OnexError(code=CoreErrorCode.OPERATION_FAILED, ...)
```

### Pattern 3: Database Operations
**Use for**: Database queries, connection management

```python
try:
    # Execute database operation
    result = await db.execute()
except OnexError:
    # Re-raise (e.g., from circuit breaker)
    raise
except ConnectionError as e:
    # Database connection failed
    raise OnexError(code=CoreErrorCode.CONNECTION_ERROR, ...)
except (TimeoutError, asyncio.TimeoutError) as e:
    # Query timeout
    raise OnexError(code=CoreErrorCode.TIMEOUT, ...)
except Exception as e:
    # Unexpected - log and wrap
    logger.error(f"Unexpected: {type(e).__name__}", exc_info=True)
    raise OnexError(code=CoreErrorCode.DATABASE_ERROR, ...)
```

### Pattern 4: Health Checks / Non-Critical Operations
**Use for**: Health checks, optional features, graceful degradation

```python
try:
    # Check health
    status = await check_service()
except ConnectionError as e:
    return (HealthStatus.UNHEALTHY, f"Connection failed: {e}")
except TimeoutError as e:
    return (HealthStatus.DEGRADED, f"Timeout: {e}")
except Exception as e:
    # Unexpected - log but don't fail
    logger.error(f"Unexpected: {type(e).__name__}", exc_info=True)
    return (HealthStatus.UNHEALTHY, f"Unexpected error: {e}")
```

## Remaining Work

### Files Requiring Refactoring (15 remaining)

From `grep "except Exception as" **/node.py`:

1. ✅ `orchestrator/v1_0_0/node.py` - DONE (3 handlers)
2. ✅ `database_adapter_effect/v1_0_0/node.py` - DONE (2 handlers)
3. ✅ `reducer/v1_0_0/node.py` - DONE (1 handler)
4. ⏳ `deployment_sender_effect/v1_0_0/node.py` - PENDING
5. ⏳ `deployment_receiver_effect/v1_0_0/node.py` - PENDING
6. ⏳ `distributed_lock_effect/v1_0_0/node.py` - PENDING
7. ⏳ `store_effect/v1_0_0/node.py` - PENDING
8. ⏳ `test_generator_effect/v1_0_0/node.py` - PENDING
9. ⏳ `llm_effect/v1_0_0/node.py` - PENDING
10. ⏳ `codegen_orchestrator/v1_0_0/node.py` - PENDING
11. ⏳ `codegen_metrics_reducer/v1_0_0/node.py` - PENDING
12. ⏳ `registry/v1_0_0/node.py` - PENDING
13. ⏳ `vault_secrets_effect/node.py` - PENDING (generated)
14. ⏳ `postgres_crud_effect/*/node.py` (3 files) - PENDING (generated)
15. ⏳ `mysql_adapter_effect/*/node.py` (2 files) - PENDING (generated)

### Prioritization for Remaining Files

**HIGH Priority** (Production-critical paths):
- `deployment_sender_effect` - Deployment operations
- `deployment_receiver_effect` - Deployment operations
- `registry/v1_0_0/node.py` - Service registry
- `llm_effect/v1_0_0/node.py` - LLM operations

**MEDIUM Priority** (Infrastructure):
- `distributed_lock_effect` - Lock management
- `store_effect` - State storage
- `codegen_orchestrator` - Code generation workflows
- `codegen_metrics_reducer` - Metrics aggregation

**LOW Priority** (Testing/Generated):
- `test_generator_effect` - Test generation
- `vault_secrets_effect` - Generated code
- `postgres_crud_effect` - Generated code
- `mysql_adapter_effect` - Generated code

## Validation Checklist

For each refactored file:
- [ ] Specific exceptions caught for known error cases
- [ ] OnexError re-raised without wrapping (preserves error context)
- [ ] Unknown exceptions logged as ERROR or WARNING (visibility)
- [ ] Unknown exceptions re-raised or wrapped (not silently caught)
- [ ] Error messages include error type (`type(e).__name__`)
- [ ] Context preserved in error details/metadata

## Benefits

1. **Better Error Visibility**: Specific exceptions clearly indicate what went wrong
2. **Easier Debugging**: Error types and stack traces preserved
3. **Proper Error Handling**: Different error types handled appropriately
4. **Production Safety**: Unexpected errors logged and raised rather than masked

## Next Steps

1. Apply refactoring patterns to remaining HIGH priority files
2. Test refactored exception handling with unit tests
3. Validate error logging includes proper context
4. Update PR #36 with refactoring results
