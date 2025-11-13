# Error Handling Strategy Comparison

## Current Implementation (Fail-Fast)

```python
async def startup(self) -> None:
    """Node startup lifecycle hook."""
    emit_log_event(LogLevel.INFO, f"{self.__class__.__name__} starting up", {...})

    try:
        # Initialize health checks
        await self._initialize_health_checks()
        emit_log_event(LogLevel.INFO, "Health checks initialized", {...})

        # Register with Consul
        if self.container.consul_client:
            await self._register_with_consul()  # ❌ If this fails, entire startup fails
            emit_log_event(LogLevel.INFO, "Consul registration complete", {...})

        # Connect to Kafka
        if self.container.kafka_client:
            await self._connect_kafka()  # ❌ If this fails, entire startup fails
            emit_log_event(LogLevel.INFO, "Kafka connection established", {...})

        emit_log_event(LogLevel.INFO, f"{self.__class__.__name__} startup complete", {...})

    except Exception as e:
        emit_log_event(LogLevel.ERROR, f"Startup failed: {e}", {...})
        await self._cleanup_partial_startup()
        raise  # ❌ Node won't start
```

**Scenario**: Consul is temporarily unavailable (network timeout)
- ❌ **Result**: Node fails to start
- ❌ **Impact**: Service unavailable even though core functionality could work
- ❌ **Recovery**: Manual restart required after Consul is available

## Required Implementation (Graceful Degradation)

```python
async def startup(self) -> None:
    """Node startup lifecycle hook with graceful degradation."""
    emit_log_event(LogLevel.INFO, f"{self.__class__.__name__} starting up", {...})

    # Phase 1: Initialize health checks
    try:
        await self._initialize_health_checks()
        emit_log_event(LogLevel.INFO, "Health checks initialized", {...})
    except Exception as e:
        emit_log_event(
            LogLevel.WARNING,
            f"Health check initialization failed: {e}",
            {"node_id": str(self.node_id), "error": str(e)}
        )  # ⚠️ Warn but continue

    # Phase 2: Register with Consul
    if self.container.consul_client:
        try:
            await self._register_with_consul()
            emit_log_event(LogLevel.INFO, "Consul registration complete", {...})
        except Exception as e:
            emit_log_event(
                LogLevel.WARNING,
                f"Consul registration failed, continuing: {e}",
                {"node_id": str(self.node_id), "error": str(e)}
            )  # ⚠️ Warn but continue

    # Phase 3: Connect to Kafka
    if self.container.kafka_client and not self.container.kafka_client.is_connected:
        try:
            await self.container.kafka_client.connect()
            emit_log_event(LogLevel.INFO, "Kafka connected", {...})
        except Exception as e:
            emit_log_event(
                LogLevel.WARNING,
                f"Kafka connection failed, continuing: {e}",
                {"node_id": str(self.node_id), "error": str(e)}
            )  # ⚠️ Warn but continue

    emit_log_event(
        LogLevel.INFO,
        f"{self.__class__.__name__} startup complete",
        {"node_id": str(self.node_id), "status": "operational"}
    )
```

**Scenario**: Consul is temporarily unavailable (network timeout)
- ✅ **Result**: Node starts successfully with degraded capabilities
- ✅ **Impact**: Core functionality available, just without Consul registration
- ✅ **Recovery**: Consul registration can be retried via background task or manual trigger

## Production Comparison

| Aspect | Fail-Fast | Graceful Degradation |
|--------|-----------|---------------------|
| **Startup success rate** | Lower | Higher |
| **Service availability** | All-or-nothing | Partial operation possible |
| **Operational complexity** | Simpler (clear failure state) | More complex (degraded states) |
| **Resilience** | Lower | Higher |
| **Debugging** | Easier (single failure point) | Harder (multiple partial states) |
| **Best for** | Development, testing | Production |

## Recommendation

**Update to graceful degradation for production resilience.**

### Implementation Changes Required

1. **Update `generate_startup_method()`** (lines 206-414)
   - Wrap each initialization step in individual try/except
   - Change from single outer try/except to per-phase error handling
   - Keep emit_log_event calls but change ERROR → WARNING for non-critical failures

2. **Update `generate_shutdown_method()`** (lines 416-584)
   - Already has some graceful handling
   - Continue on errors (already implemented)
   - Ensure all cleanup steps attempt to run even if some fail

3. **Add configuration for critical vs non-critical**
   - Allow marking certain dependencies as critical
   - Critical failures stop startup
   - Non-critical failures warn but continue

### Example Configuration

```python
startup_code = generate_startup_method(
    node_type="effect",
    dependencies=["consul", "kafka", "postgres"],
    critical_dependencies=["postgres"],  # NEW: Must succeed
    optional_dependencies=["consul", "kafka"]  # NEW: Can fail gracefully
)
```

This would generate:
```python
# Critical: PostgreSQL (must succeed)
try:
    await self._connect_postgres()
except Exception as e:
    emit_log_event(LogLevel.ERROR, f"Critical dependency failed: {e}", {...})
    await self._cleanup_partial_startup()
    raise  # ❌ Stop startup

# Optional: Consul (can fail)
try:
    await self._register_with_consul()
except Exception as e:
    emit_log_event(LogLevel.WARNING, f"Optional dependency failed: {e}", {...})
    # ⚠️ Continue startup
```

## Decision

Choose error handling strategy based on deployment environment:

- **Development/Testing**: Current fail-fast is fine
- **Production**: Graceful degradation is strongly recommended

Would you like me to implement the graceful degradation pattern?
