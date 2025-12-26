# Dispatcher Resilience Pattern

## Overview

This document describes the resilience pattern for message dispatchers in the ONEX infrastructure. The key principle is that **dispatchers own their own resilience** - the `MessageDispatchEngine` does NOT wrap dispatchers with circuit breakers or other resilience patterns.

## Design Rationale

| Principle | Explanation |
|-----------|-------------|
| **Separation of concerns** | Each dispatcher knows its specific failure modes and recovery strategies |
| **Transport-specific tuning** | Kafka dispatchers need different thresholds than HTTP dispatchers |
| **No hidden behavior** | Engine users see exactly what resilience each dispatcher provides |
| **Composability** | Dispatchers can combine circuit breakers with retry, backoff, or degradation |

## Dispatcher Implementation Pattern

### Basic Structure

```python
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.mixins import MixinAsyncCircuitBreaker
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.models.dispatch import ModelDispatchResult
from omnibase_infra.runtime import ProtocolMessageDispatcher


class MyDispatcher(MixinAsyncCircuitBreaker, ProtocolMessageDispatcher):
    """Dispatcher with built-in circuit breaker resilience."""

    def __init__(self, config: DispatcherConfig):
        self._init_circuit_breaker(
            threshold=config.failure_threshold,
            reset_timeout=config.reset_timeout_seconds,
            service_name=f"dispatcher.{config.target_service}",
            transport_type=config.transport_type,
        )

    # NOTE: ModelEventEnvelope[object] is used instead of Any to satisfy ONEX "no Any types"
    # rule. Dispatchers must accept envelopes with any payload type since the dispatch
    # engine routes based on topic/category, not payload shape. Using `object` provides
    # the same flexibility as Any but with explicit semantics.
    async def handle(self, envelope: ModelEventEnvelope[object]) -> ModelDispatchResult:
        """Handle message with circuit breaker protection."""
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("handle", envelope.correlation_id)

        try:
            result = await self._do_handle(envelope)
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()
            return result
        except Exception as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("handle", envelope.correlation_id)
            raise
```

## Transport-Specific Configuration

### Kafka Dispatcher

```python
class KafkaDispatcher(MixinAsyncCircuitBreaker, ProtocolMessageDispatcher):
    """Dispatcher for Kafka message publishing."""

    def __init__(self, kafka_producer: Any, topic: str):
        self._producer = kafka_producer
        self._topic = topic
        self._init_circuit_breaker(
            threshold=3,           # Open after 3 failures (Kafka is critical)
            reset_timeout=20.0,    # 20 seconds recovery
            service_name=f"kafka.{topic}",
            transport_type=EnumInfraTransportType.KAFKA,
        )
```

### HTTP Dispatcher

```python
class HttpDispatcher(MixinAsyncCircuitBreaker, ProtocolMessageDispatcher):
    """Dispatcher for HTTP webhook calls."""

    def __init__(self, http_client: Any, endpoint: str):
        self._client = http_client
        self._endpoint = endpoint
        self._init_circuit_breaker(
            threshold=5,           # More lenient for HTTP
            reset_timeout=60.0,    # 1 minute recovery
            service_name=f"http.{endpoint}",
            transport_type=EnumInfraTransportType.HTTP,
        )
```

### Database Dispatcher

```python
class DatabaseDispatcher(MixinAsyncCircuitBreaker, ProtocolMessageDispatcher):
    """Dispatcher for database event storage."""

    def __init__(self, db_pool: Any):
        self._pool = db_pool
        self._init_circuit_breaker(
            threshold=5,           # Standard threshold
            reset_timeout=30.0,    # 30 seconds recovery
            service_name="postgres.events",
            transport_type=EnumInfraTransportType.DATABASE,
        )
```

## What the Engine Does NOT Do

The `MessageDispatchEngine` intentionally avoids:

- Wrapping dispatcher calls with circuit breakers
- Implementing retry logic around dispatchers
- Catching and suppressing dispatcher errors (except for error aggregation)

## Dispatcher Responsibilities

Each dispatcher implementation SHOULD:

1. **Implement `MixinAsyncCircuitBreaker`** for external service calls
2. **Configure thresholds** appropriate to their transport type
3. **Raise `InfraUnavailableError`** when circuit opens (engine will capture this)
4. **Optionally combine with retry logic** for transient failures

## Combining with Retry Logic

```python
class ResilientDispatcher(MixinAsyncCircuitBreaker, ProtocolMessageDispatcher):
    """Dispatcher with circuit breaker AND retry logic."""

    async def handle(self, envelope: ModelEventEnvelope[object]) -> ModelDispatchResult:
        """Handle with retry before circuit breaker trips."""
        max_retries = 3

        for attempt in range(max_retries):
            async with self._circuit_breaker_lock:
                try:
                    await self._check_circuit_breaker("handle", envelope.correlation_id)
                except InfraUnavailableError:
                    # Circuit open - fail fast, no retry
                    raise

            try:
                result = await self._do_handle(envelope)
                async with self._circuit_breaker_lock:
                    await self._reset_circuit_breaker()
                return result

            except Exception as e:
                async with self._circuit_breaker_lock:
                    await self._record_circuit_failure("handle", envelope.correlation_id)

                if attempt == max_retries - 1:
                    raise

                # Exponential backoff before retry
                await asyncio.sleep(2 ** attempt)
```

## Configuration Guidelines

| Transport Type | Failure Threshold | Reset Timeout | Rationale |
|----------------|-------------------|---------------|-----------|
| **KAFKA** | 3 | 20s | Critical messaging, fast recovery |
| **DATABASE** | 5 | 30s | Standard durability requirements |
| **HTTP** | 5-10 | 60s | External dependencies more variable |
| **CONSUL** | 3 | 30s | Service discovery is critical |
| **VAULT** | 3 | 60s | Security service, longer recovery |
| **VALKEY** | 10 | 30s | Cache, can tolerate more failures |

## Error Handling

When a dispatcher's circuit breaker opens:

```python
try:
    await dispatcher.handle(envelope)
except InfraUnavailableError as e:
    # Circuit is open - service unavailable
    # Log and potentially route to DLQ
    logger.warning(
        "Dispatcher circuit open",
        dispatcher=dispatcher.name,
        correlation_id=envelope.correlation_id,
    )
    await dlq_handler.send(envelope, reason=str(e))
```

## Testing Dispatchers

```python
import pytest
from unittest.mock import AsyncMock


@pytest.mark.asyncio
async def test_dispatcher_opens_circuit_after_failures():
    """Test circuit breaker opens after threshold failures."""
    dispatcher = MyDispatcher(config)

    # Simulate failures up to threshold
    for _ in range(config.failure_threshold):
        with pytest.raises(SomeError):
            await dispatcher.handle(envelope)

    # Next call should raise InfraUnavailableError (circuit open)
    with pytest.raises(InfraUnavailableError) as exc_info:
        await dispatcher.handle(envelope)

    assert "circuit" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_dispatcher_recovers_after_timeout():
    """Test circuit breaker recovers after reset timeout."""
    dispatcher = MyDispatcher(config)

    # Open the circuit
    for _ in range(config.failure_threshold):
        with pytest.raises(SomeError):
            await dispatcher.handle(envelope)

    # Wait for reset timeout
    await asyncio.sleep(config.reset_timeout + 0.1)

    # Should allow a test call (HALF_OPEN state)
    # If successful, circuit closes
    await dispatcher.handle(envelope)  # Should succeed
```

## Related Patterns

- [Circuit Breaker Implementation](./circuit_breaker_implementation.md) - Core circuit breaker details
- [Error Recovery Patterns](./error_recovery_patterns.md) - Retry and backoff strategies
- [Async Thread Safety Pattern](./async_thread_safety_pattern.md) - Lock patterns for async code

## See Also

- `src/omnibase_infra/runtime/message_dispatch_engine.py` - Engine implementation
- `src/omnibase_infra/mixins/mixin_async_circuit_breaker.py` - Circuit breaker mixin
