> **Navigation**: [Home](../index.md) > Analysis > Circuit Breaker Comparison

# Circuit Breaker Implementation Comparison: omnibase_core vs omnibase_infra

## Executive Summary

**Recommendation**: **Option B - Create a handler/service circuit breaker mixin**

The omnibase_core circuit breaker is deeply integrated into the Node architecture and **cannot be directly reused** by non-node classes like KafkaEventBus. However, we should extract common circuit breaker patterns into a reusable protocol/mixin for handlers and services.

---

## Side-by-Side Comparison

### 1. omnibase_core Circuit Breaker

**Location**: `omnibase_core/src/omnibase_core/models/configuration/model_circuit_breaker.py`

**Type**: Pydantic Model (State Management)

**Purpose**: Load balancing fault tolerance for ONEX nodes

**Integration**: Built into NodeEffect base class

#### Features

| Feature | Implementation | Notes |
|---------|---------------|-------|
| **State Machine** | CLOSED → OPEN → HALF_OPEN | Standard 3-state pattern |
| **State Storage** | Pydantic model fields | Persisted in model state |
| **Thread Safety** | ❌ NOT thread-safe | Documented limitation |
| **Async Support** | ⚠️ Partial | Methods are sync, usage is async |
| **Failure Tracking** | Count + rate based | Dual thresholds |
| **Time Windows** | Sliding window (window_size_seconds) | Configurable cleanup |
| **Recovery** | Half-open with success threshold | Configurable test requests |
| **Slow Call Detection** | ✅ Yes | Optional with threshold |
| **Backoff** | ❌ No | Not implemented |
| **Metrics** | Basic (failure rate, counts) | Limited observability |

#### API Examples

```python
# Node-integrated usage
circuit_breaker = ModelCircuitBreaker(
    failure_threshold=5,
    failure_rate_threshold=0.5,
    timeout_seconds=60,
    window_size_seconds=120,
)

# Check before operation
if circuit_breaker.should_allow_request():
    try:
        result = await perform_operation()
        circuit_breaker.record_success()
    except Exception:
        circuit_breaker.record_failure()
else:
    raise CircuitOpenError()

# Slow call detection
circuit_breaker.record_slow_call(duration_ms=6000)

# Manual control
circuit_breaker.force_open()
circuit_breaker.reset_state()
```

#### Integration Pattern (NodeEffect)

```python
class NodeEffect(NodeCoreBase):
    circuit_breakers: dict[str, ModelCircuitBreaker]

    def _get_circuit_breaker(self, service_name: str) -> ModelCircuitBreaker:
        """Get or create circuit breaker for service."""
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = ModelCircuitBreaker()
        return self.circuit_breakers[service_name]

    async def execute_effect(self, contract):
        if input_data.circuit_breaker_enabled:
            cb = self._get_circuit_breaker(input_data.effect_type.value)
            if not cb.should_allow_request():
                raise CircuitBreakerOpenError()

        try:
            result = await self._execute_with_retry(input_data, transaction)
            if input_data.circuit_breaker_enabled:
                cb.record_success()
        except Exception:
            if input_data.circuit_breaker_enabled:
                cb.record_failure()
            raise
```

**Architecture Notes**:
- Circuit breaker is a **per-service registry** stored in NodeEffect
- State is **instance-specific** (not shared across nodes)
- Integrated with **retry logic** and **transaction management**
- **Not a mixin** - direct model usage

---

### 2. omnibase_infra Circuit Breaker (KafkaEventBus)

**Location**: `omnibase_infra/src/omnibase_infra/event_bus/kafka_event_bus.py`

**Type**: Inline Implementation (State Machine)

**Purpose**: Kafka connection failure protection

**Integration**: Embedded in KafkaEventBus class

#### Features

| Feature | Implementation | Notes |
|---------|---------------|-------|
| **State Machine** | CLOSED → OPEN → HALF_OPEN | Standard 3-state pattern |
| **State Storage** | Instance variables | Not persisted |
| **Thread Safety** | ✅ Thread-safe | Uses asyncio.Lock |
| **Async Support** | ✅ Full async | Async context managers |
| **Failure Tracking** | Count based only | No rate calculation |
| **Time Windows** | None | No sliding window |
| **Recovery** | Time-based auto-reset | Configurable reset timeout |
| **Slow Call Detection** | ❌ No | Not implemented |
| **Backoff** | ❌ No | Not implemented |
| **Metrics** | None | Only state tracking |

#### API Examples

```python
class KafkaEventBus:
    _circuit_state: CircuitState
    _circuit_failure_count: int
    _circuit_last_failure_time: float
    _circuit_breaker_threshold: int
    _circuit_breaker_reset_timeout: float
    _lock: asyncio.Lock

    def _check_circuit_breaker(self, correlation_id: Optional[UUID] = None) -> None:
        """Must be called while holding self._lock"""
        if self._circuit_state == CircuitState.OPEN:
            if time.time() - self._circuit_last_failure_time > self._circuit_breaker_reset_timeout:
                self._circuit_state = CircuitState.HALF_OPEN
                self._circuit_failure_count = 0
            else:
                raise InfraUnavailableError("Circuit breaker is open")

    def _record_circuit_failure(self) -> None:
        """Must be called while holding self._lock"""
        self._circuit_failure_count += 1
        self._circuit_last_failure_time = time.time()
        if self._circuit_failure_count >= self._circuit_breaker_threshold:
            self._circuit_state = CircuitState.OPEN

    def _reset_circuit_breaker(self) -> None:
        """Must be called while holding self._lock"""
        self._circuit_state = CircuitState.CLOSED
        self._circuit_failure_count = 0
```

#### Usage Pattern (KafkaEventBus)

```python
async def publish(self, topic: str, key: Optional[bytes], value: bytes):
    # Thread-safe check (caller must hold lock)
    async with self._lock:
        self._check_circuit_breaker(correlation_id=headers.correlation_id)

    try:
        await self._publish_with_retry(topic, key, value, kafka_headers, headers)

        # Success - reset circuit (thread-safe)
        async with self._lock:
            self._reset_circuit_breaker()
    except Exception:
        # Record failure (thread-safe)
        async with self._lock:
            self._record_circuit_failure()
        raise
```

**Architecture Notes**:
- Circuit breaker is **inline implementation** (not extracted)
- State is **per-instance** (one per KafkaEventBus)
- **Thread-safe** with explicit lock requirements
- Tightly coupled to Kafka infrastructure errors
- **Manual lock management** required by callers

---

## Key Differences

| Aspect | omnibase_core | omnibase_infra (KafkaEventBus) |
|--------|--------------|-------------------------------|
| **Architecture** | Pydantic model with methods | Inline state variables + methods |
| **Reusability** | Node-specific integration | Infrastructure-specific |
| **Thread Safety** | ❌ Not thread-safe | ✅ Thread-safe (with locks) |
| **State Persistence** | Model fields (serializable) | Instance variables (ephemeral) |
| **Failure Detection** | Count + rate + slow calls | Count only |
| **Time Windows** | Sliding window with cleanup | None |
| **Half-Open Testing** | Configurable request limit | Time-based auto-transition |
| **Metrics** | Basic tracking | None |
| **Lock Management** | None (not thread-safe) | Explicit asyncio.Lock |
| **Error Context** | Generic exceptions | Infrastructure-specific errors |

---

## Integration Feasibility Analysis

### Can omnibase_core circuit breaker be used by KafkaEventBus?

**Answer: No - Not directly compatible**

#### Blockers

1. **Thread Safety Incompatibility**
   - omnibase_core: NOT thread-safe (documented limitation)
   - KafkaEventBus: Requires thread-safe operations
   - **Gap**: Would need to add lock management to ModelCircuitBreaker

2. **State Management Approach**
   - omnibase_core: Pydantic model with persistent fields
   - KafkaEventBus: Inline state with explicit locking
   - **Gap**: Lock semantics don't match Pydantic model pattern

3. **Integration Pattern Mismatch**
   - omnibase_core: Designed for NodeEffect registry pattern
   - KafkaEventBus: Single circuit breaker per event bus instance
   - **Gap**: Different instantiation and ownership models

4. **Error Handling**
   - omnibase_core: Raises generic exceptions
   - KafkaEventBus: Raises infrastructure-specific errors (InfraUnavailableError)
   - **Gap**: Error types and context don't align

5. **Lock Management Requirements**
   - omnibase_core: No lock management (caller responsible)
   - KafkaEventBus: Explicit "must be called while holding lock" semantics
   - **Gap**: Would need wrapper with proper lock acquisition

#### What Would Break

```python
# This would NOT work safely:
class KafkaEventBus:
    def __init__(self):
        self._circuit_breaker = ModelCircuitBreaker()  # NOT thread-safe!
        self._lock = asyncio.Lock()

    async def publish(self, topic, key, value):
        # RACE CONDITION - should_allow_request not protected by lock
        if not self._circuit_breaker.should_allow_request():
            raise InfraUnavailableError()  # Wrong error type!

        try:
            await self._publish_with_retry(...)
            self._circuit_breaker.record_success()  # RACE CONDITION!
        except Exception:
            self._circuit_breaker.record_failure()  # RACE CONDITION!
            raise
```

---

## Architectural Recommendations

### Option A: Reuse omnibase_core Mixin (❌ Not Recommended)

**Verdict**: Not feasible without major breaking changes

**Required Changes**:
- Add thread safety to ModelCircuitBreaker
- Add async/await support to all methods
- Redesign state management for lock compatibility
- Break existing NodeEffect integration
- **Impact**: Breaking changes to omnibase_core stable API

### Option B: Create Handler/Service Circuit Breaker Mixin (✅ Recommended)

**Verdict**: Best approach - Extract common pattern

**Strategy**: Create new infrastructure mixin for handlers and services

```python
# New file: omnibase_infra/mixins/mixin_circuit_breaker.py

from typing import Protocol
from uuid import UUID
import asyncio
import time

class ProtocolCircuitBreaker(Protocol):
    """Circuit breaker protocol for infrastructure components."""

    async def check_circuit(self, correlation_id: Optional[UUID] = None) -> None:
        """Check circuit state and raise if open."""
        ...

    async def record_success(self) -> None:
        """Record successful operation."""
        ...

    async def record_failure(self) -> None:
        """Record failed operation."""
        ...

    async def get_circuit_state(self) -> CircuitState:
        """Get current circuit state."""
        ...


class MixinAsyncCircuitBreaker:
    """Thread-safe async circuit breaker mixin for infrastructure components.

    Features:
    - Thread-safe state management with asyncio.Lock
    - Configurable failure thresholds and reset timeouts
    - InfraUnavailableError integration
    - Correlation ID propagation for tracing

    Usage:
        class MyService(MixinAsyncCircuitBreaker):
            def __init__(self):
                self._init_circuit_breaker(
                    threshold=5,
                    reset_timeout=30.0
                )

            async def perform_operation(self):
                await self._check_circuit()
                try:
                    result = await external_call()
                    await self._record_circuit_success()
                    return result
                except Exception:
                    await self._record_circuit_failure()
                    raise
    """

    def _init_circuit_breaker(
        self,
        threshold: int = 5,
        reset_timeout: float = 30.0,
    ) -> None:
        """Initialize circuit breaker state."""
        self._circuit_state = CircuitState.CLOSED
        self._circuit_failure_count = 0
        self._circuit_last_failure_time = 0.0
        self._circuit_threshold = threshold
        self._circuit_reset_timeout = reset_timeout
        self._circuit_lock = asyncio.Lock()

    async def _check_circuit(self, correlation_id: Optional[UUID] = None) -> None:
        """Thread-safe circuit check."""
        async with self._circuit_lock:
            current_time = time.time()

            if self._circuit_state == CircuitState.OPEN:
                if current_time - self._circuit_last_failure_time > self._circuit_reset_timeout:
                    self._circuit_state = CircuitState.HALF_OPEN
                    self._circuit_failure_count = 0
                else:
                    context = ModelInfraErrorContext(
                        transport_type=self._get_transport_type(),
                        operation="circuit_check",
                        target_name=self._get_target_name(),
                        correlation_id=correlation_id or uuid4(),
                    )
                    raise InfraUnavailableError(
                        "Circuit breaker is open - service temporarily unavailable",
                        context=context,
                        circuit_state=self._circuit_state.value,
                    )

    async def _record_circuit_success(self) -> None:
        """Thread-safe success recording."""
        async with self._circuit_lock:
            self._circuit_state = CircuitState.CLOSED
            self._circuit_failure_count = 0

    async def _record_circuit_failure(self) -> None:
        """Thread-safe failure recording."""
        async with self._circuit_lock:
            self._circuit_failure_count += 1
            self._circuit_last_failure_time = time.time()

            if self._circuit_failure_count >= self._circuit_threshold:
                self._circuit_state = CircuitState.OPEN

    def _get_transport_type(self) -> EnumInfraTransportType:
        """Override to provide transport type for error context."""
        return EnumInfraTransportType.HTTP

    def _get_target_name(self) -> str:
        """Override to provide target name for error context."""
        return "unknown"
```

**Migration Path**:

```python
# Before (inline implementation)
class KafkaEventBus:
    def __init__(self):
        self._circuit_state = CircuitState.CLOSED
        self._circuit_failure_count = 0
        self._circuit_last_failure_time = 0.0
        self._circuit_breaker_threshold = 5
        self._circuit_breaker_reset_timeout = 30.0
        self._lock = asyncio.Lock()

    def _check_circuit_breaker(self, correlation_id=None):
        # Manual implementation...
        pass

# After (mixin-based)
class KafkaEventBus(MixinAsyncCircuitBreaker):
    def __init__(self):
        self._init_circuit_breaker(
            threshold=config.circuit_breaker_threshold,
            reset_timeout=config.circuit_breaker_reset_timeout,
        )

    def _get_transport_type(self) -> EnumInfraTransportType:
        return EnumInfraTransportType.KAFKA

    def _get_target_name(self) -> str:
        return f"kafka.{self._environment}"

    async def publish(self, topic, key, value):
        await self._check_circuit(correlation_id=headers.correlation_id)
        try:
            await self._publish_with_retry(...)
            await self._record_circuit_success()
        except Exception:
            await self._record_circuit_failure()
            raise
```

**Benefits**:
- ✅ Thread-safe by design
- ✅ Async-first implementation
- ✅ Infrastructure error integration
- ✅ Correlation ID propagation
- ✅ Reusable across handlers/services
- ✅ No breaking changes to omnibase_core

### Option C: Keep Current Implementation (⚠️ Not Recommended)

**Verdict**: Technical debt accumulation

**Pros**:
- No migration effort
- No breaking changes
- Works for current use case

**Cons**:
- Code duplication across infrastructure components
- No standardization
- Harder to test and maintain
- Pattern drift over time

### Option D: Extract Common Circuit Breaker Protocol (🔄 Alternative)

**Verdict**: Good long-term strategy

**Strategy**: Define protocol/interface, keep both implementations

```python
# omnibase_core/protocols/protocol_circuit_breaker.py
class ProtocolCircuitBreaker(Protocol):
    """Circuit breaker protocol for fault tolerance."""

    def should_allow_request(self) -> bool:
        """Check if request should be allowed."""
        ...

    def record_success(self) -> None:
        """Record successful operation."""
        ...

    def record_failure(self) -> None:
        """Record failed operation."""
        ...

    def get_state(self) -> CircuitState:
        """Get current circuit state."""
        ...


# Both implementations conform to protocol
class ModelCircuitBreaker(BaseModel):
    """Node-specific circuit breaker (sync, not thread-safe)."""

    def should_allow_request(self) -> bool: ...
    def record_success(self) -> None: ...
    def record_failure(self) -> None: ...
    def get_state(self) -> CircuitState: ...


class AsyncCircuitBreaker:
    """Infrastructure circuit breaker (async, thread-safe)."""

    async def should_allow_request(self) -> bool: ...
    async def record_success(self) -> None: ...
    async def record_failure(self) -> None: ...
    async def get_state(self) -> CircuitState: ...
```

**Benefits**:
- ✅ Defines common interface
- ✅ Allows implementation variance
- ✅ Type safety with protocol checking
- ✅ No breaking changes

**Drawbacks**:
- ⚠️ Protocol can't enforce sync vs async
- ⚠️ Still allows implementation drift

---

## Final Recommendation

### Recommended Approach: Option B + Option D Combined

**Phase 1: Create Infrastructure Mixin** (Immediate)
1. Extract KafkaEventBus circuit breaker into `MixinAsyncCircuitBreaker`
2. Add thread safety, async support, and error context integration
3. Migrate KafkaEventBus to use mixin
4. Reuse mixin for other infrastructure components (Consul, Vault, etc.)

**Phase 2: Define Protocol** (Future)
1. Create `ProtocolCircuitBreaker` in omnibase_core
2. Make both implementations conform to protocol
3. Use protocol for type checking and interface validation

**Phase 3: Standardization** (Long-term)
1. Document circuit breaker patterns for different use cases
2. Provide factory methods for common configurations
3. Add comprehensive testing framework

### Migration Plan

#### Step 1: Create Mixin (Week 1)

```bash
# Create new mixin file
touch src/omnibase_infra/mixins/mixin_circuit_breaker.py

# Implement MixinAsyncCircuitBreaker with:
- Thread-safe state management
- Async check/record methods
- Infrastructure error integration
- Configuration support
```

#### Step 2: Update KafkaEventBus (Week 1)

```python
# Before
class KafkaEventBus:
    # Inline circuit breaker implementation (150 lines)
    pass

# After
class KafkaEventBus(MixinAsyncCircuitBreaker):
    def __init__(self, config):
        self._init_circuit_breaker(
            threshold=config.circuit_breaker_threshold,
            reset_timeout=config.circuit_breaker_reset_timeout,
        )

    # Override transport type and target name
    def _get_transport_type(self): return EnumInfraTransportType.KAFKA
    def _get_target_name(self): return f"kafka.{self._environment}"
```

#### Step 3: Testing (Week 2)

```python
# Test thread safety
async def test_circuit_breaker_thread_safety():
    bus = KafkaEventBus.default()

    # Concurrent failure recording
    tasks = [bus._record_circuit_failure() for _ in range(10)]
    await asyncio.gather(*tasks)

    assert bus._circuit_state == CircuitState.OPEN

# Test correlation ID propagation
async def test_correlation_id_propagation():
    bus = KafkaEventBus.default()
    correlation_id = uuid4()

    # Open circuit
    for _ in range(5):
        await bus._record_circuit_failure()

    # Check includes correlation ID in error
    with pytest.raises(InfraUnavailableError) as exc_info:
        await bus._check_circuit(correlation_id=correlation_id)

    assert exc_info.value.model.context.correlation_id == correlation_id
```

#### Step 4: Documentation (Week 2)

```markdown
# Circuit Breaker Usage Guide

## For Infrastructure Components

Use `MixinAsyncCircuitBreaker` for thread-safe async circuit breakers:

- Event buses (Kafka, Redis)
- Service adapters (Consul, Vault)
- HTTP clients
- Database connections

## For ONEX Nodes

Use `ModelCircuitBreaker` for node-integrated circuit breakers:

- NodeEffect for external service calls
- Custom effect nodes
- Orchestrators with external dependencies
```

---

## Code Examples

### Example 1: Using MixinAsyncCircuitBreaker

```python
from omnibase_infra.mixins.mixin_circuit_breaker import MixinAsyncCircuitBreaker

class ConsulAdapter(MixinAsyncCircuitBreaker):
    """Consul service discovery adapter with circuit breaker."""

    def __init__(self, config: ModelConsulConfig):
        self.config = config
        self._init_circuit_breaker(
            threshold=config.circuit_breaker_threshold,
            reset_timeout=config.circuit_breaker_reset_timeout,
        )

    def _get_transport_type(self) -> EnumInfraTransportType:
        return EnumInfraTransportType.CONSUL

    def _get_target_name(self) -> str:
        return f"consul.{self.config.datacenter}"

    async def register_service(
        self,
        service: ModelServiceRegistration,
        correlation_id: Optional[UUID] = None,
    ) -> None:
        """Register service with circuit breaker protection."""
        await self._check_circuit(correlation_id=correlation_id)

        try:
            await self._consul_client.agent.service.register(service.model_dump())
            await self._record_circuit_success()
        except Exception as e:
            await self._record_circuit_failure()
            raise InfraConnectionError(
                f"Failed to register service {service.name}",
                context=ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.CONSUL,
                    operation="register_service",
                    target_name=f"consul.{self.config.datacenter}",
                    correlation_id=correlation_id or uuid4(),
                ),
            ) from e
```

### Example 2: Testing Thread Safety

```python
import asyncio
import pytest
from omnibase_infra.event_bus.kafka_event_bus import KafkaEventBus

@pytest.mark.asyncio
async def test_kafka_circuit_breaker_concurrent_failures():
    """Test circuit breaker handles concurrent failures safely."""
    bus = KafkaEventBus.default()
    await bus.start()

    # Simulate 10 concurrent failures
    async def record_failure():
        async with bus._lock:
            bus._record_circuit_failure()

    tasks = [record_failure() for _ in range(10)]
    await asyncio.gather(*tasks)

    # Circuit should be open (threshold = 5)
    assert bus._circuit_state == CircuitState.OPEN
    assert bus._circuit_failure_count == 10

    await bus.close()


@pytest.mark.asyncio
async def test_kafka_circuit_breaker_correlation_propagation():
    """Test correlation ID propagates through circuit breaker errors."""
    bus = KafkaEventBus.default()
    correlation_id = uuid4()

    # Open circuit
    async with bus._lock:
        for _ in range(5):
            bus._record_circuit_failure()

    # Check includes correlation ID
    with pytest.raises(InfraUnavailableError) as exc_info:
        async with bus._lock:
            bus._check_circuit_breaker(correlation_id=correlation_id)

    error = exc_info.value
    assert error.model.context.correlation_id == correlation_id
    assert error.model.context.transport_type == EnumInfraTransportType.KAFKA
```

---

## Summary

### Decision Matrix

| Option | Feasibility | Thread Safety | Reusability | Breaking Changes | Effort |
|--------|------------|---------------|-------------|------------------|--------|
| A: Reuse omnibase_core | ❌ Low | ❌ No | ⚠️ Limited | ✅ Yes | 🔴 High |
| B: Create Infra Mixin | ✅ High | ✅ Yes | ✅ High | ❌ No | 🟢 Low |
| C: Keep Current | ✅ High | ✅ Yes | ❌ None | ❌ No | 🟢 None |
| D: Extract Protocol | ✅ High | ⚠️ Mixed | ✅ High | ❌ No | 🟡 Medium |

### Final Verdict

**Recommended**: **Option B (Create Infrastructure Mixin)**

**Rationale**:
1. omnibase_core circuit breaker is node-specific and not thread-safe
2. KafkaEventBus requires thread-safe async operations
3. Creating infrastructure mixin provides reusable pattern
4. No breaking changes to existing code
5. Enables future standardization across infrastructure components

**Action Items**:
1. Create `MixinAsyncCircuitBreaker` in `src/omnibase_infra/mixins/`
2. Migrate KafkaEventBus to use mixin
3. Add comprehensive tests for thread safety
4. Document usage patterns for infrastructure components
5. Apply mixin to other adapters (Consul, Vault, etc.)

**Success Criteria**:
- ✅ All circuit breaker tests pass
- ✅ Thread safety verified with concurrent tests
- ✅ No breaking changes to KafkaEventBus API
- ✅ Mixin reused by at least 2 infrastructure components
- ✅ Documentation complete with examples
