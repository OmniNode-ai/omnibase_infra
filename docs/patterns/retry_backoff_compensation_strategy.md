# Retry, Backoff, and Compensation Strategy

## Overview

This document specifies the formal retry, backoff, and compensation strategies for ONEX infrastructure effects. It provides authoritative guidance for handling transient failures in external I/O operations.

**Scope**: Effect nodes that interact with external systems (database, Kafka, HTTP, Vault, Consul, etc.)

**Related Documents**:
- [Error Recovery Patterns](./error_recovery_patterns.md) - Implementation examples
- [Circuit Breaker Implementation](./circuit_breaker_implementation.md) - Circuit breaker details
- [Error Handling Patterns](./error_handling_patterns.md) - Error classification

---

## Architectural Responsibility: Orchestrator-Owned Retries

### Key Principle: Effects are Single-Shot Operations

**Effect nodes do NOT implement retries.** Retry logic is the exclusive responsibility of the orchestrator layer.

This is an ONEX architectural principle that ensures:
1. **Separation of concerns**: Effects handle I/O, orchestrators handle workflow coordination
2. **Testability**: Effects can be tested in isolation without retry complexity
3. **Composability**: Different orchestrators can apply different retry strategies to the same effect
4. **Observability**: Retry metrics and decisions are centralized in orchestrators

### Layer Responsibilities

| Layer | Retry Responsibility | Example |
|-------|---------------------|---------|
| **Effect Node** | **NONE** - Single-shot execution only | `NodeRegistryEffect` executes once, returns success/failure |
| **Orchestrator Node** | **FULL** - Retry policies, backoff, circuit breakers | `NodeRegistrationOrchestrator` retries failed effect calls |
| **Handler** | **NONE** - Delegates to effect | Handlers invoke effects, do not retry |

### Implementation Implications

**Effect Nodes**:
- Execute a single operation attempt
- Return structured result (`ModelBackendResult`) with success/failure status
- Include timing information (`duration_ms`) for orchestrator decisions
- Do NOT track retry counts or implement backoff
- The `retries` field was intentionally removed from effect result models (see OMN-1103)

**Orchestrator Nodes**:
- Implement retry policies via `coordination_rules.max_retries` in contract
- Configure per-error retry behavior via `error_handling.retry_policy`
- Track retry state and apply exponential backoff
- Make circuit breaker decisions based on accumulated failures

### Configuration Location

```yaml
# In orchestrator contract.yaml (NOT effect contract.yaml)
coordination_rules:
  max_retries: 3  # Workflow step retry count

error_handling:
  retry_policy:
    max_retries: 3  # Per-error retry count
    initial_backoff_seconds: 1.0
    max_backoff_seconds: 60.0
    exponential_base: 2.0
```

### Why Effects Don't Retry

1. **Single Responsibility**: Effects perform I/O - that's it
2. **Strategy Flexibility**: Orchestrators can choose retry strategy based on context
3. **Resource Efficiency**: Prevents nested retry loops (effect retries * orchestrator retries)
4. **Failure Visibility**: Orchestrators see every failure, enabling smart decisions
5. **Testing Simplicity**: Effects are deterministic - call once, get result

### Example Flow

```
Orchestrator (owns retry logic)
    │
    ├─[Attempt 1]─> Effect.execute() -> FAILURE (timeout)
    │               └─> Returns ModelBackendResult(success=False, error="timeout")
    │
    ├─[Backoff 1s]
    │
    ├─[Attempt 2]─> Effect.execute() -> FAILURE (connection refused)
    │               └─> Returns ModelBackendResult(success=False, error="connection refused")
    │
    ├─[Backoff 2s]
    │
    └─[Attempt 3]─> Effect.execute() -> SUCCESS
                    └─> Returns ModelBackendResult(success=True, duration_ms=45.2)
```

---

## Retry Policy

### Default Configuration

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `max_attempts` | 3 | 1-10 | Maximum retry attempts (inclusive of initial attempt) |
| `initial_delay_seconds` | 1.0 | 0.01-60.0 | Delay before first retry |
| `max_delay_seconds` | 60.0 | 1.0-300.0 | Maximum delay cap |
| `exponential_base` | 2.0 | 1.5-4.0 | Multiplier for exponential backoff |
| `jitter_factor` | 0.25 | 0.0-0.5 | Random jitter range (+/- percentage) |

### Retryable vs Non-Retryable Errors

**Retryable Errors** (transient, worth retrying):

| Error Type | Error Class | Rationale |
|------------|-------------|-----------|
| Connection refused | `InfraConnectionError` | Service may be starting up |
| Connection timeout | `InfraTimeoutError` | Network congestion may clear |
| Rate limited (429) | `InfraConnectionError` | Will succeed after backoff |
| Service temporarily unavailable (503) | `InfraUnavailableError` | Service may recover |
| Network partition (transient) | `InfraConnectionError` | Network may heal |
| Broker not available (Kafka) | `InfraConnectionError` | Kafka leader election |
| Lock contention | `InfraConnectionError` | Lock may be released |
| Consul session expired | `InfraConnectionError` | Session can be renewed |

**Non-Retryable Errors** (permanent, do NOT retry):

| Error Type | Error Class | Rationale |
|------------|-------------|-----------|
| Authentication failed (401) | `InfraAuthenticationError` | Credentials are invalid |
| Authorization denied (403) | `InfraAuthenticationError` | Permissions will not change |
| Resource not found (404) | `SecretResolutionError` | Resource does not exist |
| Validation error (400) | `ProtocolConfigurationError` | Input is malformed |
| Data integrity violation | `InfraConnectionError` | Constraint will not resolve |
| Schema mismatch | `ProtocolConfigurationError` | Requires code change |
| Invalid configuration | `ProtocolConfigurationError` | Configuration is wrong |
| Encryption/decryption failure | `InfraAuthenticationError` | Key mismatch |

### Retry Decision Logic

```python
from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    InfraUnavailableError,
    InfraAuthenticationError,
    ProtocolConfigurationError,
    SecretResolutionError,
)

def is_retryable(error: Exception) -> bool:
    """Determine if an error is worth retrying.

    Args:
        error: The exception that occurred

    Returns:
        True if the error is transient and worth retrying
    """
    # Non-retryable error types (permanent failures)
    if isinstance(error, (
        InfraAuthenticationError,  # Auth failures need credential refresh, not retry
        ProtocolConfigurationError,  # Config errors need code changes
        SecretResolutionError,  # Missing secrets need configuration
    )):
        return False

    # Retryable error types (transient failures)
    if isinstance(error, (
        InfraConnectionError,
        InfraTimeoutError,
        InfraUnavailableError,
    )):
        return True

    # Unknown errors - default to non-retryable for safety
    return False
```

---

## Backoff Strategy

### Exponential Backoff Formula

```
delay = min(initial_delay * (exponential_base ^ attempt), max_delay)
jittered_delay = delay * (1.0 + random.uniform(-jitter_factor, +jitter_factor))
```

**Parameters**:
- `initial_delay`: Base delay (default: 1.0 seconds)
- `exponential_base`: Multiplier per attempt (default: 2.0)
- `attempt`: Zero-indexed attempt number
- `max_delay`: Ceiling for delay (default: 60.0 seconds)
- `jitter_factor`: Random variance range (default: +/-25%)

### Backoff Sequence Examples

**Default configuration** (base=2.0, initial=1.0s, max=60.0s):

| Attempt | Base Delay | With Jitter (+/-25%) |
|---------|------------|----------------------|
| 0 | 1.0s | 0.75s - 1.25s |
| 1 | 2.0s | 1.50s - 2.50s |
| 2 | 4.0s | 3.00s - 5.00s |
| 3 | 8.0s | 6.00s - 10.00s |
| 4 | 16.0s | 12.00s - 20.00s |
| 5 | 32.0s | 24.00s - 40.00s |
| 6+ | 60.0s (capped) | 45.00s - 75.00s |

**Aggressive configuration** (base=3.0, initial=0.1s, max=30.0s):

| Attempt | Base Delay | With Jitter (+/-25%) |
|---------|------------|----------------------|
| 0 | 0.1s | 0.075s - 0.125s |
| 1 | 0.3s | 0.225s - 0.375s |
| 2 | 0.9s | 0.675s - 1.125s |
| 3 | 2.7s | 2.025s - 3.375s |
| 4 | 8.1s | 6.075s - 10.125s |
| 5 | 24.3s | 18.225s - 30.375s |
| 6+ | 30.0s (capped) | 22.50s - 37.50s |

### Jitter Specification

**Purpose**: Prevent thundering herd when multiple clients retry simultaneously.

**Formula**: `jittered_delay = delay * (1.0 + random.uniform(-0.25, +0.25))`

**Rationale**:
- +/-25% provides sufficient spread to decorrelate retries
- Preserves the exponential growth pattern
- Prevents synchronized retry storms during outages
- Small enough to maintain predictable behavior

**Implementation**:

```python
import random

def calculate_backoff_delay(
    attempt: int,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter_factor: float = 0.25,
) -> float:
    """Calculate backoff delay with jitter.

    Args:
        attempt: Zero-indexed attempt number
        initial_delay: Base delay in seconds
        max_delay: Maximum delay cap in seconds
        exponential_base: Multiplier per attempt
        jitter_factor: Random variance range (0.25 = +/-25%)

    Returns:
        Delay in seconds with jitter applied
    """
    # Calculate base exponential delay
    base_delay = initial_delay * (exponential_base ** attempt)

    # Apply ceiling
    capped_delay = min(base_delay, max_delay)

    # Apply jitter
    jitter = random.uniform(-jitter_factor, jitter_factor)
    jittered_delay = capped_delay * (1.0 + jitter)

    return jittered_delay
```

---

## Compensation Strategy for Partial Failures

### Scope and Non-Goals

**Compensation applies ONLY to external side effects (I/O operations)**.

When an effect node partially succeeds (e.g., wrote to database but Kafka publish failed), compensation addresses the external state inconsistency.

**CRITICAL NON-GOAL: Compensation does NOT imply rollback of emitted domain events.**

Domain events are **immutable facts** that represent what happened in the system. They cannot and should not be "undone". Instead, compensation creates **compensating events** that represent corrective actions.

### Domain Event Immutability Principle

```
# WRONG - Never "rollback" domain events
UserCreatedEvent -> (failure) -> DELETE UserCreatedEvent  # NEVER DO THIS

# CORRECT - Emit compensating event
UserCreatedEvent -> (failure) -> UserCreationFailedEvent  # This is correct
OrderPlacedEvent -> (failure) -> OrderCancelledEvent      # This is correct
```

**Rationale**:
1. **Auditability**: Complete history must be preserved for compliance
2. **Event sourcing**: Downstream systems may have already consumed events
3. **Idempotency**: Compensating events can be replayed safely
4. **Debugging**: Understanding what happened requires full event history

### Compensation Patterns

#### Pattern 1: Compensating Event

When partial failure occurs, emit a compensating event that reverses the logical effect:

```python
from pydantic import BaseModel, EmailStr
from uuid import UUID, uuid4

from omnibase_core.models.events import ModelEventEnvelope
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import InfraConnectionError, ModelInfraErrorContext


class ModelUserData(BaseModel):
    """Input data for user creation."""

    email: EmailStr
    username: str
    display_name: str | None = None


class UserCreatedEvent(BaseModel):
    user_id: UUID
    email: str
    created_at: str


class UserCreationFailedEvent(BaseModel):
    user_id: UUID
    original_event_id: UUID
    failure_reason: str
    failed_at: str


async def create_user_with_compensation(
    user_data: ModelUserData,
    db_client: DatabaseClient,
    event_bus: EventBus,
    correlation_id: UUID,
) -> None:
    """Create user with compensation on partial failure."""
    user_id = uuid4()

    # Step 1: Write to database
    await db_client.insert_user(user_id, user_data)

    # Step 2: Create domain event
    created_event = UserCreatedEvent(
        user_id=user_id,
        email=user_data.email,
        created_at=datetime.utcnow().isoformat(),
    )

    # Create error context for Kafka operations (per ONEX error handling patterns)
    kafka_error_context = ModelInfraErrorContext(
        transport_type=EnumInfraTransportType.KAFKA,
        operation="publish_event",
        target_name="kafka-broker",
        correlation_id=correlation_id,
    )

    try:
        # Step 3: Publish event to Kafka
        await event_bus.publish(
            topic="user.created.v1",
            event=created_event,
            correlation_id=correlation_id,
        )
    except InfraConnectionError as e:
        # Kafka failed - log with full error context and initiate compensation
        # The error already contains context from the event bus,
        # but we log additional context for debugging
        logger.warning(
            "Kafka publish failed, initiating compensation",
            extra={
                "user_id": str(user_id),
                "correlation_id": str(correlation_id),
                "transport_type": kafka_error_context.transport_type.value,
                "operation": kafka_error_context.operation,
                "target_name": kafka_error_context.target_name,
                "error": str(e),
            },
        )

        failed_event = UserCreationFailedEvent(
            user_id=user_id,
            original_event_id=correlation_id,
            failure_reason=str(e),
            failed_at=datetime.utcnow().isoformat(),
        )

        # Attempt to publish compensation (best effort)
        # If this also fails, it will be in DLQ or logged for manual intervention
        try:
            await event_bus.publish(
                topic="user.creation-failed.v1",
                event=failed_event,
                correlation_id=correlation_id,
            )
        except Exception:
            # Log for manual intervention with full context
            logger.error(
                "Failed to publish compensation event",
                extra={
                    "user_id": str(user_id),
                    "correlation_id": str(correlation_id),
                    "transport_type": kafka_error_context.transport_type.value,
                    "operation": "publish_compensation_event",
                    "target_name": kafka_error_context.target_name,
                },
            )

        # Rollback database write (external side effect compensation)
        await db_client.delete_user(user_id)

        raise
```

#### Pattern 2: Saga with Compensation Steps

For multi-step workflows, define explicit compensation for each step:

```python
from dataclasses import dataclass
from typing import Callable, Awaitable
from uuid import UUID


@dataclass
class SagaStep:
    """A step in a saga with its compensation action."""
    name: str
    action: Callable[[], Awaitable[None]]
    compensate: Callable[[], Awaitable[None]]


class CompensationFailedError(Exception):
    """Raised when one or more saga compensations fail.

    This error aggregates all compensation failures that occurred during
    saga rollback, providing visibility into which steps failed to compensate
    and why. This is critical for operational awareness when the system is
    left in an inconsistent state.
    """

    def __init__(
        self,
        message: str,
        failed_steps: list[str],
        errors: list[Exception],
    ):
        super().__init__(message)
        self.failed_steps = failed_steps
        self.errors = errors


class SagaExecutor:
    """Execute saga with automatic compensation on failure."""

    async def execute(
        self,
        steps: list[SagaStep],
        correlation_id: UUID,
    ) -> None:
        """Execute saga steps with compensation on failure.

        Args:
            steps: Ordered list of saga steps
            correlation_id: Request correlation ID

        Raises:
            CompensationFailedError: If any compensation steps fail during rollback.
                The original exception is preserved via exception chaining.
            Exception: The original exception if all compensations succeed.
        """
        completed_steps: list[SagaStep] = []

        try:
            for step in steps:
                await step.action()
                completed_steps.append(step)

        except Exception as e:
            # Compensate in reverse order, collecting errors
            compensation_errors: list[tuple[str, Exception]] = []

            for step in reversed(completed_steps):
                try:
                    await step.compensate()
                except Exception as comp_error:
                    compensation_errors.append((step.name, comp_error))
                    logger.error(
                        "Compensation failed for step %s",
                        step.name,
                        extra={
                            "correlation_id": str(correlation_id),
                            "error": str(comp_error),
                        },
                    )

            # If any compensations failed, raise aggregate error
            if compensation_errors:
                raise CompensationFailedError(
                    f"Saga failed and {len(compensation_errors)} compensation(s) also failed",
                    failed_steps=[name for name, _ in compensation_errors],
                    errors=[err for _, err in compensation_errors],
                ) from e

            raise


# Usage example
async def transfer_funds(
    from_account: str,
    to_account: str,
    amount: float,
    correlation_id: UUID,
) -> None:
    """Transfer funds between accounts with saga compensation."""
    saga = SagaExecutor()

    debit_result = None

    async def debit_source():
        nonlocal debit_result
        debit_result = await accounts_service.debit(from_account, amount)

    async def credit_destination():
        await accounts_service.credit(to_account, amount)

    async def compensate_debit():
        # Reverse the debit (credit back to source)
        await accounts_service.credit(from_account, amount)

    async def compensate_credit():
        # Reverse the credit (debit from destination)
        await accounts_service.debit(to_account, amount)

    steps = [
        SagaStep(
            name="debit_source",
            action=debit_source,
            compensate=compensate_debit,
        ),
        SagaStep(
            name="credit_destination",
            action=credit_destination,
            compensate=compensate_credit,
        ),
    ]

    await saga.execute(steps, correlation_id)
```

#### Pattern 3: Outbox Pattern (Preferred for Transactional Consistency)

For atomic consistency between database and event publishing:

```python
from uuid import UUID, uuid4
from datetime import datetime


class OutboxEntry(BaseModel):
    """Entry in the outbox table for reliable event publishing."""

    id: UUID
    aggregate_type: str
    aggregate_id: UUID
    event_type: str
    payload: BaseModel  # Accepts any Pydantic model for type safety
    created_at: datetime
    published_at: datetime | None = None
    retry_count: int = 0


class ModelUserCreatedPayload(BaseModel):
    """Payload for user created outbox event."""

    user_id: str
    email: str
    created_at: str


async def create_user_with_outbox(
    user_data: ModelUserData,
    db_client: DatabaseClient,
    correlation_id: UUID,
) -> UUID:
    """Create user with outbox pattern for reliable event publishing.

    The outbox pattern ensures atomic consistency:
    1. User creation and outbox entry are in same transaction
    2. Background worker publishes events from outbox
    3. Events are guaranteed to be published exactly once
    """
    user_id = uuid4()

    # Single atomic transaction for user + outbox entry
    async with db_client.transaction():
        # Create user
        await db_client.insert_user(user_id, user_data)

        # Create outbox entry (same transaction)
        outbox_entry = OutboxEntry(
            id=uuid4(),
            aggregate_type="User",
            aggregate_id=user_id,
            event_type="user.created.v1",
            payload=ModelUserCreatedPayload(
                user_id=str(user_id),
                email=user_data.email,
                created_at=datetime.utcnow().isoformat(),
            ),
            created_at=datetime.utcnow(),
        )
        await db_client.insert_outbox(outbox_entry)

    # Transaction committed - user and outbox entry are consistent
    # Background worker will publish event from outbox
    return user_id


# Background worker for outbox processing
async def outbox_publisher_worker(
    db_client: DatabaseClient,
    event_bus: EventBus,
) -> None:
    """Background worker that publishes events from outbox."""
    while True:
        # Fetch unpublished entries
        entries = await db_client.get_unpublished_outbox_entries(limit=100)

        for entry in entries:
            try:
                await event_bus.publish(
                    topic=entry.event_type,
                    event=entry.payload,
                    correlation_id=entry.id,
                )

                # Mark as published
                await db_client.mark_outbox_published(entry.id)

            except Exception as e:
                # Increment retry count
                await db_client.increment_outbox_retry(entry.id)

                # After max retries, move to DLQ
                if entry.retry_count >= 3:
                    await db_client.move_outbox_to_dlq(entry.id)

        await asyncio.sleep(1.0)  # Poll interval
```

---

## Per-Effect Type Configuration

Different effect types have different retry characteristics based on their failure modes and recovery patterns.

### Database (PostgreSQL)

```python
from pydantic import BaseModel, Field


class ModelDatabaseRetryConfig(BaseModel):
    """Retry configuration for database operations."""

    max_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum retry attempts for connection failures",
    )
    initial_delay_seconds: float = Field(
        default=0.5,
        ge=0.01,
        le=10.0,
        description="Initial backoff delay",
    )
    max_delay_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=120.0,
        description="Maximum backoff delay",
    )
    exponential_base: float = Field(
        default=2.0,
        ge=1.5,
        le=4.0,
        description="Exponential backoff multiplier",
    )
    connection_timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=60.0,
        description="Timeout for establishing connection",
    )
    query_timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Timeout for query execution",
    )
```

**Rationale**:
- Lower initial delay (0.5s) - database connections typically recover quickly
- Moderate max delay (30s) - don't wait too long for database
- Lower max attempts (3) - database issues often need intervention

### Kafka

```python
class ModelKafkaRetryConfig(BaseModel):
    """Retry configuration for Kafka operations."""

    max_attempts: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum retry attempts for publish failures",
    )
    initial_delay_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=30.0,
        description="Initial backoff delay",
    )
    max_delay_seconds: float = Field(
        default=60.0,
        ge=1.0,
        le=300.0,
        description="Maximum backoff delay",
    )
    exponential_base: float = Field(
        default=2.0,
        ge=1.5,
        le=4.0,
        description="Exponential backoff multiplier",
    )
    producer_timeout_ms: int = Field(
        default=30000,
        ge=1000,
        le=300000,
        description="Producer request timeout in milliseconds",
    )
    delivery_timeout_ms: int = Field(
        default=120000,
        ge=10000,
        le=600000,
        description="Total time for message delivery including retries",
    )
```

**Rationale**:
- Higher max attempts (5) - Kafka leader elections take time
- Higher max delay (60s) - broker recovery can be slow
- Long delivery timeout - Kafka handles internal retries

### HTTP/REST API

```python
class ModelHttpRetryConfig(BaseModel):
    """Retry configuration for HTTP operations."""

    max_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum retry attempts for transient failures",
    )
    initial_delay_seconds: float = Field(
        default=0.5,
        ge=0.1,
        le=10.0,
        description="Initial backoff delay",
    )
    max_delay_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=120.0,
        description="Maximum backoff delay",
    )
    exponential_base: float = Field(
        default=2.0,
        ge=1.5,
        le=4.0,
        description="Exponential backoff multiplier",
    )
    connect_timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=60.0,
        description="Timeout for establishing connection",
    )
    read_timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Timeout for receiving response",
    )
    retry_on_status_codes: list[int] = Field(
        default=[429, 500, 502, 503, 504],
        description="HTTP status codes that trigger retry",
    )
```

**Rationale**:
- Moderate settings - HTTP APIs vary widely
- Status code filtering - only retry transient failures
- Separate connect/read timeouts - different failure modes

### Vault (Secret Management)

Reference implementation: `ModelVaultRetryConfig` in `src/omnibase_infra/handlers/model_vault_retry_config.py`

```python
class ModelVaultRetryConfig(BaseModel):
    """Configuration for Vault operation retry logic with exponential backoff."""

    max_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum retry attempts for failed operations",
    )
    initial_backoff_seconds: float = Field(
        default=0.1,
        ge=0.01,
        le=10.0,
        description="Initial delay before first retry",
    )
    max_backoff_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=60.0,
        description="Maximum delay cap for exponential backoff",
    )
    exponential_base: float = Field(
        default=2.0,
        ge=1.5,
        le=4.0,
        description="Base multiplier for exponential backoff calculation",
    )
```

**Rationale**:
- Fast initial delay (0.1s) - Vault is usually fast when available
- Lower max delay (10s) - secrets are critical, fail fast if unavailable
- Lower max attempts (3) - Vault issues need investigation

### Consul (Service Discovery)

Reference implementation: `ModelConsulRetryConfig` in `src/omnibase_infra/handlers/model_consul_retry_config.py`

```python
class ModelConsulRetryConfig(BaseModel):
    """Configuration for Consul operation retry logic with exponential backoff."""

    max_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum retry attempts for failed operations",
    )
    initial_delay_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="Initial delay before first retry",
    )
    max_delay_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Maximum delay cap for exponential backoff",
    )
    exponential_base: float = Field(
        default=2.0,
        ge=1.5,
        le=4.0,
        description="Base multiplier for exponential backoff calculation",
    )
```

**Rationale**:
- Moderate initial delay (1.0s) - Consul cluster elections take time
- Higher max delay (30s) - service discovery can wait
- Moderate max attempts (3) - balance availability vs latency

### Configuration Comparison Matrix

| Effect Type | Max Attempts | Initial Delay | Max Delay | Exponential Base |
|-------------|-------------|---------------|-----------|------------------|
| Database | 3 | 0.5s | 30.0s | 2.0 |
| Kafka | 5 | 1.0s | 60.0s | 2.0 |
| HTTP | 3 | 0.5s | 30.0s | 2.0 |
| Vault | 3 | 0.1s | 10.0s | 2.0 |
| Consul | 3 | 1.0s | 30.0s | 2.0 |

---

## Integration with Circuit Breaker

Retry logic works **inside** the circuit breaker boundary:

```python
from omnibase_infra.mixins import MixinAsyncCircuitBreaker
from omnibase_infra.enums import EnumInfraTransportType


class ResilientDatabaseClient(MixinAsyncCircuitBreaker):
    """Database client with circuit breaker and retry."""

    def __init__(self, config: DatabaseConfig, retry_config: ModelDatabaseRetryConfig):
        self.config = config
        self.retry_config = retry_config

        self._init_circuit_breaker(
            threshold=5,
            reset_timeout=60.0,
            service_name="postgresql-primary",
            transport_type=EnumInfraTransportType.DATABASE,
        )

    async def execute_query(self, query: str, correlation_id: UUID) -> list[dict]:
        """Execute query with circuit breaker and retry."""

        # 1. Check circuit breaker FIRST (fail fast if open)
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("query", correlation_id)

        # 2. Retry logic operates INSIDE circuit breaker boundary
        last_error = None
        for attempt in range(self.retry_config.max_attempts):
            try:
                result = await self._do_query(query)

                # Success - reset circuit breaker
                async with self._circuit_breaker_lock:
                    await self._reset_circuit_breaker()

                return result

            except Exception as e:
                last_error = e

                # Record failure (may open circuit)
                async with self._circuit_breaker_lock:
                    await self._record_circuit_failure("query", correlation_id)

                # Check if error is retryable
                if not is_retryable(e):
                    raise

                # Check if we have attempts remaining
                if attempt >= self.retry_config.max_attempts - 1:
                    raise

                # Calculate backoff with jitter
                delay = calculate_backoff_delay(
                    attempt=attempt,
                    initial_delay=self.retry_config.initial_delay_seconds,
                    max_delay=self.retry_config.max_delay_seconds,
                    exponential_base=self.retry_config.exponential_base,
                )

                await asyncio.sleep(delay)

                # Re-check circuit breaker before next attempt
                async with self._circuit_breaker_lock:
                    await self._check_circuit_breaker("query", correlation_id)

        raise last_error
```

**Key Integration Points**:
1. Circuit breaker check BEFORE retry loop starts
2. Each retry failure records to circuit breaker
3. Circuit breaker re-checked before each retry attempt
4. Success resets circuit breaker

---

## Monitoring and Observability

### Metrics to Track

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `effect_retry_total` | Counter | effect_type, attempt, outcome | Total retry attempts |
| `effect_retry_delay_seconds` | Histogram | effect_type | Backoff delay distribution |
| `effect_compensation_total` | Counter | effect_type, outcome | Compensation attempts |
| `effect_partial_failure_total` | Counter | effect_type | Partial failure occurrences |

### Logging Guidelines

```python
import logging
from uuid import UUID

logger = logging.getLogger(__name__)


def log_retry_attempt(
    effect_type: str,
    attempt: int,
    max_attempts: int,
    delay: float,
    error: Exception,
    correlation_id: UUID,
) -> None:
    """Log retry attempt with structured context."""
    logger.warning(
        "Retrying %s operation (attempt %d/%d) after %.2fs delay",
        effect_type,
        attempt + 1,
        max_attempts,
        delay,
        extra={
            "effect_type": effect_type,
            "attempt": attempt + 1,
            "max_attempts": max_attempts,
            "delay_seconds": delay,
            "error_type": type(error).__name__,
            "correlation_id": str(correlation_id),
        },
    )


def log_retry_exhausted(
    effect_type: str,
    max_attempts: int,
    total_time: float,
    error: Exception,
    correlation_id: UUID,
) -> None:
    """Log when all retry attempts are exhausted."""
    logger.error(
        "All retry attempts exhausted for %s after %d attempts (%.2fs total)",
        effect_type,
        max_attempts,
        total_time,
        extra={
            "effect_type": effect_type,
            "max_attempts": max_attempts,
            "total_time_seconds": total_time,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "correlation_id": str(correlation_id),
        },
    )


def log_compensation_action(
    effect_type: str,
    action: str,
    success: bool,
    correlation_id: UUID,
) -> None:
    """Log compensation action."""
    level = logging.INFO if success else logging.ERROR
    logger.log(
        level,
        "Compensation %s for %s: %s",
        "succeeded" if success else "failed",
        effect_type,
        action,
        extra={
            "effect_type": effect_type,
            "compensation_action": action,
            "compensation_success": success,
            "correlation_id": str(correlation_id),
        },
    )
```

---

## Best Practices Summary

### DO

- Configure retry parameters per-effect type based on failure characteristics
- Use exponential backoff with jitter to prevent thundering herd
- Check circuit breaker before and during retry loop
- Emit compensating events (never delete domain events)
- Use the outbox pattern for transactional consistency
- Log retry attempts with full context for debugging
- Track metrics for retry and compensation operations

### DO NOT

- Retry non-retryable errors (auth failures, validation errors)
- Delete or modify emitted domain events as compensation
- Use fixed delays without jitter
- Ignore circuit breaker state during retries
- Swallow errors silently - always log and optionally re-raise
- Use unbounded retry attempts - always have a max limit

---

## Related Patterns

- [Error Recovery Patterns](./error_recovery_patterns.md) - Implementation examples
- [Circuit Breaker Implementation](./circuit_breaker_implementation.md) - Circuit breaker details
- [Error Handling Patterns](./error_handling_patterns.md) - Error classification
- [Correlation ID Tracking](./correlation_id_tracking.md) - Request tracing

## See Also

- `ModelVaultRetryConfig`: `src/omnibase_infra/handlers/model_vault_retry_config.py`
- `ModelConsulRetryConfig`: `src/omnibase_infra/handlers/model_consul_retry_config.py`
- `ModelKafkaEventBusConfig`: `src/omnibase_infra/event_bus/models/config/model_kafka_event_bus_config.py`
- `ProtocolPolicy`: `src/omnibase_infra/runtime/protocol_policy.py`
