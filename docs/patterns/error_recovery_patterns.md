> **Navigation**: [Home](../index.md) > [Patterns](README.md) > Error Recovery Patterns

# Error Recovery Patterns

## Overview

This document describes resilience patterns for handling infrastructure failures gracefully. All patterns integrate with ONEX error handling and correlation tracking.

## Recovery Strategy Selection

| Pattern | Use For | Error Types | Implementation Complexity |
|---------|---------|-------------|---------------------------|
| Exponential Backoff | `InfraConnectionError` | Transient network failures | Low |
| Circuit Breaker | `InfraUnavailableError` | Service outages | Medium |
| Graceful Degradation | `InfraTimeoutError` | Performance degradation | Medium |
| Credential Refresh | `InfraAuthenticationError` | Token expiry | Low |

## Exponential Backoff Pattern

### Use Cases
- Database connection retries
- Kafka publish failures
- HTTP API transient errors
- Consul service registration

### Implementation

```python
from typing import TypeVar, Callable
from uuid import UUID
import asyncio
from omnibase_infra.errors import InfraConnectionError

T = TypeVar("T")

async def retry_with_exponential_backoff(
    operation: Callable[[], T],
    max_attempts: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    correlation_id: UUID | None = None,
) -> T:
    """Retry operation with exponential backoff.

    Args:
        operation: Async function to retry
        max_attempts: Maximum retry attempts (default: 5)
        base_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay cap in seconds (default: 60.0)
        correlation_id: Request correlation ID for logging

    Returns:
        Operation result

    Raises:
        InfraConnectionError: After all retries exhausted
    """
    last_error: Exception | None = None

    for attempt in range(max_attempts):
        try:
            return await operation()
        except (InfraConnectionError, ConnectionError) as e:
            last_error = e

            if attempt == max_attempts - 1:
                # Final attempt failed
                raise

            # Calculate delay: 2^attempt * base_delay, capped at max_delay
            delay = min(base_delay * (2 ** attempt), max_delay)

            # Optional: Add jitter to prevent thundering herd
            import random
            jittered_delay = delay * (0.5 + random.random())

            await asyncio.sleep(jittered_delay)

    # Should never reach here, but satisfy type checker
    raise last_error if last_error else RuntimeError("Unexpected retry failure")
```

### Usage Example

```python
from omnibase_infra.errors import InfraConnectionError, ModelInfraErrorContext
from omnibase_infra.enums import EnumInfraTransportType

async def connect_to_database(correlation_id: UUID) -> asyncpg.Connection:
    async def _connect() -> asyncpg.Connection:
        try:
            return await asyncpg.connect(
                host=config.db_host,
                port=config.db_port,
            )
        except asyncpg.PostgresConnectionError as e:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="connect",
                target_name="postgresql-primary",
                correlation_id=correlation_id,
            )
            raise InfraConnectionError(
                "Database connection failed",
                context=context
            ) from e

    return await retry_with_exponential_backoff(
        operation=_connect,
        max_attempts=5,
        base_delay=1.0,
        correlation_id=correlation_id,
    )
```

## Circuit Breaker Pattern

### Use Cases
- Protecting against cascading failures
- Service mesh integration
- API gateway resilience
- Microservice communication

### Implementation

```python
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, TypeVar
from omnibase_infra.errors import InfraUnavailableError

T = TypeVar("T")

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5          # Failures before opening
    success_threshold: int = 2          # Successes to close from half-open
    timeout: timedelta = timedelta(seconds=60)  # Time before half-open attempt

class CircuitBreaker:
    """Circuit breaker for preventing cascading failures."""

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: datetime | None = None

    def _should_attempt(self) -> bool:
        """Check if request should be attempted."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if timeout elapsed
            if (self.last_failure_time and
                datetime.utcnow() - self.last_failure_time >= self.config.timeout):
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                return True
            return False

        # HALF_OPEN state - allow attempt
        return True

    def _on_success(self) -> None:
        """Record successful operation."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0

    def _on_failure(self) -> None:
        """Record failed operation."""
        self.last_failure_time = datetime.utcnow()

        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
        elif self.state == CircuitState.CLOSED:
            self.failure_count += 1
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN

    async def call(self, operation: Callable[[], T]) -> T:
        """Execute operation with circuit breaker protection.

        Args:
            operation: Async function to execute

        Returns:
            Operation result

        Raises:
            InfraUnavailableError: Circuit is open
            Exception: From operation failure
        """
        if not self._should_attempt():
            raise InfraUnavailableError(
                f"Circuit breaker is OPEN, service unavailable",
            )

        try:
            result = await operation()
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
```

### Usage Example

```python
from omnibase_infra.errors import InfraUnavailableError, ModelInfraErrorContext
from omnibase_infra.enums import EnumInfraTransportType

# Initialize circuit breaker for Kafka
kafka_circuit_breaker = CircuitBreaker(
    config=CircuitBreakerConfig(
        failure_threshold=5,
        success_threshold=2,
        timeout=timedelta(seconds=30),
    )
)

async def publish_with_circuit_breaker(
    topic: str,
    message: bytes,
    correlation_id: UUID,
) -> None:
    async def _publish() -> None:
        try:
            await kafka_producer.send(topic, message)
        except KafkaConnectionError as e:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.KAFKA,
                operation="publish_message",
                target_name=f"kafka-topic-{topic}",
                correlation_id=correlation_id,
            )
            raise InfraUnavailableError(
                "Kafka broker unavailable",
                context=context
            ) from e

    await kafka_circuit_breaker.call(_publish)
```

## Graceful Degradation Pattern

### Use Cases
- Cache fallback when database slow
- Secondary service when primary unavailable
- Read-only mode during write failures
- Stale data acceptable for non-critical operations

### Implementation

```python
from typing import TypeVar, Callable
from omnibase_infra.errors import InfraTimeoutError

T = TypeVar("T")

async def with_fallback(
    primary: Callable[[], T],
    fallback: Callable[[], T],
    correlation_id: UUID,
) -> tuple[T, bool]:
    """Execute primary operation with fallback on failure.

    Args:
        primary: Primary operation to attempt
        fallback: Fallback operation if primary fails
        correlation_id: Request tracking ID

    Returns:
        Tuple of (result, used_fallback)
    """
    try:
        result = await primary()
        return result, False
    except (InfraTimeoutError, InfraUnavailableError):
        # Primary failed, use fallback
        result = await fallback()
        return result, True
```

### Usage Example

```python
from omnibase_infra.errors import InfraTimeoutError, ModelInfraErrorContext
from omnibase_infra.enums import EnumInfraTransportType

async def get_user_data(
    user_id: str,
    correlation_id: UUID,
) -> dict[str, object]:
    """Fetch user data with cache fallback."""

    async def _fetch_from_db() -> dict[str, object]:
        try:
            query = "SELECT * FROM users WHERE id = $1"
            result = await db.fetchrow(query, user_id)
            return dict(result)
        except asyncpg.QueryCanceledError as e:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="fetch_user",
                target_name="postgresql-primary",
                correlation_id=correlation_id,
            )
            raise InfraTimeoutError(
                "Database query timeout",
                context=context
            ) from e

    async def _fetch_from_cache() -> dict[str, object]:
        # Return potentially stale cached data
        cached = await valkey.get(f"user:{user_id}")
        return json.loads(cached) if cached else {}

    user_data, used_cache = await with_fallback(
        primary=_fetch_from_db,
        fallback=_fetch_from_cache,
        correlation_id=correlation_id,
    )

    # Log if fallback was used
    if used_cache:
        logger.warning(
            "Used cache fallback for user data",
            extra={"user_id": user_id, "correlation_id": correlation_id}
        )

    return user_data
```

## Credential Refresh Pattern

### Use Cases
- OAuth token refresh
- Vault token renewal
- Service account credential rotation
- Session management

### Implementation

```python
from datetime import datetime, timedelta
from typing import Protocol
from omnibase_infra.errors import InfraAuthenticationError, ModelInfraErrorContext
from omnibase_infra.enums import EnumInfraTransportType

class Credential(Protocol):
    """Protocol for refreshable credentials."""

    @property
    def is_expired(self) -> bool:
        """Check if credential is expired."""
        ...

    async def refresh(self) -> None:
        """Refresh the credential."""
        ...

async def with_credential_refresh(
    operation: Callable[[], T],
    credential: Credential,
    correlation_id: UUID,
) -> T:
    """Execute operation with automatic credential refresh.

    Args:
        operation: Operation requiring authentication
        credential: Refreshable credential
        correlation_id: Request tracking ID

    Returns:
        Operation result
    """
    try:
        # Check if credential needs refresh before operation
        if credential.is_expired:
            await credential.refresh()

        return await operation()
    except InfraAuthenticationError:
        # Authentication failed, try refreshing credential
        await credential.refresh()
        return await operation()
```

### Usage Example

```python
from dataclasses import dataclass
from hvac import Client as VaultClient

@dataclass
class VaultCredential:
    """Vault token credential."""

    client: VaultClient
    token: str
    expiry: datetime

    @property
    def is_expired(self) -> bool:
        # Refresh 5 minutes before actual expiry
        return datetime.utcnow() >= self.expiry - timedelta(minutes=5)

    async def refresh(self) -> None:
        """Renew Vault token."""
        response = self.client.auth.token.renew_self()
        self.token = response["auth"]["client_token"]
        self.expiry = datetime.utcnow() + timedelta(
            seconds=response["auth"]["lease_duration"]
        )

async def read_secret_with_refresh(
    path: str,
    vault_cred: VaultCredential,
    correlation_id: UUID,
) -> dict[str, str]:
    """Read Vault secret with automatic token refresh."""

    async def _read_secret() -> dict[str, str]:
        try:
            response = vault_cred.client.secrets.kv.v2.read_secret_version(
                path=path
            )
            return response["data"]["data"]
        except Unauthorized as e:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.VAULT,
                operation="read_secret",
                target_name="vault-kv-v2",
                correlation_id=correlation_id,
            )
            raise InfraAuthenticationError(
                "Vault authentication failed",
                context=context
            ) from e

    return await with_credential_refresh(
        operation=_read_secret,
        credential=vault_cred,
        correlation_id=correlation_id,
    )
```

## Combining Patterns

Patterns can be composed for robust error handling:

```python
async def resilient_operation(
    operation: Callable[[], T],
    circuit_breaker: CircuitBreaker,
    correlation_id: UUID,
) -> T:
    """Combine circuit breaker with exponential backoff."""

    async def _with_retry() -> T:
        return await retry_with_exponential_backoff(
            operation=operation,
            max_attempts=3,
            correlation_id=correlation_id,
        )

    return await circuit_breaker.call(_with_retry)
```

## Related Patterns

- [Security Patterns](./security_patterns.md) - Comprehensive security guide including credential refresh and secret management
- [Error Handling Patterns](./error_handling_patterns.md) - Error classification and context
- [Error Sanitization Patterns](./error_sanitization_patterns.md) - Data classification and secure error reporting
- [Correlation ID Tracking](./correlation_id_tracking.md) - Request tracing
- [Circuit Breaker Implementation](./circuit_breaker_implementation.md) - Detailed circuit breaker guide
