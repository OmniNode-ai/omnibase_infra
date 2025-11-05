"""
Circuit Breaker Factory for ONEX Infrastructure.

Provides pre-configured circuit breakers for various infrastructure services
using the circuitbreaker library. Circuit breakers prevent cascading failures
by temporarily blocking calls to failing services.

Features:
- Service-specific circuit breaker configurations
- Integration with OnexError exception handling
- Automatic failure tracking and recovery
- Configurable failure thresholds and recovery timeouts
"""

from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from circuitbreaker import CircuitBreaker, CircuitBreakerError
from omnibase_core.core.errors.onex_error import CoreErrorCode, OnexError

# Type variable for generic function wrapping
F = TypeVar("F", bound=Callable[..., Any])


class InfrastructureCircuitBreaker(CircuitBreaker):
    """
    Extended circuit breaker with ONEX error integration.

    Converts circuit breaker errors to OnexError for consistent
    error handling across infrastructure services.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type[Exception] = Exception,
        name: str | None = None,
        fallback_function: Callable[..., Any] | None = None,
    ):
        """
        Initialize circuit breaker with infrastructure defaults.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type to track as failures
            name: Circuit breaker name for monitoring
            fallback_function: Optional fallback function when circuit is open
        """
        super().__init__(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=expected_exception,
            name=name,
            fallback_function=fallback_function,
        )

    def __call__(self, func: F) -> F:
        """
        Wrap function with circuit breaker and OnexError conversion.

        Args:
            func: Function to protect with circuit breaker

        Returns:
            Wrapped function with circuit breaker protection
        """

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return super(InfrastructureCircuitBreaker, self).__call__(func)(
                    *args, **kwargs,
                )
            except CircuitBreakerError as e:
                raise OnexError(
                    code=CoreErrorCode.SERVICE_UNAVAILABLE,
                    message=f"Circuit breaker open for {self.name or 'service'}: {e!s}",
                ) from e

        return wrapper  # type: ignore[return-value]


def create_database_circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    name: str = "database",
) -> InfrastructureCircuitBreaker:
    """
    Create a circuit breaker for database operations.

    Protects against database connection failures and query timeouts.
    Opens circuit after 5 consecutive failures, recovers after 60 seconds.

    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before attempting recovery
        name: Circuit breaker name for monitoring

    Returns:
        Configured circuit breaker for database operations

    Example:
        ```python
        @create_database_circuit_breaker()
        async def query_database(sql: str) -> list[Record]:
            async with connection_manager.acquire_connection() as conn:
                return await conn.fetch(sql)
        ```
    """
    return InfrastructureCircuitBreaker(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        expected_exception=Exception,
        name=name,
        fallback_function=None,
    )


def create_kafka_circuit_breaker(
    failure_threshold: int = 10,
    recovery_timeout: int = 30,
    name: str = "kafka",
) -> InfrastructureCircuitBreaker:
    """
    Create a circuit breaker for Kafka operations.

    Protects against Kafka broker unavailability and message delivery failures.
    Opens circuit after 10 consecutive failures, recovers after 30 seconds.

    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before attempting recovery
        name: Circuit breaker name for monitoring

    Returns:
        Configured circuit breaker for Kafka operations

    Example:
        ```python
        @create_kafka_circuit_breaker()
        async def send_kafka_message(topic: str, message: dict) -> None:
            await kafka_producer.send(topic, message)
        ```
    """
    return InfrastructureCircuitBreaker(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        expected_exception=Exception,
        name=name,
        fallback_function=None,
    )


def create_network_circuit_breaker(
    failure_threshold: int = 3,
    recovery_timeout: int = 45,
    name: str = "network",
) -> InfrastructureCircuitBreaker:
    """
    Create a circuit breaker for network operations.

    Protects against network failures and service unavailability.
    Opens circuit after 3 consecutive failures, recovers after 45 seconds.

    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before attempting recovery
        name: Circuit breaker name for monitoring

    Returns:
        Configured circuit breaker for network operations

    Example:
        ```python
        @create_network_circuit_breaker()
        async def fetch_external_api(url: str) -> dict:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                return response.json()
        ```
    """
    return InfrastructureCircuitBreaker(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        expected_exception=Exception,
        name=name,
        fallback_function=None,
    )


def create_vault_circuit_breaker(
    failure_threshold: int = 3,
    recovery_timeout: int = 90,
    name: str = "vault",
) -> InfrastructureCircuitBreaker:
    """
    Create a circuit breaker for Vault operations.

    Protects against Vault service unavailability and authentication failures.
    Opens circuit after 3 consecutive failures, recovers after 90 seconds.

    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before attempting recovery
        name: Circuit breaker name for monitoring

    Returns:
        Configured circuit breaker for Vault operations

    Example:
        ```python
        @create_vault_circuit_breaker()
        async def get_vault_secret(path: str) -> dict:
            return await vault_client.read(path)
        ```
    """
    return InfrastructureCircuitBreaker(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        expected_exception=Exception,
        name=name,
        fallback_function=None,
    )
