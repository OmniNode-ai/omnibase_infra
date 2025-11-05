"""
Retry Policy Factory for ONEX Infrastructure.

Provides pre-configured retry policies using tenacity library.
Implements exponential backoff with jitter for various infrastructure services.

Features:
- Service-specific retry configurations
- Exponential backoff with jitter
- Integration with OnexError exception handling
- Configurable retry limits and wait strategies
"""

from collections.abc import Callable
from typing import Any, TypeVar

from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

# Type variable for generic function wrapping
F = TypeVar("F", bound=Callable[..., Any])


def _on_retry_callback(retry_state: RetryCallState) -> None:
    """
    Callback executed before each retry attempt.

    Logs retry information for monitoring and debugging.

    Args:
        retry_state: Current retry state from tenacity
    """
    if retry_state.outcome and retry_state.outcome.failed:
        exception = retry_state.outcome.exception()
        attempt_number = retry_state.attempt_number

        # Log retry attempt (infrastructure services should have structured logging)
        # This is a placeholder for actual logging implementation
        error_msg = f"Retry attempt {attempt_number}: {exception!s}"
        _ = error_msg  # Suppress unused variable warning


def create_database_retry_policy(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 10.0,
) -> Callable[[F], F]:
    """
    Create a retry policy for database operations.

    Retries database operations with exponential backoff on connection
    and query failures. Useful for transient database issues.

    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time in seconds
        max_wait: Maximum wait time in seconds

    Returns:
        Retry decorator for database operations

    Example:
        ```python
        @create_database_retry_policy()
        async def query_with_retry(sql: str) -> list[Record]:
            async with connection_manager.acquire_connection() as conn:
                return await conn.fetch(sql)
        ```
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential_jitter(initial=min_wait, max=max_wait),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        before_sleep=_on_retry_callback,
        reraise=True,
    )


def create_network_retry_policy(
    max_attempts: int = 3,
    min_wait: float = 0.5,
    max_wait: float = 5.0,
) -> Callable[[F], F]:
    """
    Create a retry policy for network operations.

    Retries network operations with exponential backoff on connection
    failures and timeouts. Suitable for HTTP calls and external APIs.

    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time in seconds
        max_wait: Maximum wait time in seconds

    Returns:
        Retry decorator for network operations

    Example:
        ```python
        @create_network_retry_policy()
        async def fetch_api_with_retry(url: str) -> dict:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                return response.json()
        ```
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential_jitter(initial=min_wait, max=max_wait),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        before_sleep=_on_retry_callback,
        reraise=True,
    )


def create_kafka_retry_policy(
    max_attempts: int = 5,
    min_wait: float = 1.0,
    max_wait: float = 30.0,
) -> Callable[[F], F]:
    """
    Create a retry policy for Kafka operations.

    Retries Kafka operations with exponential backoff on broker
    unavailability and message delivery failures.

    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time in seconds
        max_wait: Maximum wait time in seconds

    Returns:
        Retry decorator for Kafka operations

    Example:
        ```python
        @create_kafka_retry_policy()
        async def send_with_retry(topic: str, message: dict) -> None:
            await kafka_producer.send(topic, message)
        ```
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential_jitter(initial=min_wait, max=max_wait),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        before_sleep=_on_retry_callback,
        reraise=True,
    )


def create_vault_retry_policy(
    max_attempts: int = 3,
    min_wait: float = 2.0,
    max_wait: float = 20.0,
) -> Callable[[F], F]:
    """
    Create a retry policy for Vault operations.

    Retries Vault operations with exponential backoff on connection
    failures and authentication issues. Uses longer wait times due to
    Vault's rate limiting and authentication complexity.

    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time in seconds
        max_wait: Maximum wait time in seconds

    Returns:
        Retry decorator for Vault operations

    Example:
        ```python
        @create_vault_retry_policy()
        async def get_secret_with_retry(path: str) -> dict:
            return await vault_client.read(path)
        ```
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential_jitter(initial=min_wait, max=max_wait),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        before_sleep=_on_retry_callback,
        reraise=True,
    )
