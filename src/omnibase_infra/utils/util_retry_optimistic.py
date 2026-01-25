# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Optimistic locking retry helper for concurrent data operations.

This module provides utilities for retrying operations that may fail due to
optimistic locking conflicts. Optimistic locking is a concurrency control
strategy where conflicts are detected at write time rather than using locks.

Use Cases:
    - Database UPDATE with version checks (row_count=0 indicates conflict)
    - CAS (Compare-And-Swap) operations in distributed systems
    - Consul KV ModifyIndex-based updates
    - Any operation where conflict detection is based on return value

Design Decisions:
    - No circuit breaker integration: Optimistic conflicts are application logic,
      not infrastructure failures. They indicate contention, not service degradation.
    - Lower initial backoff: 0.1s vs 1.0s for transient errors because conflicts
      typically resolve faster as other transactions complete.
    - Jitter by default: Critical for preventing thundering herd on high-contention
      resources where multiple retries might synchronize.
    - Caller-provided conflict check: Flexible for different conflict indicators
      (row_count=0, version mismatch, boolean flags, etc.)

Example:
    >>> import asyncio
    >>> from omnibase_infra.utils.util_retry_optimistic import (
    ...     retry_on_optimistic_conflict,
    ...     OptimisticConflictError,
    ... )
    >>>
    >>> attempt_count = 0
    >>> async def update_with_version_check():
    ...     global attempt_count
    ...     attempt_count += 1
    ...     # Simulate success after 2 conflicts
    ...     if attempt_count < 3:
    ...         return {"row_count": 0}  # Conflict
    ...     return {"row_count": 1}  # Success
    >>>
    >>> async def main():
    ...     result = await retry_on_optimistic_conflict(
    ...         update_with_version_check,
    ...         check_conflict=lambda r: r["row_count"] == 0,
    ...         max_retries=5,
    ...     )
    ...     print(f"Success after {attempt_count} attempts")
    >>>
    >>> asyncio.run(main())  # doctest: +SKIP
    Success after 3 attempts

See Also:
    - ONEX infrastructure patterns documentation
    - PostgreSQL advisory locks vs optimistic locking
    - Consul KV ModifyIndex documentation

.. versionadded:: 0.10.0
    Created for database operations requiring optimistic concurrency control.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from uuid import UUID

logger = logging.getLogger(__name__)

# Generic type for the return value of the retried function
T = TypeVar("T")


class OptimisticConflictError(Exception):
    """Exception raised when optimistic locking retries are exhausted.

    This exception indicates that an operation failed due to repeated optimistic
    locking conflicts after all retry attempts. The caller should handle this
    by either:
    - Reporting the conflict to the user
    - Using a different conflict resolution strategy
    - Applying exponential backoff at a higher level

    Note:
        This is a standard Python exception, not an ONEX error. Optimistic
        conflicts are expected application behavior, not infrastructure failures.

    Attributes:
        attempts: The total number of attempts made (including initial attempt).
        last_result: The result from the final failed attempt, useful for
            debugging or logging the conflict state.

    Example:
        >>> from omnibase_infra.utils.util_retry_optimistic import OptimisticConflictError
        >>>
        >>> try:
        ...     # ... retry logic that exhausted retries
        ...     raise OptimisticConflictError(
        ...         attempts=4,
        ...         last_result={"row_count": 0, "current_version": 5}
        ...     )
        ... except OptimisticConflictError as e:
        ...     print(f"Failed after {e.attempts} attempts")
        ...     print(f"Last conflict state: {e.last_result}")
        Failed after 4 attempts
        Last conflict state: {'row_count': 0, 'current_version': 5}
    """

    def __init__(self, *, attempts: int, last_result: object) -> None:
        """Initialize OptimisticConflictError.

        Args:
            attempts: Total number of attempts made.
            last_result: Result from the final attempt.
        """
        self.attempts = attempts
        self.last_result = last_result
        super().__init__(
            f"Optimistic locking conflict persisted after {attempts} attempts. "
            f"Last result: {last_result}"
        )


async def retry_on_optimistic_conflict(
    fn: Callable[[], Awaitable[T]],
    *,
    check_conflict: Callable[[T], bool],
    max_retries: int = 3,
    initial_backoff: float = 0.1,
    max_backoff: float = 5.0,
    backoff_multiplier: float = 2.0,
    jitter: bool = True,
    correlation_id: UUID | None = None,
) -> T:
    """Execute async function with retry on optimistic locking conflict.

    This function implements exponential backoff retry logic for operations
    that may fail due to optimistic locking conflicts. Unlike transient error
    retries, this does NOT integrate with circuit breakers since conflicts
    indicate contention, not service degradation.

    Args:
        fn: Async function to execute. Should take no arguments; use closures
            or functools.partial to bind arguments. Example:
            ``lambda: update_record(id=123, data=data)``
        check_conflict: Callable that inspects the result and returns True if
            the result indicates a conflict. Examples:
            - ``lambda r: r.row_count == 0`` (database update)
            - ``lambda r: r.success is False`` (boolean result)
            - ``lambda r: r.version != expected_version`` (version mismatch)
        max_retries: Maximum number of retry attempts after the initial attempt.
            Total attempts = max_retries + 1. Defaults to 3.
        initial_backoff: Initial backoff delay in seconds before first retry.
            Lower than transient error defaults (0.1s vs 1.0s) because conflicts
            typically resolve quickly. Defaults to 0.1.
        max_backoff: Maximum backoff delay cap in seconds. Prevents excessive
            wait times. Defaults to 5.0.
        backoff_multiplier: Multiplier for exponential backoff between retries.
            Delay doubles by default: 0.1s -> 0.2s -> 0.4s -> 0.8s. Defaults to 2.0.
        jitter: If True, adds random jitter (50-150% of delay) to prevent
            thundering herd when multiple clients retry simultaneously.
            Strongly recommended for high-contention scenarios. Defaults to True.
        correlation_id: Optional correlation ID for structured logging. When
            provided, retry attempts are logged with this ID for distributed
            tracing. Defaults to None.

    Returns:
        The result of ``fn()`` when ``check_conflict(result)`` returns False.

    Raises:
        OptimisticConflictError: If all retry attempts are exhausted and the
            operation still indicates a conflict. Contains ``attempts`` count
            and ``last_result`` for debugging.

    Example:
        Basic usage with row count check::

            from functools import partial
            from omnibase_infra.utils.util_retry_optimistic import (
                retry_on_optimistic_conflict,
                OptimisticConflictError,
            )

            async def update_with_version(id: str, data: dict, version: int):
                return await db.execute(
                    "UPDATE t SET data=$1, version=version+1 "
                    "WHERE id=$2 AND version=$3",
                    data, id, version
                )

            try:
                result = await retry_on_optimistic_conflict(
                    partial(update_with_version, "abc", {"name": "new"}, 5),
                    check_conflict=lambda r: r.row_count == 0,
                    max_retries=5,
                    correlation_id=correlation_id,
                )
            except OptimisticConflictError as e:
                logger.error(f"Update failed after {e.attempts} attempts")
                raise

    Warning:
        The ``fn`` callable should be idempotent or at least safe to retry.
        This function will call ``fn`` multiple times on conflicts.

    Note:
        Backoff timing with default parameters (jitter disabled for clarity):
        - Attempt 1: immediate
        - Attempt 2: wait 0.1s
        - Attempt 3: wait 0.2s
        - Attempt 4: wait 0.4s
        - Total max wait: ~0.7s (with max_retries=3)

    .. versionadded:: 0.10.0
    """
    backoff = initial_backoff
    last_result: T | None = None

    for attempt in range(max_retries + 1):  # +1 for initial attempt
        result = await fn()

        if not check_conflict(result):
            # Success - no conflict detected
            if attempt > 0 and correlation_id is not None:
                logger.info(
                    "Optimistic conflict resolved after %d retries",
                    attempt,
                    extra={
                        "correlation_id": str(correlation_id),
                        "total_attempts": attempt + 1,
                        "action": "optimistic_conflict_resolved",
                    },
                )
            return result

        # Conflict detected - store result for potential error reporting
        last_result = result

        if attempt == max_retries:
            # All retries exhausted
            break

        # Log retry attempt if correlation_id provided
        if correlation_id is not None:
            logger.debug(
                "Optimistic conflict detected, retrying (attempt %d/%d)",
                attempt + 1,
                max_retries + 1,
                extra={
                    "correlation_id": str(correlation_id),
                    "attempt": attempt + 1,
                    "max_attempts": max_retries + 1,
                    "backoff_seconds": backoff,
                    "action": "optimistic_conflict_retry",
                },
            )

        # Calculate delay with optional jitter
        delay = min(backoff, max_backoff)
        if jitter:
            # Apply 50-150% jitter to prevent thundering herd
            delay *= 0.5 + random.random()

        await asyncio.sleep(delay)
        backoff *= backoff_multiplier

    # All retries exhausted - raise conflict error
    # Defensive check - last_result is guaranteed to be set after at least one attempt
    if last_result is None:
        raise AssertionError("Unreachable: last_result must be set after retries")
    raise OptimisticConflictError(attempts=max_retries + 1, last_result=last_result)


__all__: list[str] = [
    "OptimisticConflictError",
    "retry_on_optimistic_conflict",
]
