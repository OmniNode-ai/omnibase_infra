# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Database transaction context manager for asyncpg.

This module provides a transaction context manager that properly wraps
database operations in transactions with configurable isolation levels,
readonly mode, and statement timeouts.

Critical Insight - Row Locks Require Explicit Transactions:
    When using ``SELECT ... FOR UPDATE`` or similar locking constructs,
    the locks are **released immediately after the SELECT** unless
    executed within an explicit transaction context.

    This is a subtle but critical behavior of PostgreSQL and asyncpg:
    without ``conn.transaction()``, each statement runs in auto-commit
    mode, causing locks to be acquired and immediately released.

    Example of INCORRECT usage (locks NOT maintained):
        ```python
        async with pool.acquire() as conn:
            # Lock is acquired but immediately released!
            rows = await conn.fetch(
                "SELECT * FROM queue WHERE status = 'pending' FOR UPDATE SKIP LOCKED"
            )
            # By this point, another worker could process the same row
            await conn.execute("UPDATE queue SET status = 'processing' WHERE id = $1", row_id)
        ```

    Example of CORRECT usage (locks maintained):
        ```python
        async with pool.acquire() as conn:
            async with conn.transaction():
                # Lock is held until transaction commits
                rows = await conn.fetch(
                    "SELECT * FROM queue WHERE status = 'pending' FOR UPDATE SKIP LOCKED"
                )
                # Lock still held - safe to update
                await conn.execute("UPDATE queue SET status = 'processing' WHERE id = $1", row_id)
        ```

    The ``transaction_context()`` function in this module encapsulates
    this pattern, ensuring locks are properly maintained throughout
    the transaction scope.

Related Implementations:
    - TransitionNotificationOutbox (runtime/transition_notification_outbox.py):
      Uses explicit transaction wrapping for SELECT FOR UPDATE SKIP LOCKED
      to safely process pending notifications with concurrent workers.

See Also:
    - PostgreSQL locking documentation: https://www.postgresql.org/docs/current/explicit-locking.html
    - asyncpg transaction documentation: https://magicstack.github.io/asyncpg/current/api/index.html#asyncpg.connection.Connection.transaction

Example:
    >>> import asyncpg
    >>> from omnibase_infra.utils import transaction_context
    >>>
    >>> async with transaction_context(pool) as conn:
    ...     await conn.execute("INSERT INTO logs (message) VALUES ($1)", "Hello")
    >>>
    >>> # With isolation level and timeout
    >>> async with transaction_context(
    ...     pool,
    ...     isolation="serializable",
    ...     timeout=5.0,
    ... ) as conn:
    ...     await conn.execute("UPDATE accounts SET balance = balance - 100 WHERE id = $1", account_id)

.. versionadded:: 0.10.0
    Created as part of database utility consolidation.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

import asyncpg

logger = logging.getLogger(__name__)


@asynccontextmanager
async def transaction_context(
    pool: asyncpg.Pool,
    *,
    isolation: str = "read_committed",
    readonly: bool = False,
    deferrable: bool = False,
    timeout: float | None = None,
    correlation_id: UUID | None = None,
) -> AsyncIterator[asyncpg.Connection]:
    """Async context manager for database transactions.

    Acquires a connection from the pool and starts a transaction with the
    specified isolation level and options. The connection is yielded for
    use within the transaction scope.

    Critical - Row Locks:
        This context manager ensures that row locks (e.g., ``FOR UPDATE``,
        ``FOR UPDATE SKIP LOCKED``) are maintained throughout the transaction.
        Without explicit transaction wrapping, asyncpg operates in auto-commit
        mode where locks are released immediately after each statement.

    Isolation Levels:
        - ``read_committed`` (default): Each statement sees a snapshot of
          committed data as of the start of that statement.
        - ``repeatable_read``: All statements in the transaction see a
          snapshot of committed data as of the transaction start.
        - ``serializable``: Strictest isolation - transactions execute as
          if they were run serially.

    Args:
        pool: asyncpg connection pool to acquire connection from.
        isolation: Transaction isolation level. One of "read_committed",
            "repeatable_read", or "serializable". Defaults to "read_committed".
        readonly: If True, the transaction is marked as read-only.
            Attempting to modify data will raise an error. Defaults to False.
        deferrable: If True, the transaction is deferrable. Only valid when
            both ``isolation="serializable"`` and ``readonly=True``.
            A deferrable transaction may block when first acquiring its
            snapshot until it can execute without conflicting with other
            serializable transactions. Defaults to False.
        timeout: Statement timeout in seconds. If provided, sets
            ``statement_timeout`` for the duration of the transaction.
            Statements exceeding this timeout will be cancelled.
            Defaults to None (no timeout).
        correlation_id: Optional correlation ID for logging. When provided,
            transaction start and commit/rollback events are logged with
            this ID for distributed tracing.

    Yields:
        asyncpg.Connection: The acquired connection within the transaction
        context. Use this connection for all queries within the transaction.

    Raises:
        asyncpg.PostgresError: For database-level errors.
        TimeoutError: If a statement exceeds the configured timeout.

    Example:
        Basic usage:

        >>> async with transaction_context(pool) as conn:
        ...     await conn.execute("INSERT INTO users (name) VALUES ($1)", "Alice")
        ...     await conn.execute("INSERT INTO audit_log (action) VALUES ($1)", "user_created")

        With SELECT FOR UPDATE:

        >>> async with transaction_context(pool) as conn:
        ...     # Lock is held until transaction commits
        ...     rows = await conn.fetch(
        ...         "SELECT * FROM jobs WHERE status = 'pending' LIMIT 1 FOR UPDATE SKIP LOCKED"
        ...     )
        ...     if rows:
        ...         await conn.execute(
        ...             "UPDATE jobs SET status = 'processing' WHERE id = $1",
        ...             rows[0]["id"]
        ...         )

        With isolation and timeout:

        >>> async with transaction_context(
        ...     pool,
        ...     isolation="serializable",
        ...     readonly=True,
        ...     timeout=10.0,
        ...     correlation_id=uuid4(),
        ... ) as conn:
        ...     totals = await conn.fetchval("SELECT SUM(amount) FROM transactions")

    Note:
        The transaction is automatically committed on successful exit from
        the context manager, or rolled back if an exception is raised.

    Warning:
        Asyncpg exception handling: This utility lets asyncpg exceptions
        propagate naturally without wrapping them in ONEX errors. This is
        intentional as it keeps the utility simple and composable. Callers
        should handle asyncpg exceptions as appropriate for their use case.

    Related:
        - OMN-1139: TransitionNotificationOutbox uses this pattern for
          SELECT FOR UPDATE SKIP LOCKED with concurrent workers.
    """
    async with pool.acquire() as conn:
        # Log transaction start if correlation_id provided
        if correlation_id is not None:
            logger.debug(
                "Starting database transaction",
                extra={
                    "correlation_id": str(correlation_id),
                    "isolation": isolation,
                    "readonly": readonly,
                    "deferrable": deferrable,
                    "timeout": timeout,
                },
            )

        try:
            async with conn.transaction(
                isolation=isolation,
                readonly=readonly,
                deferrable=deferrable,
            ):
                # Set statement timeout if provided
                # Uses LOCAL to scope timeout to this transaction only
                if timeout is not None:
                    timeout_ms = int(timeout * 1000)
                    await conn.execute("SET LOCAL statement_timeout = $1", timeout_ms)

                yield conn

            # Log successful commit if correlation_id provided
            if correlation_id is not None:
                logger.debug(
                    "Database transaction committed",
                    extra={
                        "correlation_id": str(correlation_id),
                    },
                )

        except Exception:
            # Log rollback if correlation_id provided
            # Transaction is automatically rolled back by asyncpg
            if correlation_id is not None:
                logger.debug(
                    "Database transaction rolled back",
                    extra={
                        "correlation_id": str(correlation_id),
                    },
                )
            raise


__all__: list[str] = [
    "transaction_context",
]
