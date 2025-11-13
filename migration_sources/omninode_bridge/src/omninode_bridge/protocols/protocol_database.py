"""
Database Protocol for type-safe database operations.

This module defines the SupportsQuery protocol for structural typing
of database connections, enabling type-safe dependency injection
without tight coupling to specific database implementations.

ONEX Compliance:
- Suffix-based naming: SupportsQuery (Protocol suffix implicit)
- Protocol-based typing for flexibility
- Async-first design

Usage:
    ```python
    from omninode_bridge.protocols import SupportsQuery

    async def process_events(db: SupportsQuery) -> list[dict[str, Any]]:
        query = "SELECT * FROM events WHERE session_id = $1"
        results = await db.execute_query(query, session_id)
        return results
    ```
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SupportsQuery(Protocol):
    """
    Protocol for database connections supporting async query execution.

    This protocol defines the minimal interface required for database
    operations in the omninode_bridge infrastructure. Any database
    connection manager implementing these methods can be used with
    components requiring database access.

    Methods:
        execute_query: Execute async SQL query with parameters
        initialize: Initialize database connection pool (optional)
        close: Close database connections gracefully (optional)

    Example:
        ```python
        from omninode_bridge.protocols import SupportsQuery
        from omninode_bridge.infrastructure.postgres_connection_manager import (
            PostgresConnectionManager
        )

        # PostgresConnectionManager implements SupportsQuery protocol
        db: SupportsQuery = PostgresConnectionManager(config)
        await db.initialize()

        # Type-safe query execution
        results = await db.execute_query(
            "SELECT * FROM events WHERE id = $1",
            event_id
        )
        ```

    Protocol Benefits:
        - Type safety without tight coupling
        - Easy testing with mock implementations
        - Flexibility for different database backends
        - Static type checking with mypy/pyright
    """

    async def execute_query(self, query: str, *args: Any) -> list[dict[str, Any]]:
        """
        Execute async SQL query with optional parameters.

        Args:
            query: SQL query string with parameter placeholders ($1, $2, etc.)
            *args: Query parameters to bind to placeholders

        Returns:
            List of result rows as dictionaries mapping column names to values.
            Empty list if no results.

        Raises:
            Exception: Database-specific exceptions for query failures

        Example:
            ```python
            results = await db.execute_query(
                "SELECT event_id, timestamp FROM events WHERE session_id = $1",
                session_id
            )
            for row in results:
                print(f"Event {row['event_id']} at {row['timestamp']}")
            ```
        """
        ...

    async def initialize(self) -> None:
        """
        Initialize database connection pool.

        Optional method for setting up connection pools, migrations,
        or other initialization tasks.

        Example:
            ```python
            await db.initialize()
            # Connection pool ready for queries
            ```
        """
        ...

    async def close(self) -> None:
        """
        Close database connections gracefully.

        Optional method for cleanup, closing connection pools, and
        releasing resources.

        Example:
            ```python
            try:
                # Use database
                results = await db.execute_query(...)
            finally:
                await db.close()
            ```
        """
        ...
