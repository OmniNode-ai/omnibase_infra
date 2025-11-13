#!/usr/bin/env python3
"""
Mock Database Client for integration testing.

Provides in-memory database simulation for testing without real PostgreSQL.
"""

from collections import defaultdict
from datetime import UTC, datetime
from typing import Any, Optional
from uuid import uuid4


class MockDatabaseClient:
    """
    Mock database client for testing.

    Simulates PostgreSQL operations with in-memory storage.
    """

    def __init__(self):
        """Initialize mock database."""
        self.tables: dict[str, dict[str, Any]] = defaultdict(dict)
        self.records_created = 0
        self.records_updated = 0
        self.records_deleted = 0
        self.query_count = 0
        self.is_connected = True
        self.transaction_active = False

    async def connect(self) -> None:
        """Connect to mock database (no-op)."""
        self.is_connected = True

    async def disconnect(self) -> None:
        """Disconnect from mock database."""
        self.is_connected = False

    async def execute(self, query: str, *args: Any) -> int:
        """
        Execute SQL query (simplified simulation).

        Args:
            query: SQL query string
            *args: Query parameters

        Returns:
            Number of rows affected
        """
        self.query_count += 1

        query_lower = query.lower()

        if "insert" in query_lower:
            self.records_created += 1
            return 1
        elif "update" in query_lower:
            self.records_updated += 1
            return 1
        elif "delete" in query_lower:
            self.records_deleted += 1
            return 1

        return 0

    async def fetch_one(self, query: str, *args: Any) -> Optional[dict[str, Any]]:
        """
        Fetch single row.

        Args:
            query: SQL query string
            *args: Query parameters

        Returns:
            Row dict or None
        """
        self.query_count += 1
        rows = await self.fetch_all(query, *args)
        return rows[0] if rows else None

    async def fetch_all(self, query: str, *args: Any) -> list[dict[str, Any]]:
        """
        Fetch all rows.

        Args:
            query: SQL query string
            *args: Query parameters

        Returns:
            List of row dicts
        """
        self.query_count += 1

        # Extract table name (simplified)
        query_lower = query.lower()
        table_name = "unknown"

        if "from" in query_lower:
            parts = query_lower.split("from")
            if len(parts) > 1:
                table_parts = parts[1].strip().split()
                if table_parts:
                    table_name = table_parts[0]

        # Return all records from table
        if table_name in self.tables:
            return list(self.tables[table_name].values())

        return []

    async def begin_transaction(self) -> None:
        """Begin transaction."""
        self.transaction_active = True

    async def commit(self) -> None:
        """Commit transaction."""
        self.transaction_active = False

    async def rollback(self) -> None:
        """Rollback transaction."""
        self.transaction_active = False

    async def insert(
        self, table: str, record: dict[str, Any], returning: Optional[str] = None
    ) -> Optional[Any]:
        """
        Insert record into table.

        Args:
            table: Table name
            record: Record data
            returning: Column to return

        Returns:
            Returned value if specified
        """
        record_id = record.get("id") or str(uuid4())
        record_copy = record.copy()

        if "id" not in record_copy:
            record_copy["id"] = record_id

        if "created_at" not in record_copy:
            record_copy["created_at"] = datetime.now(UTC)

        self.tables[table][record_id] = record_copy
        self.records_created += 1

        if returning:
            return record_copy.get(returning)

        return record_id

    async def update(self, table: str, record_id: str, updates: dict[str, Any]) -> bool:
        """
        Update record in table.

        Args:
            table: Table name
            record_id: Record ID
            updates: Fields to update

        Returns:
            True if updated, False if not found
        """
        if table not in self.tables or record_id not in self.tables[table]:
            return False

        self.tables[table][record_id].update(updates)
        self.tables[table][record_id]["updated_at"] = datetime.now(UTC)
        self.records_updated += 1

        return True

    async def delete(self, table: str, record_id: str) -> bool:
        """
        Delete record from table.

        Args:
            table: Table name
            record_id: Record ID

        Returns:
            True if deleted, False if not found
        """
        if table not in self.tables or record_id not in self.tables[table]:
            return False

        del self.tables[table][record_id]
        self.records_deleted += 1

        return True

    async def get(self, table: str, record_id: str) -> Optional[dict[str, Any]]:
        """
        Get record by ID.

        Args:
            table: Table name
            record_id: Record ID

        Returns:
            Record dict or None
        """
        if table not in self.tables:
            return None

        return self.tables[table].get(record_id)

    def clear(self) -> None:
        """Clear all tables and reset counters."""
        self.tables.clear()
        self.records_created = 0
        self.records_updated = 0
        self.records_deleted = 0
        self.query_count = 0

    def get_table(self, name: str) -> dict[str, Any]:
        """Get all records from table."""
        return self.tables.get(name, {})

    def get_metrics(self) -> dict[str, int]:
        """Get mock database metrics."""
        return {
            "records_created": self.records_created,
            "records_updated": self.records_updated,
            "records_deleted": self.records_deleted,
            "query_count": self.query_count,
            "tables": len(self.tables),
            "total_records": sum(len(records) for records in self.tables.values()),
        }
