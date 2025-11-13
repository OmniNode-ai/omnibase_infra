#!/usr/bin/env python3
"""Fixtures for infrastructure integration tests."""

from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest


@pytest.fixture
def database_adapter_node():
    """Create a mock database adapter node for SQL injection testing."""

    class MockDatabaseAdapterNode:
        """Mock database adapter node for testing SQL injection protection."""

        def __init__(self):
            self.node_id = str(uuid4())
            self._handle_insert = AsyncMock()
            self._handle_query = AsyncMock()
            self._handle_update = AsyncMock()
            self._handle_delete = AsyncMock()

            # Configure default successful responses (matching ModelDatabaseOperationOutput structure)
            inserted_id = str(uuid4())
            self._handle_insert.return_value = Mock(
                success=True, result_data={"id": inserted_id}
            )
            # Mock query returns the "inserted" item with full entity structure
            # Tests expect items to have fields from the inserted entity (e.g., file_hash)
            self._handle_query.return_value = Mock(
                success=True,
                result_data={
                    "items": [
                        {
                            "id": inserted_id,
                            "file_hash": "abc123def456789abcdef0123456789abcdef0123456789abcdef01234567890",
                            "namespace": "test_app",
                            "stamp_data": {"type": "test"},
                        }
                    ]
                },
            )
            self._handle_update.return_value = Mock(
                success=True, result_data={"updated": 1}
            )
            self._handle_delete.return_value = Mock(
                success=True, result_data={"deleted": 1}
            )

    return MockDatabaseAdapterNode()
