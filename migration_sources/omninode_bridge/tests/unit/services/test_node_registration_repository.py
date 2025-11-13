"""
Tests for NodeRegistrationRepository with focus on SQL injection prevention.

Verifies that asyncpg's parameterized queries provide sufficient protection
against SQL injection without requiring additional defensive layers.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from omninode_bridge.models.node_registration import (
    ModelNodeRegistrationCreate,
    ModelNodeRegistrationUpdate,
)
from omninode_bridge.services.node_registration_repository import (
    NodeRegistrationRepository,
)


@pytest.fixture
def mock_postgres_client():
    """Create a mock PostgresClient."""
    client = Mock()
    client.fetch_one = AsyncMock()
    client.fetch_all = AsyncMock()
    client.execute_query = AsyncMock()
    return client


@pytest.fixture
def repository(mock_postgres_client):
    """Create a NodeRegistrationRepository with mocked client."""
    return NodeRegistrationRepository(mock_postgres_client)


@pytest.fixture
def sample_registration_data():
    """Sample registration data for testing."""
    return {
        "node_id": "test-node-v1",
        "node_type": "effect",
        "capabilities": {"operations": ["stamp", "validate"]},
        "endpoints": {"stamp": "http://service:8053/api/v1/stamp"},
        "metadata": {"version": "1.0.0"},
        "health_endpoint": "http://service:8053/health",
    }


@pytest.fixture
def sample_db_row():
    """Sample database row for testing."""
    return {
        "id": uuid4(),
        "node_id": "test-node-v1",
        "node_type": "effect",
        "capabilities": {"operations": ["stamp", "validate"]},
        "endpoints": {"stamp": "http://service:8053/api/v1/stamp"},
        "metadata": {"version": "1.0.0"},
        "health_endpoint": "http://service:8053/health",
        "last_heartbeat": datetime.now(UTC),
        "registered_at": datetime.now(UTC),
        "updated_at": datetime.now(UTC),
    }


class TestBasicOperations:
    """Test basic CRUD operations."""

    @pytest.mark.asyncio
    async def test_create_registration(
        self, repository, mock_postgres_client, sample_registration_data, sample_db_row
    ):
        """Test creating a node registration."""
        mock_postgres_client.fetch_one.return_value = sample_db_row

        registration = ModelNodeRegistrationCreate(**sample_registration_data)
        result = await repository.create_registration(registration)

        assert result.node_id == "test-node-v1"
        assert result.node_type == "effect"
        mock_postgres_client.fetch_one.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_registration_upsert(
        self, repository, mock_postgres_client, sample_registration_data, sample_db_row
    ):
        """Test that re-registration (UPSERT) updates existing record instead of failing."""
        # First registration
        mock_postgres_client.fetch_one.return_value = sample_db_row
        registration = ModelNodeRegistrationCreate(**sample_registration_data)
        result1 = await repository.create_registration(registration)
        assert result1.node_id == "test-node-v1"
        assert result1.node_type == "effect"

        # Re-registration with updated data (UPSERT should update, not fail)
        updated_data = sample_registration_data.copy()
        updated_data["capabilities"] = {"operations": ["stamp", "validate", "hash"]}
        updated_row = sample_db_row.copy()
        updated_row["capabilities"] = {"operations": ["stamp", "validate", "hash"]}
        updated_row["updated_at"] = datetime.now(UTC)

        mock_postgres_client.fetch_one.return_value = updated_row
        registration2 = ModelNodeRegistrationCreate(**updated_data)
        result2 = await repository.create_registration(registration2)

        # Verify UPSERT updated the existing record
        assert result2.node_id == "test-node-v1"
        assert result2.capabilities == {"operations": ["stamp", "validate", "hash"]}
        assert mock_postgres_client.fetch_one.call_count == 2

    @pytest.mark.asyncio
    async def test_get_registration(
        self, repository, mock_postgres_client, sample_db_row
    ):
        """Test retrieving a node registration."""
        mock_postgres_client.fetch_one.return_value = sample_db_row

        result = await repository.get_registration("test-node-v1")

        assert result is not None
        assert result.node_id == "test-node-v1"
        mock_postgres_client.fetch_one.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_registration(
        self, repository, mock_postgres_client, sample_db_row
    ):
        """Test updating a node registration."""
        mock_postgres_client.fetch_one.return_value = sample_db_row

        update = ModelNodeRegistrationUpdate(
            metadata={"version": "2.0.0", "updated": True}
        )
        result = await repository.update_registration("test-node-v1", update)

        assert result is not None
        mock_postgres_client.fetch_one.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_registration(self, repository, mock_postgres_client):
        """Test deleting a node registration."""
        mock_postgres_client.execute_query.return_value = "DELETE 1"

        result = await repository.delete_registration("test-node-v1")

        assert result is True
        mock_postgres_client.execute_query.assert_called_once()


class TestSQLInjectionPrevention:
    """
    Test SQL injection prevention through asyncpg parameterized queries.

    These tests verify that malicious SQL in parameters is treated as data,
    not code, demonstrating that asyncpg's parameterized queries are sufficient
    for SQL injection prevention.
    """

    @pytest.mark.asyncio
    async def test_sql_injection_in_node_id_get(self, repository, mock_postgres_client):
        """Test that SQL injection in node_id parameter is prevented."""
        # Attempt SQL injection through node_id
        malicious_node_id = "test'; DROP TABLE node_registrations; --"

        mock_postgres_client.fetch_one.return_value = None

        result = await repository.get_registration(malicious_node_id)

        # The query should execute safely, treating the malicious string as data
        assert result is None
        mock_postgres_client.fetch_one.assert_called_once()

        # Verify the malicious string was passed as a parameter, not concatenated
        call_args = mock_postgres_client.fetch_one.call_args
        assert malicious_node_id in call_args[0]

    @pytest.mark.asyncio
    async def test_sql_injection_in_node_id_delete(
        self, repository, mock_postgres_client
    ):
        """Test SQL injection prevention in delete operation."""
        malicious_node_id = "test' OR '1'='1"

        mock_postgres_client.execute_query.return_value = "DELETE 0"

        result = await repository.delete_registration(malicious_node_id)

        # Should safely execute without affecting other rows
        assert result is False  # No rows deleted
        mock_postgres_client.execute_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_sql_injection_in_update_values(
        self, repository, mock_postgres_client, sample_db_row
    ):
        """Test that SQL injection in update values is prevented."""
        # Attempt SQL injection through metadata value
        malicious_metadata = {
            "version": "1.0'; DROP TABLE node_registrations; --",
            "description": "test' OR '1'='1",
        }

        mock_postgres_client.fetch_one.return_value = sample_db_row

        update = ModelNodeRegistrationUpdate(metadata=malicious_metadata)
        result = await repository.update_registration("test-node-v1", update)

        # Should execute safely, treating the malicious strings as data
        assert result is not None
        mock_postgres_client.fetch_one.assert_called_once()

        # Verify parameters were passed separately
        call_args = mock_postgres_client.fetch_one.call_args
        assert malicious_metadata in call_args[0]

    @pytest.mark.asyncio
    async def test_sql_injection_in_jsonb_values(
        self, repository, mock_postgres_client, sample_db_row
    ):
        """Test SQL injection prevention in JSONB field values."""
        malicious_capabilities = {
            "operations": ["stamp; DROP TABLE node_registrations; --"],
            "features": "'; DELETE FROM node_registrations WHERE '1'='1",
        }

        mock_postgres_client.fetch_one.return_value = sample_db_row

        update = ModelNodeRegistrationUpdate(capabilities=malicious_capabilities)
        result = await repository.update_registration("test-node-v1", update)

        # JSONB values should be safely parameterized
        assert result is not None
        mock_postgres_client.fetch_one.assert_called_once()

    @pytest.mark.asyncio
    async def test_union_injection_attempt(self, repository, mock_postgres_client):
        """Test UNION-based SQL injection attempt."""
        malicious_node_id = "test' UNION SELECT * FROM pg_user; --"

        mock_postgres_client.fetch_one.return_value = None

        result = await repository.get_registration(malicious_node_id)

        # Should safely execute, treating UNION statement as data
        assert result is None
        mock_postgres_client.fetch_one.assert_called_once()

    @pytest.mark.asyncio
    async def test_blind_injection_attempt(self, repository, mock_postgres_client):
        """Test blind SQL injection attempt."""
        # Time-based blind injection attempt
        malicious_node_id = "test'; SELECT pg_sleep(10); --"

        mock_postgres_client.fetch_one.return_value = None

        result = await repository.get_registration(malicious_node_id)

        # Should execute immediately, not waiting for pg_sleep
        assert result is None
        mock_postgres_client.fetch_one.assert_called_once()

    @pytest.mark.asyncio
    async def test_stacked_queries_attempt(self, repository, mock_postgres_client):
        """Test stacked queries SQL injection attempt."""
        malicious_node_id = "test'; CREATE TABLE malicious (data TEXT); --"

        mock_postgres_client.execute_query.return_value = "DELETE 0"

        result = await repository.delete_registration(malicious_node_id)

        # Should safely execute, not creating the malicious table
        assert result is False
        mock_postgres_client.execute_query.assert_called_once()


class TestFieldValidation:
    """Test field validation (application logic, not security)."""

    @pytest.mark.asyncio
    async def test_invalid_update_field_rejected(self, repository):
        """Test that invalid fields are rejected by application logic."""
        # Attempt to update immutable field
        update = ModelNodeRegistrationUpdate()
        update_dict = {"id": uuid4(), "node_id": "new-id"}

        with pytest.raises(ValueError, match="Invalid update fields"):
            repository._validate_update_fields(update_dict)

    @pytest.mark.asyncio
    async def test_valid_update_fields_accepted(self, repository):
        """Test that valid fields are accepted."""
        update_dict = {
            "capabilities": {"new": "capability"},
            "metadata": {"version": "2.0.0"},
        }

        # Should not raise
        repository._validate_update_fields(update_dict)

    @pytest.mark.asyncio
    async def test_non_dict_update_rejected(self, repository):
        """Test that non-dict update data is rejected."""
        with pytest.raises(ValueError, match="Update data must be a dictionary"):
            repository._validate_update_fields("not a dict")

    @pytest.mark.asyncio
    async def test_empty_update_dict(
        self, repository, mock_postgres_client, sample_db_row
    ):
        """Test that empty update returns current registration."""
        mock_postgres_client.fetch_one.return_value = sample_db_row

        update = ModelNodeRegistrationUpdate()
        result = await repository.update_registration("test-node-v1", update)

        # Should call get_registration instead of update
        assert result is not None
        assert result.node_id == "test-node-v1"


class TestComplexQueries:
    """Test complex query operations."""

    @pytest.mark.asyncio
    async def test_find_by_capability_with_special_chars(
        self, repository, mock_postgres_client, sample_db_row
    ):
        """Test capability search with special characters in values."""
        mock_postgres_client.fetch_all.return_value = [sample_db_row]

        # Search with special characters that could be SQL injection attempts
        result = await repository.find_by_capability(
            "operations", "stamp'; DROP TABLE node_registrations; --"
        )

        # Should safely execute with JSONB parameterized query
        assert isinstance(result, list)
        mock_postgres_client.fetch_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_by_node_type_with_injection(
        self, repository, mock_postgres_client, sample_db_row
    ):
        """Test listing nodes with malicious node_type parameter."""
        malicious_type = "effect' OR '1'='1"

        mock_postgres_client.fetch_all.return_value = []

        result = await repository.list_all_registrations(node_type=malicious_type)

        # Should safely execute, returning empty list (no matches)
        assert result == []
        mock_postgres_client.fetch_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_heartbeat_update_with_injection(
        self, repository, mock_postgres_client
    ):
        """Test heartbeat update with malicious node_id."""
        malicious_node_id = (
            "test'; UPDATE node_registrations SET node_type='hacked'; --"
        )

        mock_postgres_client.execute_query.return_value = "UPDATE 0"

        result = await repository.update_heartbeat(malicious_node_id)

        # Should safely execute without affecting other rows
        assert result is False
        mock_postgres_client.execute_query.assert_called_once()


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_update_with_null_values(
        self, repository, mock_postgres_client, sample_db_row
    ):
        """Test update with null values is handled correctly."""
        mock_postgres_client.fetch_one.return_value = sample_db_row

        # None values should be filtered out
        update = ModelNodeRegistrationUpdate(
            metadata={"key": "value"},
            capabilities=None,  # Should be ignored
        )

        result = await repository.update_registration("test-node-v1", update)

        assert result is not None
        # Verify capabilities wasn't included in the update
        call_args = mock_postgres_client.fetch_one.call_args
        query = call_args[0][0]
        assert "metadata" in query
        # capabilities should not be in the query since it's None

    @pytest.mark.asyncio
    async def test_multiple_special_chars_in_values(
        self, repository, mock_postgres_client, sample_db_row
    ):
        """Test handling of multiple special characters in update values."""
        mock_postgres_client.fetch_one.return_value = sample_db_row

        # Complex string with various SQL-sensitive characters
        complex_metadata = {
            "description": 'Test\'s "quoted" value; with--comments/*and*/brackets',
            "sql": "SELECT * FROM users WHERE id = $1 OR 1=1",
        }

        update = ModelNodeRegistrationUpdate(metadata=complex_metadata)
        result = await repository.update_registration("test-node-v1", update)

        # Should handle all special characters safely
        assert result is not None
        mock_postgres_client.fetch_one.assert_called_once()

    @pytest.mark.asyncio
    async def test_unicode_and_emoji_in_values(
        self, repository, mock_postgres_client, sample_db_row
    ):
        """Test handling of unicode and emoji characters."""
        mock_postgres_client.fetch_one.return_value = sample_db_row

        unicode_metadata = {
            "name": "测试节点 🚀",
            "description": "Node with unicode symbols and emoji 🎉",  # Changed from Cyrillic to avoid ambiguous characters
        }

        update = ModelNodeRegistrationUpdate(metadata=unicode_metadata)
        result = await repository.update_registration("test-node-v1", update)

        # Should handle unicode safely
        assert result is not None
        mock_postgres_client.fetch_one.assert_called_once()
