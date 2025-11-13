"""
Tests for ActionDedupService - Action deduplication with TTL-based expiration.

Verifies:
- Duplicate detection via composite key (workflow_key, action_id)
- Result hash validation
- TTL expiration behavior
- Concurrent access handling
- Metrics tracking
- Error handling and recovery
"""

import hashlib
import json

# Import service directly (avoid infrastructure imports that require omnibase_core)
import sys
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

sys.path.insert(0, "src")
from omninode_bridge.services.action_dedup import ActionDedupService


@pytest.fixture
def mock_postgres_client():
    """Create a mock PostgresClient."""
    client = Mock()
    client.pool = Mock()  # Mock pool to simulate connected state
    client.fetch_one = AsyncMock()
    client.execute_query = AsyncMock()
    return client


@pytest.fixture
def dedup_service(mock_postgres_client):
    """Create an ActionDedupService with mocked client."""
    return ActionDedupService(mock_postgres_client)


@pytest.fixture
def sample_action_id():
    """Generate a sample action ID."""
    return uuid4()


@pytest.fixture
def sample_result_hash():
    """Generate a sample SHA256 result hash."""
    result = {"status": "completed", "items": 100}
    return hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()


class TestShouldProcess:
    """Test should_process method for duplicate detection."""

    @pytest.mark.asyncio
    async def test_should_process_new_action(
        self, dedup_service, mock_postgres_client, sample_action_id
    ):
        """Test that a new action should be processed."""
        # Mock: action not found in database
        mock_postgres_client.fetch_one.return_value = None

        result = await dedup_service.should_process("workflow-123", sample_action_id)

        assert result is True
        assert dedup_service._metrics["dedup_checks_total"] == 1
        assert dedup_service._metrics["dedup_hits_total"] == 0
        mock_postgres_client.fetch_one.assert_called_once()

    @pytest.mark.asyncio
    async def test_should_process_duplicate_action(
        self, dedup_service, mock_postgres_client, sample_action_id
    ):
        """Test that a duplicate action should not be processed."""
        # Mock: action exists in database
        mock_postgres_client.fetch_one.return_value = {"exists": 1}

        result = await dedup_service.should_process("workflow-123", sample_action_id)

        assert result is False
        assert dedup_service._metrics["dedup_checks_total"] == 1
        assert dedup_service._metrics["dedup_hits_total"] == 1
        mock_postgres_client.fetch_one.assert_called_once()

    @pytest.mark.asyncio
    async def test_should_process_database_error(
        self, dedup_service, mock_postgres_client, sample_action_id
    ):
        """Test that database errors default to processing."""
        # Mock: database error
        mock_postgres_client.fetch_one.side_effect = Exception("Database error")

        result = await dedup_service.should_process("workflow-123", sample_action_id)

        # Should default to processing on error (safe failure mode)
        assert result is True
        mock_postgres_client.fetch_one.assert_called_once()

    @pytest.mark.asyncio
    async def test_should_process_no_connection(
        self, dedup_service, mock_postgres_client, sample_action_id
    ):
        """Test that missing connection raises RuntimeError."""
        mock_postgres_client.pool = None

        with pytest.raises(RuntimeError, match="PostgreSQL client not connected"):
            await dedup_service.should_process("workflow-123", sample_action_id)

    @pytest.mark.asyncio
    async def test_should_process_multiple_workflows(
        self, dedup_service, mock_postgres_client, sample_action_id
    ):
        """Test that same action_id in different workflows is allowed."""
        # First workflow: not duplicate
        mock_postgres_client.fetch_one.return_value = None
        result1 = await dedup_service.should_process("workflow-1", sample_action_id)
        assert result1 is True

        # Second workflow: also not duplicate (different workflow_key)
        mock_postgres_client.fetch_one.return_value = None
        result2 = await dedup_service.should_process("workflow-2", sample_action_id)
        assert result2 is True

        # Both should have been checked
        assert mock_postgres_client.fetch_one.call_count == 2


class TestRecordProcessed:
    """Test record_processed method for tracking processed actions."""

    @pytest.mark.asyncio
    async def test_record_processed_success(
        self, dedup_service, mock_postgres_client, sample_action_id, sample_result_hash
    ):
        """Test successful recording of processed action."""
        mock_postgres_client.execute_query.return_value = "INSERT 0 1"

        await dedup_service.record_processed(
            "workflow-123", sample_action_id, sample_result_hash, ttl_hours=6
        )

        assert dedup_service._metrics["dedup_records_total"] == 1
        mock_postgres_client.execute_query.assert_called_once()

        # Verify the SQL call
        call_args = mock_postgres_client.execute_query.call_args
        assert "INSERT INTO action_dedup_log" in call_args[0][0]
        assert call_args[0][1] == "workflow-123"
        assert call_args[0][2] == sample_action_id
        assert call_args[0][3] == sample_result_hash

    @pytest.mark.asyncio
    async def test_record_processed_invalid_hash(
        self, dedup_service, mock_postgres_client, sample_action_id
    ):
        """Test that invalid result hash raises ValueError."""
        invalid_hash = "not-a-valid-sha256-hash"

        with pytest.raises(ValueError, match="64-character SHA256 hex string"):
            await dedup_service.record_processed(
                "workflow-123", sample_action_id, invalid_hash
            )

        # Should not have recorded
        assert dedup_service._metrics["dedup_records_total"] == 0

    @pytest.mark.asyncio
    async def test_record_processed_database_error(
        self, dedup_service, mock_postgres_client, sample_action_id, sample_result_hash
    ):
        """Test that database errors are handled gracefully."""
        mock_postgres_client.execute_query.side_effect = Exception("Database error")

        # Should not raise - errors are logged and swallowed
        await dedup_service.record_processed(
            "workflow-123", sample_action_id, sample_result_hash
        )

        # Metrics should not be updated on error
        assert dedup_service._metrics["dedup_records_total"] == 0

    @pytest.mark.asyncio
    async def test_record_processed_no_connection(
        self, dedup_service, mock_postgres_client, sample_action_id, sample_result_hash
    ):
        """Test that missing connection raises RuntimeError."""
        mock_postgres_client.pool = None

        with pytest.raises(RuntimeError, match="PostgreSQL client not connected"):
            await dedup_service.record_processed(
                "workflow-123", sample_action_id, sample_result_hash
            )

    @pytest.mark.asyncio
    async def test_record_processed_custom_ttl(
        self, dedup_service, mock_postgres_client, sample_action_id, sample_result_hash
    ):
        """Test recording with custom TTL."""
        mock_postgres_client.execute_query.return_value = "INSERT 0 1"

        await dedup_service.record_processed(
            "workflow-123", sample_action_id, sample_result_hash, ttl_hours=12
        )

        # Verify TTL was used
        call_args = mock_postgres_client.execute_query.call_args
        expires_at = call_args[0][5]
        processed_at = call_args[0][4]

        # Should be 12 hours apart (with small tolerance for execution time)
        time_diff = (expires_at - processed_at).total_seconds()
        assert 12 * 3600 - 5 < time_diff < 12 * 3600 + 5


class TestCleanupExpired:
    """Test cleanup_expired method for TTL-based cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_expired_success(self, dedup_service, mock_postgres_client):
        """Test successful cleanup of expired entries."""
        # Mock: 42 entries deleted
        mock_postgres_client.execute_query.return_value = "DELETE 42"

        deleted_count = await dedup_service.cleanup_expired()

        assert deleted_count == 42
        assert dedup_service._metrics["dedup_cleanup_deleted_total"] == 42
        mock_postgres_client.execute_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_expired_no_entries(
        self, dedup_service, mock_postgres_client
    ):
        """Test cleanup when no entries are expired."""
        # Mock: 0 entries deleted
        mock_postgres_client.execute_query.return_value = "DELETE 0"

        deleted_count = await dedup_service.cleanup_expired()

        assert deleted_count == 0
        assert dedup_service._metrics["dedup_cleanup_deleted_total"] == 0

    @pytest.mark.asyncio
    async def test_cleanup_expired_database_error(
        self, dedup_service, mock_postgres_client
    ):
        """Test that database errors are handled gracefully."""
        mock_postgres_client.execute_query.side_effect = Exception("Database error")

        deleted_count = await dedup_service.cleanup_expired()

        # Should return 0 on error (not raise)
        assert deleted_count == 0
        assert dedup_service._metrics["dedup_cleanup_deleted_total"] == 0

    @pytest.mark.asyncio
    async def test_cleanup_expired_no_connection(
        self, dedup_service, mock_postgres_client
    ):
        """Test that missing connection raises RuntimeError."""
        mock_postgres_client.pool = None

        with pytest.raises(RuntimeError, match="PostgreSQL client not connected"):
            await dedup_service.cleanup_expired()

    @pytest.mark.asyncio
    async def test_cleanup_expired_malformed_result(
        self, dedup_service, mock_postgres_client
    ):
        """Test handling of malformed database result."""
        # Mock: malformed result
        mock_postgres_client.execute_query.return_value = "INVALID RESULT"

        deleted_count = await dedup_service.cleanup_expired()

        # Should handle gracefully and return 0
        assert deleted_count == 0


class TestGetDedupEntry:
    """Test get_dedup_entry method for retrieving dedup records."""

    @pytest.mark.asyncio
    async def test_get_dedup_entry_exists(
        self, dedup_service, mock_postgres_client, sample_action_id, sample_result_hash
    ):
        """Test retrieving an existing dedup entry."""
        now = datetime.now(UTC)
        expires_at = now + timedelta(hours=6)

        mock_postgres_client.fetch_one.return_value = {
            "workflow_key": "workflow-123",
            "action_id": sample_action_id,
            "result_hash": sample_result_hash,
            "processed_at": now,
            "expires_at": expires_at,
        }

        entry = await dedup_service.get_dedup_entry("workflow-123", sample_action_id)

        assert entry is not None
        # Check that it's a dictionary with expected keys
        assert isinstance(entry, dict)
        assert entry["workflow_key"] == "workflow-123"
        assert entry["action_id"] == sample_action_id
        assert entry["result_hash"] == sample_result_hash

    @pytest.mark.asyncio
    async def test_get_dedup_entry_not_exists(
        self, dedup_service, mock_postgres_client, sample_action_id
    ):
        """Test retrieving a non-existent dedup entry."""
        mock_postgres_client.fetch_one.return_value = None

        entry = await dedup_service.get_dedup_entry("workflow-123", sample_action_id)

        assert entry is None

    @pytest.mark.asyncio
    async def test_get_dedup_entry_database_error(
        self, dedup_service, mock_postgres_client, sample_action_id
    ):
        """Test that database errors return None."""
        mock_postgres_client.fetch_one.side_effect = Exception("Database error")

        entry = await dedup_service.get_dedup_entry("workflow-123", sample_action_id)

        assert entry is None

    @pytest.mark.asyncio
    async def test_get_dedup_entry_no_connection(
        self, dedup_service, mock_postgres_client, sample_action_id
    ):
        """Test that missing connection raises RuntimeError."""
        mock_postgres_client.pool = None

        with pytest.raises(RuntimeError, match="PostgreSQL client not connected"):
            await dedup_service.get_dedup_entry("workflow-123", sample_action_id)


class TestMetrics:
    """Test metrics tracking functionality."""

    @pytest.mark.asyncio
    async def test_get_metrics(self, dedup_service, mock_postgres_client):
        """Test retrieving metrics."""
        metrics = dedup_service.get_metrics()

        assert "dedup_checks_total" in metrics
        assert "dedup_hits_total" in metrics
        assert "dedup_records_total" in metrics
        assert "dedup_cleanup_deleted_total" in metrics
        assert all(isinstance(v, int) for v in metrics.values())

    @pytest.mark.asyncio
    async def test_reset_metrics(
        self, dedup_service, mock_postgres_client, sample_action_id
    ):
        """Test resetting metrics."""
        # Generate some metrics
        mock_postgres_client.fetch_one.return_value = {"exists": 1}
        await dedup_service.should_process("workflow-123", sample_action_id)

        # Verify metrics exist
        assert dedup_service._metrics["dedup_checks_total"] > 0

        # Reset
        dedup_service.reset_metrics()

        # Verify all metrics are 0
        metrics = dedup_service.get_metrics()
        assert all(v == 0 for v in metrics.values())

    @pytest.mark.asyncio
    async def test_metrics_isolation(self, dedup_service):
        """Test that get_metrics returns a copy (not reference)."""
        metrics1 = dedup_service.get_metrics()
        metrics1["dedup_checks_total"] = 999

        metrics2 = dedup_service.get_metrics()

        # Original metrics should not be affected
        assert metrics2["dedup_checks_total"] == 0


class TestConcurrentAccess:
    """Test concurrent access handling."""

    @pytest.mark.asyncio
    async def test_concurrent_should_process(self, dedup_service, mock_postgres_client):
        """Test concurrent calls to should_process."""
        import asyncio

        action_ids = [uuid4() for _ in range(10)]

        # Mock: all new actions
        mock_postgres_client.fetch_one.return_value = None

        # Run concurrently
        results = await asyncio.gather(
            *[
                dedup_service.should_process("workflow-123", action_id)
                for action_id in action_ids
            ]
        )

        # All should return True
        assert all(results)
        # Should have checked all 10
        assert dedup_service._metrics["dedup_checks_total"] == 10
        assert dedup_service._metrics["dedup_hits_total"] == 0

    @pytest.mark.asyncio
    async def test_concurrent_record_processed(
        self, dedup_service, mock_postgres_client, sample_result_hash
    ):
        """Test concurrent calls to record_processed."""
        import asyncio

        action_ids = [uuid4() for _ in range(10)]
        mock_postgres_client.execute_query.return_value = "INSERT 0 1"

        # Run concurrently
        await asyncio.gather(
            *[
                dedup_service.record_processed(
                    "workflow-123", action_id, sample_result_hash
                )
                for action_id in action_ids
            ]
        )

        # Should have recorded all 10
        assert dedup_service._metrics["dedup_records_total"] == 10


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    @pytest.mark.asyncio
    async def test_full_dedup_workflow(
        self, dedup_service, mock_postgres_client, sample_result_hash
    ):
        """Test a complete deduplication workflow."""
        action_id = uuid4()

        # 1. Check if should process (first time - yes)
        mock_postgres_client.fetch_one.return_value = None
        should_process1 = await dedup_service.should_process("workflow-123", action_id)
        assert should_process1 is True

        # 2. Process action and record
        mock_postgres_client.execute_query.return_value = "INSERT 0 1"
        await dedup_service.record_processed(
            "workflow-123", action_id, sample_result_hash
        )

        # 3. Check again (duplicate - no)
        mock_postgres_client.fetch_one.return_value = {"exists": 1}
        should_process2 = await dedup_service.should_process("workflow-123", action_id)
        assert should_process2 is False

        # Verify metrics
        assert dedup_service._metrics["dedup_checks_total"] == 2
        assert dedup_service._metrics["dedup_hits_total"] == 1
        assert dedup_service._metrics["dedup_records_total"] == 1

    @pytest.mark.asyncio
    async def test_retry_scenario(
        self, dedup_service, mock_postgres_client, sample_result_hash
    ):
        """Test retry scenario with duplicate detection."""
        action_id = uuid4()

        # First attempt
        mock_postgres_client.fetch_one.return_value = None
        should_process1 = await dedup_service.should_process("workflow-123", action_id)
        assert should_process1 is True

        mock_postgres_client.execute_query.return_value = "INSERT 0 1"
        await dedup_service.record_processed(
            "workflow-123", action_id, sample_result_hash
        )

        # Retry (simulated)
        mock_postgres_client.fetch_one.return_value = {"exists": 1}
        should_process2 = await dedup_service.should_process("workflow-123", action_id)
        assert should_process2 is False  # Should skip retry

    @pytest.mark.asyncio
    async def test_cleanup_after_ttl(self, dedup_service, mock_postgres_client):
        """Test cleanup after TTL expiration."""
        # Record some actions
        mock_postgres_client.execute_query.return_value = "INSERT 0 1"
        action_ids = [uuid4() for _ in range(5)]
        sample_hash = hashlib.sha256(b"test").hexdigest()

        for action_id in action_ids:
            await dedup_service.record_processed(
                "workflow-123", action_id, sample_hash, ttl_hours=1
            )

        # Simulate time passing and cleanup
        mock_postgres_client.execute_query.return_value = "DELETE 5"
        deleted_count = await dedup_service.cleanup_expired()

        assert deleted_count == 5
        assert dedup_service._metrics["dedup_cleanup_deleted_total"] == 5
