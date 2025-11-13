"""
Unit tests for CanonicalStoreService.

Tests cover:
- get_state retrieval and error handling
- try_commit success scenarios (optimistic locking)
- try_commit conflict scenarios (version mismatch)
- Event emission verification (StateCommitted, StateConflict)
- Provenance tracking validation
- Metrics instrumentation
- Input validation and edge cases

ONEX v2.0 Compliance:
- Comprehensive test coverage for all methods
- Mock-based testing for database and Kafka
- Validation of error handling and edge cases
"""

# Direct imports to avoid omnibase_core dependency chain through infrastructure/__init__
import json
import sys
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, "src")

# Import ModelWorkflowState directly to avoid infrastructure.__init__ import chain
from omninode_bridge.infrastructure.entities.model_workflow_state import (
    ModelWorkflowState,
)
from omninode_bridge.services.canonical_store import (
    CanonicalStoreService,
    EventStateCommitted,
    EventStateConflict,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_postgres_client():
    """Mock PostgreSQL client."""
    client = MagicMock()
    client.fetch_one = AsyncMock()
    return client


@pytest.fixture
def mock_kafka_client():
    """Mock Kafka client."""
    client = MagicMock()
    client.publish_event = AsyncMock()
    return client


@pytest.fixture
def service(mock_postgres_client, mock_kafka_client):
    """CanonicalStoreService instance with mocked dependencies."""
    return CanonicalStoreService(mock_postgres_client, mock_kafka_client)


@pytest.fixture
def sample_workflow_state():
    """Sample workflow state for testing."""
    return {
        "workflow_key": "test-workflow-123",
        "version": 1,
        "state": {"items": [], "count": 0},
        "updated_at": datetime.now(UTC),
        "schema_version": 1,
        "provenance": {
            "effect_id": "effect-001",
            "timestamp": datetime.now(UTC).isoformat(),
            "action_id": "action-001",
        },
    }


@pytest.fixture
def sample_provenance():
    """Sample provenance metadata."""
    return {
        "effect_id": "effect-002",
        "timestamp": datetime.now(UTC).isoformat(),
        "action_id": "action-002",
    }


# ============================================================================
# Test get_state
# ============================================================================


@pytest.mark.asyncio
async def test_get_state_success(service, mock_postgres_client, sample_workflow_state):
    """Test successful state retrieval."""
    # Setup mock
    mock_postgres_client.fetch_one.return_value = sample_workflow_state

    # Execute
    result = await service.get_state("test-workflow-123")

    # Verify
    assert isinstance(result, ModelWorkflowState)
    assert result.workflow_key == "test-workflow-123"
    assert result.version == 1
    assert result.state == {"items": [], "count": 0}
    assert result.provenance["effect_id"] == "effect-001"

    # Verify database query
    mock_postgres_client.fetch_one.assert_called_once()
    call_args = mock_postgres_client.fetch_one.call_args
    assert "SELECT workflow_key, version, state" in call_args[0][0]
    assert call_args[0][1] == "test-workflow-123"

    # Verify metrics
    assert service._metrics_get_state_total == 1
    assert service._metrics_get_state_errors == 0


@pytest.mark.asyncio
async def test_get_state_not_found(service, mock_postgres_client):
    """Test state retrieval when workflow not found."""
    # Setup mock to return None (not found)
    mock_postgres_client.fetch_one.return_value = None

    # Execute and verify exception
    with pytest.raises(RuntimeError) as exc_info:
        await service.get_state("nonexistent-workflow")

    assert "Workflow state not found" in str(exc_info.value)
    assert "nonexistent-workflow" in str(exc_info.value)

    # Verify metrics
    assert service._metrics_get_state_total == 1
    assert service._metrics_get_state_errors == 1


@pytest.mark.asyncio
async def test_get_state_empty_workflow_key(service):
    """Test state retrieval with empty workflow_key."""
    # Test empty string
    with pytest.raises(ValueError) as exc_info:
        await service.get_state("")
    assert "workflow_key must be non-empty" in str(exc_info.value)

    # Test whitespace-only string
    with pytest.raises(ValueError) as exc_info:
        await service.get_state("   ")
    assert "workflow_key must be non-empty" in str(exc_info.value)

    # Note: Validation errors occur before metrics are incremented,
    # so errors counter should still be 0
    assert service._metrics_get_state_errors == 0


@pytest.mark.asyncio
async def test_get_state_database_error(service, mock_postgres_client):
    """Test state retrieval when database error occurs."""
    # Setup mock to raise exception
    mock_postgres_client.fetch_one.side_effect = Exception("Database connection lost")

    # Execute and verify exception
    with pytest.raises(RuntimeError) as exc_info:
        await service.get_state("test-workflow")

    assert "Database error retrieving workflow state" in str(exc_info.value)

    # Verify metrics
    assert service._metrics_get_state_total == 1
    assert service._metrics_get_state_errors == 1


# ============================================================================
# Test try_commit - Success Scenarios
# ============================================================================


@pytest.mark.asyncio
async def test_try_commit_success(
    service, mock_postgres_client, mock_kafka_client, sample_provenance
):
    """Test successful state commit with version increment."""
    # Setup mock - simulate successful UPDATE (returns updated row)
    updated_row = {
        "workflow_key": "test-workflow-123",
        "version": 2,  # Incremented from 1
        "state": {"items": [1], "count": 1},
        "updated_at": datetime.now(UTC),
        "schema_version": 1,
        "provenance": sample_provenance,
    }
    mock_postgres_client.fetch_one.return_value = updated_row

    # Execute
    result = await service.try_commit(
        workflow_key="test-workflow-123",
        expected_version=1,
        state_prime={"items": [1], "count": 1},
        provenance=sample_provenance,
    )

    # Verify result type and content
    assert isinstance(result, EventStateCommitted)
    assert result.workflow_key == "test-workflow-123"
    assert result.new_version == 2
    assert result.state_snapshot == {"items": [1], "count": 1}
    assert result.provenance == sample_provenance
    assert result.event_type == "state_committed"

    # Verify database UPDATE query
    mock_postgres_client.fetch_one.assert_called_once()
    call_args = mock_postgres_client.fetch_one.call_args
    assert "UPDATE workflow_state" in call_args[0][0]
    assert "SET" in call_args[0][0]
    assert "version = version + 1" in call_args[0][0]
    assert "WHERE workflow_key = $3 AND version = $4" in call_args[0][0]
    # Verify parameters (service converts dicts to JSON strings)
    assert json.loads(call_args[0][1]) == {"items": [1], "count": 1}  # state_prime
    assert json.loads(call_args[0][2]) == sample_provenance  # provenance
    assert call_args[0][3] == "test-workflow-123"  # workflow_key
    assert call_args[0][4] == 1  # expected_version

    # Verify Kafka event published
    mock_kafka_client.publish_event.assert_called_once()
    kafka_call_args = mock_kafka_client.publish_event.call_args
    assert kafka_call_args[1]["topic"] == "omninode_bridge_state_committed_v1"
    assert kafka_call_args[1]["key"] == "test-workflow-123"
    event_dict = kafka_call_args[1]["event"]
    assert event_dict["event_type"] == "state_committed"
    assert event_dict["new_version"] == 2

    # Verify metrics
    assert service._metrics_commits_total == 1
    assert service._metrics_conflicts_total == 0


# ============================================================================
# Test try_commit - Conflict Scenarios
# ============================================================================


@pytest.mark.asyncio
async def test_try_commit_version_conflict(
    service,
    mock_postgres_client,
    mock_kafka_client,
    sample_provenance,
    sample_workflow_state,
):
    """Test state commit with version conflict (concurrent modification)."""
    # Setup mocks
    # First call (UPDATE) returns None (conflict)
    # Second call (get_state for conflict info) returns current state
    mock_postgres_client.fetch_one.side_effect = [
        None,  # UPDATE failed (version mismatch)
        {
            "workflow_key": "test-workflow-123",
            "version": 2,  # Actual version is 2, expected was 1
            "state": {"items": [99], "count": 1},
            "updated_at": datetime.now(UTC),
            "schema_version": 1,
            "provenance": {
                "effect_id": "other-effect",
                "timestamp": datetime.now(UTC).isoformat(),
            },
        },
    ]

    # Execute
    result = await service.try_commit(
        workflow_key="test-workflow-123",
        expected_version=1,
        state_prime={"items": [1], "count": 1},
        provenance=sample_provenance,
    )

    # Verify result type and content
    assert isinstance(result, EventStateConflict)
    assert result.workflow_key == "test-workflow-123"
    assert result.expected_version == 1
    assert result.actual_version == 2
    assert result.reason == "concurrent_modification"
    assert result.event_type == "state_conflict"

    # Verify database calls
    assert mock_postgres_client.fetch_one.call_count == 2
    # First call: UPDATE attempt
    # Second call: get_state for conflict info

    # Verify Kafka conflict event published
    mock_kafka_client.publish_event.assert_called_once()
    kafka_call_args = mock_kafka_client.publish_event.call_args
    assert kafka_call_args[1]["topic"] == "omninode_bridge_state_conflicts_v1"
    assert kafka_call_args[1]["key"] == "test-workflow-123"
    event_dict = kafka_call_args[1]["event"]
    assert event_dict["event_type"] == "state_conflict"
    assert event_dict["expected_version"] == 1
    assert event_dict["actual_version"] == 2

    # Verify metrics
    assert service._metrics_commits_total == 0
    assert service._metrics_conflicts_total == 1


# ============================================================================
# Test try_commit - Input Validation
# ============================================================================


@pytest.mark.asyncio
async def test_try_commit_empty_workflow_key(service, sample_provenance):
    """Test commit with empty workflow_key."""
    with pytest.raises(ValueError) as exc_info:
        await service.try_commit(
            workflow_key="",
            expected_version=1,
            state_prime={"items": []},
            provenance=sample_provenance,
        )
    assert "workflow_key must be non-empty" in str(exc_info.value)


@pytest.mark.asyncio
async def test_try_commit_invalid_version(service, sample_provenance):
    """Test commit with invalid expected_version."""
    # Version < 1
    with pytest.raises(ValueError) as exc_info:
        await service.try_commit(
            workflow_key="test-workflow",
            expected_version=0,
            state_prime={"items": []},
            provenance=sample_provenance,
        )
    assert "expected_version must be >= 1" in str(exc_info.value)

    # Negative version
    with pytest.raises(ValueError) as exc_info:
        await service.try_commit(
            workflow_key="test-workflow",
            expected_version=-1,
            state_prime={"items": []},
            provenance=sample_provenance,
        )
    assert "expected_version must be >= 1" in str(exc_info.value)


@pytest.mark.asyncio
async def test_try_commit_invalid_state_prime(service, sample_provenance):
    """Test commit with invalid state_prime."""
    # Non-dict state_prime
    with pytest.raises(ValueError) as exc_info:
        await service.try_commit(
            workflow_key="test-workflow",
            expected_version=1,
            state_prime="not a dict",
            provenance=sample_provenance,
        )
    assert "state_prime must be dict" in str(exc_info.value)

    # Empty dict state_prime
    with pytest.raises(ValueError) as exc_info:
        await service.try_commit(
            workflow_key="test-workflow",
            expected_version=1,
            state_prime={},
            provenance=sample_provenance,
        )
    assert "state_prime cannot be empty" in str(exc_info.value)


@pytest.mark.asyncio
async def test_try_commit_invalid_provenance(service):
    """Test commit with invalid provenance."""
    # Non-dict provenance
    with pytest.raises(ValueError) as exc_info:
        await service.try_commit(
            workflow_key="test-workflow",
            expected_version=1,
            state_prime={"items": []},
            provenance="not a dict",
        )
    assert "provenance must be dict" in str(exc_info.value)

    # Missing required fields
    with pytest.raises(ValueError) as exc_info:
        await service.try_commit(
            workflow_key="test-workflow",
            expected_version=1,
            state_prime={"items": []},
            provenance={"effect_id": "only-one-field"},
        )
    assert "provenance missing required fields" in str(exc_info.value)
    assert "timestamp" in str(exc_info.value)


# ============================================================================
# Test Event Publishing
# ============================================================================


@pytest.mark.asyncio
async def test_event_publishing_without_kafka_client(
    mock_postgres_client, sample_provenance
):
    """Test service works without Kafka client (events logged but not published)."""
    # Create service without Kafka client
    service = CanonicalStoreService(mock_postgres_client, kafka_client=None)

    # Setup mock for successful commit
    updated_row = {
        "workflow_key": "test-workflow",
        "version": 2,
        "state": {"items": [1]},
        "updated_at": datetime.now(UTC),
        "schema_version": 1,
        "provenance": sample_provenance,
    }
    mock_postgres_client.fetch_one.return_value = updated_row

    # Execute - should succeed without Kafka
    result = await service.try_commit(
        workflow_key="test-workflow",
        expected_version=1,
        state_prime={"items": [1]},
        provenance=sample_provenance,
    )

    # Verify success
    assert isinstance(result, EventStateCommitted)
    assert result.new_version == 2


@pytest.mark.asyncio
async def test_event_publishing_kafka_error_doesnt_fail_commit(
    service, mock_postgres_client, mock_kafka_client, sample_provenance
):
    """Test that Kafka publishing error doesn't fail the commit."""
    # Setup mock for successful commit
    updated_row = {
        "workflow_key": "test-workflow",
        "version": 2,
        "state": {"items": [1]},
        "updated_at": datetime.now(UTC),
        "schema_version": 1,
        "provenance": sample_provenance,
    }
    mock_postgres_client.fetch_one.return_value = updated_row

    # Setup Kafka to fail
    mock_kafka_client.publish_event.side_effect = Exception("Kafka connection failed")

    # Execute - should still succeed despite Kafka error
    result = await service.try_commit(
        workflow_key="test-workflow",
        expected_version=1,
        state_prime={"items": [1]},
        provenance=sample_provenance,
    )

    # Verify commit succeeded
    assert isinstance(result, EventStateCommitted)
    assert result.new_version == 2

    # Verify metrics still tracked
    assert service._metrics_commits_total == 1


# ============================================================================
# Test Metrics
# ============================================================================


@pytest.mark.asyncio
async def test_get_metrics(
    service, mock_postgres_client, mock_kafka_client, sample_provenance
):
    """Test metrics collection and retrieval."""
    # Perform various operations
    # 1. Successful get_state
    mock_postgres_client.fetch_one.return_value = {
        "workflow_key": "test",
        "version": 1,
        "state": {"items": []},  # Must be non-empty dict
        "updated_at": datetime.now(UTC),
        "schema_version": 1,
        "provenance": sample_provenance,
    }
    await service.get_state("test")

    # 2. Failed get_state
    mock_postgres_client.fetch_one.return_value = None
    try:
        await service.get_state("nonexistent")
    except RuntimeError:
        pass

    # 3. Successful commit
    mock_postgres_client.fetch_one.return_value = {
        "workflow_key": "test",
        "version": 2,
        "state": {"data": "new"},
        "updated_at": datetime.now(UTC),
        "schema_version": 1,
        "provenance": sample_provenance,
    }
    await service.try_commit("test", 1, {"data": "new"}, sample_provenance)

    # 4. Conflict
    mock_postgres_client.fetch_one.side_effect = [
        None,  # UPDATE failed
        {
            "workflow_key": "test",
            "version": 3,
            "state": {"data": "conflict"},
            "updated_at": datetime.now(UTC),
            "schema_version": 1,
            "provenance": sample_provenance,
        },
    ]
    await service.try_commit("test", 2, {"data": "newer"}, sample_provenance)

    # Verify metrics
    metrics = service.get_metrics()
    assert metrics["canonical_commits_total"] == 1
    assert metrics["canonical_conflicts_total"] == 1
    # get_state_total: 1 (explicit) + 1 (failed) + 1 (conflict lookup) = 3
    assert metrics["canonical_get_state_total"] == 3
    assert metrics["canonical_get_state_errors"] == 1


# ============================================================================
# Test Edge Cases
# ============================================================================


@pytest.mark.asyncio
async def test_concurrent_commits_serialization(
    service, mock_postgres_client, mock_kafka_client, sample_provenance
):
    """
    Test that concurrent commits are properly serialized by database.

    This test simulates two concurrent commits:
    1. First commit succeeds (version 1 → 2)
    2. Second commit conflicts (still at version 1, but DB is at version 2)
    """
    # Setup mocks for two sequential commits
    # First commit succeeds
    mock_postgres_client.fetch_one.return_value = {
        "workflow_key": "test-workflow",
        "version": 2,
        "state": {"value": "first"},
        "updated_at": datetime.now(UTC),
        "schema_version": 1,
        "provenance": sample_provenance,
    }

    result1 = await service.try_commit(
        workflow_key="test-workflow",
        expected_version=1,
        state_prime={"value": "first"},
        provenance=sample_provenance,
    )

    assert isinstance(result1, EventStateCommitted)
    assert result1.new_version == 2

    # Second commit conflicts (tries to update from version 1, but DB is at version 2)
    mock_postgres_client.fetch_one.side_effect = [
        None,  # UPDATE fails (version mismatch)
        {
            "workflow_key": "test-workflow",
            "version": 2,
            "state": {"value": "first"},
            "updated_at": datetime.now(UTC),
            "schema_version": 1,
            "provenance": sample_provenance,
        },
    ]

    result2 = await service.try_commit(
        workflow_key="test-workflow",
        expected_version=1,
        state_prime={"value": "second"},
        provenance=sample_provenance,
    )

    assert isinstance(result2, EventStateConflict)
    assert result2.expected_version == 1
    assert result2.actual_version == 2

    # Verify metrics
    assert service._metrics_commits_total == 1
    assert service._metrics_conflicts_total == 1


@pytest.mark.asyncio
async def test_large_state_handling(
    service, mock_postgres_client, mock_kafka_client, sample_provenance
):
    """Test handling of large state objects (JSONB scalability)."""
    # Create a large state with 1000 items
    large_state = {
        "items": [{"id": i, "data": f"item-{i}" * 100} for i in range(1000)],
        "count": 1000,
    }

    # Setup mock
    mock_postgres_client.fetch_one.return_value = {
        "workflow_key": "large-workflow",
        "version": 2,
        "state": large_state,
        "updated_at": datetime.now(UTC),
        "schema_version": 1,
        "provenance": sample_provenance,
    }

    # Execute
    result = await service.try_commit(
        workflow_key="large-workflow",
        expected_version=1,
        state_prime=large_state,
        provenance=sample_provenance,
    )

    # Verify success
    assert isinstance(result, EventStateCommitted)
    assert len(result.state_snapshot["items"]) == 1000
