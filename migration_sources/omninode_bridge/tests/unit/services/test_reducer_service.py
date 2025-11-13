"""
Unit tests for ReducerService.

Tests deduplication, conflict retry logic, jittered backoff,
and metrics tracking.

Pure Reducer Refactor - Wave 3, Workstream 3B
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from omninode_bridge.infrastructure.entities.model_action import ModelAction
from omninode_bridge.infrastructure.entities.model_workflow_state import (
    ModelWorkflowState,
)
from omninode_bridge.nodes.reducer.v1_0_0.models.model_output_state import (
    ModelReducerOutputState,
)
from omninode_bridge.services.action_dedup import ActionDedupService
from omninode_bridge.services.canonical_store import (
    CanonicalStoreService,
    EventStateCommitted,
    EventStateConflict,
)
from omninode_bridge.services.kafka_client import KafkaClient
from omninode_bridge.services.projection_store import ProjectionStoreService
from omninode_bridge.services.reducer_service import ReducerService


# Module-level fixture to force ImportError fallback path
# This simulates omnibase_core.models.contracts not being available
@pytest.fixture(autouse=True)
def force_reducer_fallback_path():
    """
    Force ReducerService to use the fallback path by making ModelContractReducer
    import fail. This ensures tests work with simple mocks rather than full contracts.
    """
    import builtins

    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "omnibase_core.models.contracts.model_contract_reducer":
            raise ImportError("Simulated ImportError for testing fallback path")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        yield


@pytest.fixture
def mock_reducer():
    """Create mock reducer with execute_reduction method."""
    reducer = MagicMock()  # Use MagicMock instead of AsyncMock for the reducer object

    # Explicitly configure the reducer to NOT have a process method
    # This ensures the service uses execute_reduction path
    if hasattr(reducer, "process"):
        del reducer.process  # Remove the auto-created process attribute

    # Create an async mock for execute_reduction that returns the expected result
    # Handles both contract-based and fallback (None) calling patterns
    async def mock_execute_reduction(contract_or_none=None):
        # Return same result regardless of whether contract is passed
        return ModelReducerOutputState(
            aggregation_type="namespace_grouping",
            total_items=1,
            total_size_bytes=1024,
            namespaces=["omninode.test"],
            aggregations={
                "omninode.test": {
                    "total_stamps": 1,
                    "total_size_bytes": 1024,
                    "file_types": ["application/pdf"],
                }
            },
            fsm_states={"workflow-123": "PROCESSING"},
        )

    reducer.execute_reduction = AsyncMock(side_effect=mock_execute_reduction)

    return reducer


@pytest.fixture
def mock_canonical_store():
    """Create mock canonical store."""
    store = AsyncMock(spec=CanonicalStoreService)
    store.get_state = AsyncMock(
        return_value=ModelWorkflowState(
            workflow_key="workflow-123",
            version=1,
            state={"aggregations": {"omninode.test": {}}},
            updated_at=datetime.now(UTC),
            schema_version=1,
            provenance={
                "effect_id": "test-effect",
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )
    )
    store.try_commit = AsyncMock(
        return_value=EventStateCommitted(
            workflow_key="workflow-123",
            new_version=2,
            state_snapshot={"aggregations": {"omninode.test": {}}},
            provenance={
                "effect_id": "test-effect",
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )
    )
    return store


@pytest.fixture
def mock_projection_store():
    """Create mock projection store."""
    return AsyncMock(spec=ProjectionStoreService)


@pytest.fixture
def mock_action_dedup():
    """Create mock action dedup service."""
    dedup = AsyncMock(spec=ActionDedupService)
    dedup.should_process = AsyncMock(return_value=True)
    dedup.record_processed = AsyncMock()
    return dedup


@pytest.fixture
def mock_kafka_client():
    """Create mock Kafka client."""
    client = AsyncMock(spec=KafkaClient)
    client.publish_event = AsyncMock()
    return client


@pytest.fixture
def reducer_service(
    mock_reducer,
    mock_canonical_store,
    mock_projection_store,
    mock_action_dedup,
    mock_kafka_client,
):
    """Create ReducerService with mocked dependencies."""
    return ReducerService(
        reducer=mock_reducer,
        canonical_store=mock_canonical_store,
        projection_store=mock_projection_store,
        action_dedup=mock_action_dedup,
        kafka_client=mock_kafka_client,
        max_attempts=3,
        backoff_base_ms=10,
        backoff_cap_ms=250,
    )


@pytest.fixture
def sample_action():
    """Create sample action for testing."""
    return ModelAction(
        action_id=uuid4(),
        workflow_key="workflow-123",
        epoch=1,
        lease_id=uuid4(),
        payload={"operation": "add_stamp", "data": {"file_hash": "abc123"}},
    )


# ============================================================================
# Deduplication Tests
# ============================================================================


@pytest.mark.asyncio
async def test_deduplication_skips_duplicate_action(
    reducer_service, mock_action_dedup, sample_action
):
    """Test that duplicate actions are skipped via deduplication."""
    # Setup: Action already processed
    mock_action_dedup.should_process.return_value = False

    # Execute
    await reducer_service.handle_action(sample_action)

    # Verify: Dedup check was called
    mock_action_dedup.should_process.assert_called_once_with(
        sample_action.workflow_key, sample_action.action_id
    )

    # Verify: Reducer was NOT called
    reducer_service.reducer.execute_reduction.assert_not_called()

    # Verify: Metrics updated
    assert reducer_service.metrics.duplicate_actions_skipped == 1
    assert reducer_service.metrics.successful_actions == 0


@pytest.mark.asyncio
async def test_deduplication_processes_new_action(
    reducer_service, mock_action_dedup, sample_action
):
    """Test that new (non-duplicate) actions are processed."""
    # Setup: Action not yet processed
    mock_action_dedup.should_process.return_value = True

    # Execute
    await reducer_service.handle_action(sample_action)

    # Verify: Dedup check was called
    mock_action_dedup.should_process.assert_called_once()

    # Verify: Reducer WAS called
    reducer_service.reducer.execute_reduction.assert_called_once()

    # Verify: Action recorded after processing
    mock_action_dedup.record_processed.assert_called_once()

    # Verify: Metrics updated
    assert reducer_service.metrics.duplicate_actions_skipped == 0
    assert reducer_service.metrics.successful_actions == 1


@pytest.mark.asyncio
async def test_deduplication_records_result_hash(
    reducer_service, mock_action_dedup, sample_action
):
    """Test that result hash is recorded for deduplication."""
    # Execute
    await reducer_service.handle_action(sample_action)

    # Verify: record_processed called with hash
    call_args = mock_action_dedup.record_processed.call_args
    assert call_args is not None

    workflow_key, action_id, result_hash = call_args[0][:3]

    assert workflow_key == sample_action.workflow_key
    assert action_id == sample_action.action_id
    assert isinstance(result_hash, str)
    assert len(result_hash) == 64  # SHA256 hex string


# ============================================================================
# Conflict Resolution Tests
# ============================================================================


@pytest.mark.asyncio
async def test_conflict_retry_succeeds_on_second_attempt(
    reducer_service, mock_canonical_store, sample_action
):
    """Test that conflict retry succeeds on second attempt."""
    # Setup: First commit fails with conflict, second succeeds
    mock_canonical_store.try_commit.side_effect = [
        EventStateConflict(
            workflow_key="workflow-123",
            expected_version=1,
            actual_version=2,
            reason="concurrent_modification",
        ),
        EventStateCommitted(
            workflow_key="workflow-123",
            new_version=3,
            state_snapshot={"aggregations": {"omninode.test": {}}},
            provenance={
                "effect_id": "test",
                "timestamp": datetime.now(UTC).isoformat(),
            },
        ),
    ]

    # Also need to return different versions on state reads
    mock_canonical_store.get_state.side_effect = [
        ModelWorkflowState(
            workflow_key="workflow-123",
            version=1,
            state={"aggregations": {"omninode.test": {}}},
            updated_at=datetime.now(UTC),
            schema_version=1,
            provenance={
                "effect_id": "test-effect",
                "timestamp": datetime.now(UTC).isoformat(),
            },
        ),
        ModelWorkflowState(
            workflow_key="workflow-123",
            version=2,
            state={"aggregations": {"omninode.test": {}}},
            updated_at=datetime.now(UTC),
            schema_version=1,
            provenance={
                "effect_id": "test-effect",
                "timestamp": datetime.now(UTC).isoformat(),
            },
        ),
    ]

    # Execute
    await reducer_service.handle_action(sample_action)

    # Verify: get_state called twice (for both attempts)
    assert mock_canonical_store.get_state.call_count == 2

    # Verify: try_commit called twice (conflict + success)
    assert mock_canonical_store.try_commit.call_count == 2

    # Verify: Metrics updated
    assert reducer_service.metrics.successful_actions == 1
    assert reducer_service.metrics.conflict_attempts_total == 1
    assert reducer_service.metrics.total_backoff_time_ms > 0


@pytest.mark.asyncio
async def test_conflict_max_retries_exceeded(
    reducer_service, mock_canonical_store, mock_kafka_client, sample_action
):
    """Test that max retries publishes ReducerGaveUp event."""
    # Setup: All commits fail with conflict
    mock_canonical_store.try_commit.return_value = EventStateConflict(
        workflow_key="workflow-123",
        expected_version=1,
        actual_version=2,
        reason="concurrent_modification",
    )

    # Execute
    await reducer_service.handle_action(sample_action)

    # Verify: try_commit called max_attempts times
    assert mock_canonical_store.try_commit.call_count == reducer_service.max_attempts

    # Verify: ReducerGaveUp event published
    gave_up_calls = [
        call
        for call in mock_kafka_client.publish_event.call_args_list
        if "reducer_gave_up" in call[1]["topic"]
    ]
    assert len(gave_up_calls) == 1

    gave_up_event = gave_up_calls[0][1]["event"]
    assert gave_up_event["workflow_key"] == sample_action.workflow_key
    assert gave_up_event["action_id"] == str(sample_action.action_id)
    assert gave_up_event["attempts"] == reducer_service.max_attempts

    # Verify: Metrics updated
    assert reducer_service.metrics.failed_actions == 1
    assert (
        reducer_service.metrics.conflict_attempts_total == reducer_service.max_attempts
    )


# ============================================================================
# Jittered Backoff Tests
# ============================================================================


def test_backoff_delay_range_attempt_1(reducer_service):
    """Test backoff delay range for first attempt."""
    # First attempt: random(10, 20)
    for _ in range(10):
        delay = reducer_service._backoff_delay(1)
        assert 10 <= delay <= 20


def test_backoff_delay_range_attempt_3(reducer_service):
    """Test backoff delay range for third attempt."""
    # Third attempt: random(10, 80)
    for _ in range(10):
        delay = reducer_service._backoff_delay(3)
        assert 10 <= delay <= 80


def test_backoff_delay_respects_cap(reducer_service):
    """Test that backoff delay respects cap."""
    # Large attempt should hit cap (250ms)
    for _ in range(10):
        delay = reducer_service._backoff_delay(10)
        assert 10 <= delay <= 250


def test_backoff_delay_randomization(reducer_service):
    """Test that backoff delay is randomized (prevents thundering herd)."""
    # Generate 100 delays for attempt 3
    delays = [reducer_service._backoff_delay(3) for _ in range(100)]

    # Should see variance (not all the same)
    unique_delays = set(delays)
    assert len(unique_delays) > 10, "Backoff should be randomized"


# ============================================================================
# Result Hashing Tests
# ============================================================================


def test_hash_result_deterministic(reducer_service):
    """Test that result hashing is deterministic."""
    result1 = {"a": 1, "b": 2, "c": 3}
    result2 = {"c": 3, "a": 1, "b": 2}  # Different order

    hash1 = reducer_service._hash_result(result1)
    hash2 = reducer_service._hash_result(result2)

    assert hash1 == hash2, "Hash should be deterministic (order-independent)"


def test_hash_result_format(reducer_service):
    """Test that result hash is SHA256 hex string."""
    result = {"test": "data"}
    hash_val = reducer_service._hash_result(result)

    assert isinstance(hash_val, str)
    assert len(hash_val) == 64  # SHA256 hex = 64 chars
    assert all(c in "0123456789abcdef" for c in hash_val)


def test_hash_result_different_for_different_data(reducer_service):
    """Test that different results produce different hashes."""
    result1 = {"a": 1}
    result2 = {"a": 2}

    hash1 = reducer_service._hash_result(result1)
    hash2 = reducer_service._hash_result(result2)

    assert hash1 != hash2


# ============================================================================
# Metrics Tests
# ============================================================================


@pytest.mark.asyncio
async def test_metrics_success_rate(reducer_service, sample_action):
    """Test success rate metric calculation."""
    # Execute successful action
    await reducer_service.handle_action(sample_action)

    metrics = reducer_service.get_metrics()
    assert metrics.success_rate == 100.0
    assert metrics.successful_actions == 1
    assert metrics.failed_actions == 0


@pytest.mark.asyncio
async def test_metrics_avg_conflicts_per_action(
    reducer_service, mock_canonical_store, sample_action
):
    """Test average conflicts per action metric."""
    # Setup: Two conflicts before success
    mock_canonical_store.try_commit.side_effect = [
        EventStateConflict(
            workflow_key="workflow-123",
            expected_version=1,
            actual_version=2,
        ),
        EventStateConflict(
            workflow_key="workflow-123",
            expected_version=2,
            actual_version=3,
        ),
        EventStateCommitted(
            workflow_key="workflow-123",
            new_version=4,
            state_snapshot={"aggregations": {"omninode.test": {}}},
            provenance={
                "effect_id": "test",
                "timestamp": datetime.now(UTC).isoformat(),
            },
        ),
    ]

    mock_canonical_store.get_state.side_effect = [
        ModelWorkflowState(
            workflow_key="workflow-123",
            version=i,
            state={"aggregations": {"omninode.test": {}}},
            updated_at=datetime.now(UTC),
            schema_version=1,
            provenance={
                "effect_id": "test-effect",
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )
        for i in range(1, 4)
    ]

    # Execute
    await reducer_service.handle_action(sample_action)

    # Verify metrics
    metrics = reducer_service.get_metrics()
    assert metrics.conflict_attempts_total == 2
    assert metrics.avg_conflicts_per_action == 2.0


def test_metrics_reset(reducer_service):
    """Test metrics reset functionality."""
    # Set some metrics
    reducer_service.metrics.successful_actions = 10
    reducer_service.metrics.failed_actions = 2
    reducer_service.metrics.conflict_attempts_total = 5

    # Reset
    reducer_service.reset_metrics()

    # Verify all counters reset
    metrics = reducer_service.get_metrics()
    assert metrics.successful_actions == 0
    assert metrics.failed_actions == 0
    assert metrics.conflict_attempts_total == 0


# ============================================================================
# Edge Cases
# ============================================================================


@pytest.mark.asyncio
async def test_invalid_action_workflow_key_raises_error(reducer_service):
    """Test that invalid action raises Pydantic ValidationError."""
    from pydantic import ValidationError

    # Pydantic validates at creation time, preventing empty workflow_key
    with pytest.raises(
        ValidationError, match="String should have at least 1 character"
    ):
        invalid_action = ModelAction(
            action_id=uuid4(),
            workflow_key="",  # Empty workflow key - Pydantic will reject this
            epoch=1,
            lease_id=uuid4(),
            payload={},
        )


@pytest.mark.asyncio
async def test_service_without_dedup_processes_action(
    mock_reducer,
    mock_canonical_store,
    mock_projection_store,
    mock_kafka_client,
    sample_action,
):
    """Test that service works without action_dedup (optional dependency)."""
    # Create service WITHOUT action_dedup
    service = ReducerService(
        reducer=mock_reducer,
        canonical_store=mock_canonical_store,
        projection_store=mock_projection_store,
        action_dedup=None,  # No dedup
        kafka_client=mock_kafka_client,
    )

    # Execute - should work without dedup
    await service.handle_action(sample_action)

    # Verify: Reducer was called (no dedup check blocked it)
    mock_reducer.execute_reduction.assert_called_once()


@pytest.mark.asyncio
async def test_service_without_kafka_processes_action(
    mock_reducer,
    mock_canonical_store,
    mock_projection_store,
    mock_action_dedup,
    sample_action,
):
    """Test that service works without Kafka (optional dependency)."""
    # Create service WITHOUT kafka_client
    service = ReducerService(
        reducer=mock_reducer,
        canonical_store=mock_canonical_store,
        projection_store=mock_projection_store,
        action_dedup=mock_action_dedup,
        kafka_client=None,  # No Kafka
    )

    # Execute - should work without Kafka
    await service.handle_action(sample_action)

    # Verify: Reducer was called
    mock_reducer.execute_reduction.assert_called_once()

    # Verify: Action successful despite no Kafka
    assert service.metrics.successful_actions == 1


@pytest.mark.asyncio
async def test_error_in_reducer_triggers_retry(
    reducer_service, mock_canonical_store, sample_action
):
    """Test that errors in reducer trigger retry with backoff."""
    # Setup: Reducer fails twice, then succeeds
    reducer_service.reducer.execute_reduction.side_effect = [
        RuntimeError("Reducer error"),
        RuntimeError("Reducer error again"),
        ModelReducerOutputState(
            aggregation_type="namespace_grouping",
            total_items=1,
            total_size_bytes=1024,
            namespaces=["omninode.test"],
            aggregations={"omninode.test": {}},
        ),
    ]

    # Execute
    await reducer_service.handle_action(sample_action)

    # Verify: Reducer called 3 times (2 failures + 1 success)
    assert reducer_service.reducer.execute_reduction.call_count == 3

    # Verify: Final attempt succeeded
    assert reducer_service.metrics.successful_actions == 1

    # Verify: Backoff applied
    assert reducer_service.metrics.total_backoff_time_ms > 0
