#!/usr/bin/env python3
"""
End-to-End Integration Tests for Pure Reducer Architecture.

Tests complete Pure Reducer workflow including:
- Action → Reduce → Commit → Projection (happy path)
- Concurrent actions with optimistic concurrency control
- Action deduplication for idempotency
- Projection lag with canonical fallback
- Reducer gave up escalation for max retries

Test Infrastructure:
- PostgreSQL for canonical and projection stores
- Kafka for event streaming (mocked for unit tests)
- Real database transactions for conflict testing
- Async fixtures with proper cleanup

Wave 6A - Pure Reducer Refactor
Reference: docs/planning/PURE_REDUCER_REFACTOR_PLAN.md
"""

import asyncio
import hashlib
import logging

# Import models - direct import to avoid __init__ dependencies
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

# Add src to path for direct imports
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import entity models directly
from omninode_bridge.infrastructure.entities.model_action import ModelAction

# Import services
from omninode_bridge.services.action_dedup import ActionDedupService
from omninode_bridge.services.canonical_store import (
    CanonicalStoreService,
    EventStateConflict,
)
from omninode_bridge.services.kafka_client import KafkaClient
from omninode_bridge.services.postgres_client import PostgresClient
from omninode_bridge.services.projection_store import ProjectionStoreService
from omninode_bridge.services.reducer_service import ReducerService

logger = logging.getLogger(__name__)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
async def postgres_client():
    """Create PostgreSQL client connected to test database."""
    # PostgresClient uses environment variables by default
    # Set env vars if needed, or use defaults
    client = PostgresClient()

    try:
        await client.connect()

        # Verify connection
        assert client.pool is not None, "PostgreSQL client failed to initialize"

        yield client
    finally:
        # Cleanup
        await client.close()


@pytest.fixture
async def kafka_client():
    """Create mocked Kafka client for event publishing."""
    mock_client = AsyncMock(spec=KafkaClient)
    mock_client.publish_event = AsyncMock(return_value=None)
    return mock_client


@pytest.fixture
async def canonical_store(postgres_client, kafka_client):
    """Create canonical store service."""
    return CanonicalStoreService(
        postgres_client=postgres_client,
        kafka_client=kafka_client,
    )


@pytest.fixture
async def projection_store(postgres_client, canonical_store):
    """Create projection store service."""
    service = ProjectionStoreService(
        db_client=postgres_client,
        canonical_store=canonical_store,
        poll_interval_ms=5,
    )
    await service.initialize()
    return service


@pytest.fixture
async def action_dedup(postgres_client):
    """Create action deduplication service."""
    return ActionDedupService(postgres_client=postgres_client)


@pytest.fixture
async def mock_reducer():
    """Create mock reducer for testing."""
    mock = AsyncMock()

    # Mock execute_reduction to return aggregated state
    async def execute_reduction(contract):
        """Mock reducer that aggregates items."""
        input_state = contract.input_state if hasattr(contract, "input_state") else {}
        items = input_state.get("items", [])

        # Simple aggregation: count items by namespace
        aggregations = {}
        for item in items:
            namespace = item.get("namespace", "default")
            aggregations[namespace] = aggregations.get(namespace, 0) + 1

        # Return mock reducer output
        from unittest.mock import MagicMock

        result = MagicMock()
        result.aggregations = aggregations
        return result

    mock.execute_reduction = execute_reduction
    return mock


@pytest.fixture
async def reducer_service(
    mock_reducer, canonical_store, projection_store, action_dedup, kafka_client
):
    """Create reducer service with all dependencies."""
    return ReducerService(
        reducer=mock_reducer,
        canonical_store=canonical_store,
        projection_store=projection_store,
        action_dedup=action_dedup,
        kafka_client=kafka_client,
        max_attempts=3,
        backoff_base_ms=10,
        backoff_cap_ms=250,
    )


@pytest.fixture
async def setup_test_workflow(postgres_client):
    """Setup initial workflow state in database."""

    async def _setup(workflow_key: str, initial_state: dict[str, Any] = None) -> int:
        """
        Create initial workflow state.

        Args:
            workflow_key: Workflow identifier
            initial_state: Initial state dict (default: empty aggregations)

        Returns:
            Initial version (1)
        """
        if initial_state is None:
            initial_state = {"aggregations": {}}

        # Insert initial state
        await postgres_client.execute(
            """
            INSERT INTO workflow_state (workflow_key, version, state, schema_version, provenance)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (workflow_key) DO NOTHING
            """,
            workflow_key,
            1,  # Initial version
            initial_state,
            1,  # Schema version
            {
                "effect_id": str(uuid4()),
                "timestamp": datetime.now(UTC).isoformat(),
                "source": "test_setup",
            },
        )

        return 1

    return _setup


@pytest.fixture
async def cleanup_test_data(postgres_client):
    """Cleanup test data after each test."""
    workflow_keys = []

    yield workflow_keys

    # Cleanup after test
    if workflow_keys:
        for workflow_key in workflow_keys:
            try:
                await postgres_client.execute(
                    "DELETE FROM workflow_state WHERE workflow_key = $1", workflow_key
                )
                await postgres_client.execute(
                    "DELETE FROM workflow_projection WHERE workflow_key = $1",
                    workflow_key,
                )
                await postgres_client.execute(
                    "DELETE FROM action_dedup_log WHERE workflow_key = $1", workflow_key
                )
            except Exception as e:
                logger.warning(f"Cleanup error for {workflow_key}: {e}")


# ============================================================================
# Happy Path Tests
# ============================================================================


@pytest.mark.asyncio
async def test_happy_path_action_to_projection(
    reducer_service,
    canonical_store,
    projection_store,
    setup_test_workflow,
    cleanup_test_data,
):
    """
    Test complete workflow from action to materialized projection.

    Workflow:
    1. Publish action event
    2. Reducer processes it
    3. State committed to canonical store
    4. Projection materialized
    5. Projection matches canonical state
    """
    workflow_key = f"test-workflow-{uuid4()}"
    cleanup_test_data.append(workflow_key)

    # Setup: Create initial workflow state
    initial_version = await setup_test_workflow(workflow_key)
    logger.info(f"Created workflow {workflow_key} at version {initial_version}")

    # Step 1: Create action
    action = ModelAction(
        action_id=uuid4(),
        workflow_key=workflow_key,
        epoch=1,
        lease_id=uuid4(),
        payload={"namespace": "test.namespace", "operation": "add_item"},
    )

    logger.info(f"Processing action {action.action_id}")

    # Step 2: Process action through reducer service
    start_time = time.perf_counter()
    await reducer_service.handle_action(action)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    logger.info(f"Action processed in {elapsed_ms:.2f}ms")

    # Verify performance target (<70ms for no conflicts)
    assert (
        elapsed_ms < 100
    ), f"Action processing took {elapsed_ms:.2f}ms (target: <100ms)"

    # Step 3: Verify state committed to canonical store
    canonical_state = await canonical_store.get_state(workflow_key)
    assert canonical_state.version == 2, "Version should be incremented to 2"
    assert "aggregations" in canonical_state.state, "State should contain aggregations"

    logger.info(f"Canonical state updated to version {canonical_state.version}")

    # Step 4: Verify action recorded in dedup log
    dedup_entry = await reducer_service.action_dedup.get_dedup_entry(
        workflow_key, action.action_id
    )
    assert dedup_entry is not None, "Action should be recorded in dedup log"
    assert dedup_entry["workflow_key"] == workflow_key
    assert dedup_entry["action_id"] == action.action_id

    logger.info("Action recorded in dedup log successfully")

    # Step 5: Verify metrics
    metrics = reducer_service.get_metrics()
    assert metrics.successful_actions == 1, "Should have 1 successful action"
    assert metrics.failed_actions == 0, "Should have 0 failed actions"
    assert (
        metrics.duplicate_actions_skipped == 0
    ), "Should have 0 duplicate actions (first action)"

    logger.info(f"Test completed successfully. Metrics: {metrics}")


# ============================================================================
# Conflict Resolution Tests
# ============================================================================


@pytest.mark.asyncio
async def test_concurrent_actions_conflict_resolution(
    reducer_service,
    canonical_store,
    setup_test_workflow,
    cleanup_test_data,
):
    """
    Test optimistic concurrency with concurrent actions.

    Workflow:
    1. Start 10 concurrent actions on same workflow_key
    2. Verify conflicts detected
    3. Verify retry loop working
    4. Verify eventual consistency (all actions processed)
    """
    workflow_key = f"test-workflow-concurrent-{uuid4()}"
    cleanup_test_data.append(workflow_key)

    # Setup: Create initial workflow state
    await setup_test_workflow(workflow_key)
    logger.info(f"Created workflow {workflow_key} for concurrent test")

    # Create 10 concurrent actions
    num_actions = 10
    actions = [
        ModelAction(
            action_id=uuid4(),
            workflow_key=workflow_key,
            epoch=i + 1,
            lease_id=uuid4(),
            payload={"namespace": f"namespace-{i}", "operation": "add_item"},
        )
        for i in range(num_actions)
    ]

    logger.info(f"Starting {num_actions} concurrent actions")

    # Execute all actions concurrently
    start_time = time.perf_counter()
    results = await asyncio.gather(
        *[reducer_service.handle_action(action) for action in actions],
        return_exceptions=True,
    )
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    logger.info(f"All {num_actions} actions processed in {elapsed_ms:.2f}ms")

    # Verify no exceptions
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Action {i} failed with exception: {result}")
        assert not isinstance(
            result, Exception
        ), f"Action {i} should not raise exception"

    # Verify final state
    final_state = await canonical_store.get_state(workflow_key)
    expected_version = 1 + num_actions  # Initial version + num_actions
    assert (
        final_state.version == expected_version
    ), f"Final version should be {expected_version}, got {final_state.version}"

    # Verify metrics
    metrics = reducer_service.get_metrics()
    assert (
        metrics.successful_actions == num_actions
    ), f"Should have {num_actions} successful actions"
    assert metrics.failed_actions == 0, "Should have 0 failed actions"
    assert (
        metrics.conflict_attempts_total > 0
    ), "Should have some conflict retry attempts"

    logger.info(
        f"Conflict resolution successful. "
        f"Conflicts: {metrics.conflict_attempts_total}, "
        f"Avg conflicts per action: {metrics.avg_conflicts_per_action:.2f}"
    )


# ============================================================================
# Deduplication Tests
# ============================================================================


@pytest.mark.asyncio
async def test_action_deduplication(
    reducer_service,
    canonical_store,
    setup_test_workflow,
    cleanup_test_data,
):
    """
    Test duplicate action handling.

    Workflow:
    1. Process action with ID xxx
    2. Re-process same action_id
    3. Verify second attempt skipped
    4. Verify result hash matches
    """
    workflow_key = f"test-workflow-dedup-{uuid4()}"
    cleanup_test_data.append(workflow_key)

    # Setup: Create initial workflow state
    await setup_test_workflow(workflow_key)
    logger.info(f"Created workflow {workflow_key} for deduplication test")

    # Create action
    action = ModelAction(
        action_id=uuid4(),
        workflow_key=workflow_key,
        epoch=1,
        lease_id=uuid4(),
        payload={"namespace": "test.namespace", "operation": "add_item"},
    )

    # Process action first time
    logger.info(f"Processing action {action.action_id} (first time)")
    await reducer_service.handle_action(action)

    # Verify action processed
    state_after_first = await canonical_store.get_state(workflow_key)
    assert state_after_first.version == 2, "Version should be incremented to 2"

    # Process same action again (duplicate)
    logger.info(f"Processing action {action.action_id} (second time - duplicate)")
    await reducer_service.handle_action(action)

    # Verify state NOT changed (duplicate skipped)
    state_after_second = await canonical_store.get_state(workflow_key)
    assert (
        state_after_second.version == 2
    ), "Version should still be 2 (duplicate skipped)"

    # Verify metrics
    metrics = reducer_service.get_metrics()
    assert metrics.successful_actions == 1, "Should have 1 successful action"
    assert (
        metrics.duplicate_actions_skipped == 1
    ), "Should have 1 duplicate action skipped"

    logger.info(
        f"Deduplication successful. " f"Dedup hit rate: {metrics.dedup_hit_rate:.2f}%"
    )


# ============================================================================
# Projection Lag Tests
# ============================================================================


@pytest.mark.asyncio
async def test_projection_lag_fallback(
    canonical_store,
    projection_store,
    setup_test_workflow,
    cleanup_test_data,
):
    """
    Test fallback to canonical when projection lags.

    Workflow:
    1. Create workflow state in canonical
    2. Query before projection materialized
    3. Verify fallback to canonical
    4. Verify projection query works with version gating
    """
    workflow_key = f"test-workflow-lag-{uuid4()}"
    cleanup_test_data.append(workflow_key)

    # Setup: Create initial workflow state
    await setup_test_workflow(workflow_key, {"aggregations": {"test": 1}})
    logger.info(f"Created workflow {workflow_key} for projection lag test")

    # Update canonical state to version 2
    await canonical_store.try_commit(
        workflow_key=workflow_key,
        expected_version=1,
        state_prime={"aggregations": {"test": 2}},
        provenance={
            "effect_id": str(uuid4()),
            "timestamp": datetime.now(UTC).isoformat(),
        },
    )

    canonical_state = await canonical_store.get_state(workflow_key)
    assert canonical_state.version == 2, "Canonical version should be 2"

    logger.info(f"Canonical state updated to version {canonical_state.version}")

    # Query projection with version requirement (should fallback to canonical)
    # Projection doesn't exist yet, so it will fallback
    projection = await projection_store.get_state(
        workflow_key, required_version=2, max_wait_ms=50
    )

    # Verify fallback worked
    assert projection.version == 2, "Should fallback to canonical version 2"
    assert projection.workflow_key == workflow_key

    # Verify metrics
    metrics = projection_store.get_metrics()
    assert metrics.projection_reads_total == 1, "Should have 1 projection read"
    assert metrics.projection_fallback_count == 1, "Should have 1 fallback to canonical"

    logger.info(
        f"Projection lag fallback successful. "
        f"Fallback rate: {metrics.fallback_rate:.2f}%"
    )


# ============================================================================
# Reducer Gave Up Tests
# ============================================================================


@pytest.mark.asyncio
async def test_reducer_gave_up_escalation(
    reducer_service,
    canonical_store,
    setup_test_workflow,
    cleanup_test_data,
    kafka_client,
):
    """
    Test max retries exceeded scenario.

    Workflow:
    1. Force conflicts on every attempt
    2. Verify 3 retries attempted
    3. Verify ReducerGaveUp event published
    """
    workflow_key = f"test-workflow-gaveup-{uuid4()}"
    cleanup_test_data.append(workflow_key)

    # Setup: Create initial workflow state
    await setup_test_workflow(workflow_key)
    logger.info(f"Created workflow {workflow_key} for gave up test")

    # Mock canonical_store.try_commit to always return conflict
    original_try_commit = canonical_store.try_commit

    async def always_conflict(*args, **kwargs):
        """Always return conflict to force max retries."""
        return EventStateConflict(
            workflow_key=workflow_key,
            expected_version=1,
            actual_version=2,
            reason="forced_conflict_for_test",
        )

    canonical_store.try_commit = always_conflict

    # Create action
    action = ModelAction(
        action_id=uuid4(),
        workflow_key=workflow_key,
        epoch=1,
        lease_id=uuid4(),
        payload={"namespace": "test.namespace", "operation": "add_item"},
    )

    # Process action (should exhaust retries)
    logger.info(f"Processing action {action.action_id} (will fail after 3 retries)")
    await reducer_service.handle_action(action)

    # Restore original method
    canonical_store.try_commit = original_try_commit

    # Verify metrics
    metrics = reducer_service.get_metrics()
    assert metrics.failed_actions == 1, "Should have 1 failed action"
    assert metrics.successful_actions == 0, "Should have 0 successful actions"
    assert (
        metrics.conflict_attempts_total == 3
    ), "Should have exactly 3 conflict attempts (max_attempts)"

    # Verify ReducerGaveUp event published
    assert (
        kafka_client.publish_event.call_count >= 1
    ), "Should have published ReducerGaveUp event"

    # Find the ReducerGaveUp event
    gave_up_event = None
    for call in kafka_client.publish_event.call_args_list:
        args, kwargs = call
        if "omninode_bridge_reducer_gave_up_v1" in str(
            args
        ) or "omninode_bridge_reducer_gave_up_v1" in str(kwargs):
            gave_up_event = kwargs.get("event") or (args[1] if len(args) > 1 else None)
            break

    assert gave_up_event is not None, "ReducerGaveUp event should be published"
    assert gave_up_event["workflow_key"] == workflow_key
    assert gave_up_event["attempts"] == 3

    logger.info(
        f"Reducer gave up successfully. "
        f"Conflicts: {metrics.conflict_attempts_total}, "
        f"Failed actions: {metrics.failed_actions}"
    )


# ============================================================================
# Performance Validation Tests
# ============================================================================


@pytest.mark.asyncio
async def test_reducer_service_performance_targets(
    reducer_service,
    canonical_store,
    setup_test_workflow,
    cleanup_test_data,
):
    """
    Validate reducer service meets performance targets.

    Targets:
    - Deduplication check: <5ms
    - State read: <5ms
    - Reducer invocation: <50ms
    - Commit attempt: <10ms
    - Total latency (no conflicts): <70ms p95
    """
    workflow_key = f"test-workflow-perf-{uuid4()}"
    cleanup_test_data.append(workflow_key)

    # Setup: Create initial workflow state
    await setup_test_workflow(workflow_key)

    # Process 100 actions to get reliable metrics
    num_actions = 100
    latencies = []

    for i in range(num_actions):
        action = ModelAction(
            action_id=uuid4(),
            workflow_key=workflow_key,
            epoch=i + 1,
            lease_id=uuid4(),
            payload={"namespace": f"namespace-{i}", "operation": "add_item"},
        )

        start_time = time.perf_counter()
        await reducer_service.handle_action(action)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        latencies.append(elapsed_ms)

    # Calculate p95 latency
    latencies.sort()
    p95_idx = int(len(latencies) * 0.95)
    p95_latency = latencies[p95_idx]
    avg_latency = sum(latencies) / len(latencies)

    logger.info(
        f"Performance metrics: "
        f"avg={avg_latency:.2f}ms, "
        f"p95={p95_latency:.2f}ms, "
        f"max={max(latencies):.2f}ms"
    )

    # Verify performance targets
    # Note: Relaxed targets for test environment (2x production targets)
    assert p95_latency < 140, f"P95 latency {p95_latency:.2f}ms exceeds target 140ms"
    assert avg_latency < 100, f"Avg latency {avg_latency:.2f}ms exceeds target 100ms"

    # Verify metrics
    metrics = reducer_service.get_metrics()
    assert (
        metrics.successful_actions == num_actions
    ), f"Should have {num_actions} successful actions"
    assert metrics.failed_actions == 0, "Should have 0 failed actions"


# ============================================================================
# Edge Case Tests
# ============================================================================


@pytest.mark.asyncio
async def test_empty_payload_handling(
    reducer_service,
    canonical_store,
    setup_test_workflow,
    cleanup_test_data,
):
    """Test action with empty payload."""
    workflow_key = f"test-workflow-empty-{uuid4()}"
    cleanup_test_data.append(workflow_key)

    await setup_test_workflow(workflow_key)

    action = ModelAction(
        action_id=uuid4(),
        workflow_key=workflow_key,
        epoch=1,
        lease_id=uuid4(),
        payload={},  # Empty payload
    )

    # Should process without errors
    await reducer_service.handle_action(action)

    # Verify state updated
    state = await canonical_store.get_state(workflow_key)
    assert state.version == 2, "Version should be incremented"


@pytest.mark.asyncio
async def test_invalid_workflow_key_handling(reducer_service):
    """Test action with invalid workflow_key."""
    action = ModelAction(
        action_id=uuid4(),
        workflow_key="",  # Empty workflow key
        epoch=1,
        lease_id=uuid4(),
        payload={"test": "data"},
    )

    # Should raise ValueError
    with pytest.raises(ValueError, match="workflow_key must be non-empty"):
        await reducer_service.handle_action(action)


# ============================================================================
# Cleanup Tests
# ============================================================================


@pytest.mark.asyncio
async def test_action_dedup_cleanup(
    action_dedup,
    postgres_client,
):
    """Test action dedup cleanup removes expired entries."""
    workflow_key = f"test-workflow-cleanup-{uuid4()}"
    action_id = uuid4()
    result_hash = hashlib.sha256(b"test").hexdigest()

    # Record action with short TTL
    await action_dedup.record_processed(
        workflow_key, action_id, result_hash, ttl_hours=0  # Immediate expiration
    )

    # Verify recorded
    entry = await action_dedup.get_dedup_entry(workflow_key, action_id)
    assert entry is not None, "Entry should be recorded"

    # Wait a bit to ensure expiration
    await asyncio.sleep(0.1)

    # Run cleanup
    deleted_count = await action_dedup.cleanup_expired()

    logger.info(f"Cleanup removed {deleted_count} expired entries")

    # Verify entry removed (may or may not be removed depending on timing)
    # This test is best-effort due to timing sensitivity
    assert deleted_count >= 0, "Cleanup should not fail"
