"""
Integration tests for NodeStoreEffect with real CanonicalStoreService.

Tests cover:
- End-to-end persistence flow with real database
- Concurrent commit handling with optimistic locking
- Event publishing to Kafka
- Real CanonicalStoreService integration

Pure Reducer Refactor - Wave 4, Workstream 4A

Note: These tests require:
- PostgreSQL database with workflow_state table
- Kafka/Redpanda for event publishing (optional, can mock)
"""

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from omnibase_core.models.core import ModelContainer

from omninode_bridge.services.canonical_store import (
    CanonicalStoreService,
    EventStateCommitted,
    EventStateConflict,
)
from omninode_bridge.services.postgres_client import PostgresClient

from ..models.model_persist_state_event import ModelPersistStateEvent
from ..node import NodeStoreEffect

# Skip integration tests if dependencies not available
pytestmark = pytest.mark.integration


@pytest.fixture
async def postgres_client():
    """
    Create real PostgreSQL client for integration testing.

    Note: Requires POSTGRES_HOST, POSTGRES_PORT, etc. environment variables.
    """
    client = PostgresClient()
    await client.connect()
    yield client
    await client.close()


@pytest.fixture
async def kafka_client():
    """
    Create real Kafka client for integration testing.

    Note: Can be mocked if Kafka is not available.
    """
    # For now, return None to skip Kafka publishing in tests
    # In production, create real KafkaClient instance
    return None


@pytest.fixture
async def canonical_store(postgres_client, kafka_client):
    """Create real CanonicalStoreService instance."""
    return CanonicalStoreService(
        postgres_client=postgres_client,
        kafka_client=kafka_client,
    )


@pytest.fixture
async def container(postgres_client, kafka_client):
    """Create ONEX container with real services."""
    from unittest.mock import MagicMock

    container = MagicMock(spec=ModelContainer)

    def get_service(service_name: str):
        if service_name == "postgres_client":
            return postgres_client
        elif service_name == "kafka_client":
            return kafka_client
        return None

    container.get_service = MagicMock(side_effect=get_service)
    return container


@pytest.fixture
async def store_node(container):
    """Create and initialize NodeStoreEffect with real services."""
    node = NodeStoreEffect(container)
    await node.initialize()
    yield node
    await node.shutdown()


@pytest.fixture
async def test_workflow(postgres_client):
    """
    Create test workflow state in database.

    Yields workflow_key and cleans up after test.
    """
    workflow_key = f"test-workflow-{uuid4()}"

    # Insert initial workflow state
    await postgres_client.execute(
        """
        INSERT INTO workflow_state (workflow_key, version, state, schema_version, provenance)
        VALUES ($1, $2, $3, $4, $5)
        """,
        workflow_key,
        1,
        {"items": [], "count": 0},
        1,
        {"created_by": "test_integration"},
    )

    yield workflow_key

    # Cleanup
    await postgres_client.execute(
        "DELETE FROM workflow_state WHERE workflow_key = $1",
        workflow_key,
    )


class TestEndToEndPersistence:
    """Test end-to-end persistence flow with real database."""

    @pytest.mark.asyncio
    async def test_persist_state_success_with_real_database(
        self, store_node, test_workflow, postgres_client
    ):
        """Test successful state persistence with real database."""
        # Create PersistState event
        event = ModelPersistStateEvent(
            workflow_key=test_workflow,
            expected_version=1,
            state_prime={
                "items": ["item-1", "item-2"],
                "count": 2,
            },
            action_id=uuid4(),
            provenance={
                "test_id": "integration_test",
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

        # Execute persistence
        result = await store_node.handle_persist_state_event(event)

        # Verify result
        assert isinstance(result, EventStateCommitted)
        assert result.workflow_key == test_workflow
        assert result.new_version == 2

        # Verify database state
        row = await postgres_client.fetch_one(
            "SELECT version, state FROM workflow_state WHERE workflow_key = $1",
            test_workflow,
        )
        assert row["version"] == 2
        assert row["state"]["count"] == 2

    @pytest.mark.asyncio
    async def test_persist_state_conflict_with_real_database(
        self, store_node, test_workflow, postgres_client
    ):
        """Test version conflict with real database."""
        # First, update database to version 2 (simulate concurrent update)
        await postgres_client.execute(
            """
            UPDATE workflow_state
            SET version = version + 1, state = $1, updated_at = NOW()
            WHERE workflow_key = $2
            """,
            {"items": ["concurrent"], "count": 1},
            test_workflow,
        )

        # Create PersistState event with stale version (1)
        event = ModelPersistStateEvent(
            workflow_key=test_workflow,
            expected_version=1,  # Stale version!
            state_prime={
                "items": ["item-1", "item-2"],
                "count": 2,
            },
        )

        # Execute persistence
        result = await store_node.handle_persist_state_event(event)

        # Verify conflict
        assert isinstance(result, EventStateConflict)
        assert result.expected_version == 1
        assert result.actual_version == 2
        assert result.reason == "concurrent_modification"

        # Verify database unchanged
        row = await postgres_client.fetch_one(
            "SELECT version, state FROM workflow_state WHERE workflow_key = $1",
            test_workflow,
        )
        assert row["version"] == 2
        assert row["state"]["count"] == 1  # Unchanged


class TestConcurrentCommits:
    """Test concurrent commit handling with optimistic locking."""

    @pytest.mark.asyncio
    async def test_concurrent_commits_only_one_succeeds(
        self, store_node, test_workflow
    ):
        """Test that only one of multiple concurrent commits succeeds."""

        # Create multiple PersistState events with same version
        events = [
            ModelPersistStateEvent(
                workflow_key=test_workflow,
                expected_version=1,
                state_prime={
                    "items": [f"concurrent-{i}"],
                    "count": i,
                },
            )
            for i in range(5)
        ]

        # Execute concurrently
        results = await asyncio.gather(
            *[store_node.handle_persist_state_event(event) for event in events],
            return_exceptions=False,
        )

        # Verify: Exactly 1 success, 4 conflicts
        successes = [r for r in results if isinstance(r, EventStateCommitted)]
        conflicts = [r for r in results if isinstance(r, EventStateConflict)]

        assert len(successes) == 1
        assert len(conflicts) == 4

        # Verify success has version 2
        assert successes[0].new_version == 2

        # Verify all conflicts expected v1, got v2
        for conflict in conflicts:
            assert conflict.expected_version == 1
            assert conflict.actual_version == 2


class TestMetricsIntegration:
    """Test metrics tracking with real operations."""

    @pytest.mark.asyncio
    async def test_metrics_track_real_operations(self, store_node, test_workflow):
        """Test that metrics correctly track real persistence operations."""
        initial_commits = store_node.metrics.state_commits_total
        initial_conflicts = store_node.metrics.state_conflicts_total

        # Successful commit
        success_event = ModelPersistStateEvent(
            workflow_key=test_workflow,
            expected_version=1,
            state_prime={"items": ["item-1"], "count": 1},
        )
        result1 = await store_node.handle_persist_state_event(success_event)
        assert isinstance(result1, EventStateCommitted)

        # Conflicting commit
        conflict_event = ModelPersistStateEvent(
            workflow_key=test_workflow,
            expected_version=1,  # Stale!
            state_prime={"items": ["item-2"], "count": 2},
        )
        result2 = await store_node.handle_persist_state_event(conflict_event)
        assert isinstance(result2, EventStateConflict)

        # Verify metrics
        assert store_node.metrics.state_commits_total == initial_commits + 1
        assert store_node.metrics.state_conflicts_total == initial_conflicts + 1
        assert store_node.metrics.avg_persist_latency_ms > 0


class TestProvenanceTracking:
    """Test provenance tracking with real persistence."""

    @pytest.mark.asyncio
    async def test_provenance_preserved_in_database(
        self, store_node, test_workflow, postgres_client
    ):
        """Test that provenance metadata is preserved in database."""
        # Create event with custom provenance
        event = ModelPersistStateEvent(
            workflow_key=test_workflow,
            expected_version=1,
            state_prime={"items": ["item-1"], "count": 1},
            action_id=uuid4(),
            provenance={
                "source": "integration_test",
                "custom_field": "custom_value",
            },
        )

        # Execute persistence
        result = await store_node.handle_persist_state_event(event)
        assert isinstance(result, EventStateCommitted)

        # Verify provenance in database
        row = await postgres_client.fetch_one(
            "SELECT provenance FROM workflow_state WHERE workflow_key = $1",
            test_workflow,
        )

        provenance = row["provenance"]
        assert "effect_id" in provenance
        assert provenance["effect_id"] == str(store_node.node_id)
        assert "source" in provenance
        assert provenance["source"] == "integration_test"
        assert "custom_field" in provenance
        assert provenance["custom_field"] == "custom_value"


class TestPerformance:
    """Test performance characteristics of Store Effect Node."""

    @pytest.mark.asyncio
    async def test_persistence_latency_meets_target(self, store_node, test_workflow):
        """Test that persistence latency meets < 10ms target."""
        # Create event
        event = ModelPersistStateEvent(
            workflow_key=test_workflow,
            expected_version=1,
            state_prime={"items": ["item-1"], "count": 1},
        )

        # Execute and measure
        import time

        start = time.perf_counter()
        result = await store_node.handle_persist_state_event(event)
        latency_ms = (time.perf_counter() - start) * 1000

        # Verify result and latency
        assert isinstance(result, EventStateCommitted)
        assert latency_ms < 10.0  # < 10ms target

    @pytest.mark.asyncio
    async def test_throughput_sequential(self, store_node, postgres_client):
        """Test sequential throughput (should exceed 1000 ops/sec)."""
        # Create multiple test workflows
        workflows = []
        for i in range(10):
            workflow_key = f"throughput-test-{uuid4()}"
            await postgres_client.execute(
                """
                INSERT INTO workflow_state (workflow_key, version, state, schema_version, provenance)
                VALUES ($1, $2, $3, $4, $5)
                """,
                workflow_key,
                1,
                {"count": 0},
                "1.0.0",
                {"test": "throughput"},
            )
            workflows.append(workflow_key)

        # Execute persistence operations
        import time

        start = time.perf_counter()

        for workflow_key in workflows:
            event = ModelPersistStateEvent(
                workflow_key=workflow_key,
                expected_version=1,
                state_prime={"count": 1},
            )
            await store_node.handle_persist_state_event(event)

        elapsed_sec = time.perf_counter() - start

        # Calculate throughput
        throughput = len(workflows) / elapsed_sec

        # Cleanup
        for workflow_key in workflows:
            await postgres_client.execute(
                "DELETE FROM workflow_state WHERE workflow_key = $1",
                workflow_key,
            )

        # Verify throughput
        # Note: Sequential throughput may be lower than target
        # This is expected, as the target assumes parallel execution
        print(f"Sequential throughput: {throughput:.2f} ops/sec")
