"""
Unit tests for NodeStoreEffect - PersistState Event Handling.

Tests cover:
- Successful state persistence (StateCommitted)
- Version conflict handling (StateConflict)
- Error handling and recovery
- Provenance generation and tracking
- Metrics recording

Pure Reducer Refactor - Wave 4, Workstream 4A
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from omnibase_core import EnumCoreErrorCode, ModelOnexError
from omnibase_core.models.core import ModelContainer

from omninode_bridge.services.canonical_store import (
    EventStateCommitted,
    EventStateConflict,
)

from ..models.model_persist_state_event import ModelPersistStateEvent
from ..node import NodeStoreEffect


@pytest.fixture
def mock_container():
    """Create mock ONEX container with required services."""
    container = MagicMock(spec=ModelContainer)

    # Mock PostgresClient
    postgres_client = AsyncMock()
    postgres_client.fetch_one = AsyncMock()
    postgres_client.execute = AsyncMock()

    # Mock KafkaClient
    kafka_client = AsyncMock()
    kafka_client.publish_event = AsyncMock()

    # Configure container.get_service() to return mocks
    def get_service(service_name: str):
        if service_name == "postgres_client":
            return postgres_client
        elif service_name == "kafka_client":
            return kafka_client
        return None

    container.get_service = MagicMock(side_effect=get_service)

    return container


@pytest.fixture
async def store_node(mock_container):
    """Create and initialize NodeStoreEffect instance."""
    node = NodeStoreEffect(mock_container)
    await node.initialize()
    return node


@pytest.fixture
def persist_state_event():
    """Create sample PersistState event for testing."""
    return ModelPersistStateEvent(
        workflow_key="test-workflow-123",
        expected_version=1,
        state_prime={
            "aggregations": {
                "omninode.services.metadata": {
                    "total_stamps": 100,
                    "total_size_bytes": 1024000,
                }
            }
        },
        action_id=uuid4(),
        provenance={
            "reducer_id": "reducer-456",
            "timestamp": datetime.now(UTC).isoformat(),
        },
    )


class TestPersistStateSuccess:
    """Test successful state persistence scenarios."""

    @pytest.mark.asyncio
    async def test_persist_state_success_returns_state_committed(
        self, store_node, persist_state_event
    ):
        """Test that successful persistence returns StateCommitted event."""
        # Mock CanonicalStoreService.try_commit() to return success
        expected_result = EventStateCommitted(
            workflow_key=persist_state_event.workflow_key,
            new_version=2,
            state_snapshot=persist_state_event.state_prime,
            provenance={
                "effect_id": str(store_node.node_id),
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

        with patch.object(
            store_node._canonical_store,
            "try_commit",
            new=AsyncMock(return_value=expected_result),
        ):
            result = await store_node.handle_persist_state_event(persist_state_event)

        # Verify result
        assert isinstance(result, EventStateCommitted)
        assert result.workflow_key == persist_state_event.workflow_key
        assert result.new_version == 2
        assert result.state_snapshot == persist_state_event.state_prime

    @pytest.mark.asyncio
    async def test_persist_state_success_records_metrics(
        self, store_node, persist_state_event
    ):
        """Test that successful persistence increments metrics correctly."""
        # Mock CanonicalStoreService.try_commit()
        expected_result = EventStateCommitted(
            workflow_key=persist_state_event.workflow_key,
            new_version=2,
            state_snapshot=persist_state_event.state_prime,
            provenance={},
        )

        with patch.object(
            store_node._canonical_store,
            "try_commit",
            new=AsyncMock(return_value=expected_result),
        ):
            initial_commits = store_node.metrics.state_commits_total
            await store_node.handle_persist_state_event(persist_state_event)

        # Verify metrics
        assert store_node.metrics.state_commits_total == initial_commits + 1
        assert store_node.metrics.events_published_total > 0
        assert store_node.metrics.persist_errors_total == 0

    @pytest.mark.asyncio
    async def test_persist_state_success_builds_provenance(
        self, store_node, persist_state_event
    ):
        """Test that provenance metadata is correctly built and merged."""
        # Capture try_commit arguments
        commit_args = {}

        async def capture_commit(**kwargs):
            commit_args.update(kwargs)
            return EventStateCommitted(
                workflow_key=kwargs["workflow_key"],
                new_version=2,
                state_snapshot=kwargs["state_prime"],
                provenance=kwargs["provenance"],
            )

        with patch.object(
            store_node._canonical_store,
            "try_commit",
            new=AsyncMock(side_effect=capture_commit),
        ):
            await store_node.handle_persist_state_event(persist_state_event)

        # Verify provenance
        provenance = commit_args["provenance"]
        assert "effect_id" in provenance
        assert provenance["effect_id"] == str(store_node.node_id)
        assert "timestamp" in provenance
        assert "correlation_id" in provenance
        assert "reducer_id" in provenance  # Merged from event.provenance


class TestPersistStateConflict:
    """Test version conflict scenarios."""

    @pytest.mark.asyncio
    async def test_persist_state_conflict_returns_state_conflict(
        self, store_node, persist_state_event
    ):
        """Test that version conflict returns StateConflict event."""
        # Mock CanonicalStoreService.try_commit() to return conflict
        expected_result = EventStateConflict(
            workflow_key=persist_state_event.workflow_key,
            expected_version=1,
            actual_version=2,
            reason="concurrent_modification",
        )

        with patch.object(
            store_node._canonical_store,
            "try_commit",
            new=AsyncMock(return_value=expected_result),
        ):
            result = await store_node.handle_persist_state_event(persist_state_event)

        # Verify result
        assert isinstance(result, EventStateConflict)
        assert result.workflow_key == persist_state_event.workflow_key
        assert result.expected_version == 1
        assert result.actual_version == 2

    @pytest.mark.asyncio
    async def test_persist_state_conflict_records_metrics(
        self, store_node, persist_state_event
    ):
        """Test that version conflict increments conflict metrics."""
        # Mock CanonicalStoreService.try_commit()
        expected_result = EventStateConflict(
            workflow_key=persist_state_event.workflow_key,
            expected_version=1,
            actual_version=2,
            reason="version_mismatch",
        )

        with patch.object(
            store_node._canonical_store,
            "try_commit",
            new=AsyncMock(return_value=expected_result),
        ):
            initial_conflicts = store_node.metrics.state_conflicts_total
            await store_node.handle_persist_state_event(persist_state_event)

        # Verify metrics
        assert store_node.metrics.state_conflicts_total == initial_conflicts + 1
        assert store_node.metrics.events_published_total > 0
        assert store_node.metrics.persist_errors_total == 0


class TestPersistStateErrors:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_persist_state_error_raises_onex_error(
        self, store_node, persist_state_event
    ):
        """Test that persistence errors raise OnexError."""
        # Mock CanonicalStoreService.try_commit() to raise exception
        with patch.object(
            store_node._canonical_store,
            "try_commit",
            new=AsyncMock(side_effect=RuntimeError("Database connection failed")),
        ):
            with pytest.raises(ModelOnexError) as exc_info:
                await store_node.handle_persist_state_event(persist_state_event)

        # Verify error
        assert exc_info.value.error_code == EnumCoreErrorCode.INTERNAL_ERROR
        assert "Failed to persist state" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_persist_state_error_records_metrics(
        self, store_node, persist_state_event
    ):
        """Test that persistence errors increment error metrics."""
        # Mock CanonicalStoreService.try_commit()
        with patch.object(
            store_node._canonical_store,
            "try_commit",
            new=AsyncMock(side_effect=RuntimeError("Database error")),
        ):
            initial_errors = store_node.metrics.persist_errors_total

            with pytest.raises(ModelOnexError):
                await store_node.handle_persist_state_event(persist_state_event)

        # Verify metrics
        assert store_node.metrics.persist_errors_total == initial_errors + 1


class TestMetricsTracking:
    """Test metrics tracking functionality."""

    def test_metrics_track_latency(self, store_node):
        """Test that latency is tracked correctly."""
        # Record some latency samples
        store_node.metrics.record_commit_success(10.5)
        store_node.metrics.record_commit_success(12.3)
        store_node.metrics.record_conflict(8.7)

        # Verify average latency
        assert store_node.metrics.avg_persist_latency_ms > 0
        assert store_node.metrics.state_commits_total == 2
        assert store_node.metrics.state_conflicts_total == 1

    def test_metrics_calculate_success_rate(self, store_node):
        """Test success rate calculation."""
        # Record operations
        store_node.metrics.record_commit_success(10.0)
        store_node.metrics.record_commit_success(12.0)
        store_node.metrics.record_conflict(8.0)

        # Verify success rate (2 successes / 3 total = 66.67%)
        success_rate = store_node.metrics.get_success_rate()
        assert 66.0 < success_rate < 67.0

    def test_metrics_calculate_conflict_rate(self, store_node):
        """Test conflict rate calculation."""
        # Record operations
        store_node.metrics.record_commit_success(10.0)
        store_node.metrics.record_conflict(8.0)
        store_node.metrics.record_conflict(9.0)

        # Verify conflict rate (2 conflicts / 3 total = 66.67%)
        conflict_rate = store_node.metrics.get_conflict_rate()
        assert 66.0 < conflict_rate < 67.0

    def test_metrics_to_dict(self, store_node):
        """Test metrics serialization to dict."""
        # Record some operations
        store_node.metrics.record_commit_success(10.0)
        store_node.metrics.record_conflict(8.0)

        # Get metrics dict
        metrics_dict = store_node.metrics.to_dict()

        # Verify structure
        assert "state_commits_total" in metrics_dict
        assert "state_conflicts_total" in metrics_dict
        assert "success_rate_pct" in metrics_dict
        assert "conflict_rate_pct" in metrics_dict
        assert metrics_dict["state_commits_total"] == 1
        assert metrics_dict["state_conflicts_total"] == 1


class TestHealthStatus:
    """Test health status reporting."""

    @pytest.mark.asyncio
    async def test_get_health_status_includes_dependencies(self, store_node):
        """Test that health status includes dependency information."""
        health = store_node.get_health_status()

        # Verify structure
        assert "status" in health
        assert "node_id" in health
        assert "dependencies" in health
        assert "metrics" in health
        assert "performance" in health

        # Verify dependencies
        deps = health["dependencies"]
        assert deps["postgres_client"] is True
        assert deps["kafka_client"] is True
        assert deps["canonical_store"] is True

    @pytest.mark.asyncio
    async def test_get_health_status_includes_performance(self, store_node):
        """Test that health status includes performance metrics."""
        # Record some operations
        store_node.metrics.record_commit_success(10.0)
        store_node.metrics.record_conflict(8.0)

        health = store_node.get_health_status()
        performance = health["performance"]

        # Verify performance metrics
        assert "success_rate_pct" in performance
        assert "conflict_rate_pct" in performance
        assert "error_rate_pct" in performance
        assert "avg_latency_ms" in performance
        assert performance["success_rate_pct"] == 50.0  # 1 success / 2 total
