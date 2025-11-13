#!/usr/bin/env python3
"""
Unit tests for ProjectionStoreService.

Tests cover:
- Version gating behavior with polling
- Canonical fallback scenarios
- Projection lag handling
- Metrics tracking
- Error handling

Pure Reducer Refactor - Wave 2, Workstream 2B
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from omninode_bridge.infrastructure.entities.model_workflow_state import (
    ModelWorkflowState,
)
from omninode_bridge.services.projection_store import (
    ProjectionStoreMetrics,
    ProjectionStoreService,
)


class MockCanonicalStore:
    """Mock canonical store for testing."""

    def __init__(self):
        self._states: dict[str, ModelWorkflowState] = {}

    def add_state(
        self,
        workflow_key: str,
        version: int,
        tag: str = "PROCESSING",
        namespace: str = "test",
    ) -> ModelWorkflowState:
        """Add a mock state to the canonical store."""
        state = ModelWorkflowState(
            workflow_key=workflow_key,
            version=version,
            state={"tag": tag, "namespace": namespace, "items": []},
            updated_at=datetime.now(UTC),
            schema_version=1,
            provenance={
                "effect_id": f"effect-{workflow_key}",
                "timestamp": datetime.now(UTC).isoformat(),
                "action_id": f"action-{workflow_key}",
                "namespace": namespace,
            },
        )
        self._states[workflow_key] = state
        return state

    async def get_state(self, workflow_key: str) -> ModelWorkflowState:
        """Get state from canonical store."""
        if workflow_key not in self._states:
            raise KeyError(f"Workflow {workflow_key} not found in canonical store")
        return self._states[workflow_key]


@pytest.fixture
def mock_db_client():
    """Create mock database client."""
    client = AsyncMock()
    client.fetchrow = AsyncMock()
    return client


@pytest.fixture
def mock_canonical_store():
    """Create mock canonical store."""
    return MockCanonicalStore()


@pytest.fixture
def projection_store(mock_db_client, mock_canonical_store):
    """Create projection store service."""
    return ProjectionStoreService(
        db_client=mock_db_client,
        canonical_store=mock_canonical_store,
        poll_interval_ms=5,
    )


@pytest.mark.asyncio
class TestProjectionStoreService:
    """Test suite for ProjectionStoreService."""

    async def test_initialize(self, projection_store):
        """Test service initialization."""
        assert not projection_store._initialized

        await projection_store.initialize()
        assert projection_store._initialized

        # Idempotent initialization
        await projection_store.initialize()
        assert projection_store._initialized

    async def test_get_state_no_version_requirement(
        self, projection_store, mock_db_client
    ):
        """Test get_state without version requirement (fast path)."""
        # Setup mock projection
        mock_db_client.fetchrow.return_value = {
            "workflow_key": "wf-123",
            "version": 3,
            "tag": "PROCESSING",
            "last_action": "StampContent",
            "namespace": "production",
            "updated_at": datetime.now(UTC),
            "indices": None,
            "extras": None,
        }

        # Get state without version requirement
        projection = await projection_store.get_state("wf-123")

        # Verify result
        assert projection.workflow_key == "wf-123"
        assert projection.version == 3
        assert projection.tag == "PROCESSING"
        assert projection.namespace == "production"

        # Verify metrics
        metrics = projection_store.get_metrics()
        assert metrics.projection_reads_total == 1
        assert metrics.projection_fallback_count == 0

    async def test_get_state_version_satisfied_immediately(
        self, projection_store, mock_db_client
    ):
        """Test get_state when projection version already satisfies requirement."""
        # Setup mock projection with version 5
        mock_db_client.fetchrow.return_value = {
            "workflow_key": "wf-456",
            "version": 5,
            "tag": "COMPLETED",
            "last_action": "Complete",
            "namespace": "production",
            "updated_at": datetime.now(UTC),
            "indices": None,
            "extras": None,
        }

        # Get state with version requirement 3 (already satisfied)
        projection = await projection_store.get_state(
            "wf-456", required_version=3, max_wait_ms=100
        )

        # Verify result
        assert projection.workflow_key == "wf-456"
        assert projection.version == 5
        assert projection.version >= 3

        # Verify metrics (no wait since version already satisfied)
        metrics = projection_store.get_metrics()
        assert metrics.projection_reads_total == 1
        assert metrics.projection_wait_count == 0
        assert metrics.projection_fallback_count == 0

    async def test_get_state_version_gating_with_wait(
        self, projection_store, mock_db_client
    ):
        """Test get_state with version gating - projection catches up during wait."""
        call_count = 0

        async def mock_fetchrow(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            # First 2 calls: version 3 (not ready)
            # Third call: version 5 (ready)
            version = 3 if call_count < 3 else 5

            return {
                "workflow_key": "wf-789",
                "version": version,
                "tag": "PROCESSING",
                "last_action": "Process",
                "namespace": "production",
                "updated_at": datetime.now(UTC),
                "indices": None,
                "extras": None,
            }

        mock_db_client.fetchrow = mock_fetchrow

        # Get state with version requirement 5 (should wait and poll)
        projection = await projection_store.get_state(
            "wf-789", required_version=5, max_wait_ms=100
        )

        # Verify result
        assert projection.workflow_key == "wf-789"
        assert projection.version == 5
        assert call_count >= 3  # Should have polled multiple times

        # Verify metrics
        metrics = projection_store.get_metrics()
        assert metrics.projection_reads_total == 1
        assert metrics.projection_wait_count == 1
        assert metrics.total_wait_time_ms > 0
        assert metrics.projection_fallback_count == 0

    async def test_get_state_version_timeout_fallback(
        self, projection_store, mock_db_client, mock_canonical_store
    ):
        """Test get_state with version timeout - should fall back to canonical."""
        # Setup mock projection with stale version (always returns v3)
        mock_db_client.fetchrow.return_value = {
            "workflow_key": "wf-timeout",
            "version": 3,
            "tag": "PROCESSING",
            "last_action": "Process",
            "namespace": "production",
            "updated_at": datetime.now(UTC),
            "indices": None,
            "extras": None,
        }

        # Add canonical state with version 10
        mock_canonical_store.add_state(
            workflow_key="wf-timeout",
            version=10,
            tag="COMPLETED",
            namespace="production",
        )

        # Get state with version requirement 10, short timeout (should fallback)
        projection = await projection_store.get_state(
            "wf-timeout", required_version=10, max_wait_ms=20
        )

        # Verify result from canonical fallback
        assert projection.workflow_key == "wf-timeout"
        assert projection.version == 10  # From canonical
        assert projection.tag == "COMPLETED"  # From canonical

        # Verify metrics
        metrics = projection_store.get_metrics()
        assert metrics.projection_reads_total == 1
        assert metrics.projection_fallback_count == 1
        assert metrics.fallback_rate == 100.0

    async def test_get_state_projection_not_found_fallback(
        self, projection_store, mock_db_client, mock_canonical_store
    ):
        """Test get_state when projection not found - should fall back to canonical."""
        # Setup mock to raise KeyError (projection not found)
        mock_db_client.fetchrow.return_value = None

        # Add canonical state
        mock_canonical_store.add_state(
            workflow_key="wf-notfound", version=2, tag="PENDING", namespace="production"
        )

        # Get state (should fallback immediately)
        projection = await projection_store.get_state("wf-notfound")

        # Verify result from canonical fallback
        assert projection.workflow_key == "wf-notfound"
        assert projection.version == 2
        assert projection.tag == "PENDING"

        # Verify metrics
        metrics = projection_store.get_metrics()
        assert metrics.projection_reads_total == 1
        assert metrics.projection_fallback_count == 1

    async def test_get_state_canonical_not_found_raises(
        self, projection_store, mock_db_client, mock_canonical_store
    ):
        """Test get_state when workflow not found anywhere - should raise KeyError."""
        # Setup mock to raise KeyError (projection not found)
        mock_db_client.fetchrow.return_value = None

        # No canonical state added

        # Get state should raise KeyError
        with pytest.raises(KeyError, match="not found in canonical store"):
            await projection_store.get_state("wf-nonexistent")

    async def test_get_version(self, projection_store, mock_db_client):
        """Test get_version method."""
        # Setup mock projection
        mock_db_client.fetchrow.return_value = {
            "workflow_key": "wf-version",
            "version": 7,
            "tag": "PROCESSING",
            "last_action": "Process",
            "namespace": "production",
            "updated_at": datetime.now(UTC),
            "indices": None,
            "extras": None,
        }

        # Get version
        version = await projection_store.get_version("wf-version")

        # Verify result
        assert version == 7

    async def test_get_version_not_found_raises(self, projection_store, mock_db_client):
        """Test get_version when workflow not found - should raise KeyError."""
        # Setup mock to return None (not found)
        mock_db_client.fetchrow.return_value = None

        # Get version should raise KeyError
        with pytest.raises(KeyError, match="not found in projection"):
            await projection_store.get_version("wf-nonexistent")

    async def test_get_state_empty_workflow_key_raises(self, projection_store):
        """Test get_state with empty workflow_key - should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            await projection_store.get_state("")

        with pytest.raises(ValueError, match="cannot be empty"):
            await projection_store.get_state("   ")

    async def test_get_version_empty_workflow_key_raises(self, projection_store):
        """Test get_version with empty workflow_key - should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            await projection_store.get_version("")

        with pytest.raises(ValueError, match="cannot be empty"):
            await projection_store.get_version("   ")

    async def test_metrics_tracking(
        self, projection_store, mock_db_client, mock_canonical_store
    ):
        """Test comprehensive metrics tracking."""
        # Setup mock projection
        mock_db_client.fetchrow.return_value = {
            "workflow_key": "wf-metrics",
            "version": 5,
            "tag": "PROCESSING",
            "last_action": "Process",
            "namespace": "production",
            "updated_at": datetime.now(UTC),
            "indices": None,
            "extras": None,
        }

        # Perform multiple reads
        await projection_store.get_state("wf-metrics")  # Read 1
        await projection_store.get_state("wf-metrics")  # Read 2
        await projection_store.get_state("wf-metrics", required_version=3)  # Read 3

        # Setup fallback scenario
        mock_db_client.fetchrow.return_value = None
        mock_canonical_store.add_state(
            workflow_key="wf-fallback", version=2, tag="PENDING"
        )
        await projection_store.get_state("wf-fallback")  # Read 4 (fallback)

        # Get metrics
        metrics = projection_store.get_metrics()

        # Verify metrics
        assert metrics.projection_reads_total == 4
        assert metrics.projection_fallback_count == 1
        assert metrics.fallback_rate == 25.0
        assert len(metrics.read_latencies_ms) == 4
        assert metrics.p95_read_latency_ms >= 0

    async def test_canonical_to_projection_conversion(
        self, projection_store, mock_canonical_store
    ):
        """Test conversion from canonical state to projection."""
        # Create canonical state
        canonical = mock_canonical_store.add_state(
            workflow_key="wf-convert",
            version=8,
            tag="COMPLETED",
            namespace="staging",
        )

        # Convert to projection
        projection = projection_store._canonical_to_projection(canonical)

        # Verify conversion
        assert projection.workflow_key == "wf-convert"
        assert projection.version == 8
        assert projection.tag == "COMPLETED"
        assert projection.namespace == "staging"
        assert projection.last_action == "action-wf-convert"
        assert projection.indices is None
        assert projection.extras is None

    async def test_poll_interval_configuration(
        self, mock_db_client, mock_canonical_store
    ):
        """Test custom poll interval configuration."""
        # Create service with custom poll interval
        service = ProjectionStoreService(
            db_client=mock_db_client,
            canonical_store=mock_canonical_store,
            poll_interval_ms=10,  # 10ms instead of default 5ms
        )

        assert service._poll_interval_ms == 10


class TestProjectionStoreMetrics:
    """Test suite for ProjectionStoreMetrics."""

    def test_fallback_rate_calculation(self):
        """Test fallback rate calculation."""
        metrics = ProjectionStoreMetrics()

        # No reads yet
        assert metrics.fallback_rate == 0.0

        # 1 fallback out of 4 reads = 25%
        metrics.projection_reads_total = 4
        metrics.projection_fallback_count = 1
        assert metrics.fallback_rate == 25.0

        # 3 fallbacks out of 4 reads = 75%
        metrics.projection_fallback_count = 3
        assert metrics.fallback_rate == 75.0

    def test_avg_wait_time_calculation(self):
        """Test average wait time calculation."""
        metrics = ProjectionStoreMetrics()

        # No waits yet
        assert metrics.avg_wait_time_ms == 0.0

        # 3 waits with 150ms total = 50ms avg
        metrics.projection_wait_count = 3
        metrics.total_wait_time_ms = 150.0
        assert metrics.avg_wait_time_ms == 50.0

    def test_p95_read_latency_calculation(self):
        """Test 95th percentile latency calculation."""
        metrics = ProjectionStoreMetrics()

        # No reads yet
        assert metrics.p95_read_latency_ms == 0.0

        # Add latencies: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        metrics.read_latencies_ms = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # P95 should be around 9.5 (95th percentile of 10 values)
        assert 9.0 <= metrics.p95_read_latency_ms <= 10.0

        # Add more latencies with outlier
        metrics.read_latencies_ms = [1, 1, 1, 1, 1, 1, 1, 1, 1, 100]
        # P95 should be around 100 (outlier at 95th percentile)
        assert metrics.p95_read_latency_ms >= 10.0
