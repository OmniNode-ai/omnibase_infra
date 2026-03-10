# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Unit tests for BatchResponsePublisher (OMN-478).

Tests validate:
- Batch publishing with size threshold flush
- Batch publishing with timeout flush
- Graceful shutdown flushes pending responses
- Error handling for partial batch failures
- Response ordering preserved within batches
- Metrics tracking
- Configuration validation and clamping
- Integration with RuntimeHostProcess config
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from omnibase_infra.runtime.batch_response_publisher import BatchResponsePublisher
from omnibase_infra.runtime.models.model_batch_publisher_config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_FLUSH_INTERVAL_MS,
    MAX_BATCH_SIZE,
    MAX_FLUSH_INTERVAL_MS,
    MIN_BATCH_SIZE,
    MIN_FLUSH_INTERVAL_MS,
    ModelBatchPublisherConfig,
)
from omnibase_infra.runtime.models.model_batch_publisher_metrics import (
    ModelBatchPublisherMetrics,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_publish_fn() -> AsyncMock:
    """Create a mock publish function."""
    return AsyncMock()


@pytest.fixture
def publisher(mock_publish_fn: AsyncMock) -> BatchResponsePublisher:
    """Create a BatchResponsePublisher with default settings."""
    return BatchResponsePublisher(
        publish_fn=mock_publish_fn,
        topic="test-responses",
        batch_size=3,
        flush_interval_ms=100,
    )


def _make_envelope(correlation_id: str | None = None) -> dict[str, object]:
    """Create a test response envelope."""
    return {
        "success": True,
        "status": "ok",
        "correlation_id": correlation_id or str(uuid4()),
    }


# =============================================================================
# Configuration Tests
# =============================================================================


class TestBatchPublisherConfig:
    """Tests for BatchResponsePublisher configuration."""

    def test_default_config(self, mock_publish_fn: AsyncMock) -> None:
        """Config model uses correct defaults."""
        config = ModelBatchPublisherConfig()
        assert config.batch_size == DEFAULT_BATCH_SIZE
        assert config.flush_interval_ms == DEFAULT_FLUSH_INTERVAL_MS
        assert config.enabled is False

    def test_config_property(self, publisher: BatchResponsePublisher) -> None:
        """Config property returns effective configuration."""
        config = publisher.config
        assert config.batch_size == 3
        assert config.flush_interval_ms == 100.0
        assert config.enabled is True

    def test_batch_size_clamped_below_min(self, mock_publish_fn: AsyncMock) -> None:
        """Batch size below minimum is clamped."""
        pub = BatchResponsePublisher(
            publish_fn=mock_publish_fn,
            topic="test",
            batch_size=0,
        )
        assert pub.config.batch_size == MIN_BATCH_SIZE

    def test_batch_size_clamped_above_max(self, mock_publish_fn: AsyncMock) -> None:
        """Batch size above maximum is clamped."""
        pub = BatchResponsePublisher(
            publish_fn=mock_publish_fn,
            topic="test",
            batch_size=9999,
        )
        assert pub.config.batch_size == MAX_BATCH_SIZE

    def test_flush_interval_clamped_below_min(self, mock_publish_fn: AsyncMock) -> None:
        """Flush interval below minimum is clamped."""
        pub = BatchResponsePublisher(
            publish_fn=mock_publish_fn,
            topic="test",
            flush_interval_ms=1.0,
        )
        assert pub.config.flush_interval_ms == MIN_FLUSH_INTERVAL_MS

    def test_flush_interval_clamped_above_max(self, mock_publish_fn: AsyncMock) -> None:
        """Flush interval above maximum is clamped."""
        pub = BatchResponsePublisher(
            publish_fn=mock_publish_fn,
            topic="test",
            flush_interval_ms=99999.0,
        )
        assert pub.config.flush_interval_ms == MAX_FLUSH_INTERVAL_MS


# =============================================================================
# Lifecycle Tests
# =============================================================================


class TestBatchPublisherLifecycle:
    """Tests for start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_sets_running(self, publisher: BatchResponsePublisher) -> None:
        """Start sets is_running to True."""
        assert not publisher.is_running
        await publisher.start()
        assert publisher.is_running
        await publisher.stop()

    @pytest.mark.asyncio
    async def test_stop_sets_not_running(
        self, publisher: BatchResponsePublisher
    ) -> None:
        """Stop sets is_running to False."""
        await publisher.start()
        await publisher.stop()
        assert not publisher.is_running

    @pytest.mark.asyncio
    async def test_double_start_is_safe(
        self, publisher: BatchResponsePublisher
    ) -> None:
        """Calling start twice is safe (idempotent)."""
        await publisher.start()
        await publisher.start()  # Should not raise
        assert publisher.is_running
        await publisher.stop()

    @pytest.mark.asyncio
    async def test_double_stop_is_safe(self, publisher: BatchResponsePublisher) -> None:
        """Calling stop twice is safe (idempotent)."""
        await publisher.start()
        await publisher.stop()
        await publisher.stop()  # Should not raise

    @pytest.mark.asyncio
    async def test_stop_flushes_pending(
        self,
        publisher: BatchResponsePublisher,
        mock_publish_fn: AsyncMock,
    ) -> None:
        """Stop flushes any pending responses."""
        await publisher.start()

        # Enqueue 2 responses (below batch_size=3, so no auto-flush)
        await publisher.enqueue(_make_envelope())
        await publisher.enqueue(_make_envelope())
        assert publisher.pending_count == 2
        assert mock_publish_fn.call_count == 0

        await publisher.stop()

        # Both should be published during stop
        assert mock_publish_fn.call_count == 2
        assert publisher.pending_count == 0


# =============================================================================
# Batch Flush Tests
# =============================================================================


class TestBatchFlush:
    """Tests for batch flush behavior."""

    @pytest.mark.asyncio
    async def test_flush_on_size_threshold(
        self,
        publisher: BatchResponsePublisher,
        mock_publish_fn: AsyncMock,
    ) -> None:
        """Responses are flushed when batch_size threshold is reached."""
        await publisher.start()

        try:
            # Enqueue batch_size (3) responses
            for _ in range(3):
                await publisher.enqueue(_make_envelope())

            # Should have been flushed immediately
            assert mock_publish_fn.call_count == 3
            assert publisher.pending_count == 0
        finally:
            await publisher.stop()

    @pytest.mark.asyncio
    async def test_flush_on_timeout(
        self,
        mock_publish_fn: AsyncMock,
    ) -> None:
        """Responses are flushed when flush interval elapses."""
        # Use very short flush interval for testing
        pub = BatchResponsePublisher(
            publish_fn=mock_publish_fn,
            topic="test",
            batch_size=100,  # High threshold, won't trigger by size
            flush_interval_ms=50,  # 50ms flush interval
        )

        await pub.start()

        try:
            await pub.enqueue(_make_envelope())
            assert pub.pending_count == 1

            # Wait for flush interval to trigger
            await asyncio.sleep(0.15)

            # Should have been flushed by timeout
            assert mock_publish_fn.call_count == 1
            assert pub.pending_count == 0
        finally:
            await pub.stop()

    @pytest.mark.asyncio
    async def test_manual_flush(
        self,
        publisher: BatchResponsePublisher,
        mock_publish_fn: AsyncMock,
    ) -> None:
        """flush_all() drains the buffer immediately."""
        await publisher.start()

        try:
            await publisher.enqueue(_make_envelope())
            await publisher.enqueue(_make_envelope())
            assert publisher.pending_count == 2

            await publisher.flush_all()

            assert mock_publish_fn.call_count == 2
            assert publisher.pending_count == 0
        finally:
            await publisher.stop()

    @pytest.mark.asyncio
    async def test_flush_empty_buffer_is_noop(
        self,
        publisher: BatchResponsePublisher,
        mock_publish_fn: AsyncMock,
    ) -> None:
        """Flushing an empty buffer does nothing."""
        await publisher.start()

        try:
            await publisher.flush_all()
            assert mock_publish_fn.call_count == 0
        finally:
            await publisher.stop()


# =============================================================================
# Ordering Tests
# =============================================================================


class TestResponseOrdering:
    """Tests for response ordering preservation."""

    @pytest.mark.asyncio
    async def test_ordering_preserved_within_batch(
        self,
        publisher: BatchResponsePublisher,
        mock_publish_fn: AsyncMock,
    ) -> None:
        """Responses within a batch are published in insertion order."""
        await publisher.start()

        try:
            envelopes = [_make_envelope(f"corr-{i}") for i in range(3)]
            for env in envelopes:
                await publisher.enqueue(env)

            # Verify order of publish calls
            assert mock_publish_fn.call_count == 3
            for i, call in enumerate(mock_publish_fn.call_args_list):
                published_envelope = call[0][0]
                assert published_envelope["correlation_id"] == f"corr-{i}"
        finally:
            await publisher.stop()


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for partial batch failure handling."""

    @pytest.mark.asyncio
    async def test_partial_batch_failure(
        self,
        mock_publish_fn: AsyncMock,
    ) -> None:
        """Failed publishes in a batch are counted but don't block others."""
        # Second publish call fails
        mock_publish_fn.side_effect = [None, RuntimeError("bus down"), None]

        pub = BatchResponsePublisher(
            publish_fn=mock_publish_fn,
            topic="test",
            batch_size=3,
        )

        await pub.start()

        try:
            for _ in range(3):
                await pub.enqueue(_make_envelope())

            # All 3 were attempted
            assert mock_publish_fn.call_count == 3

            # Metrics reflect partial failure
            metrics = pub.metrics
            assert metrics.total_enqueued == 3
            assert metrics.total_published == 2
            assert metrics.total_failed == 1
        finally:
            await pub.stop()

    @pytest.mark.asyncio
    async def test_all_fail_in_batch(
        self,
        mock_publish_fn: AsyncMock,
    ) -> None:
        """When all publishes fail, metrics track all failures."""
        mock_publish_fn.side_effect = RuntimeError("bus down")

        pub = BatchResponsePublisher(
            publish_fn=mock_publish_fn,
            topic="test",
            batch_size=2,
        )

        await pub.start()

        try:
            await pub.enqueue(_make_envelope())
            await pub.enqueue(_make_envelope())

            metrics = pub.metrics
            assert metrics.total_enqueued == 2
            assert metrics.total_published == 0
            assert metrics.total_failed == 2
        finally:
            await pub.stop()


# =============================================================================
# Metrics Tests
# =============================================================================


class TestMetrics:
    """Tests for metrics tracking."""

    @pytest.mark.asyncio
    async def test_metrics_initial_state(
        self, publisher: BatchResponsePublisher
    ) -> None:
        """Initial metrics are all zero."""
        metrics = publisher.metrics
        assert metrics.total_enqueued == 0
        assert metrics.total_published == 0
        assert metrics.total_failed == 0
        assert metrics.total_batches_flushed == 0

    @pytest.mark.asyncio
    async def test_size_flush_counted(
        self,
        publisher: BatchResponsePublisher,
        mock_publish_fn: AsyncMock,
    ) -> None:
        """Size-triggered flushes are counted in metrics."""
        await publisher.start()

        try:
            for _ in range(3):
                await publisher.enqueue(_make_envelope())

            metrics = publisher.metrics
            assert metrics.total_size_flushes == 1
            assert metrics.total_batches_flushed == 1
        finally:
            await publisher.stop()

    @pytest.mark.asyncio
    async def test_timeout_flush_counted(
        self,
        mock_publish_fn: AsyncMock,
    ) -> None:
        """Timeout-triggered flushes are counted in metrics."""
        pub = BatchResponsePublisher(
            publish_fn=mock_publish_fn,
            topic="test",
            batch_size=100,
            flush_interval_ms=50,
        )

        await pub.start()

        try:
            await pub.enqueue(_make_envelope())
            await asyncio.sleep(0.15)

            metrics = pub.metrics
            assert metrics.total_timeout_flushes >= 1
        finally:
            await pub.stop()

    @pytest.mark.asyncio
    async def test_metrics_snapshot_is_copy(
        self, publisher: BatchResponsePublisher
    ) -> None:
        """Metrics property returns a copy, not a reference."""
        m1 = publisher.metrics
        m2 = publisher.metrics
        assert m1 is not m2


# =============================================================================
# RuntimeHostProcess Integration Tests
# =============================================================================


class TestRuntimeHostProcessIntegration:
    """Tests for batch publisher integration with RuntimeHostProcess config."""

    def test_batch_disabled_by_default(self) -> None:
        """Batch publishing is disabled by default."""
        from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
        from omnibase_infra.runtime.service_runtime_host_process import (
            RuntimeHostProcess,
        )
        from tests.helpers.runtime_helpers import make_runtime_config

        bus = EventBusInmemory()
        config = make_runtime_config()
        process = RuntimeHostProcess(event_bus=bus, config=config)
        assert process.batch_publisher is None
        assert process.batch_response_enabled is False

    def test_batch_enabled_via_config(self) -> None:
        """Batch publishing is enabled via config dict."""
        from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
        from omnibase_infra.runtime.service_runtime_host_process import (
            RuntimeHostProcess,
        )
        from tests.helpers.runtime_helpers import make_runtime_config

        bus = EventBusInmemory()
        config = make_runtime_config(
            batch_response_enabled=True,
            batch_response_size=5,
            batch_flush_interval_ms=200,
        )
        process = RuntimeHostProcess(event_bus=bus, config=config)
        assert process.batch_publisher is not None
        assert process.batch_response_enabled is True
        assert process.batch_publisher.config.batch_size == 5
        assert process.batch_publisher.config.flush_interval_ms == 200.0

    def test_batch_enabled_via_string_config(self) -> None:
        """Batch publishing enabled with string 'true' value."""
        from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
        from omnibase_infra.runtime.service_runtime_host_process import (
            RuntimeHostProcess,
        )
        from tests.helpers.runtime_helpers import make_runtime_config

        bus = EventBusInmemory()
        config = make_runtime_config(batch_response_enabled="true")
        process = RuntimeHostProcess(event_bus=bus, config=config)
        assert process.batch_response_enabled is True

    @pytest.mark.asyncio
    async def test_health_check_includes_batch_info(self) -> None:
        """Health check includes batch publisher status."""
        from unittest.mock import AsyncMock, patch

        from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
        from omnibase_infra.runtime.service_runtime_host_process import (
            RuntimeHostProcess,
        )
        from tests.helpers.runtime_helpers import (
            make_runtime_config,
            seed_mock_handlers,
        )

        bus = EventBusInmemory()
        config = make_runtime_config(batch_response_enabled=True)
        process = RuntimeHostProcess(event_bus=bus, config=config)

        # Seed mock handlers to pass the no-handlers check
        seed_mock_handlers(process)

        with patch.object(process, "_validate_architecture", new_callable=AsyncMock):
            await process.start()

        try:
            health = await process.health_check()
            assert "batch_response_enabled" in health
            assert health["batch_response_enabled"] is True
            assert "batch_response_pending" in health
        finally:
            await process.stop()
