# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Unit tests for ServiceRetryWorker.

This module tests:
    - Single poll-and-retry cycle (poll_and_retry)
    - Fetch pending retries with SELECT FOR UPDATE SKIP LOCKED
    - Successful delivery updates status to SUCCEEDED
    - Failed delivery increments attempt_count and schedules next retry
    - DLQ escalation when max retries exceeded
    - Exponential backoff calculation
    - Circuit breaker state and error handling
    - Configuration validation
    - Graceful shutdown signaling
    - Health status reporting
    - ModelRetryResult metrics

All tests mock asyncpg pool - no real database required.

Related Tickets:
    - OMN-1454: Implement RetryWorker for subscription notification delivery
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

pytestmark = pytest.mark.unit

from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    InfraUnavailableError,
)
from omnibase_infra.services.retry_worker.config_retry_worker import ConfigRetryWorker
from omnibase_infra.services.retry_worker.models.model_delivery_attempt import (
    EnumDeliveryStatus,
    ModelDeliveryAttempt,
)
from omnibase_infra.services.retry_worker.models.model_retry_result import (
    ModelRetryResult,
)
from omnibase_infra.services.retry_worker.service_retry_worker import (
    ServiceRetryWorker,
)

# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def config() -> ConfigRetryWorker:
    """Create a test configuration."""
    return ConfigRetryWorker(
        postgres_dsn="postgresql://test:test@localhost:5432/test_db",
        poll_interval_seconds=10,
        batch_size=10,
        max_retry_attempts=3,
        backoff_base_seconds=10.0,
        backoff_max_seconds=600.0,
        backoff_multiplier=2.0,
        delivery_timeout_seconds=5.0,
        circuit_breaker_threshold=3,
        circuit_breaker_reset_timeout=30.0,
    )


@pytest.fixture
def mock_pool() -> MagicMock:
    """Create a mock asyncpg pool."""
    pool = MagicMock()
    conn = AsyncMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
    return pool


@pytest.fixture
def mock_deliver_fn() -> AsyncMock:
    """Create a mock delivery function."""
    return AsyncMock()


@pytest.fixture
def worker(
    mock_pool: MagicMock,
    config: ConfigRetryWorker,
    mock_deliver_fn: AsyncMock,
) -> ServiceRetryWorker:
    """Create a ServiceRetryWorker with mocked dependencies."""
    return ServiceRetryWorker(
        pool=mock_pool,
        config=config,
        deliver_fn=mock_deliver_fn,
    )


def _make_attempt_row(
    *,
    attempt_count: int = 1,
    max_attempts: int = 3,
    status: str = "failed",
) -> dict:
    """Create a mock database row for a delivery attempt."""
    now = datetime.now(UTC)
    return {
        "id": uuid4(),
        "subscription_id": uuid4(),
        "notification_payload": '{"event": "test"}',
        "status": status,
        "attempt_count": attempt_count,
        "max_attempts": max_attempts,
        "next_retry_at": now - timedelta(seconds=10),
        "last_error": "ConnectionError: timeout",
        "created_at": now - timedelta(hours=1),
        "updated_at": now - timedelta(minutes=5),
    }


# =========================================================================
# Configuration Tests
# =========================================================================


class TestConfigRetryWorker:
    """Tests for ConfigRetryWorker validation."""

    def test_default_values(self) -> None:
        """Default config values are sensible."""
        config = ConfigRetryWorker(
            postgres_dsn="postgresql://test:test@localhost:5432/test",
        )
        assert config.poll_interval_seconds == 30
        assert config.batch_size == 50
        assert config.max_retry_attempts == 5
        assert config.backoff_base_seconds == 60.0
        assert config.backoff_max_seconds == 3600.0
        assert config.backoff_multiplier == 2.0

    def test_custom_values(self, config: ConfigRetryWorker) -> None:
        """Custom config values are preserved."""
        assert config.poll_interval_seconds == 10
        assert config.batch_size == 10
        assert config.max_retry_attempts == 3

    def test_invalid_poll_interval(self) -> None:
        """Poll interval below minimum raises validation error."""
        with pytest.raises(Exception):
            ConfigRetryWorker(
                postgres_dsn="postgresql://test:test@localhost:5432/test",
                poll_interval_seconds=1,  # Below ge=5
            )

    def test_invalid_batch_size(self) -> None:
        """Batch size below minimum raises validation error."""
        with pytest.raises(Exception):
            ConfigRetryWorker(
                postgres_dsn="postgresql://test:test@localhost:5432/test",
                batch_size=0,  # Below ge=1
            )


# =========================================================================
# Model Tests
# =========================================================================


class TestModelDeliveryAttempt:
    """Tests for ModelDeliveryAttempt."""

    def test_creation(self) -> None:
        """Can create a delivery attempt model."""
        now = datetime.now(UTC)
        attempt = ModelDeliveryAttempt(
            id=uuid4(),
            subscription_id=uuid4(),
            notification_payload='{"test": true}',
            status=EnumDeliveryStatus.FAILED,
            attempt_count=2,
            max_attempts=5,
            next_retry_at=now + timedelta(minutes=5),
            last_error="timeout",
            created_at=now,
            updated_at=now,
        )
        assert attempt.status == EnumDeliveryStatus.FAILED
        assert attempt.attempt_count == 2

    def test_frozen(self) -> None:
        """Model is immutable."""
        now = datetime.now(UTC)
        attempt = ModelDeliveryAttempt(
            id=uuid4(),
            subscription_id=uuid4(),
            notification_payload="{}",
            status=EnumDeliveryStatus.PENDING,
            created_at=now,
            updated_at=now,
        )
        with pytest.raises(Exception):
            attempt.status = EnumDeliveryStatus.FAILED  # type: ignore[misc]


class TestModelRetryResult:
    """Tests for ModelRetryResult."""

    def test_bool_true_when_retries_attempted(self) -> None:
        """Result is truthy when retries were attempted."""
        now = datetime.now(UTC)
        result = ModelRetryResult(
            correlation_id=uuid4(),
            started_at=now,
            completed_at=now,
            retries_attempted=3,
            retries_succeeded=2,
            retries_failed=1,
        )
        assert bool(result) is True

    def test_bool_false_when_no_retries(self) -> None:
        """Result is falsy when no retries were attempted."""
        now = datetime.now(UTC)
        result = ModelRetryResult(
            correlation_id=uuid4(),
            started_at=now,
            completed_at=now,
            retries_attempted=0,
        )
        assert bool(result) is False


# =========================================================================
# Backoff Tests
# =========================================================================


class TestBackoffCalculation:
    """Tests for exponential backoff."""

    def test_first_retry_uses_base(self, worker: ServiceRetryWorker) -> None:
        """First retry (attempt 0) uses base delay."""
        before = datetime.now(UTC)
        next_retry = worker.calculate_next_retry_at(0)
        # Base is 10s, so delay should be approximately 10s from now
        expected_min = before + timedelta(seconds=9)
        expected_max = before + timedelta(seconds=12)
        assert expected_min <= next_retry <= expected_max

    def test_exponential_growth(self, worker: ServiceRetryWorker) -> None:
        """Backoff grows exponentially with attempt count."""
        before = datetime.now(UTC)
        retry_0 = worker.calculate_next_retry_at(0)
        retry_1 = worker.calculate_next_retry_at(1)
        retry_2 = worker.calculate_next_retry_at(2)

        # With base=10, multiplier=2:
        # attempt 0: 10s, attempt 1: 20s, attempt 2: 40s
        delta_0 = (retry_0 - before).total_seconds()
        delta_1 = (retry_1 - before).total_seconds()
        delta_2 = (retry_2 - before).total_seconds()

        assert delta_1 > delta_0
        assert delta_2 > delta_1

    def test_backoff_respects_max(self, worker: ServiceRetryWorker) -> None:
        """Backoff is capped at max_seconds."""
        before = datetime.now(UTC)
        # With base=10, multiplier=2, attempt=100 would be huge
        # but max=600 caps it
        next_retry = worker.calculate_next_retry_at(100)
        max_delta = (next_retry - before).total_seconds()
        assert max_delta <= 601  # 600 + small margin


# =========================================================================
# Poll and Retry Tests
# =========================================================================


class TestPollAndRetry:
    """Tests for the poll_and_retry cycle."""

    async def test_no_pending_retries(
        self, worker: ServiceRetryWorker, mock_pool: MagicMock
    ) -> None:
        """Returns empty result when no pending retries found."""
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)

        result = await worker.poll_and_retry()

        assert result.retries_attempted == 0
        assert result.retries_succeeded == 0
        assert result.retries_failed == 0
        assert result.moved_to_dlq == 0
        assert not bool(result)

    async def test_successful_delivery(
        self,
        worker: ServiceRetryWorker,
        mock_pool: MagicMock,
        mock_deliver_fn: AsyncMock,
    ) -> None:
        """Successful delivery updates status and increments counters."""
        row = _make_attempt_row(attempt_count=1, max_attempts=3)
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[row])
        conn.execute = AsyncMock(return_value="UPDATE 1")
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)

        result = await worker.poll_and_retry()

        assert result.retries_attempted == 1
        assert result.retries_succeeded == 1
        assert result.retries_failed == 0
        assert result.moved_to_dlq == 0
        mock_deliver_fn.assert_awaited_once_with(row["notification_payload"])

    async def test_failed_delivery_schedules_retry(
        self,
        worker: ServiceRetryWorker,
        mock_pool: MagicMock,
        mock_deliver_fn: AsyncMock,
    ) -> None:
        """Failed delivery increments attempt_count and schedules next retry."""
        row = _make_attempt_row(attempt_count=1, max_attempts=3)
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[row])
        conn.execute = AsyncMock(return_value="UPDATE 1")
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)

        mock_deliver_fn.side_effect = ConnectionError("delivery failed")

        result = await worker.poll_and_retry()

        assert result.retries_attempted == 1
        assert result.retries_succeeded == 0
        assert result.retries_failed == 1
        assert len(result.errors) == 1
        assert "ConnectionError" in result.errors[0][1]

    async def test_dlq_escalation(
        self,
        worker: ServiceRetryWorker,
        mock_pool: MagicMock,
        mock_deliver_fn: AsyncMock,
    ) -> None:
        """Attempts exceeding max retries are moved to DLQ."""
        # attempt_count=3 with max_attempts=3 -> should DLQ
        row = _make_attempt_row(attempt_count=3, max_attempts=3)
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[row])
        conn.execute = AsyncMock(return_value="UPDATE 1")
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)

        result = await worker.poll_and_retry()

        assert result.retries_attempted == 1
        assert result.moved_to_dlq == 1
        assert result.retries_succeeded == 0
        # Delivery function should NOT be called for DLQ
        mock_deliver_fn.assert_not_awaited()

    async def test_mixed_results(
        self,
        worker: ServiceRetryWorker,
        mock_pool: MagicMock,
        mock_deliver_fn: AsyncMock,
    ) -> None:
        """Handles mix of success, failure, and DLQ in one cycle."""
        rows = [
            _make_attempt_row(attempt_count=1, max_attempts=3),  # Will succeed
            _make_attempt_row(attempt_count=1, max_attempts=3),  # Will fail
            _make_attempt_row(attempt_count=3, max_attempts=3),  # Will DLQ
        ]

        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=rows)
        conn.execute = AsyncMock(return_value="UPDATE 1")
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)

        # First call succeeds, second fails
        mock_deliver_fn.side_effect = [None, ValueError("bad payload")]

        result = await worker.poll_and_retry()

        assert result.retries_attempted == 3
        assert result.retries_succeeded == 1
        assert result.retries_failed == 1
        assert result.moved_to_dlq == 1

    async def test_stores_last_result(
        self,
        worker: ServiceRetryWorker,
        mock_pool: MagicMock,
    ) -> None:
        """Last result is stored for health checks."""
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)

        assert worker.last_result is None
        await worker.poll_and_retry()
        assert worker.last_result is not None
        assert isinstance(worker.last_result, ModelRetryResult)


# =========================================================================
# Circuit Breaker Tests
# =========================================================================


class TestCircuitBreaker:
    """Tests for circuit breaker integration."""

    async def test_circuit_opens_after_threshold(
        self,
        config: ConfigRetryWorker,
        mock_deliver_fn: AsyncMock,
    ) -> None:
        """Circuit opens after consecutive database failures."""
        import asyncpg

        pool = MagicMock()
        conn = AsyncMock()
        conn.fetch = AsyncMock(side_effect=asyncpg.PostgresConnectionError("down"))
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        worker = ServiceRetryWorker(
            pool=pool,
            config=config,
            deliver_fn=mock_deliver_fn,
        )

        # Trip circuit breaker (threshold=3)
        for _ in range(3):
            with pytest.raises(InfraConnectionError):
                await worker.poll_and_retry()

        # Now circuit should be open
        with pytest.raises(InfraUnavailableError):
            await worker.poll_and_retry()


# =========================================================================
# Graceful Shutdown Tests
# =========================================================================


class TestGracefulShutdown:
    """Tests for graceful shutdown."""

    def test_stop_sets_event(self, worker: ServiceRetryWorker) -> None:
        """stop() sets the shutdown event."""
        assert not worker._shutdown_event.is_set()
        worker.stop()
        assert worker._shutdown_event.is_set()

    async def test_run_exits_on_stop(
        self,
        worker: ServiceRetryWorker,
        mock_pool: MagicMock,
    ) -> None:
        """run() exits when stop() is called."""
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)

        # Stop after a short delay
        async def stop_after_delay() -> None:
            await asyncio.sleep(0.1)
            worker.stop()

        task = asyncio.create_task(stop_after_delay())
        await asyncio.wait_for(worker.run(), timeout=5.0)
        await task


# =========================================================================
# Health Check Tests
# =========================================================================


class TestHealthCheck:
    """Tests for health status reporting."""

    def test_health_without_result(self, worker: ServiceRetryWorker) -> None:
        """Health status returns None for last_result before first run."""
        status = worker.get_health_status()
        assert status["service"] == "retry-worker"
        assert status["last_result"] is None
        assert "circuit_breaker" in status
        assert "config" in status

    async def test_health_with_result(
        self,
        worker: ServiceRetryWorker,
        mock_pool: MagicMock,
    ) -> None:
        """Health status includes last result after a run."""
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)

        await worker.poll_and_retry()

        status = worker.get_health_status()
        assert status["last_result"] is not None
        last = status["last_result"]
        assert isinstance(last, dict)
        assert "correlation_id" in last
        assert "retries_attempted" in last
        assert "duration_ms" in last


# =========================================================================
# Enum Tests
# =========================================================================


class TestEnumDeliveryStatus:
    """Tests for EnumDeliveryStatus values."""

    def test_all_statuses(self) -> None:
        """All expected status values exist."""
        assert EnumDeliveryStatus.PENDING.value == "pending"
        assert EnumDeliveryStatus.FAILED.value == "failed"
        assert EnumDeliveryStatus.SUCCEEDED.value == "succeeded"
        assert EnumDeliveryStatus.DLQ.value == "dlq"

    def test_string_enum(self) -> None:
        """Status values are strings for SQL compatibility."""
        for status in EnumDeliveryStatus:
            assert isinstance(status.value, str)
