# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for health check endpoints in ServiceLlmCostAggregator.

Tests:
    - EnumHealthStatus enum values
    - _determine_health_status() state machine logic
    - _health_handler() HTTP response codes and body
    - _liveness_handler() alive/dead detection
    - _readiness_handler() dependency checking

All tests mock aiohttp and asyncpg -- no real infrastructure required.

Related Tickets:
    - OMN-2240: E1-T4 LLM cost aggregation service
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web

from omnibase_infra.services.observability.llm_cost_aggregation.config import (
    ConfigLlmCostAggregation,
)
from omnibase_infra.services.observability.llm_cost_aggregation.consumer import (
    ConsumerMetrics,
    EnumHealthStatus,
    ServiceLlmCostAggregator,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config() -> ConfigLlmCostAggregation:
    """Create a minimal config for testing."""
    return ConfigLlmCostAggregation(
        postgres_dsn="postgresql://test:test@localhost:5432/test",
        kafka_bootstrap_servers="localhost:9092",
        health_check_port=18089,
        health_check_staleness_seconds=300,
        health_check_poll_staleness_seconds=60,
        startup_grace_period_seconds=60.0,
    )


@pytest.fixture
def service(config: ConfigLlmCostAggregation) -> ServiceLlmCostAggregator:
    """Create a service instance without starting it."""
    return ServiceLlmCostAggregator(config)


@pytest.fixture
def mock_request() -> MagicMock:
    """Create a mock aiohttp web request."""
    return MagicMock(spec=web.Request)


def _make_metrics_snapshot(
    *,
    messages_received: int = 0,
    messages_processed: int = 0,
    messages_failed: int = 0,
    messages_skipped: int = 0,
    batches_processed: int = 0,
    aggregations_written: int = 0,
    commit_failures: int = 0,
    last_poll_at: str | None = None,
    last_successful_write_at: str | None = None,
    last_commit_failure_at: str | None = None,
    started_at: str | None = None,
) -> dict[str, object]:
    """Build a metrics snapshot dict matching ConsumerMetrics.snapshot() output."""
    if started_at is None:
        started_at = datetime.now(UTC).isoformat()
    return {
        "messages_received": messages_received,
        "messages_processed": messages_processed,
        "messages_failed": messages_failed,
        "messages_skipped": messages_skipped,
        "batches_processed": batches_processed,
        "aggregations_written": aggregations_written,
        "commit_failures": commit_failures,
        "last_poll_at": last_poll_at,
        "last_successful_write_at": last_successful_write_at,
        "last_commit_failure_at": last_commit_failure_at,
        "started_at": started_at,
    }


# =============================================================================
# Tests: EnumHealthStatus
# =============================================================================


@pytest.mark.unit
class TestEnumHealthStatus:
    """Tests for EnumHealthStatus enum."""

    def test_enum_values(self) -> None:
        """HEALTHY, DEGRADED, UNHEALTHY exist with correct string values."""
        assert EnumHealthStatus.HEALTHY == "healthy"
        assert EnumHealthStatus.DEGRADED == "degraded"
        assert EnumHealthStatus.UNHEALTHY == "unhealthy"

    def test_enum_is_str(self) -> None:
        """EnumHealthStatus members are strings (StrEnum)."""
        assert isinstance(EnumHealthStatus.HEALTHY, str)
        assert isinstance(EnumHealthStatus.DEGRADED, str)
        assert isinstance(EnumHealthStatus.UNHEALTHY, str)

    def test_enum_value_attribute(self) -> None:
        """The .value attribute matches the string representation."""
        assert EnumHealthStatus.HEALTHY.value == "healthy"
        assert EnumHealthStatus.DEGRADED.value == "degraded"
        assert EnumHealthStatus.UNHEALTHY.value == "unhealthy"


# =============================================================================
# Tests: _determine_health_status
# =============================================================================


@pytest.mark.unit
class TestDetermineHealthStatus:
    """Tests for _determine_health_status logic."""

    def test_healthy_when_running_normally(
        self, service: ServiceLlmCostAggregator
    ) -> None:
        """All conditions good returns HEALTHY."""
        service._running = True
        now = datetime.now(UTC)
        snapshot = _make_metrics_snapshot(
            messages_received=10,
            last_poll_at=(now - timedelta(seconds=5)).isoformat(),
            last_successful_write_at=(now - timedelta(seconds=10)).isoformat(),
            started_at=(now - timedelta(minutes=5)).isoformat(),
        )
        circuit_state: dict[str, object] = {"state": "closed"}

        result = service._determine_health_status(snapshot, circuit_state)

        assert result == EnumHealthStatus.HEALTHY

    def test_degraded_when_circuit_breaker_open(
        self, service: ServiceLlmCostAggregator
    ) -> None:
        """Circuit breaker in OPEN state returns DEGRADED."""
        service._running = True
        snapshot = _make_metrics_snapshot()
        circuit_state: dict[str, object] = {"state": "open"}

        result = service._determine_health_status(snapshot, circuit_state)

        assert result == EnumHealthStatus.DEGRADED

    def test_degraded_when_circuit_breaker_half_open(
        self, service: ServiceLlmCostAggregator
    ) -> None:
        """Circuit breaker in HALF_OPEN state returns DEGRADED."""
        service._running = True
        snapshot = _make_metrics_snapshot()
        circuit_state: dict[str, object] = {"state": "half_open"}

        result = service._determine_health_status(snapshot, circuit_state)

        assert result == EnumHealthStatus.DEGRADED

    def test_degraded_when_poll_stale(self, service: ServiceLlmCostAggregator) -> None:
        """Last poll time exceeds stale threshold returns DEGRADED."""
        service._running = True
        now = datetime.now(UTC)
        stale_poll = now - timedelta(
            seconds=service._config.health_check_poll_staleness_seconds + 30
        )
        snapshot = _make_metrics_snapshot(
            last_poll_at=stale_poll.isoformat(),
            last_successful_write_at=now.isoformat(),
            messages_received=10,
            started_at=(now - timedelta(minutes=5)).isoformat(),
        )
        circuit_state: dict[str, object] = {"state": "closed"}

        result = service._determine_health_status(snapshot, circuit_state)

        assert result == EnumHealthStatus.DEGRADED

    def test_degraded_when_write_stale(self, service: ServiceLlmCostAggregator) -> None:
        """Last write time exceeds stale threshold returns DEGRADED."""
        service._running = True
        now = datetime.now(UTC)
        stale_write = now - timedelta(
            seconds=service._config.health_check_staleness_seconds + 30
        )
        snapshot = _make_metrics_snapshot(
            messages_received=10,
            last_poll_at=now.isoformat(),
            last_successful_write_at=stale_write.isoformat(),
            started_at=(now - timedelta(minutes=10)).isoformat(),
        )
        circuit_state: dict[str, object] = {"state": "closed"}

        result = service._determine_health_status(snapshot, circuit_state)

        assert result == EnumHealthStatus.DEGRADED

    def test_healthy_during_startup_grace(
        self, service: ServiceLlmCostAggregator
    ) -> None:
        """Within startup grace period and no writes yet returns HEALTHY."""
        service._running = True
        now = datetime.now(UTC)
        # Started 10 seconds ago, grace period is 60 seconds
        snapshot = _make_metrics_snapshot(
            last_successful_write_at=None,
            started_at=(now - timedelta(seconds=10)).isoformat(),
        )
        circuit_state: dict[str, object] = {"state": "closed"}

        result = service._determine_health_status(snapshot, circuit_state)

        assert result == EnumHealthStatus.HEALTHY

    def test_degraded_after_startup_grace_no_writes(
        self, service: ServiceLlmCostAggregator
    ) -> None:
        """Past startup grace period with no writes returns DEGRADED."""
        service._running = True
        now = datetime.now(UTC)
        # Started 120 seconds ago, grace period is 60 seconds
        snapshot = _make_metrics_snapshot(
            last_successful_write_at=None,
            started_at=(now - timedelta(seconds=120)).isoformat(),
        )
        circuit_state: dict[str, object] = {"state": "closed"}

        result = service._determine_health_status(snapshot, circuit_state)

        assert result == EnumHealthStatus.DEGRADED

    def test_unhealthy_when_not_running(
        self, service: ServiceLlmCostAggregator
    ) -> None:
        """Service not running returns UNHEALTHY."""
        service._running = False
        snapshot = _make_metrics_snapshot()
        circuit_state: dict[str, object] = {"state": "closed"}

        result = service._determine_health_status(snapshot, circuit_state)

        assert result == EnumHealthStatus.UNHEALTHY

    def test_healthy_when_no_started_at_and_no_writes(
        self, service: ServiceLlmCostAggregator
    ) -> None:
        """No started_at and no writes returns HEALTHY (fallback)."""
        service._running = True
        snapshot = _make_metrics_snapshot(
            last_successful_write_at=None,
            started_at=None,
        )
        # Remove started_at key entirely to test the None branch
        snapshot.pop("started_at", None)
        circuit_state: dict[str, object] = {"state": "closed"}

        result = service._determine_health_status(snapshot, circuit_state)

        assert result == EnumHealthStatus.HEALTHY

    def test_healthy_when_write_fresh_and_messages_received(
        self, service: ServiceLlmCostAggregator
    ) -> None:
        """Recent write with received messages returns HEALTHY."""
        service._running = True
        now = datetime.now(UTC)
        snapshot = _make_metrics_snapshot(
            messages_received=50,
            last_poll_at=now.isoformat(),
            last_successful_write_at=(now - timedelta(seconds=10)).isoformat(),
            started_at=(now - timedelta(minutes=10)).isoformat(),
        )
        circuit_state: dict[str, object] = {"state": "closed"}

        result = service._determine_health_status(snapshot, circuit_state)

        assert result == EnumHealthStatus.HEALTHY


# =============================================================================
# Tests: _health_handler
# =============================================================================


@pytest.mark.unit
class TestHealthHandler:
    """Tests for _health_handler HTTP endpoint."""

    async def test_returns_200_for_healthy(
        self, service: ServiceLlmCostAggregator, mock_request: MagicMock
    ) -> None:
        """HTTP 200 with healthy status JSON when service is healthy."""
        service._running = True
        now = datetime.now(UTC)
        service.metrics.last_poll_at = now
        service.metrics.last_successful_write_at = now
        service.metrics.started_at = now - timedelta(minutes=5)

        # Mock the writer's circuit breaker state
        mock_writer = MagicMock()
        mock_writer.get_circuit_breaker_state.return_value = {"state": "closed"}
        service._writer = mock_writer

        response = await service._health_handler(mock_request)

        assert response.status == 200
        body = json.loads(response.body)
        assert body["status"] == "healthy"
        assert body["consumer_running"] is True

    async def test_returns_503_for_degraded(
        self, service: ServiceLlmCostAggregator, mock_request: MagicMock
    ) -> None:
        """HTTP 503 with degraded status when circuit breaker is open."""
        service._running = True
        now = datetime.now(UTC)
        service.metrics.started_at = now - timedelta(minutes=5)

        mock_writer = MagicMock()
        mock_writer.get_circuit_breaker_state.return_value = {"state": "open"}
        service._writer = mock_writer

        response = await service._health_handler(mock_request)

        assert response.status == 503
        body = json.loads(response.body)
        assert body["status"] == "degraded"

    async def test_returns_503_for_unhealthy(
        self, service: ServiceLlmCostAggregator, mock_request: MagicMock
    ) -> None:
        """HTTP 503 with unhealthy status when service is not running."""
        service._running = False
        service._writer = None

        response = await service._health_handler(mock_request)

        assert response.status == 503
        body = json.loads(response.body)
        assert body["status"] == "unhealthy"
        assert body["consumer_running"] is False

    async def test_response_includes_metrics(
        self, service: ServiceLlmCostAggregator, mock_request: MagicMock
    ) -> None:
        """Response body includes consumer metrics snapshot fields."""
        service._running = True
        now = datetime.now(UTC)
        service.metrics.messages_processed = 42
        service.metrics.messages_failed = 3
        service.metrics.batches_processed = 5
        service.metrics.aggregations_written = 100
        service.metrics.last_poll_at = now
        service.metrics.last_successful_write_at = now
        service.metrics.started_at = now - timedelta(minutes=5)

        mock_writer = MagicMock()
        mock_writer.get_circuit_breaker_state.return_value = {"state": "closed"}
        service._writer = mock_writer

        response = await service._health_handler(mock_request)

        body = json.loads(response.body)
        assert body["messages_processed"] == 42
        assert body["messages_failed"] == 3
        assert body["batches_processed"] == 5
        assert body["aggregations_written"] == 100
        assert body["consumer_id"] == service._consumer_id
        assert "last_poll_time" in body
        assert "last_successful_write" in body
        assert "circuit_breaker_state" in body

    async def test_handles_no_writer(
        self, service: ServiceLlmCostAggregator, mock_request: MagicMock
    ) -> None:
        """Health handler works gracefully when writer is None."""
        service._running = True
        service._writer = None
        now = datetime.now(UTC)
        service.metrics.started_at = now - timedelta(seconds=5)

        response = await service._health_handler(mock_request)

        # Should still return a valid response (HEALTHY during grace, no circuit state)
        assert response.status in (200, 503)
        body = json.loads(response.body)
        assert "status" in body


# =============================================================================
# Tests: _liveness_handler
# =============================================================================


@pytest.mark.unit
class TestLivenessHandler:
    """Tests for _liveness_handler Kubernetes liveness probe."""

    async def test_alive_when_running(
        self, service: ServiceLlmCostAggregator, mock_request: MagicMock
    ) -> None:
        """Returns 200 with 'alive' when _running=True."""
        service._running = True

        response = await service._liveness_handler(mock_request)

        assert response.status == 200
        body = json.loads(response.body)
        assert body["status"] == "alive"
        assert body["consumer_id"] == service._consumer_id

    async def test_dead_when_not_running(
        self, service: ServiceLlmCostAggregator, mock_request: MagicMock
    ) -> None:
        """Returns 503 with 'dead' when _running=False."""
        service._running = False

        response = await service._liveness_handler(mock_request)

        assert response.status == 503
        body = json.loads(response.body)
        assert body["status"] == "dead"


# =============================================================================
# Tests: _readiness_handler
# =============================================================================


@pytest.mark.unit
class TestReadinessHandler:
    """Tests for _readiness_handler Kubernetes readiness probe."""

    async def test_ready_when_all_deps_ok(
        self, service: ServiceLlmCostAggregator, mock_request: MagicMock
    ) -> None:
        """Returns 200 when pool, consumer, writer connected and running."""
        service._running = True
        service._pool = MagicMock()
        service._consumer = MagicMock()

        mock_writer = MagicMock()
        mock_writer.get_circuit_breaker_state.return_value = {"state": "closed"}
        service._writer = mock_writer

        response = await service._readiness_handler(mock_request)

        assert response.status == 200
        body = json.loads(response.body)
        assert body["status"] == "ready"
        assert body["dependencies"]["postgres_pool"] is True
        assert body["dependencies"]["kafka_consumer"] is True
        assert body["dependencies"]["writer"] is True
        assert body["dependencies"]["circuit_breaker"] is True

    async def test_not_ready_when_pool_missing(
        self, service: ServiceLlmCostAggregator, mock_request: MagicMock
    ) -> None:
        """Returns 503 when pool is None."""
        service._running = True
        service._pool = None
        service._consumer = MagicMock()

        mock_writer = MagicMock()
        mock_writer.get_circuit_breaker_state.return_value = {"state": "closed"}
        service._writer = mock_writer

        response = await service._readiness_handler(mock_request)

        assert response.status == 503
        body = json.loads(response.body)
        assert body["status"] == "not_ready"
        assert body["dependencies"]["postgres_pool"] is False

    async def test_not_ready_when_consumer_missing(
        self, service: ServiceLlmCostAggregator, mock_request: MagicMock
    ) -> None:
        """Returns 503 when Kafka consumer is None."""
        service._running = True
        service._pool = MagicMock()
        service._consumer = None

        mock_writer = MagicMock()
        mock_writer.get_circuit_breaker_state.return_value = {"state": "closed"}
        service._writer = mock_writer

        response = await service._readiness_handler(mock_request)

        assert response.status == 503
        body = json.loads(response.body)
        assert body["status"] == "not_ready"
        assert body["dependencies"]["kafka_consumer"] is False

    async def test_not_ready_when_writer_missing(
        self, service: ServiceLlmCostAggregator, mock_request: MagicMock
    ) -> None:
        """Returns 503 when writer is None."""
        service._running = True
        service._pool = MagicMock()
        service._consumer = MagicMock()
        service._writer = None

        response = await service._readiness_handler(mock_request)

        assert response.status == 503
        body = json.loads(response.body)
        assert body["status"] == "not_ready"
        assert body["dependencies"]["writer"] is False

    async def test_not_ready_when_circuit_breaker_open(
        self, service: ServiceLlmCostAggregator, mock_request: MagicMock
    ) -> None:
        """Returns 503 when circuit breaker is open."""
        service._running = True
        service._pool = MagicMock()
        service._consumer = MagicMock()

        mock_writer = MagicMock()
        mock_writer.get_circuit_breaker_state.return_value = {"state": "open"}
        service._writer = mock_writer

        response = await service._readiness_handler(mock_request)

        assert response.status == 503
        body = json.loads(response.body)
        assert body["status"] == "not_ready"
        assert body["dependencies"]["circuit_breaker"] is False

    async def test_not_ready_when_not_running(
        self, service: ServiceLlmCostAggregator, mock_request: MagicMock
    ) -> None:
        """Returns 503 when service is not running even if deps are present."""
        service._running = False
        service._pool = MagicMock()
        service._consumer = MagicMock()

        mock_writer = MagicMock()
        mock_writer.get_circuit_breaker_state.return_value = {"state": "closed"}
        service._writer = mock_writer

        response = await service._readiness_handler(mock_request)

        assert response.status == 503
        body = json.loads(response.body)
        assert body["status"] == "not_ready"
        assert body["consumer_running"] is False
