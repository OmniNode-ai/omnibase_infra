# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for event bus readiness checking (OMN-1931).

Tests the readiness API across EventBusInmemory and mocked EventBusKafka,
ModelEventBusReadiness model, RuntimeHostProcess.readiness_check(), and
ServiceHealth /ready endpoint separation.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models import ModelEventBusReadiness
from tests.conftest import make_test_node_identity

pytestmark = pytest.mark.unit


class TestModelEventBusReadiness:
    """Test the ModelEventBusReadiness Pydantic model."""

    def test_ready_state(self) -> None:
        """Test model construction for a ready state."""
        model = ModelEventBusReadiness(
            is_ready=True,
            consumers_started=True,
            assignments={"topic-a": [0, 1]},
            consume_tasks_alive={"topic-a": True},
            required_topics=("topic-a",),
            required_topics_ready=True,
        )
        assert model.is_ready is True
        assert model.required_topics_ready is True
        assert model.last_error == ""

    def test_not_ready_state(self) -> None:
        """Test model construction for a not-ready state."""
        model = ModelEventBusReadiness(
            is_ready=False,
            consumers_started=True,
            assignments={},
            consume_tasks_alive={},
            required_topics=("topic-a",),
            required_topics_ready=False,
            last_error="No partition assignments",
        )
        assert model.is_ready is False
        assert model.last_error == "No partition assignments"

    def test_frozen_immutability(self) -> None:
        """Test that model is immutable (frozen=True)."""
        model = ModelEventBusReadiness(
            is_ready=True,
            consumers_started=True,
            required_topics=(),
            required_topics_ready=True,
        )
        with pytest.raises(Exception):
            model.is_ready = False  # type: ignore[misc]

    def test_no_extra_fields(self) -> None:
        """Test that extra fields are forbidden."""
        with pytest.raises(Exception):
            ModelEventBusReadiness(
                is_ready=True,
                consumers_started=True,
                required_topics=(),
                required_topics_ready=True,
                unknown_field="bad",  # type: ignore[call-arg]
            )


class TestInmemoryReadiness:
    """Test EventBusInmemory.get_readiness_status()."""

    @pytest.fixture
    def event_bus(self) -> EventBusInmemory:
        return EventBusInmemory(environment="test", group="test-group")

    @pytest.mark.asyncio
    async def test_not_ready_before_start(self, event_bus: EventBusInmemory) -> None:
        """Inmemory bus is not ready before start."""
        readiness = await event_bus.get_readiness_status()
        assert readiness.is_ready is False
        assert readiness.consumers_started is False

    @pytest.mark.asyncio
    async def test_ready_after_start(self, event_bus: EventBusInmemory) -> None:
        """Inmemory bus is always ready once started."""
        await event_bus.start()
        readiness = await event_bus.get_readiness_status()
        assert readiness.is_ready is True
        assert readiness.consumers_started is True
        assert readiness.required_topics_ready is True
        await event_bus.close()

    @pytest.mark.asyncio
    async def test_not_ready_after_close(self, event_bus: EventBusInmemory) -> None:
        """Inmemory bus reverts to not-ready after close."""
        await event_bus.start()
        await event_bus.close()
        readiness = await event_bus.get_readiness_status()
        assert readiness.is_ready is False

    @pytest.mark.asyncio
    async def test_subscribe_with_required_for_readiness(
        self, event_bus: EventBusInmemory
    ) -> None:
        """Inmemory subscribe accepts required_for_readiness parameter."""
        await event_bus.start()
        identity = make_test_node_identity()

        async def handler(msg: object) -> None:
            pass

        unsubscribe = await event_bus.subscribe(
            topic="test-topic",
            node_identity=identity,
            on_message=handler,
            required_for_readiness=True,
        )
        # Should still be ready (inmemory ignores this parameter)
        readiness = await event_bus.get_readiness_status()
        assert readiness.is_ready is True

        await unsubscribe()
        await event_bus.close()


class TestKafkaReadinessTracking:
    """Test EventBusKafka required_for_readiness tracking.

    Uses mocking to avoid real Kafka connections while verifying
    the readiness tracking logic.
    """

    @pytest.mark.asyncio
    async def test_required_topics_tracked_on_subscribe(self) -> None:
        """Verify that subscribe() with required_for_readiness=True tracks the topic."""
        from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
        from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

        config = ModelKafkaEventBusConfig(
            bootstrap_servers="localhost:9092",
            environment="test",
        )
        bus = EventBusKafka(config=config)

        # Before any subscriptions, no required topics
        assert len(bus._required_topics) == 0

        # Mock internal methods to avoid actual Kafka connection
        bus._started = True
        bus._start_consumer_for_topic = AsyncMock()  # type: ignore[method-assign]
        bus._validate_topic_name = MagicMock()  # type: ignore[method-assign]

        identity = make_test_node_identity()

        async def handler(msg: object) -> None:
            pass

        # Subscribe with required_for_readiness=True
        await bus.subscribe(
            topic="required-topic",
            node_identity=identity,
            on_message=handler,
            required_for_readiness=True,
        )
        assert "required-topic" in bus._required_topics

        # Subscribe without required_for_readiness
        await bus.subscribe(
            topic="optional-topic",
            node_identity=identity,
            on_message=handler,
            required_for_readiness=False,
        )
        assert "optional-topic" not in bus._required_topics
        assert len(bus._required_topics) == 1

    @pytest.mark.asyncio
    async def test_readiness_not_ready_when_not_started(self) -> None:
        """Bus reports not ready when not started."""
        from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
        from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

        config = ModelKafkaEventBusConfig(
            bootstrap_servers="localhost:9092",
            environment="test",
        )
        bus = EventBusKafka(config=config)
        readiness = await bus.get_readiness_status()
        assert readiness.is_ready is False
        assert readiness.consumers_started is False

    @pytest.mark.asyncio
    async def test_readiness_with_no_required_topics(self) -> None:
        """Bus with no required topics is ready once started."""
        from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
        from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

        config = ModelKafkaEventBusConfig(
            bootstrap_servers="localhost:9092",
            environment="test",
        )
        bus = EventBusKafka(config=config)
        bus._started = True
        readiness = await bus.get_readiness_status()
        assert readiness.is_ready is True
        assert readiness.required_topics_ready is True

    @pytest.mark.asyncio
    async def test_readiness_not_ready_missing_assignments(self) -> None:
        """Bus with required topics but no consumer assignments is not ready."""
        from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
        from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

        config = ModelKafkaEventBusConfig(
            bootstrap_servers="localhost:9092",
            environment="test",
        )
        bus = EventBusKafka(config=config)
        bus._started = True
        bus._required_topics = {"important-topic"}
        # No consumers registered -> not ready
        readiness = await bus.get_readiness_status()
        assert readiness.is_ready is False
        assert readiness.required_topics_ready is False

    @pytest.mark.asyncio
    async def test_readiness_ready_with_assignments(self) -> None:
        """Bus is ready when required topics have consumers with assignments."""
        from aiokafka import TopicPartition

        from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
        from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

        config = ModelKafkaEventBusConfig(
            bootstrap_servers="localhost:9092",
            environment="test",
        )
        bus = EventBusKafka(config=config)
        bus._started = True
        bus._required_topics = {"important-topic"}

        # Mock consumer with assignments
        mock_consumer = MagicMock()
        mock_consumer.assignment.return_value = {
            TopicPartition("important-topic", 0),
            TopicPartition("important-topic", 1),
        }
        bus._consumers["important-topic"] = mock_consumer

        # Mock alive consumer task
        mock_task = MagicMock()
        mock_task.done.return_value = False
        bus._consumer_tasks["important-topic"] = mock_task

        readiness = await bus.get_readiness_status()
        assert readiness.is_ready is True
        assert readiness.required_topics_ready is True
        assert readiness.assignments["important-topic"] == [0, 1]
        assert readiness.consume_tasks_alive["important-topic"] is True

    @pytest.mark.asyncio
    async def test_readiness_regression_dead_task(self) -> None:
        """Readiness flips to False when a required topic's consume task dies."""
        from aiokafka import TopicPartition

        from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
        from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

        config = ModelKafkaEventBusConfig(
            bootstrap_servers="localhost:9092",
            environment="test",
        )
        bus = EventBusKafka(config=config)
        bus._started = True
        bus._required_topics = {"important-topic"}

        mock_consumer = MagicMock()
        mock_consumer.assignment.return_value = {
            TopicPartition("important-topic", 0),
        }
        bus._consumers["important-topic"] = mock_consumer

        # Task has crashed (done() returns True)
        mock_task = MagicMock()
        mock_task.done.return_value = True
        bus._consumer_tasks["important-topic"] = mock_task

        readiness = await bus.get_readiness_status()
        assert readiness.is_ready is False
        assert readiness.consume_tasks_alive["important-topic"] is False

    @pytest.mark.asyncio
    async def test_readiness_regression_empty_assignments(self) -> None:
        """Readiness flips to False when consumer has no partition assignments."""
        from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
        from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

        config = ModelKafkaEventBusConfig(
            bootstrap_servers="localhost:9092",
            environment="test",
        )
        bus = EventBusKafka(config=config)
        bus._started = True
        bus._required_topics = {"important-topic"}

        # Consumer exists but with empty assignments (rebalance revoked)
        mock_consumer = MagicMock()
        mock_consumer.assignment.return_value = set()
        bus._consumers["important-topic"] = mock_consumer

        mock_task = MagicMock()
        mock_task.done.return_value = False
        bus._consumer_tasks["important-topic"] = mock_task

        readiness = await bus.get_readiness_status()
        assert readiness.is_ready is False
        assert readiness.required_topics_ready is False


class TestRuntimeHostProcessReadiness:
    """Test RuntimeHostProcess.readiness_check()."""

    @pytest.mark.asyncio
    async def test_ready_when_running_and_bus_ready(self) -> None:
        """Runtime is ready when running, not draining, and bus is ready."""
        from omnibase_infra.runtime.service_runtime_host_process import (
            RuntimeHostProcess,
        )

        mock_event_bus = AsyncMock()
        mock_event_bus.get_readiness_status = AsyncMock(
            return_value=ModelEventBusReadiness(
                is_ready=True,
                consumers_started=True,
                required_topics=(),
                required_topics_ready=True,
            )
        )

        runtime = RuntimeHostProcess.__new__(RuntimeHostProcess)
        runtime._event_bus = mock_event_bus
        runtime._is_running = True
        runtime._is_draining = False

        result = await runtime.readiness_check()
        assert result["ready"] is True
        assert result["is_running"] is True
        assert result["is_draining"] is False

    @pytest.mark.asyncio
    async def test_not_ready_when_draining(self) -> None:
        """Runtime is not ready when draining (graceful shutdown)."""
        from omnibase_infra.runtime.service_runtime_host_process import (
            RuntimeHostProcess,
        )

        mock_event_bus = AsyncMock()
        mock_event_bus.get_readiness_status = AsyncMock(
            return_value=ModelEventBusReadiness(
                is_ready=True,
                consumers_started=True,
                required_topics=(),
                required_topics_ready=True,
            )
        )

        runtime = RuntimeHostProcess.__new__(RuntimeHostProcess)
        runtime._event_bus = mock_event_bus
        runtime._is_running = True
        runtime._is_draining = True

        result = await runtime.readiness_check()
        assert result["ready"] is False

    @pytest.mark.asyncio
    async def test_not_ready_when_bus_not_ready(self) -> None:
        """Runtime is not ready when event bus is not ready."""
        from omnibase_infra.runtime.service_runtime_host_process import (
            RuntimeHostProcess,
        )

        mock_event_bus = AsyncMock()
        mock_event_bus.get_readiness_status = AsyncMock(
            return_value=ModelEventBusReadiness(
                is_ready=False,
                consumers_started=True,
                required_topics=("topic-a",),
                required_topics_ready=False,
            )
        )

        runtime = RuntimeHostProcess.__new__(RuntimeHostProcess)
        runtime._event_bus = mock_event_bus
        runtime._is_running = True
        runtime._is_draining = False

        result = await runtime.readiness_check()
        assert result["ready"] is False

    @pytest.mark.asyncio
    async def test_fallback_for_bus_without_readiness(self) -> None:
        """Runtime falls back to health_check() for buses without get_readiness_status."""
        from omnibase_infra.runtime.service_runtime_host_process import (
            RuntimeHostProcess,
        )

        mock_event_bus = AsyncMock()
        # No get_readiness_status method
        del mock_event_bus.get_readiness_status
        mock_event_bus.health_check = AsyncMock(return_value={"healthy": True})

        runtime = RuntimeHostProcess.__new__(RuntimeHostProcess)
        runtime._event_bus = mock_event_bus
        runtime._is_running = True
        runtime._is_draining = False

        result = await runtime.readiness_check()
        assert result["ready"] is True
        assert result["event_bus_readiness"]["fallback"] is True


class TestServiceHealthReadyEndpoint:
    """Test ServiceHealth /ready endpoint separation."""

    @pytest.mark.asyncio
    async def test_ready_returns_200(self) -> None:
        """GET /ready returns 200 when runtime is ready."""
        from aiohttp.test_utils import TestClient, TestServer

        from omnibase_infra.services.service_health import ServiceHealth

        mock_runtime = AsyncMock()
        mock_runtime.health_check = AsyncMock(
            return_value={"healthy": True, "degraded": False}
        )
        mock_runtime.readiness_check = AsyncMock(
            return_value={"ready": True, "is_running": True, "is_draining": False}
        )

        service = ServiceHealth(runtime=mock_runtime, port=0)
        # Manually construct the app to test routes
        service._app = service._app or __import__("aiohttp").web.Application()
        from aiohttp import web

        app = web.Application()
        app.router.add_get("/health", service._handle_health)
        app.router.add_get("/ready", service._handle_readiness)

        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/ready")
            assert resp.status == 200
            data = await resp.json()
            assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_not_ready_returns_503(self) -> None:
        """GET /ready returns 503 when runtime is not ready."""
        from aiohttp.test_utils import TestClient, TestServer

        from omnibase_infra.services.service_health import ServiceHealth

        mock_runtime = AsyncMock()
        mock_runtime.readiness_check = AsyncMock(
            return_value={"ready": False, "is_running": True, "is_draining": False}
        )

        service = ServiceHealth(runtime=mock_runtime, port=0)
        from aiohttp import web

        app = web.Application()
        app.router.add_get("/ready", service._handle_readiness)

        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/ready")
            assert resp.status == 503
            data = await resp.json()
            assert data["status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_health_and_ready_are_independent(self) -> None:
        """GET /health and GET /ready can return different statuses."""
        from aiohttp.test_utils import TestClient, TestServer

        from omnibase_infra.services.service_health import ServiceHealth

        mock_runtime = AsyncMock()
        # Health says healthy (process is alive)
        mock_runtime.health_check = AsyncMock(
            return_value={
                "healthy": True,
                "degraded": False,
                "is_running": True,
                "is_draining": False,
                "pending_message_count": 0,
                "event_bus": {"healthy": True},
                "event_bus_healthy": True,
                "failed_handlers": {},
                "registered_handlers": ["http"],
                "handlers": {},
                "no_handlers_registered": False,
            }
        )
        # Readiness says not ready (Kafka consumers not assigned)
        mock_runtime.readiness_check = AsyncMock(
            return_value={"ready": False, "is_running": True, "is_draining": False}
        )

        service = ServiceHealth(runtime=mock_runtime, port=0)
        from aiohttp import web

        app = web.Application()
        app.router.add_get("/health", service._handle_health)
        app.router.add_get("/ready", service._handle_readiness)

        async with TestClient(TestServer(app)) as client:
            health_resp = await client.get("/health")
            ready_resp = await client.get("/ready")

            # Health: 200 (process alive), Ready: 503 (consumers not ready)
            assert health_resp.status == 200
            assert ready_resp.status == 503
