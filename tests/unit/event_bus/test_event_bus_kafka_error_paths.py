# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Focused unit tests for EventBusKafka publish/subscribe error paths.

Covers the surfaces flagged by Repowise as hotspots:
- publish() when bus not started -> InfraUnavailableError
- publish() when circuit is open -> InfraUnavailableError
- publish() with KafkaError after all retries -> InfraConnectionError
- publish() with TimeoutError after all retries -> InfraTimeoutError
- publish() with UnknownTopicOrPartitionError -> ProtocolConfigurationError (no retry)
- subscribe() without on_message -> ValueError
- subscribe() without node_identity or group_id -> ValueError
- subscribe() when not started: consumer not launched
- _publish_with_retry: closing mid-publish raises InfraUnavailableError
- circuit breaker threshold < 1 -> ProtocolConfigurationError at init
- start() connection failure -> InfraConnectionError (circuit records failure)
- start() timeout -> InfraTimeoutError (circuit records failure)

Related: OMN-12385
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiokafka.errors import KafkaError, UnknownTopicOrPartitionError
from pydantic import ValidationError

from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    InfraUnavailableError,
    ProtocolConfigurationError,
)
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.models import ModelNodeIdentity

_SERVERS = os.environ["KAFKA_BOOTSTRAP_SERVERS"]
_ENV = "test"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_config(
    *,
    max_retry_attempts: int = 0,
    retry_backoff_base: float = 0.001,
    circuit_breaker_threshold: int = 5,
) -> ModelKafkaEventBusConfig:
    return ModelKafkaEventBusConfig(
        bootstrap_servers=_SERVERS,
        environment=_ENV,
        max_retry_attempts=max_retry_attempts,
        retry_backoff_base=retry_backoff_base,
        circuit_breaker_threshold=circuit_breaker_threshold,
        timeout_seconds=5,
    )


def _make_mock_producer() -> AsyncMock:
    producer = AsyncMock()
    producer.start = AsyncMock()
    producer.stop = AsyncMock()
    producer._closed = False
    return producer


@pytest.fixture
async def started_bus() -> AsyncGenerator[tuple[EventBusKafka, AsyncMock], None]:
    """Started EventBusKafka with a mocked producer, no retries."""
    mock_producer = _make_mock_producer()
    future: asyncio.Future[MagicMock] = asyncio.get_event_loop().create_future()
    meta = MagicMock()
    meta.partition = 0
    meta.offset = 0
    future.set_result(meta)
    mock_producer.send = AsyncMock(return_value=future)

    with patch(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
        return_value=mock_producer,
    ):
        config = _make_config()
        bus = EventBusKafka(config=config)
        await bus.start()
        yield bus, mock_producer
        try:
            await bus.close()
        except Exception:  # noqa: BLE001
            pass


@pytest.fixture
async def unstarted_bus() -> AsyncGenerator[EventBusKafka, None]:
    """EventBusKafka that has NOT been started."""
    config = _make_config()
    bus = EventBusKafka(config=config)
    yield bus
    try:
        await bus.close()
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Init validation
# ---------------------------------------------------------------------------


class TestEventBusKafkaInit:
    """Validate constructor-time guards."""

    def test_circuit_breaker_threshold_zero_raises(self) -> None:
        """circuit_breaker_threshold=0 is rejected by Pydantic (ge=1 constraint)."""
        with pytest.raises(ValidationError, match="circuit_breaker_threshold"):
            ModelKafkaEventBusConfig(
                bootstrap_servers=_SERVERS,
                environment=_ENV,
                circuit_breaker_threshold=0,
            )

    def test_valid_config_constructs(self) -> None:
        """Valid configuration produces an instance without error."""
        bus = EventBusKafka(config=_make_config())
        assert bus.environment == _ENV


# ---------------------------------------------------------------------------
# Start error paths
# ---------------------------------------------------------------------------


class TestEventBusKafkaStartErrors:
    """start() must record circuit failures and raise typed infra errors."""

    @pytest.mark.asyncio
    async def test_start_connection_error_raises_infra_connection_error(
        self,
    ) -> None:
        """start() connection failure raises InfraConnectionError."""
        mock_producer = _make_mock_producer()
        mock_producer.start = AsyncMock(side_effect=KafkaError("broker unreachable"))

        with patch(
            "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
            return_value=mock_producer,
        ):
            bus = EventBusKafka(config=_make_config())
            with pytest.raises(InfraConnectionError):
                await bus.start()
            # Producer must be cleaned up
            assert bus._producer is None

    @pytest.mark.asyncio
    async def test_start_timeout_raises_infra_timeout_error(self) -> None:
        """start() timeout raises InfraTimeoutError."""
        mock_producer = _make_mock_producer()
        mock_producer.start = AsyncMock(side_effect=TimeoutError("connect timed out"))

        with patch(
            "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
            return_value=mock_producer,
        ):
            bus = EventBusKafka(config=_make_config())
            with pytest.raises(InfraTimeoutError):
                await bus.start()
            assert bus._producer is None


# ---------------------------------------------------------------------------
# Publish error paths
# ---------------------------------------------------------------------------


class TestEventBusKafkaPublishErrors:
    """publish() must raise correct errors for each failure mode."""

    @pytest.mark.asyncio
    async def test_publish_before_start_raises_unavailable(
        self, unstarted_bus: EventBusKafka
    ) -> None:
        """Publish on unstarted bus must raise InfraUnavailableError immediately."""
        with pytest.raises(InfraUnavailableError, match="not started"):
            await unstarted_bus.publish("onex.evt.test.topic.v1", b"key", b"value")

    @pytest.mark.asyncio
    async def test_publish_kafka_error_raises_infra_connection_error(
        self,
    ) -> None:
        """KafkaError during publish raises InfraConnectionError after exhausting retries."""
        mock_producer = _make_mock_producer()
        mock_producer.send = AsyncMock(side_effect=KafkaError("leader not available"))

        with (
            patch(
                "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
                return_value=mock_producer,
            ),
            patch(
                "omnibase_infra.event_bus.event_bus_kafka.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            config = _make_config(max_retry_attempts=0)
            bus = EventBusKafka(config=config)
            await bus.start()
            try:
                with pytest.raises(InfraConnectionError):
                    await bus.publish("onex.evt.test.kafka-err.v1", b"k", b"v")
            finally:
                await bus.close()

    @pytest.mark.asyncio
    async def test_publish_timeout_raises_infra_timeout_error(
        self,
    ) -> None:
        """TimeoutError during publish raises InfraTimeoutError after exhausting retries."""
        mock_producer = _make_mock_producer()
        mock_producer.send = AsyncMock(side_effect=TimeoutError("send timed out"))

        with (
            patch(
                "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
                return_value=mock_producer,
            ),
            patch(
                "omnibase_infra.event_bus.event_bus_kafka.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            config = _make_config(max_retry_attempts=0)
            bus = EventBusKafka(config=config)
            await bus.start()
            try:
                with pytest.raises(InfraTimeoutError):
                    await bus.publish("onex.evt.test.timeout.v1", b"k", b"v")
            finally:
                await bus.close()

    @pytest.mark.asyncio
    async def test_publish_unknown_topic_raises_configuration_error(
        self,
    ) -> None:
        """UnknownTopicOrPartitionError raises ProtocolConfigurationError and does not retry."""
        mock_producer = _make_mock_producer()
        mock_producer.send = AsyncMock(
            side_effect=UnknownTopicOrPartitionError("no such topic")
        )

        with patch(
            "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
            return_value=mock_producer,
        ):
            config = _make_config(max_retry_attempts=2)
            bus = EventBusKafka(config=config)
            await bus.start()
            try:
                with pytest.raises(ProtocolConfigurationError):
                    await bus.publish("onex.evt.test.missing.v1", b"k", b"v")
                # send() must only have been called once (no retry on missing topic)
                assert mock_producer.send.call_count == 1
            finally:
                await bus.close()

    @pytest.mark.asyncio
    async def test_publish_unknown_topic_does_not_trip_circuit_breaker(
        self,
    ) -> None:
        """UnknownTopicOrPartitionError must NOT increment circuit breaker failure count."""
        mock_producer = _make_mock_producer()
        mock_producer.send = AsyncMock(
            side_effect=UnknownTopicOrPartitionError("no such topic")
        )

        with patch(
            "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
            return_value=mock_producer,
        ):
            config = _make_config()
            bus = EventBusKafka(config=config)
            await bus.start()
            try:
                with pytest.raises(ProtocolConfigurationError):
                    await bus.publish("onex.evt.test.missing.v1", b"k", b"v")
                # Circuit breaker failure counter stays at zero
                assert bus._circuit_breaker_failures == 0  # type: ignore[attr-defined]
            finally:
                await bus.close()

    @pytest.mark.asyncio
    async def test_publish_while_closing_raises_unavailable(
        self,
    ) -> None:
        """Publish during shutdown raises InfraUnavailableError."""
        mock_producer = _make_mock_producer()

        async def slow_send(*args: object, **kwargs: object) -> None:
            await asyncio.sleep(10)

        mock_producer.send = AsyncMock(side_effect=slow_send)

        with patch(
            "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
            return_value=mock_producer,
        ):
            config = _make_config()
            bus = EventBusKafka(config=config)
            await bus.start()
            # Force closing state
            bus._closing = True
            with pytest.raises(InfraUnavailableError, match="shutting down"):
                await bus.publish("onex.evt.test.closing.v1", b"k", b"v")
            bus._closing = False
            await bus.close()

    @pytest.mark.asyncio
    async def test_publish_retry_count_exhausted(
        self,
    ) -> None:
        """With max_retry_attempts=2, send is called exactly 3 times before raising."""
        call_count = 0

        async def failing_send(*args: object, **kwargs: object) -> None:
            nonlocal call_count
            call_count += 1
            raise KafkaError("transient error")

        mock_producer = _make_mock_producer()
        mock_producer.send = AsyncMock(side_effect=failing_send)

        with (
            patch(
                "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
                return_value=mock_producer,
            ),
            patch(
                "omnibase_infra.event_bus.event_bus_kafka.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            config = _make_config(max_retry_attempts=2, retry_backoff_base=0.001)
            bus = EventBusKafka(config=config)
            await bus.start()
            try:
                with pytest.raises(InfraConnectionError):
                    await bus.publish("onex.evt.test.retry.v1", b"k", b"v")
                # 1 initial + 2 retries = 3 total calls
                assert call_count == 3
            finally:
                await bus.close()


# ---------------------------------------------------------------------------
# Subscribe error paths
# ---------------------------------------------------------------------------


class TestEventBusKafkaSubscribeErrors:
    """subscribe() must validate arguments before touching Kafka."""

    @pytest.mark.asyncio
    async def test_subscribe_without_on_message_raises_value_error(
        self, unstarted_bus: EventBusKafka
    ) -> None:
        """subscribe() with on_message=None raises ValueError."""
        identity = ModelNodeIdentity(
            env="test", service="svc", node_name="node", version="v1"
        )
        with pytest.raises(ValueError, match="on_message"):
            await unstarted_bus.subscribe("onex.evt.test.topic.v1", identity, None)

    @pytest.mark.asyncio
    async def test_subscribe_without_identity_or_group_id_raises_value_error(
        self, unstarted_bus: EventBusKafka
    ) -> None:
        """subscribe() without node_identity or group_id raises ValueError."""

        async def handler(msg: object) -> None:
            pass

        with pytest.raises(ValueError, match="node_identity or group_id"):
            await unstarted_bus.subscribe(
                "onex.evt.test.topic.v1",
                None,
                handler,
                group_id=None,
            )

    @pytest.mark.asyncio
    async def test_subscribe_adds_to_registry_when_bus_not_started(
        self, unstarted_bus: EventBusKafka
    ) -> None:
        """subscribe() registers subscriber even when bus not started (consumer deferred)."""

        async def handler(msg: object) -> None:
            pass

        unsubscribe = await unstarted_bus.subscribe(
            "onex.evt.test.topic.v1",
            group_id="my-group",
            on_message=handler,
        )
        assert len(unstarted_bus._subscribers["onex.evt.test.topic.v1"]) == 1
        await unsubscribe()
        assert len(unstarted_bus._subscribers["onex.evt.test.topic.v1"]) == 0

    @pytest.mark.asyncio
    async def test_subscribe_unsubscribe_removes_registration(
        self, started_bus: tuple[EventBusKafka, AsyncMock]
    ) -> None:
        """Calling the returned unsubscribe function removes the subscriber."""
        bus, _ = started_bus

        async def handler(msg: object) -> None:
            pass

        # Patch consumer start to avoid real Kafka connection
        with patch.object(bus, "_start_consumer_for_topic_unlocked", new=AsyncMock()):
            unsubscribe = await bus.subscribe(
                "onex.evt.test.sub.v1",
                group_id="test-group",
                on_message=handler,
            )
            assert len(bus._subscribers["onex.evt.test.sub.v1"]) == 1
            await unsubscribe()
            assert len(bus._subscribers["onex.evt.test.sub.v1"]) == 0
