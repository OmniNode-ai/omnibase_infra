# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for transient metadata retry in consumer start.

Verifies that EventBusKafka._start_consumer_for_topic retries when the broker
returns transient metadata errors, which Redpanda --smp 1 --mode dev-container
can raise while topic metadata converges.

Without the fix, InvalidPartitionsError fell to the generic except Exception
block and raised InfraConnectionError immediately. With the fix, it enters the
same exponential-backoff retry loop as UnknownTopicOrPartitionError.

Pairs with tests/unit/event_bus/ unit tests and the runtime boot smoke CI job
wired in OMN-9120.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from aiokafka.errors import BrokerNotAvailableError, InvalidPartitionsError

from omnibase_infra.errors import InfraTimeoutError
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

pytestmark = pytest.mark.integration


@pytest.fixture
def kafka_config() -> ModelKafkaEventBusConfig:
    return ModelKafkaEventBusConfig(
        bootstrap_servers="localhost:19092",
        timeout_seconds=10,
    )


@pytest.mark.asyncio
async def test_invalid_partitions_error_is_retried_and_recovers(
    kafka_config: ModelKafkaEventBusConfig,
) -> None:
    """Consumer start recovers after InvalidPartitionsError on first attempt.

    Simulates the Redpanda --smp 1 scenario: broker returns InvalidPartitionsError
    on the first consumer.start() call (metadata not yet stable), then succeeds
    on the second attempt. The subscription must complete successfully.
    """
    bus = EventBusKafka(config=kafka_config)

    producer_mock = MagicMock()
    producer_mock.start = AsyncMock()
    producer_mock.stop = AsyncMock()

    with patch(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
        return_value=producer_mock,
    ):
        await bus.start()

    # First consumer.start() raises InvalidPartitionsError; second call succeeds.
    consumer_first = MagicMock()
    consumer_first.start = AsyncMock(side_effect=InvalidPartitionsError())
    consumer_first.stop = AsyncMock()

    consumer_second = MagicMock()
    consumer_second.start = AsyncMock()
    consumer_second.stop = AsyncMock()

    call_count = 0

    def make_consumer(*args: object, **kwargs: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return consumer_first
        return consumer_second

    with (
        patch(
            "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaConsumer",
            side_effect=make_consumer,
        ),
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        await bus.subscribe(
            "onex.evt.integration.invalid-partitions-retry.v1",
            on_message=AsyncMock(),
            group_id="integration-invalid-partitions-retry-group",
        )

    # Both consumers were constructed (one retry)
    assert call_count == 2
    # Failed consumer was stopped for cleanup
    consumer_first.stop.assert_awaited_once()
    # Successful consumer was started
    consumer_second.start.assert_awaited_once()


@pytest.mark.asyncio
async def test_broker_not_available_error_is_retried_and_recovers(
    kafka_config: ModelKafkaEventBusConfig,
) -> None:
    """Consumer start recovers after BrokerNotAvailableError on first attempt.

    Redpanda can return BrokerNotAvailableError while an auto-created topic is
    visible to the broker but metadata is not yet stable enough for aiokafka's
    consumer startup path.
    """
    bus = EventBusKafka(config=kafka_config)

    producer_mock = MagicMock()
    producer_mock.start = AsyncMock()
    producer_mock.stop = AsyncMock()

    with patch(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
        return_value=producer_mock,
    ):
        await bus.start()

    consumer_first = MagicMock()
    consumer_first.start = AsyncMock(side_effect=BrokerNotAvailableError())
    consumer_first.stop = AsyncMock()

    consumer_second = MagicMock()
    consumer_second.start = AsyncMock()
    consumer_second.stop = AsyncMock()

    call_count = 0

    def make_consumer(*args: object, **kwargs: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return consumer_first
        return consumer_second

    with (
        patch(
            "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaConsumer",
            side_effect=make_consumer,
        ),
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        await bus.subscribe(
            "onex.evt.integration.broker-not-available-retry.v1",
            on_message=AsyncMock(),
            group_id="integration-broker-not-available-retry-group",
        )

    assert call_count == 2
    consumer_first.stop.assert_awaited_once()
    consumer_second.start.assert_awaited_once()


@pytest.mark.asyncio
async def test_invalid_partitions_error_exhausts_budget_raises_infra_timeout(
    kafka_config: ModelKafkaEventBusConfig,
) -> None:
    """After the retry budget is exhausted, InfraTimeoutError is raised.

    Simulates a permanently broken Redpanda partition metadata state: every
    consumer.start() call raises InvalidPartitionsError. Once the timeout
    budget is consumed, the caller must receive InfraTimeoutError (not
    InfraConnectionError, which would indicate a network failure).
    """
    bus = EventBusKafka(config=kafka_config)

    producer_mock = MagicMock()
    producer_mock.start = AsyncMock()
    producer_mock.stop = AsyncMock()

    with patch(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
        return_value=producer_mock,
    ):
        await bus.start()

    consumer_mock = MagicMock()
    consumer_mock.start = AsyncMock(side_effect=InvalidPartitionsError())
    consumer_mock.stop = AsyncMock()

    # Exhaust the deadline immediately by making get_event_loop().time() always
    # return a value past the deadline on the first retry check.
    original_time = asyncio.get_event_loop().time

    call_count = 0

    def time_side_effect() -> float:
        nonlocal call_count
        call_count += 1
        # First call: returns deadline-setting value (loop.time() in deadline calc)
        # Second+ calls: return deadline+1 so remaining <= backoff immediately
        if call_count <= 1:
            return original_time()
        return original_time() + 1000.0

    with (
        patch(
            "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaConsumer",
            return_value=consumer_mock,
        ),
        patch(
            "omnibase_infra.event_bus.event_bus_kafka.asyncio.get_event_loop"
        ) as mock_loop,
    ):
        mock_loop_instance = MagicMock()
        mock_loop_instance.time = time_side_effect
        mock_loop.return_value = mock_loop_instance

        with pytest.raises(InfraTimeoutError):
            await bus.subscribe(
                "onex.evt.integration.invalid-partitions-timeout.v1",
                on_message=AsyncMock(),
                group_id="integration-invalid-partitions-timeout-group",
            )
