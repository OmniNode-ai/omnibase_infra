# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Both producer surfaces obey one connect-retry dial, measured in attempts (OMN-19043).

``test_kafka_connect_retry_omn18925.py`` shows that the policy model resolves
the same values for both surfaces. This module shows the same thing through
behaviour. One ``KAFKA_MAX_RETRY_ATTEMPTS`` value is set, and the DI producer
surface (``ProviderKafkaProducer``) and the event-bus surface
(``EventBusKafka``) each make exactly that many retries against a broker that
never answers. A surface with its own default would make a different number
of attempts, which is the drift the ticket's AC3 falsifier names.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, call, patch

import pytest

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import InfraTimeoutError
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config.model_kafka_event_bus_config import (
    ModelKafkaEventBusConfig,
)
from omnibase_infra.runtime.models.model_kafka_producer_config import (
    ModelKafkaProducerConfig,
)
from omnibase_infra.runtime.providers.provider_kafka_producer import (
    ProviderKafkaProducer,
)


class _NeverAnswers:
    """A producer whose every ``start()`` outruns the per-attempt deadline."""

    def __init__(self) -> None:
        self.calls = 0
        self.stop = AsyncMock()

    async def start(self) -> None:
        self.calls += 1
        await asyncio.sleep(30.0)


@pytest.fixture(autouse=True)
def _one_declared_dial(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAFKA_MAX_RETRY_ATTEMPTS", "2")
    monkeypatch.setenv("KAFKA_RETRY_BACKOFF_BASE", "0.001")


@pytest.mark.asyncio
async def test_provider_surface_makes_the_declared_number_of_attempts() -> None:
    producer = _NeverAnswers()

    with patch("aiokafka.AIOKafkaProducer", return_value=producer):
        with patch(
            "omnibase_infra.event_bus.kafka_connect_retry.asyncio.sleep",
            wraps=asyncio.sleep,
        ) as sleep:
            with pytest.raises(InfraTimeoutError, match=r"1\.0s") as raised:
                await ProviderKafkaProducer(
                    ModelKafkaProducerConfig(
                        bootstrap_servers="silent-broker:19092",
                        timeout_seconds=1.0,
                    )
                ).create()

    assert producer.calls == 3  # 1 + the declared 2 retries
    # The long sleeps belong to the stalled producer and are cancelled by
    # each deadline. The two intervening sleeps must be awaited in order.
    assert sleep.await_args_list == [
        call(30.0),
        call(0.001),
        call(30.0),
        call(0.002),
        call(30.0),
    ]
    assert raised.value.model.context["timeout_seconds"] == 1.0
    assert raised.value.model.context["connect_attempts"] == 3
    assert raised.value.model.context["connect_bound_seconds"] == pytest.approx(3.003)
    assert raised.value.model.context["transport_type"] == EnumInfraTransportType.KAFKA
    assert isinstance(raised.value.__cause__, TimeoutError)
    assert producer.stop.await_count == 3


@pytest.mark.asyncio
async def test_event_bus_surface_makes_the_declared_number_of_attempts() -> None:
    producer = _NeverAnswers()
    bus = EventBusKafka(
        config=ModelKafkaEventBusConfig.default().model_copy(
            update={"bootstrap_servers": "silent-broker:19092", "timeout_seconds": 1}
        )
    )

    with patch(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
        return_value=producer,
    ):
        with patch(
            "omnibase_infra.event_bus.kafka_connect_retry.asyncio.sleep",
            wraps=asyncio.sleep,
        ) as sleep:
            with pytest.raises(InfraTimeoutError, match="1s") as raised:
                await bus.start()

    assert producer.calls == 3  # the same dial, the same count
    assert sleep.await_args_list == [
        call(30.0),
        call(0.001),
        call(30.0),
        call(0.002),
        call(30.0),
    ]
    assert raised.value.model.context["timeout_seconds"] == 1
    assert raised.value.model.context["transport_type"] == EnumInfraTransportType.KAFKA
    assert isinstance(raised.value.__cause__, TimeoutError)
    assert producer.stop.await_count == 3
