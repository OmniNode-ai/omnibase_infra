# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""MSK IAM auth coverage for direct aiokafka call sites."""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_core.models.contracts.subcontracts import (
    ModelReplyTopics,
    ModelRequestResponseConfig,
    ModelRequestResponseInstance,
)
from omnibase_infra.event_bus.kafka_auth import build_aiokafka_auth_kwargs
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.nodes.node_dlq_replay_effect import engine_dlq_replay
from omnibase_infra.nodes.node_dlq_replay_effect.engine_dlq_replay import (
    DLQConsumer,
    DLQProducer,
    DLQQuarantineProducer,
    ModelDlqReplayEngineConfig,
)
from omnibase_infra.runtime.models.model_kafka_producer_config import (
    ModelKafkaProducerConfig,
)
from omnibase_infra.runtime.providers.provider_kafka_producer import (
    ProviderKafkaProducer,
)
from omnibase_infra.runtime.request_response_wiring import RequestResponseWiring

pytestmark = pytest.mark.unit


class _FakeAiokafkaClient:
    created: ClassVar[list[_FakeAiokafkaClient]] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.args = args
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.__class__.created.append(self)

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    def __aiter__(self) -> _FakeAiokafkaClient:
        return self

    async def __anext__(self) -> object:
        await asyncio.sleep(3600)
        raise StopAsyncIteration


@pytest.fixture(autouse=True)
def _reset_fake_clients() -> None:
    _FakeAiokafkaClient.created = []


def test_ssl_ca_file_builds_aiokafka_ssl_context() -> None:
    with patch("ssl.create_default_context") as create_context:
        context = object()
        create_context.return_value = context
        config = ModelKafkaEventBusConfig(
            bootstrap_servers="b-1.example:9098",
            security_protocol="SSL",
            ssl_ca_file="/etc/ssl/certs/custom-ca.pem",
        )

        kwargs = build_aiokafka_auth_kwargs(config)

    assert kwargs["ssl_context"] is context
    assert "ssl_cafile" not in kwargs
    create_context.assert_called_once_with(cafile="/etc/ssl/certs/custom-ca.pem")


@pytest.mark.asyncio
async def test_provider_kafka_producer_passes_msk_auth_kwargs() -> None:
    auth_kwargs = {"security_protocol": "SASL_SSL", "sasl_mechanism": "OAUTHBEARER"}
    config = ModelKafkaProducerConfig(max_request_size=8_000_000)

    with (
        patch(
            "omnibase_infra.runtime.providers.provider_kafka_producer.build_aiokafka_auth_kwargs_from_env",
            return_value=auth_kwargs,
        ),
        patch("aiokafka.AIOKafkaProducer") as producer_cls,
    ):
        producer = MagicMock()
        producer.start = AsyncMock()
        producer_cls.return_value = producer
        await ProviderKafkaProducer(config).create()

    call_kwargs = producer_cls.call_args.kwargs
    assert call_kwargs["max_request_size"] == 8_000_000
    assert call_kwargs["security_protocol"] == "SASL_SSL"
    assert call_kwargs["sasl_mechanism"] == "OAUTHBEARER"


@pytest.mark.asyncio
async def test_request_response_wiring_passes_msk_auth_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import omnibase_infra.runtime.request_response_wiring as module

    monkeypatch.setattr(module, "AIOKafkaConsumer", _FakeAiokafkaClient)
    monkeypatch.setattr(
        module,
        "build_aiokafka_auth_kwargs_from_env",
        lambda: {"security_protocol": "SASL_SSL"},
    )
    event_bus = MagicMock()
    event_bus.publish = AsyncMock()
    config = ModelRequestResponseConfig(
        instances=[
            ModelRequestResponseInstance(
                name="rpc",
                request_topic="onex.cmd.rpc.request.v1",
                reply_topics=ModelReplyTopics(
                    completed="onex.evt.rpc.completed.v1",
                    failed="onex.evt.rpc.failed.v1",
                ),
            )
        ]
    )
    wiring = RequestResponseWiring(
        event_bus=event_bus,
        environment="test",
        app_name="app",
        bootstrap_servers="localhost:9092",
    )

    await wiring.wire_request_response(config)
    await wiring.cleanup()

    assert _FakeAiokafkaClient.created[0].kwargs["security_protocol"] == "SASL_SSL"


@pytest.mark.asyncio
async def test_dlq_replay_clients_pass_msk_auth_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_kwargs: dict[str, Any] = {"security_protocol": "SASL_SSL"}
    monkeypatch.setattr(engine_dlq_replay, "AIOKafkaConsumer", _FakeAiokafkaClient)
    monkeypatch.setattr(engine_dlq_replay, "AIOKafkaProducer", _FakeAiokafkaClient)
    monkeypatch.setattr(
        engine_dlq_replay,
        "build_aiokafka_auth_kwargs_from_env",
        lambda: auth_kwargs,
    )
    config = ModelDlqReplayEngineConfig(
        bootstrap_servers="localhost:9092",
        dlq_topic="onex.dlq.omnibase-infra.events.v1",
    )

    consumer = DLQConsumer(config)
    producer = DLQProducer(config)
    quarantine = DLQQuarantineProducer(config)
    await consumer.start()
    await producer.start()
    await quarantine.start()
    await consumer.stop()
    await producer.stop()
    await quarantine.stop()

    assert [
        client.kwargs["security_protocol"] for client in _FakeAiokafkaClient.created
    ] == [
        "SASL_SSL",
        "SASL_SSL",
        "SASL_SSL",
    ]
