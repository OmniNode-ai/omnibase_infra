# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Real Kafka clients time out against a TCP peer that never replies (OMN-19043).

The peer accepts and reads Kafka requests but deliberately sends no response.
Neither producer construction nor the retry helper is mocked: deadlines must
cancel real network I/O, and cleanup must close every accepted connection.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import InfraTimeoutError
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config.model_kafka_connect_retry_policy import (
    ModelKafkaConnectRetryPolicy,
)
from omnibase_infra.event_bus.models.config.model_kafka_event_bus_config import (
    ModelKafkaEventBusConfig,
)
from omnibase_infra.runtime.models.model_kafka_producer_config import (
    ModelKafkaProducerConfig,
)
from omnibase_infra.runtime.providers.provider_kafka_producer import (
    ProviderKafkaProducer,
)

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["provider", "event_bus"])
@pytest.mark.parametrize("retries", [0, 2])
async def test_silent_peer_exhausts_declared_retries_with_typed_timeout(
    monkeypatch: pytest.MonkeyPatch, surface: str, retries: int
) -> None:
    monkeypatch.setenv("KAFKA_MAX_RETRY_ATTEMPTS", str(retries))
    monkeypatch.setenv("KAFKA_RETRY_BACKOFF_BASE", "0.01")
    monkeypatch.setenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT")

    requests: list[bytes] = []
    connections: list[asyncio.Task[None]] = []
    writers: list[asyncio.StreamWriter] = []

    async def hold_response(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            size = int.from_bytes(await reader.readexactly(4), "big")
            assert 0 < size < 1_048_588
            requests.append(await reader.readexactly(size))
            # EOF must come from the client's failed-attempt cleanup, before
            # the test closes its own sockets in the finally block below.
            assert await reader.read() == b""
        finally:
            writer.close()
            await writer.wait_closed()

    def accept(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writers.append(writer)
        connections.append(asyncio.create_task(hold_response(reader, writer)))

    server = await asyncio.start_server(accept, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    bootstrap_servers = f"127.0.0.1:{port}"
    config = ModelKafkaEventBusConfig.default().model_copy(
        update={"bootstrap_servers": bootstrap_servers, "timeout_seconds": 1}
    )
    policy = ModelKafkaConnectRetryPolicy.from_bus_config(
        config, attempt_timeout_seconds=1.0
    )
    bus = EventBusKafka(config=config)
    provider = ProviderKafkaProducer(
        ModelKafkaProducerConfig(
            bootstrap_servers=bootstrap_servers, timeout_seconds=1.0
        )
    )
    try:
        started = time.monotonic()
        with pytest.raises(InfraTimeoutError, match=r"1(?:\.0)?s") as raised:
            async with asyncio.timeout(policy.total_bound_seconds + 5.0):
                if surface == "provider":
                    await provider.create()
                else:
                    await bus.start()
        elapsed = time.monotonic() - started

        # Server-side observations verify attempts and cleanup independently
        # of the error metadata reported by the client.
        assert len(connections) == retries + 1
        await asyncio.wait_for(asyncio.gather(*connections), timeout=2.0)
        assert len(requests) == retries + 1
        assert all(requests)
        assert policy.total_bound_seconds <= elapsed < policy.total_bound_seconds + 5.0
        assert isinstance(raised.value.__cause__, TimeoutError)
        context = raised.value.model.context
        assert context["transport_type"] == EnumInfraTransportType.KAFKA
        assert context["operation"] == "start"
        assert context["timeout_seconds"] == 1.0
        if surface == "provider":
            assert context["connect_attempts"] == retries + 1
            assert context["connect_bound_seconds"] == pytest.approx(
                policy.total_bound_seconds
            )
    finally:
        server.close()
        await server.wait_closed()
        await bus.close()
        for writer in writers:
            writer.close()
        if connections:
            await asyncio.gather(*connections, return_exceptions=True)
