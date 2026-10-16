# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Exercise the shared verdict through the public producer boundaries.

Only aiokafka clients are replaced. EventBusKafka runs its startup cleanup,
bounded retry and publish loop; the gateway runs KafkaTransport and its own
error translation. A permanent verdict must retain each caller's handling,
and a transient control must still spend the configured retry budget.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import UUID

import pytest
from aiokafka.errors import (
    AuthenticationFailedError,
    ClusterAuthorizationFailedError,
    KafkaConnectionError,
    TopicAuthorizationFailedError,
)

from omnibase_infra.enums.enum_kafka_permanent_failure import EnumKafkaPermanentFailure
from omnibase_infra.errors import (
    EventTopicAuthorizationError,
    InfraConnectionError,
    InfraUnavailableError,
)
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.kafka_transport import KafkaTransport
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.handlers.kafka_permanent_failure import (
    classify_permanent_kafka_failure,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.models import (
    ModelGatewayTenantIdentity,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_gateway_forwarder import (
    GatewayEgressDeniedError,
)
from omnibase_infra.runtime.gateway_forwarder import TransportGatewayBus

pytestmark = pytest.mark.integration

_TOPIC = "onex.evt.omnibase-infra.inference-response.v1"


def _config() -> ModelKafkaEventBusConfig:
    return ModelKafkaEventBusConfig(
        bootstrap_servers="localhost:19092",
        environment="test",
        timeout_seconds=1,
        max_retry_attempts=2,
        retry_backoff_base=0.001,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("permanent", [True, False], ids=["wrapped-auth", "transient"])
async def test_start_preserves_the_verdict_through_cleanup_and_translation(
    permanent: bool,
) -> None:
    cause = AuthenticationFailedError() if permanent else KafkaConnectionError()
    failure = RuntimeError("client wrapper")
    failure.__cause__ = cause
    producer = AsyncMock()
    producer.start.side_effect = failure

    with patch(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
        return_value=producer,
    ) as factory:
        bus = EventBusKafka(config=_config())
        try:
            with pytest.raises(InfraConnectionError) as raised:
                await asyncio.wait_for(bus.start(), timeout=2)
            expected_attempts = 1 if permanent else _config().max_retry_attempts + 1
            assert factory.call_count == expected_attempts
            assert producer.start.await_count == expected_attempts
            assert producer.stop.await_count == expected_attempts
            assert raised.value.__cause__ is failure
            expected = (
                EnumKafkaPermanentFailure.CONNECT_AUTHENTICATION_REJECTED
                if permanent
                else None
            )
            assert classify_permanent_kafka_failure(raised.value) is expected
        finally:
            await bus.close()


@pytest.mark.asyncio
async def test_publish_refusal_keeps_its_type_and_the_next_publish_works() -> None:
    refusal = TopicAuthorizationFailedError(_TOPIC)
    producer = AsyncMock()
    producer._closed = False
    accepted: asyncio.Future[SimpleNamespace] = (
        asyncio.get_running_loop().create_future()
    )
    accepted.set_result(SimpleNamespace(topic=_TOPIC, partition=0, offset=19))
    producer.send.side_effect = [refusal, accepted]

    with patch(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
        return_value=producer,
    ):
        bus = EventBusKafka(config=_config())
        await bus.start()
        try:
            with pytest.raises(EventTopicAuthorizationError) as raised:
                await asyncio.wait_for(bus.publish(_TOPIC, b"key", b"denied"), 2)
            assert raised.value.topic == _TOPIC
            assert (
                classify_permanent_kafka_failure(raised.value)
                is EnumKafkaPermanentFailure.TOPIC_AUTHORIZATION_REFUSED
            )
            assert producer.send.await_count == 1
            await bus.publish(_TOPIC, b"key", b"accepted")
            assert producer.send.await_count == 2
        finally:
            await bus.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error_type", "gateway_error", "kind"),
    [
        (
            TopicAuthorizationFailedError,
            GatewayEgressDeniedError,
            EnumKafkaPermanentFailure.TOPIC_AUTHORIZATION_REFUSED,
        ),
        (
            ClusterAuthorizationFailedError,
            GatewayEgressDeniedError,
            EnumKafkaPermanentFailure.CLUSTER_AUTHORIZATION_REFUSED,
        ),
        (
            AuthenticationFailedError,
            InfraUnavailableError,
            EnumKafkaPermanentFailure.CONNECT_AUTHENTICATION_REJECTED,
        ),
        (KafkaConnectionError, InfraUnavailableError, None),
    ],
)
async def test_gateway_keeps_its_own_handling_and_preserves_delivery_coordinates(
    error_type: type[Exception],
    gateway_error: type[Exception],
    kind: EnumKafkaPermanentFailure | None,
) -> None:
    refusal = error_type()
    producer = AsyncMock()
    producer.send_and_wait.side_effect = [
        refusal,
        SimpleNamespace(topic=_TOPIC, partition=2, offset=19092),
    ]
    identity = ModelGatewayTenantIdentity(
        tenant_id=UUID("19092000-0000-4000-8000-000000019092"),
        tenant_slug="shared-verdict",
        principal_id="t-19092000000040008000000000019092",
    )

    with patch(
        "omnibase_infra.event_bus.kafka_transport.AIOKafkaProducer",
        return_value=producer,
    ):
        transport = KafkaTransport(config=_config())
        await transport.start()
        gateway = TransportGatewayBus(transport, identity=identity)
        try:
            with pytest.raises(gateway_error) as raised:
                await gateway.publish(_TOPIC, b"key", b"denied")
            assert raised.value.__cause__ is refusal
            assert classify_permanent_kafka_failure(raised.value) is kind
            assert producer.send_and_wait.await_count == 1
            receipt = await gateway.publish(_TOPIC, b"key", b"accepted")
            assert receipt is not None
            assert (receipt.topic, receipt.partition, receipt.offset) == (
                _TOPIC,
                2,
                19092,
            )
            assert producer.send_and_wait.await_count == 2
        finally:
            await transport.close()
