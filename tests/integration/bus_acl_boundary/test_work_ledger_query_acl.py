# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20740: the real SCRAM broker enforces the derived query ACL.

Only a throwaway broker is mutated. The six shipped query grants are applied
to non-superuser developer/lane identities; an authenticated outsider has none.
Topics are renamed into the test namespace to honor the ledger test-write guard.
The resource patterns, operations and actual client group prefix are unchanged.
This tests broker authorization and client round trips, not database parity.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.errors import TopicAuthorizationFailedError
from pydantic import BaseModel, SecretStr

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.lane_client_transport_binding import (
    bind_lane_client_transport,
)
from omnibase_infra.event_bus.model_lane_client_transport import (
    ModelLaneClientTransport,
)
from omnibase_infra.handlers.handler_work_ledger_query import HandlerWorkLedgerQuery
from omnibase_infra.models.model_work_ledger_query_contract import (
    ModelWorkLedgerQueryContract,
)
from tests.integration.customer_path import redpanda_sasl_harness as harness

pytestmark = [
    pytest.mark.integration,
    pytest.mark.kafka,
    pytest.mark.slow,
    pytest.mark.timeout(600),
]
ROOT = Path(__file__).resolve().parents[3]
REQUEST = "onex.cmd.omnimarket.work-ledger-query-requested.v1"
COMPLETED = "onex.evt.omnimarket.work-ledger-query-completed.v1"
PREFIX = "local.omnimarket.work_ledger_query_client.consume.v1."
TOPICS = {
    REQUEST: "test.cmd.work-ledger-query.v1",
    COMPLETED: "test.evt.work-ledger-query.v1",
}


class Query(BaseModel):
    correlation_id: UUID
    query: str


class Result(BaseModel):
    correlation_id: UUID
    query: str
    rows: tuple[str, ...]
    error: str | None = None


CONTRACT = ModelWorkLedgerQueryContract(
    command_topic=TOPICS[REQUEST],
    completed_topic=TOPICS[COMPLETED],
    request_model=Query,
    result_model=Result,
)


def query_grants() -> list[dict[str, str]]:
    grants = json.loads(
        (ROOT / "deploy/lab/developer-principal-derived-grants.json").read_text()
    )["grants"]
    selected = [g for g in grants if g["resource"] in (*TOPICS, PREFIX)]
    assert len(selected) == 6
    assert all(g["broker"] == "dev" for g in selected)
    return selected


@pytest.fixture(scope="module")
def broker() -> Iterator[harness.RedpandaSasl]:
    # Never resolve/adopt a declared broker: all mutation is confined to our own.
    owned = harness.start_redpanda_sasl()
    try:
        for topic in TOPICS.values():
            owned.create_topic(topic)
        for name in ("dev-ledger-reader", "lane-ledger-reader", "outsider"):
            principal = harness.RuntimePrincipal(name, "synthetic-ledger-test-password")
            owned.create_principal(principal)
            if name == "outsider":
                continue
            for grant in query_grants():
                resource = TOPICS.get(grant["resource"], grant["resource"])
                owned.rpk(
                    "security",
                    "acl",
                    "create",
                    "--allow-principal",
                    f"User:{name}",
                    "--operation",
                    grant["operation"].lower(),
                    "--topic" if grant["resource_type"] == "TOPIC" else "--group",
                    resource,
                    "--resource-pattern-type",
                    grant["pattern"].lower(),
                )
        yield owned
    finally:
        harness.stop_redpanda(owned)


def transport(broker: harness.RedpandaSasl, principal: str) -> ModelLaneClientTransport:
    return ModelLaneClientTransport(
        lane="dev",
        bootstrap_servers=broker.bootstrap,
        security_protocol="SASL_PLAINTEXT",
        sasl_mechanism=broker.mechanism,
        sasl_username=principal,
        sasl_password=SecretStr(
            broker.password
            if principal == broker.username
            else "synthetic-ledger-test-password"
        ),
        declared_in=Path(__file__),
    )


@pytest.mark.parametrize("principal", ["dev-ledger-reader", "lane-ledger-reader"])
async def test_developer_and_lane_can_query(
    broker: harness.RedpandaSasl, principal: str
) -> None:
    with bind_lane_client_transport(transport(broker, broker.username)):
        server = EventBusKafka.from_bootstrap(broker.bootstrap)
        await server.start()

        async def reply(message):
            raw = json.loads(message.value)
            result = Result(
                correlation_id=raw["correlation_id"],
                query=raw["payload"]["query"],
                rows=("2026-10-25T00:00:00Z | STATUS | lane=test",),
            )
            envelope = ModelEventEnvelope[dict[str, object]](
                payload=result.model_dump(mode="json"),
                correlation_id=result.correlation_id,
                parent_envelope_id=UUID(raw["envelope_id"]),
                event_type="test.ledger-query-completed.v1",
            )
            await server.publish(
                CONTRACT.completed_topic, None, envelope.model_dump_json().encode()
            )

        await server.subscribe(
            CONTRACT.command_topic,
            on_message=reply,
            group_id=f"test-query-node.{principal}",
            auto_offset_reset="latest",
        )
    try:
        with bind_lane_client_transport(transport(broker, principal)):
            bus = EventBusKafka.from_bootstrap(broker.bootstrap)
            try:
                await bus.start()
                async with HandlerWorkLedgerQuery(bus, contract=CONTRACT) as client:
                    answer = await client.query({"query": "rows"})
                    assert answer.rows == ("2026-10-25T00:00:00Z | STATUS | lane=test",)
                    assert client.group_id.startswith(PREFIX)
            finally:
                await bus.close()
    finally:
        await server.close()


async def test_other_authenticated_principal_cannot_publish_query(
    broker: harness.RedpandaSasl,
) -> None:
    producer = AIOKafkaProducer(
        bootstrap_servers=broker.bootstrap,
        security_protocol="SASL_PLAINTEXT",
        sasl_mechanism=broker.mechanism,
        sasl_plain_username="outsider",
        sasl_plain_password="synthetic-ledger-test-password",
        enable_idempotence=False,
    )
    try:
        await producer.start()
        with pytest.raises(TopicAuthorizationFailedError):
            await producer.send_and_wait(CONTRACT.command_topic, b"{}")
    finally:
        await producer.stop()


async def test_other_authenticated_principal_cannot_read_replies(
    broker: harness.RedpandaSasl,
) -> None:
    consumer = AIOKafkaConsumer(
        CONTRACT.completed_topic,
        bootstrap_servers=broker.bootstrap,
        security_protocol="SASL_PLAINTEXT",
        sasl_mechanism=broker.mechanism,
        sasl_plain_username="outsider",
        sasl_plain_password="synthetic-ledger-test-password",
        group_id=PREFIX + "outsider",
    )
    try:
        with pytest.raises(TopicAuthorizationFailedError):
            await consumer.start()
            await consumer.getmany(timeout_ms=1000)
    finally:
        await consumer.stop()


async def test_other_authenticated_principal_cannot_open_client(
    broker: harness.RedpandaSasl,
) -> None:
    from omnibase_infra.errors import InfraConnectionError

    with bind_lane_client_transport(transport(broker, "outsider")):
        bus = EventBusKafka.from_bootstrap(broker.bootstrap)
        try:
            with pytest.raises(
                InfraConnectionError, match=r"[Aa]uthoriz|[Pp]ermission|consumer"
            ):
                await bus.start()
                async with HandlerWorkLedgerQuery(bus, contract=CONTRACT):
                    pytest.fail("outsider acquired the query reply subscription")
        finally:
            await bus.close()
