# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Work-ledger queries as a granted developer and an unrelated SCRAM principal.

Only the six dev query grants derived from the shipped bindings are installed.
The responder uses the read node's wire shape; its domain replay is owned and
tested by omnimarket. This exercises real Kafka authorization and the actual
infra client, without a database, private checkout, or hosted-runner secret.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
from uuid import uuid4

import pytest
import yaml
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.errors import GroupAuthorizationFailedError, TopicAuthorizationFailedError

from omnibase_infra.event_bus.kafka_auth import build_aiokafka_auth_kwargs
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.handlers.handler_broker_grant_derive import HandlerBrokerGrantDerive
from omnibase_infra.handlers.handler_work_ledger_query import (
    HandlerWorkLedgerQuery,
    load_work_ledger_query_client_config,
)
from omnibase_infra.nodes.node_broker_grant_derive_compute.models.model_broker_grant_bindings import (
    ModelBrokerGrantBindings,
)

from .redpanda_sasl_harness import RedpandaSasl, RuntimePrincipal

pytestmark = [
    pytest.mark.integration,
    pytest.mark.kafka,
    pytest.mark.slow,
    pytest.mark.timeout(600),
    pytest.mark.xdist_group("omn18012_redpanda_sasl"),
]
ROOT = Path(__file__).parents[3]


def _transport(
    broker: RedpandaSasl, principal: RuntimePrincipal
) -> ModelKafkaEventBusConfig:
    return ModelKafkaEventBusConfig(
        bootstrap_servers=broker.bootstrap,
        security_protocol="SASL_PLAINTEXT",
        sasl_mechanism=broker.mechanism,
        sasl_plain_username=principal.username,
        sasl_plain_password=principal.password,
    )


def _query_grants():
    contract = load_work_ledger_query_client_config()
    bindings = ModelBrokerGrantBindings.model_validate(
        yaml.safe_load(
            (ROOT / "deploy/lab/developer-principal-bindings.yaml").read_text()
        )
    )
    return tuple(
        g
        for g in HandlerBrokerGrantDerive().handle(bindings).grants
        if g.resource
        in {
            contract.command_topic,
            contract.completed_topic,
            contract.consumer_group_prefix,
        }
    )


def _grant_query(broker: RedpandaSasl, principal: RuntimePrincipal) -> None:
    grants = _query_grants()
    assert len(grants) == 6
    assert {g.broker for g in grants} == {"dev"}
    for grant in grants:
        broker.rpk(
            "security",
            "acl",
            "create",
            "--allow-principal",
            f"User:{principal.username}",
            "--operation",
            grant.operation.lower(),
            f"--{grant.resource_type.lower()}",
            grant.resource,
            "--resource-pattern-type",
            grant.pattern.lower(),
        )


async def _reply_once(broker: RedpandaSasl, ready: asyncio.Event) -> None:
    contract = load_work_ledger_query_client_config()
    auth = build_aiokafka_auth_kwargs(
        _transport(broker, RuntimePrincipal(broker.username, broker.password))
    )
    consumer = AIOKafkaConsumer(
        contract.command_topic,
        bootstrap_servers=broker.bootstrap,
        group_id=f"omn20740-node-{uuid4().hex}",
        auto_offset_reset="latest",
        **auth,
    )
    producer = AIOKafkaProducer(bootstrap_servers=broker.bootstrap, **auth)
    try:
        await consumer.start()
        for partition in consumer.assignment():
            await consumer.position(partition)
        await producer.start()
        ready.set()
        message = await consumer.getone()
        request = json.loads(message.value)
        cid = request["correlation_id"]
        answer = {
            "correlation_id": cid,
            "query": request["payload"]["query"],
            "window_since": None,
            "window_until": "2026-10-25T00:01:00Z",
            "matched": 1,
            "rows": [{"raw_row": "2026-10-25T00:00:00Z | STATUS | lane=fixture |"}],
            "freshness": {
                "newest_row_ts": "2026-10-25T00:00:00Z",
                "newest_projected_at": "2026-10-25T00:00:01Z",
            },
            "parity": {"status": "unmeasured", "days": []},
            "error": None,
        }
        await producer.send_and_wait(
            contract.completed_topic,
            json.dumps(
                {
                    "correlation_id": cid,
                    "parent_envelope_id": request["envelope_id"],
                    "payload": answer,
                }
            ).encode(),
        )
    finally:
        await producer.stop()
        await consumer.stop()


def test_developer_can_query_but_an_unrelated_principal_is_refused(
    redpanda_sasl: RedpandaSasl,
) -> None:
    broker = redpanda_sasl
    # No test-created identities or grants on an adopted shared lane. The
    # hosted job supplies no declaration and uses the throwaway harness.
    assert broker.owned, (
        "ACL regression requires the throwaway broker, never a shared lane"
    )
    contract = load_work_ledger_query_client_config()
    for topic in (contract.command_topic, contract.completed_topic):
        broker.create_topic(topic)
    developer = RuntimePrincipal(
        f"dev-omn20740-{uuid4().hex[:8]}", "omn20740-synthetic-developer"
    )
    unrelated = RuntimePrincipal(
        f"unrelated-omn20740-{uuid4().hex[:8]}", "omn20740-synthetic-unrelated"
    )
    for principal in (developer, unrelated):
        broker.create_principal(principal)
        assert principal.username not in broker.superusers()

    async def before_grant() -> None:
        with pytest.raises(
            (TopicAuthorizationFailedError, GroupAuthorizationFailedError)
        ):
            await HandlerWorkLedgerQuery(_transport(broker, developer)).query(
                {"query": "rows"}, timeout_s=30
            )

    # Failing-first on the same authenticated principal: absent grant is a
    # broker refusal; installing exactly the derived dev grants enables it.
    asyncio.run(before_grant())
    _grant_query(broker, developer)

    async def exercise() -> None:
        ready = asyncio.Event()
        task = asyncio.create_task(_reply_once(broker, ready))
        try:
            await asyncio.wait_for(ready.wait(), timeout=30)
            answer = await HandlerWorkLedgerQuery(_transport(broker, developer)).query(
                {"query": "rows"}, timeout_s=30
            )
            assert answer["matched"] == 1
            assert answer["freshness"] is not None
            assert answer["parity"] is not None
            await task
            with pytest.raises(
                (TopicAuthorizationFailedError, GroupAuthorizationFailedError)
            ):
                await HandlerWorkLedgerQuery(_transport(broker, unrelated)).query(
                    {"query": "rows"}, timeout_s=30
                )
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    asyncio.run(exercise())
