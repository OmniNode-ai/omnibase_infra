# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The read client uses the granted group and refuses unusable answers."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel, ConfigDict

from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.model_event_headers import ModelEventHeaders
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.handlers.handler_work_ledger_query import (
    HandlerWorkLedgerQuery,
    WorkLedgerQueryError,
    load_work_ledger_query_contract,
)
from omnibase_infra.models.model_work_ledger_query_contract import (
    ModelWorkLedgerQueryContract,
)


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")
    correlation_id: UUID
    query: str
    since: str | None = None


class Answer(BaseModel):
    correlation_id: UUID
    query: str
    rows: tuple[str, ...] = ()
    freshness: dict[str, object] = {}
    parity: dict[str, object] = {}
    error: str | None = None


@pytest.fixture
def contract() -> ModelWorkLedgerQueryContract:
    return ModelWorkLedgerQueryContract(
        command_topic="test.cmd.ledger-query.v1",
        completed_topic="test.evt.ledger-query-completed.v1",
        request_model=Request,
        result_model=Answer,
    )


@pytest.mark.asyncio
async def test_query_subscribes_before_publish_and_preserves_freshness(contract):
    bus = AsyncMock(spec_set=EventBusKafka)
    unsubscribe = AsyncMock()
    bus.subscribe.return_value = unsubscribe
    client = HandlerWorkLedgerQuery(bus, contract=contract)

    async def publish(topic, key, value):
        command = json.loads(value)
        callback = bus.subscribe.call_args.kwargs["on_message"]
        assert bus.subscribe.call_args.kwargs["group_id"].startswith(
            "local.omnimarket.work_ledger_query_client.consume.v1."
        )
        # Another caller's reply must never answer this call.
        for correlation in (str(uuid4()), command["correlation_id"]):
            await callback(
                ModelEventMessage(
                    topic=contract.completed_topic,
                    key=None,
                    headers=ModelEventHeaders(
                        source="test", event_type="test", timestamp=datetime.now(UTC)
                    ),
                    value=json.dumps(
                        {
                            "correlation_id": correlation,
                            "payload": {
                                "correlation_id": correlation,
                                "query": "rows",
                                "rows": ["row"],
                                "freshness": {"rows_read": 1},
                                "parity": {"status": "unmeasured"},
                            },
                        }
                    ).encode(),
                )
            )

    bus.publish.side_effect = publish
    async with client:
        answer = await client.query({"query": "rows"})
    assert answer.rows == ("row",)
    assert answer.freshness == {"rows_read": 1}
    assert answer.parity == {"status": "unmeasured"}
    unsubscribe.assert_awaited_once()


@pytest.mark.asyncio
async def test_node_error_is_not_an_empty_success(contract):
    bus = AsyncMock(spec_set=EventBusKafka)
    client = HandlerWorkLedgerQuery(bus, contract=contract)

    async def publish(topic, key, value):
        command = json.loads(value)
        await bus.subscribe.call_args.kwargs["on_message"](
            ModelEventMessage(
                topic=contract.completed_topic,
                key=None,
                headers=ModelEventHeaders(
                    source="test", event_type="test", timestamp=datetime.now(UTC)
                ),
                value=json.dumps(
                    {
                        "correlation_id": command["correlation_id"],
                        "payload": {
                            "correlation_id": command["correlation_id"],
                            "query": "rows",
                            "error": "database read refused",
                        },
                    }
                ).encode(),
            )
        )

    bus.publish.side_effect = publish
    async with client:
        with pytest.raises(WorkLedgerQueryError, match="database read refused"):
            await client.query({"query": "rows"})


@pytest.mark.asyncio
async def test_timeout_and_invalid_request_do_not_leave_pending_calls(contract):
    bus = AsyncMock(spec_set=EventBusKafka)
    client = HandlerWorkLedgerQuery(bus, contract=contract)
    async with client:
        with pytest.raises(ValueError):
            await client.query({"query": "rows", "principal": "developer"})
        bus.publish.assert_not_awaited()
        with pytest.raises(TimeoutError):
            await client.query({"query": "rows"}, timeout_s=0.01)
        assert not client._pending


@pytest.mark.asyncio
async def test_broker_refusal_propagates(contract):
    bus = AsyncMock(spec_set=EventBusKafka)
    bus.subscribe.side_effect = PermissionError("topic authorization refused")
    with pytest.raises(PermissionError, match="authorization refused"):
        async with HandlerWorkLedgerQuery(bus, contract=contract):
            pytest.fail("a refused principal reached query")
    bus.publish.assert_not_awaited()


@pytest.mark.asyncio
async def test_concurrent_queries_get_unique_correlations_and_their_own_replies(
    contract,
):
    bus = AsyncMock(spec_set=EventBusKafka)
    published = []

    async def publish(topic, key, value):
        published.append(json.loads(value))
        if len(published) != 12:
            return
        for command in reversed(published):
            correlation = command["correlation_id"]
            await bus.subscribe.call_args.kwargs["on_message"](
                ModelEventMessage(
                    topic=contract.completed_topic,
                    headers=ModelEventHeaders(
                        source="test", event_type="test", timestamp=datetime.now(UTC)
                    ),
                    value=json.dumps(
                        {
                            "correlation_id": correlation,
                            "payload": {
                                "correlation_id": correlation,
                                "query": command["payload"]["query"],
                                "rows": [command["payload"]["query"]],
                            },
                        }
                    ).encode(),
                )
            )

    bus.publish.side_effect = publish
    async with HandlerWorkLedgerQuery(bus, contract=contract) as client:
        answers = await asyncio.gather(
            *(client.query({"query": str(i)}) for i in range(12))
        )
    assert len({command["correlation_id"] for command in published}) == 12
    assert [answer.rows for answer in answers] == [(str(i),) for i in range(12)]


@pytest.mark.asyncio
async def test_correlated_malformed_reply_fails_immediately(contract):
    bus = AsyncMock(spec_set=EventBusKafka)

    async def publish(topic, key, value):
        command = json.loads(value)
        await bus.subscribe.call_args.kwargs["on_message"](
            ModelEventMessage(
                topic=contract.completed_topic,
                headers=ModelEventHeaders(
                    source="test", event_type="test", timestamp=datetime.now(UTC)
                ),
                value=json.dumps(
                    {"correlation_id": command["correlation_id"], "payload": {}}
                ).encode(),
            )
        )

    bus.publish.side_effect = publish
    async with HandlerWorkLedgerQuery(bus, contract=contract) as client:
        with pytest.raises(WorkLedgerQueryError, match="validation errors"):
            await client.query({"query": "rows"})


@pytest.mark.parametrize("model_shape", ["qualified", "mapping"])
def test_contract_loader_uses_installed_topics_and_shared_models(
    monkeypatch, model_shape
):
    from omnibase_infra.handlers import handler_work_ledger_query as module

    def declared(cls):
        if model_shape == "qualified":
            return f"{__name__}.{cls.__name__}"
        return {"module": __name__, "class": cls.__name__}

    monkeypatch.setattr(
        module,
        "node_contract",
        lambda name: {
            "runtime_dispatch": {
                "command_topic": "test.changed-command",
                "terminal_events": {"success": "test.changed-result"},
            },
            "input_model": declared(Request),
            "output_model": declared(Answer),
        },
    )
    loaded = load_work_ledger_query_contract()
    assert loaded.command_topic == "test.changed-command"
    assert loaded.completed_topic == "test.changed-result"
    assert loaded.request_model is Request
    assert loaded.result_model is Answer


def test_missing_contract_is_a_refusal(monkeypatch):
    from omnibase_infra.cli.contract_registry import RegistryUnresolvedError
    from omnibase_infra.handlers import handler_work_ledger_query as module

    monkeypatch.setattr(module, "node_contract", lambda name: {})
    with pytest.raises(RegistryUnresolvedError, match="contract is unreadable"):
        load_work_ledger_query_contract()
