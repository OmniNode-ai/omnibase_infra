# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev ledger readers use the granted group and fail closed on bad answers."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from click.testing import CliRunner

from omnibase_infra.cli.cli_ledger import ledger_group
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.handlers.handler_work_ledger_query import (
    HandlerWorkLedgerQuery,
    WorkLedgerQueryError,
    load_work_ledger_query_client_config,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    state = SimpleNamespace(
        sent=[], assigned=False, consumer_options={}, producer_options={}
    )
    consumer = AsyncMock()
    producer = AsyncMock()
    consumer.assignment = lambda: {"partition-0"}

    async def position(partition: object) -> int:
        state.assigned = True
        return 0

    consumer.position.side_effect = position

    async def send(topic: str, value: bytes, *, key: bytes) -> None:
        assert state.assigned, "publish raced the reply consumer's starting position"
        state.sent.append((topic, json.loads(value), key))

    producer.send_and_wait.side_effect = send

    async def getone() -> SimpleNamespace:
        envelope = state.sent[-1][1]
        payload = {
            "correlation_id": envelope["correlation_id"],
            "query": envelope["payload"]["query"],
            "matched": 1,
            "rows": [{"raw_row": "a full row"}],
            "freshness": {"newest_row_ts": "2026-10-25T00:00:00Z"},
            "parity": {"status": "unmeasured"},
            "error": None,
        }
        payload.update(getattr(state, "reply_override", {}))
        return SimpleNamespace(
            value=json.dumps(
                {"correlation_id": envelope["correlation_id"], "payload": payload}
            ).encode()
        )

    consumer.getone.side_effect = getone

    def make_consumer(*args: object, **kwargs: object) -> AsyncMock:
        state.consumer_options = kwargs
        return consumer

    def make_producer(**kwargs: object) -> AsyncMock:
        state.producer_options = kwargs
        return producer

    monkeypatch.setattr(
        "omnibase_infra.handlers.handler_work_ledger_query.AIOKafkaConsumer",
        make_consumer,
    )
    monkeypatch.setattr(
        "omnibase_infra.handlers.handler_work_ledger_query.AIOKafkaProducer",
        make_producer,
    )
    state.consumer, state.producer = consumer, producer
    return state


async def test_query_uses_the_dev_grant_and_preserves_freshness(
    wire: SimpleNamespace,
) -> None:
    config = load_work_ledger_query_client_config()
    client = HandlerWorkLedgerQuery(
        ModelKafkaEventBusConfig(bootstrap_servers="broker.invalid:9092")
    )
    result = await client.query({"query": "rows", "term": "OMN-20740"})
    assert wire.consumer_options["group_id"].startswith(config.consumer_group_prefix)
    assert wire.consumer_options["enable_auto_commit"] is False
    assert wire.consumer_options["auto_offset_reset"] == "latest"
    assert wire.sent[0][0] == config.command_topic
    assert wire.sent[0][1]["payload"]["term"] == "OMN-20740"
    assert result["rows"] == [{"raw_row": "a full row"}]
    assert result["freshness"] == {"newest_row_ts": "2026-10-25T00:00:00Z"}
    assert result["parity"] == {"status": "unmeasured"}
    wire.consumer.stop.assert_awaited_once()
    wire.producer.stop.assert_awaited_once()


async def test_database_error_is_never_an_empty_success(wire: SimpleNamespace) -> None:
    wire.reply_override = {"error": "database unavailable"}
    client = HandlerWorkLedgerQuery(
        ModelKafkaEventBusConfig(bootstrap_servers="broker.invalid:9092")
    )
    with pytest.raises(WorkLedgerQueryError, match="database"):
        await client.query({"query": "rows"})
    wire.consumer.stop.assert_awaited_once()
    wire.producer.stop.assert_awaited_once()


async def test_mismatched_payload_is_refused(wire: SimpleNamespace) -> None:
    wire.reply_override = {"correlation_id": str(uuid4())}
    client = HandlerWorkLedgerQuery(
        ModelKafkaEventBusConfig(bootstrap_servers="broker.invalid:9092")
    )
    with pytest.raises(WorkLedgerQueryError, match="correlation"):
        await client.query({"query": "rows"})


async def test_timeout_closes_both_clients(wire: SimpleNamespace) -> None:
    import asyncio

    wire.consumer.getone.side_effect = asyncio.TimeoutError
    client = HandlerWorkLedgerQuery(
        ModelKafkaEventBusConfig(bootstrap_servers="broker.invalid:9092")
    )
    with pytest.raises(TimeoutError):
        await client.query({"query": "rows"}, timeout_s=1)
    wire.consumer.stop.assert_awaited_once()
    wire.producer.stop.assert_awaited_once()


def test_read_command_refuses_prod_before_resolving_a_credential() -> None:
    result = CliRunner().invoke(ledger_group, ["query", "--bus-lane", "prod"])
    assert result.exit_code == 2
    assert "prod" in result.output and "dev" in result.output


def test_ledger_ask_query_reaches_the_client(
    monkeypatch: pytest.MonkeyPatch, wire: SimpleNamespace
) -> None:
    monkeypatch.setattr(
        "omnibase_infra.handlers.handler_work_ledger_query.resolve_work_ledger_query_transport",
        lambda lane: ModelKafkaEventBusConfig(bootstrap_servers="broker.invalid:9092"),
    )
    result = CliRunner().invoke(
        ledger_group,
        [
            "query",
            "--term",
            "OMN-20740",
            "--since",
            "2026-10-25T00:00:00Z",
            "--lane",
            "some-lane",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["matched"] == 1
    request = wire.sent[0][1]["payload"]
    assert request["query"] == "rows"
    assert request["lane"] == "some-lane"
    assert request["since"] == "2026-10-25T00:00:00Z"


@pytest.mark.parametrize("reader", ["ledger-ask", "catchup"])
def test_existing_reader_stdin_protocol_preserves_correlation_and_filters(
    reader: str, monkeypatch: pytest.MonkeyPatch, wire: SimpleNamespace
) -> None:
    """Both existing readers use handler_ledger_source.read_db_lines's protocol."""
    monkeypatch.setattr(
        "omnibase_infra.handlers.handler_work_ledger_query.resolve_work_ledger_query_transport",
        lambda lane: ModelKafkaEventBusConfig(bootstrap_servers="broker.invalid:9092"),
    )
    cid = str(uuid4())
    request = {"query": "rows", "correlation_id": cid, "since": "2026-10-25T00:00:00Z"}
    if reader == "ledger-ask":
        request["term"] = "OMN-20740"
    result = CliRunner().invoke(
        ledger_group,
        ["query", "--bus-lane", "dev", "--stdin"],
        input=json.dumps(request),
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["correlation_id"] == cid
    assert wire.sent[0][1]["payload"] == request


@pytest.mark.parametrize("data", ["[]", "{", "null", '"a string"'])
def test_stdin_refuses_non_object_input_before_connecting(
    data: str, wire: SimpleNamespace
) -> None:
    result = CliRunner().invoke(ledger_group, ["query", "--stdin"], input=data)
    assert result.exit_code == 2
    assert "JSON object" in result.output
    assert not wire.sent


def test_stdin_refuses_filter_overrides(wire: SimpleNamespace) -> None:
    result = CliRunner().invoke(
        ledger_group, ["query", "--stdin", "--limit", "1"], input='{"query":"rows"}'
    )
    assert result.exit_code == 2
    assert "filter options" in result.output
    assert not wire.sent


async def test_missing_freshness_is_refused(wire: SimpleNamespace) -> None:
    wire.reply_override = {"freshness": None}
    client = HandlerWorkLedgerQuery(
        ModelKafkaEventBusConfig(bootstrap_servers="broker.invalid:9092")
    )
    with pytest.raises(WorkLedgerQueryError, match="freshness"):
        await client.query({"query": "rows"})


async def test_shared_terminal_noise_cannot_answer_the_request(
    wire: SimpleNamespace,
) -> None:
    good = wire.consumer.getone.side_effect
    noise = iter(
        [b"{", json.dumps({"correlation_id": str(uuid4()), "payload": {}}).encode()]
    )

    async def getone() -> SimpleNamespace:
        value = next(noise, None)
        return SimpleNamespace(value=value) if value is not None else await good()

    wire.consumer.getone.side_effect = getone
    client = HandlerWorkLedgerQuery(
        ModelKafkaEventBusConfig(bootstrap_servers="broker.invalid:9092")
    )
    assert (await client.query({"query": "rows"}))["matched"] == 1


def test_cli_refusal_is_nonzero_and_cannot_leak_a_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refused(lane: str) -> ModelKafkaEventBusConfig:
        raise RuntimeError("postgresql://user:synthetic-secret@invalid/db")

    monkeypatch.setattr(
        "omnibase_infra.handlers.handler_work_ledger_query.resolve_work_ledger_query_transport",
        refused,
    )
    result = CliRunner().invoke(ledger_group, ["query"])
    assert result.exit_code == 1
    assert not result.stdout
    assert "refused (RuntimeError)" in result.stderr
    assert "synthetic-secret" not in result.output
