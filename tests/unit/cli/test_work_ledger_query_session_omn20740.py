# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A reader session shares its connection without racing correlated replies."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from omnibase_infra.event_bus.kafka_transport import KafkaTransport
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.handlers.handler_work_ledger_query import HandlerWorkLedgerQuery

pytestmark = pytest.mark.unit


async def test_session_reuses_one_transport_and_serializes_replies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = AsyncMock(spec=KafkaTransport)
    requests = []
    active = 0

    async def send(topic: str, *, value: bytes, **kwargs: object) -> None:
        nonlocal active
        active += 1
        assert active == 1, "another query could consume this query's reply"
        requests.append(json.loads(value))
        await asyncio.sleep(0)

    async def poll(**kwargs: object) -> list[SimpleNamespace]:
        nonlocal active
        await asyncio.sleep(0)
        request = requests[-1]
        active -= 1
        return [
            SimpleNamespace(
                value=json.dumps(
                    {
                        "correlation_id": request["correlation_id"],
                        "payload": {
                            **request["payload"],
                            "freshness": {},
                            "parity": {"status": "unmeasured"},
                        },
                    }
                ).encode()
            )
        ]

    transport.send.side_effect = send
    transport.poll.side_effect = poll
    factory = Mock(return_value=transport)
    monkeypatch.setattr(
        "omnibase_infra.handlers.handler_work_ledger_query.KafkaTransport", factory
    )
    client = HandlerWorkLedgerQuery(ModelKafkaEventBusConfig())
    async with client:
        first, second = await asyncio.gather(
            client.query({"query": "rows", "term": "first"}),
            client.query({"query": "rows", "term": "second"}),
        )
        assert first["term"] == "first"
        assert second["term"] == "second"
        assert first["correlation_id"] != second["correlation_id"]
        factory.assert_called_once()
        transport.start.assert_awaited_once()
        transport.close.assert_not_awaited()
    transport.close.assert_awaited_once()


async def test_session_timeout_closes_transport_before_next_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed = AsyncMock(spec=KafkaTransport)
    failed.poll.side_effect = TimeoutError
    replacement = AsyncMock(spec=KafkaTransport)
    replacement.poll.side_effect = TimeoutError
    factory = Mock(side_effect=[failed, replacement])
    monkeypatch.setattr(
        "omnibase_infra.handlers.handler_work_ledger_query.KafkaTransport", factory
    )
    client = HandlerWorkLedgerQuery(ModelKafkaEventBusConfig())
    async with client:
        for transport in (failed, replacement):
            with pytest.raises(TimeoutError):
                await client.query({"query": "rows"}, timeout_s=1)
            transport.close.assert_awaited_once()
    assert factory.call_count == 2
    failed.close.assert_awaited_once()
    replacement.close.assert_awaited_once()


async def test_waiting_query_deadline_does_not_close_the_active_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = AsyncMock(spec=KafkaTransport)
    polling = asyncio.Event()
    release_reply = asyncio.Event()
    request = {}

    async def send(topic: str, *, value: bytes, **kwargs: object) -> None:
        request.update(json.loads(value))

    async def poll(**kwargs: object) -> list[SimpleNamespace]:
        polling.set()
        await release_reply.wait()
        return [
            SimpleNamespace(
                value=json.dumps(
                    {
                        "correlation_id": request["correlation_id"],
                        "payload": {
                            **request["payload"],
                            "freshness": {},
                            "parity": {},
                        },
                    }
                ).encode()
            )
        ]

    transport.send.side_effect = send
    transport.poll.side_effect = poll
    monkeypatch.setattr(
        "omnibase_infra.handlers.handler_work_ledger_query.KafkaTransport",
        Mock(return_value=transport),
    )
    async with HandlerWorkLedgerQuery(ModelKafkaEventBusConfig()) as client:
        active = asyncio.create_task(client.query({"query": "rows"}, timeout_s=1))
        try:
            await asyncio.wait_for(polling.wait(), timeout=1)
            with pytest.raises(TimeoutError):
                await client.query({"query": "rows"}, timeout_s=0.01)
            transport.close.assert_not_awaited()
            transport.send.assert_awaited_once()
            release_reply.set()
            assert (await active)["query"] == "rows"
        finally:
            release_reply.set()
            await active
    transport.close.assert_awaited_once()


@pytest.mark.parametrize("cancel", [False, True])
async def test_interrupted_session_read_closes_the_connection(
    cancel: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = AsyncMock(spec=KafkaTransport)
    polling = asyncio.Event()

    async def poll(**kwargs: object) -> None:
        polling.set()
        await asyncio.Event().wait()

    transport.poll.side_effect = poll
    monkeypatch.setattr(
        "omnibase_infra.handlers.handler_work_ledger_query.KafkaTransport",
        Mock(return_value=transport),
    )
    async with HandlerWorkLedgerQuery(ModelKafkaEventBusConfig()) as client:
        task = asyncio.create_task(
            client.query({"query": "rows"}, timeout_s=1 if cancel else 0.05)
        )
        await asyncio.wait_for(polling.wait(), timeout=1)
        if cancel:
            task.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
            await task
        transport.close.assert_awaited_once()
    transport.close.assert_awaited_once()
