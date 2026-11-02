# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Boundary -> Kafka DLQ serializer -> parser -> replay, with transport doubles.

No broker is contacted. The production serializers, classification and replay
handler run together, and only Kafka sends and dispatch failures are doubled.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch
from uuid import UUID

import pytest
from aiokafka import AIOKafkaProducer

from omnibase_core.models.event_bus.model_event_headers import ModelEventHeaders
from omnibase_core.models.event_bus.model_event_message import ModelEventMessage
from omnibase_infra.dlq.models.enum_replay_status import EnumReplayStatus
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.nodes.node_dlq_replay_effect.engine_dlq_replay import (
    DLQConsumer,
    DLQProducer,
    DLQQuarantineProducer,
    ModelDlqReplayEngineConfig,
)
from omnibase_infra.nodes.node_dlq_replay_effect.handlers.handler_dlq_replay import (
    DlqConsumerDrainState,
    HandlerDlqReplay,
)
from omnibase_infra.nodes.node_dlq_replay_effect.models.model_dlq_message import (
    ModelDlqMessage,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import _make_event_bus_callback
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

_COMMAND = "onex.cmd.omnimarket.delegate-skill.v1"
_FAILED = "onex.evt.omnimarket.delegate-skill-failed.v1"
_DLQ = "onex.dlq.omnibase-infra.commands.v1"
_CORRELATION = UUID("95c783ae-79b6-47bb-a1d9-2edc46f66a45")


def _message(body: bytes) -> ModelEventMessage:
    return ModelEventMessage(
        topic=_COMMAND,
        key=None,
        value=body,
        headers=ModelEventHeaders(
            correlation_id=_CORRELATION,
            timestamp=datetime.now(UTC),
            source="omn21324",
            event_type=_COMMAND,
        ),
        offset="1",
        partition=0,
    )


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "error", "retryable"),
    [
        (b'{"task":"x"}', RuntimeError("ProtocolConfigurationError: refused"), False),
        (b"\xff", None, False),
        (b'{"task":"x"}', RuntimeError("TimeoutError: transient"), True),
    ],
)
async def test_boundary_dlq_replay_emits_no_second_terminal_for_refusal(
    monkeypatch: pytest.MonkeyPatch,
    body: bytes,
    error: Exception | None,
    retryable: bool,
) -> None:
    monkeypatch.setenv("ONEX_BOUNDARY_DLQ_ENABLED", "true")
    bus = EventBusKafka(ModelKafkaEventBusConfig(dead_letter_topic=_DLQ))
    kafka = AsyncMock(spec=AIOKafkaProducer)
    kafka.send_and_wait.return_value = object()
    bus._producer = kafka
    bus.publish = AsyncMock()
    engine = MessageDispatchEngine()
    engine.freeze()
    callback = _make_event_bus_callback(
        _COMMAND,
        engine,
        event_bus=bus,
        allowed_dispatcher_ids={"test-dispatcher"},
        failure_terminal_topics=(_FAILED,),
    )
    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._dispatch_to_contract_scope",
        new=AsyncMock(side_effect=error),
    ):
        await callback(_message(body))

    assert bus.publish.await_count == 1
    terminal = json.loads(bus.publish.await_args.args[2])
    assert bus.publish.await_args.args[0] == _FAILED
    assert terminal["correlation_id"] == str(_CORRELATION)
    assert terminal["payload"]["retryable"] is retryable
    records = [
        call for call in kafka.send_and_wait.await_args_list if call.args[0] == _DLQ
    ]
    assert len(records) == 1
    record = ModelDlqMessage.from_kafka_message(
        json.loads(records[0].kwargs["value"]), 1, 0
    )
    assert record.correlation_id == _CORRELATION

    config = ModelDlqReplayEngineConfig(
        bootstrap_servers="localhost:9092", dlq_topic=_DLQ
    )
    producer, quarantine = DLQProducer(config), DLQQuarantineProducer(config)
    replay_kafka = AsyncMock(spec=AIOKafkaProducer)
    replay_kafka.send_and_wait.return_value = object()
    producer._producer = replay_kafka
    producer._started = True
    quarantine._producer = replay_kafka
    quarantine._started = True
    handler = HandlerDlqReplay(
        consumers={_DLQ: DLQConsumer(config)},
        producer=producer,
        quarantine_producer=quarantine,
        tracking=None,
    )
    result = await handler._process_message(record, config, DlqConsumerDrainState())
    assert result.status == (
        EnumReplayStatus.COMPLETED if retryable else EnumReplayStatus.QUARANTINED
    )
    command_sends = [
        call
        for call in replay_kafka.send_and_wait.await_args_list
        if call.args[0] == _COMMAND
    ]
    assert len(command_sends) == int(retryable)
    if retryable:
        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._dispatch_to_contract_scope",
            new=AsyncMock(side_effect=error),
        ):
            await callback(_message(command_sends[0].kwargs["value"]))
    else:
        assert replay_kafka.send_and_wait.await_args.args[0] == config.quarantine_topic
    assert bus.publish.await_count == (2 if retryable else 1)
