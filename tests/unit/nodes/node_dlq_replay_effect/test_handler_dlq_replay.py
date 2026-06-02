# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for the contract-native DLQ replay node (OMN-12619).

Proves the OMN-12619 invariants without a live broker by injecting fakes for
the consumer / replay producer / quarantine producer / tracking service:

    1. A non-replayable message is QUARANTINED (published to the quarantine
       topic) — never silently dropped.
    2. An eligible message is replayed exactly once to its original topic.
    3. A replay publish failure is recorded as FAILED (never a false
       COMPLETED / false success).
    4. A quarantine publish failure is recorded as FAILED (still not silent
       loss — the failure is durable).
    5. dlq_replay_history records every terminal outcome.
    6. Eligibility is decided by the reused should_replay() (max-retry,
       non-retryable error type) — not a reimplementation.
    7. The QUARANTINED enum member and onex.dlq.omnibase-infra.quarantine.v1 constant exist.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import uuid4

import pytest

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.dlq.models.enum_replay_status import EnumReplayStatus
from omnibase_infra.dlq.models.model_dlq_replay_record import ModelDlqReplayRecord
from omnibase_infra.event_bus.topic_constants import TOPIC_DLQ_QUARANTINE
from omnibase_infra.nodes.node_dlq_replay_effect.engine_dlq_replay import (
    DLQ_REPLAY_CONSUMER_GROUP,
    ModelDlqReplayEngineConfig,
    should_replay,
)
from omnibase_infra.nodes.node_dlq_replay_effect.handlers.handler_dlq_replay import (
    HandlerDlqReplay,
)
from omnibase_infra.nodes.node_dlq_replay_effect.models.model_dlq_message import (
    ModelDlqMessage,
)
from omnibase_infra.nodes.node_dlq_replay_effect.models.model_dlq_replay_run_result import (
    ModelDlqReplayRunResult,
)

pytestmark = pytest.mark.unit


def _make_message(
    *,
    retry_count: int = 0,
    error_type: str = "InfraConnectionError",
    topic: str = "dev.orders.command.v1",
) -> ModelDlqMessage:
    return ModelDlqMessage(
        original_topic=topic,
        original_key="k",
        original_value='{"hello": "world"}',
        original_offset="10",
        original_partition=0,
        failure_reason="boom",
        failure_timestamp="2026-06-02T00:00:00Z",
        correlation_id=uuid4(),
        retry_count=retry_count,
        error_type=error_type,
        dlq_offset=42,
        dlq_partition=1,
        raw_payload={"original_topic": topic},
    )


def _config(**overrides: object) -> ModelDlqReplayEngineConfig:
    base: dict[str, object] = {
        "bootstrap_servers": "localhost:9092",
        "dlq_topic": "onex.dlq.omnibase-infra.events.v1",
        "max_replay_count": 5,
    }
    base.update(overrides)
    return ModelDlqReplayEngineConfig(**base)  # type: ignore[arg-type]


class _FakeConsumer:
    def __init__(
        self, messages: list[ModelDlqMessage], config: ModelDlqReplayEngineConfig
    ) -> None:
        self._messages = messages
        self.config = config

    async def consume_messages(self) -> AsyncIterator[ModelDlqMessage]:
        for message in self._messages:
            yield message


class _FakeProducer:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.replayed: list[tuple[str, object]] = []

    async def replay_message(
        self, message: ModelDlqMessage, replay_correlation_id: object
    ) -> None:
        if self.fail:
            raise RuntimeError("kafka unavailable")
        self.replayed.append((message.original_topic, message.correlation_id))


class _FakeQuarantineProducer:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.quarantined: list[tuple[ModelDlqMessage, str]] = []

    async def quarantine_message(
        self, message: ModelDlqMessage, reason: str, quarantine_correlation_id: object
    ) -> None:
        if self.fail:
            raise RuntimeError("quarantine broker down")
        self.quarantined.append((message, reason))


class _FakeTracking:
    is_tracking_enabled = True

    def __init__(self) -> None:
        self.records: list[ModelDlqReplayRecord] = []

    async def record_replay_attempt(self, record: ModelDlqReplayRecord) -> None:
        self.records.append(record)


def _handler(
    messages: list[ModelDlqMessage],
    config: ModelDlqReplayEngineConfig,
    *,
    replay_fail: bool = False,
    quarantine_fail: bool = False,
    tracking: _FakeTracking | None = None,
) -> tuple[HandlerDlqReplay, _FakeProducer, _FakeQuarantineProducer]:
    producer = _FakeProducer(fail=replay_fail)
    quarantine = _FakeQuarantineProducer(fail=quarantine_fail)
    handler = HandlerDlqReplay(
        consumer=_FakeConsumer(messages, config),  # type: ignore[arg-type]
        producer=producer,  # type: ignore[arg-type]
        quarantine_producer=quarantine,  # type: ignore[arg-type]
        tracking=tracking,  # type: ignore[arg-type]
    )
    return handler, producer, quarantine


def test_quarantine_constants_exist() -> None:
    assert TOPIC_DLQ_QUARANTINE == "onex.dlq.omnibase-infra.quarantine.v1"
    assert EnumReplayStatus.QUARANTINED.value == "quarantined"
    assert DLQ_REPLAY_CONSUMER_GROUP == "onex-dlq-replay"


def test_should_replay_is_reused_for_eligibility() -> None:
    # Non-retryable error type -> not eligible (reused predicate, not reimplemented)
    msg = _make_message(error_type="ValidationError")
    eligible, reason = should_replay(msg, _config())
    assert eligible is False
    assert "Non-retryable" in reason


async def test_non_replayable_message_is_quarantined_not_dropped() -> None:
    # retry_count >= max -> non-replayable
    msg = _make_message(retry_count=9)
    tracking = _FakeTracking()
    handler, producer, quarantine = _handler([msg], _config(), tracking=tracking)

    result = await handler.run()

    assert isinstance(result, ModelDlqReplayRunResult)
    assert result.quarantined == 1
    assert result.completed == 0
    assert result.failed == 0
    # message was published to quarantine, NOT dropped, NOT replayed
    assert len(quarantine.quarantined) == 1
    assert len(producer.replayed) == 0
    # tracking recorded a QUARANTINED (not SKIPPED) terminal outcome
    assert len(tracking.records) == 1
    assert tracking.records[0].replay_status == EnumReplayStatus.QUARANTINED
    assert tracking.records[0].success is False


async def test_eligible_message_replayed_exactly_once() -> None:
    msg = _make_message(retry_count=0, error_type="InfraConnectionError")
    tracking = _FakeTracking()
    handler, producer, quarantine = _handler([msg], _config(), tracking=tracking)

    result = await handler.run()

    assert result.completed == 1
    assert result.quarantined == 0
    assert len(producer.replayed) == 1  # exactly once
    assert len(quarantine.quarantined) == 0
    assert tracking.records[0].replay_status == EnumReplayStatus.COMPLETED
    assert tracking.records[0].success is True


async def test_replay_failure_records_failed_not_false_success() -> None:
    msg = _make_message(retry_count=0, error_type="InfraConnectionError")
    tracking = _FakeTracking()
    handler, _producer, _quarantine = _handler(
        [msg], _config(), replay_fail=True, tracking=tracking
    )

    result = await handler.run()

    assert result.completed == 0
    assert result.failed == 1
    # The record must be FAILED with success=False — never a false COMPLETED
    assert len(tracking.records) == 1
    assert tracking.records[0].replay_status == EnumReplayStatus.FAILED
    assert tracking.records[0].success is False


async def test_quarantine_failure_records_failed_not_silent_loss() -> None:
    msg = _make_message(retry_count=9)  # non-replayable
    tracking = _FakeTracking()
    handler, _producer, _quarantine = _handler(
        [msg], _config(), quarantine_fail=True, tracking=tracking
    )

    result = await handler.run()

    assert result.quarantined == 0
    assert result.failed == 1
    # Failed quarantine is still durable evidence — not a silent drop
    assert len(tracking.records) == 1
    assert tracking.records[0].replay_status == EnumReplayStatus.FAILED
    assert tracking.records[0].success is False


async def test_dry_run_publishes_nothing() -> None:
    eligible = _make_message(retry_count=0, error_type="InfraConnectionError")
    non_replayable = _make_message(retry_count=9)
    handler, producer, quarantine = _handler(
        [eligible, non_replayable], _config(dry_run=True)
    )

    result = await handler.run()

    assert result.dry_run is True
    assert result.pending == 2
    assert len(producer.replayed) == 0
    assert len(quarantine.quarantined) == 0


async def test_handle_envelope_returns_typed_output() -> None:
    msg = _make_message(retry_count=0, error_type="InfraConnectionError")
    handler, _producer, _quarantine = _handler([msg], _config())
    correlation_id = uuid4()
    envelope: ModelEventEnvelope[ModelDlqReplayRunResult] = ModelEventEnvelope(
        payload=ModelDlqReplayRunResult(
            dlq_topic="onex.dlq.omnibase-infra.events.v1",
            total_processed=0,
            completed=0,
            quarantined=0,
            failed=0,
            pending=0,
            dry_run=False,
        ),
        correlation_id=correlation_id,
    )

    output = await handler.handle(envelope)

    assert output.correlation_id == correlation_id
    assert output.input_envelope_id == envelope.envelope_id
    assert isinstance(output.result, ModelDlqReplayRunResult)
    assert output.result.completed == 1
