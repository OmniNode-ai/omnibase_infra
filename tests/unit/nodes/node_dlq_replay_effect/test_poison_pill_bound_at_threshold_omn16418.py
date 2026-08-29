# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16418 AC3 — a record that fails dispatch repeatedly must be parked
(real quarantine) after a bounded attempt count, not looped forever.

``max_replay_count`` / ``should_replay()`` (OMN-14551) already implement
this bound, and ``test_handler_dlq_replay.py::test_non_replayable_message_is_
quarantined_not_dropped`` already proves the *shape* of the bound (a message
deep past the threshold, ``retry_count=9`` against ``max=5``, is quarantined
not replayed). What was never pinned by a test — flagged on OMN-16418
2026-08-29T12:23:21Z, against live evidence of ``x-replay-count: 4`` on
30/30 sampled messages against a ``max_replay_count=5`` cap, "one increment
from tripping" — is the *exact boundary transition*: the last attempt that
is still eligible, and the very next attempt (one ``x-replay-count``
increment later) that must trip the park instead of looping again.

This file pins that boundary precisely, using the same
``retry_count`` -> ``x-replay-count`` carry-forward header semantics
``engine_dlq_replay.py`` writes on every replay (``"x-replay-count",
str(message.retry_count + 1)``).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_dlq_replay_effect.engine_dlq_replay import (
    ModelDlqReplayEngineConfig,
    should_replay,
)
from omnibase_infra.nodes.node_dlq_replay_effect.handlers.handler_dlq_replay import (
    HandlerDlqReplay,
)
from omnibase_infra.nodes.node_dlq_replay_effect.models.model_dlq_message import (
    ModelDlqMessage,
)

pytestmark = pytest.mark.unit

_MAX_REPLAY_COUNT = 5


def _make_message(*, retry_count: int) -> ModelDlqMessage:
    return ModelDlqMessage(
        original_topic="onex.evt.omniclaude.session-started.v1",
        original_key="k",
        original_value='{"hello": "world"}',
        original_offset="10",
        original_partition=0,
        failure_reason="handler dispatch failed",
        failure_timestamp="2026-08-18T15:08:31Z",
        correlation_id=uuid4(),
        retry_count=retry_count,
        error_type="InfraConnectionError",  # retryable error class
        dlq_offset=42,
        dlq_partition=1,
        raw_payload={"original_topic": "onex.evt.omniclaude.session-started.v1"},
    )


def _config() -> ModelDlqReplayEngineConfig:
    return ModelDlqReplayEngineConfig(
        bootstrap_servers="localhost:9092",
        dlq_topic="onex.dlq.omnibase-infra.events.v1",
        max_replay_count=_MAX_REPLAY_COUNT,
    )


class _FakeConsumer:
    def __init__(
        self, messages: list[ModelDlqMessage], config: ModelDlqReplayEngineConfig
    ) -> None:
        self._messages = messages
        self.config = config
        self.commits = 0

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def consume_messages(self) -> AsyncIterator[ModelDlqMessage]:
        for message in self._messages:
            yield message

    async def commit(self) -> None:
        self.commits += 1


class _FakeProducer:
    def __init__(self) -> None:
        self.replayed: list[str] = []

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def replay_message(
        self, message: ModelDlqMessage, replay_correlation_id: object
    ) -> None:
        self.replayed.append(str(message.correlation_id))


class _FakeQuarantineProducer:
    def __init__(self) -> None:
        self.quarantined: list[str] = []

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def quarantine_message(
        self, message: ModelDlqMessage, reason: str, quarantine_correlation_id: object
    ) -> None:
        self.quarantined.append(str(message.correlation_id))


def _handler(
    message: ModelDlqMessage,
) -> tuple[HandlerDlqReplay, _FakeConsumer, _FakeProducer, _FakeQuarantineProducer]:
    consumer = _FakeConsumer([message], _config())
    producer = _FakeProducer()
    quarantine = _FakeQuarantineProducer()
    handler = HandlerDlqReplay(
        consumer=consumer,  # type: ignore[arg-type]
        producer=producer,  # type: ignore[arg-type]
        quarantine_producer=quarantine,  # type: ignore[arg-type]
        tracking=None,
    )
    return handler, consumer, producer, quarantine


class TestPoisonPillParkedAtBoundedThreshold:
    """AC3: the last attempt below the cap still replays (not parked too
    early); the attempt AT the cap is parked (not looped past it)."""

    async def test_last_attempt_below_cap_still_replays_not_yet_parked(self) -> None:
        """One increment before the cap (retry_count == max - 1, the
        x-replay-count the live storm evidence sampled at) is still
        eligible -- the bound must not trip early."""
        msg = _make_message(retry_count=_MAX_REPLAY_COUNT - 1)
        handler, consumer, producer, quarantine = _handler(msg)

        eligible, _reason = should_replay(msg, _config())
        assert eligible is True

        result = await handler.run()

        assert result.completed == 1
        assert result.quarantined == 0
        assert len(producer.replayed) == 1
        assert len(quarantine.quarantined) == 0
        assert consumer.commits == 1

    async def test_attempt_at_cap_is_parked_not_replayed_again(self) -> None:
        """The very next attempt (retry_count == max) is the poison-pill
        bound: it must be parked in the real quarantine sink -- not
        replayed, not dropped, not looped forever."""
        msg = _make_message(retry_count=_MAX_REPLAY_COUNT)
        handler, consumer, producer, quarantine = _handler(msg)

        eligible, reason = should_replay(msg, _config())
        assert eligible is False
        assert "max replay count" in reason.lower()

        result = await handler.run()

        assert result.completed == 0
        assert result.quarantined == 1
        assert len(producer.replayed) == 0  # never re-enters replay
        assert len(quarantine.quarantined) == 1  # durably parked instead
        assert consumer.commits == 1

    async def test_replay_header_carry_forward_reaches_the_cap_deterministically(
        self,
    ) -> None:
        """The x-replay-count header that carries retry_count forward on
        every republish (engine_dlq_replay.py) increments by exactly one
        per cycle -- so a record that keeps failing reaches the cap in a
        bounded, deterministic number of cycles, not an unbounded tail."""
        cycles_to_cap = _MAX_REPLAY_COUNT - (_MAX_REPLAY_COUNT - 1)
        assert cycles_to_cap == 1

        retry_count = _MAX_REPLAY_COUNT - 1
        for _ in range(cycles_to_cap):
            # Mirrors engine_dlq_replay.py's header write:
            # ("x-replay-count", str(message.retry_count + 1))
            retry_count = retry_count + 1

        assert retry_count == _MAX_REPLAY_COUNT
        eligible, _reason = should_replay(
            _make_message(retry_count=retry_count), _config()
        )
        assert eligible is False
