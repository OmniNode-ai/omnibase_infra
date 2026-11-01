# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21324 — undecodable delegate-skill command bodies and non-retryable replay.

Measured in the malformed-and-injected-tasks eval on the delegate-skill command
topic:

* an empty or non-JSON body terminalized with ``failure_code`` None;
* a JSON-array body terminalized under a freshly minted correlation instead of
  the one on its Kafka header;
* a command refused as non-retryable terminalized six times, because the DLQ
  record's ``error_type`` is the boundary wrapper's class name, which
  ``should_replay`` does not know, so replay re-dispatched it five times.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from omnibase_core.models.event_bus.model_event_headers import ModelEventHeaders
from omnibase_core.models.event_bus.model_event_message import ModelEventMessage
from omnibase_infra.enums import EnumNonRetryableErrorCategory
from omnibase_infra.nodes.node_dlq_replay_effect.engine_dlq_replay import (
    ModelDlqReplayEngineConfig,
    should_replay,
)
from omnibase_infra.nodes.node_dlq_replay_effect.models.model_dlq_message import (
    ModelDlqMessage,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _make_event_bus_callback,
)
from omnibase_infra.runtime.boundary_failure_terminal import classify_boundary_failure
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
from omnibase_infra.utils.util_error_sanitization import sanitize_error_message

_COMMAND_TOPIC = "onex.cmd.omnimarket.delegate-skill.v1"
_FAILED_TOPIC = "onex.evt.omnimarket.delegate-skill-failed.v1"
_HEADER_CORRELATION = UUID("95c783ae-79b6-47bb-a1d9-2edc46f66a45")
_CONFIG = ModelDlqReplayEngineConfig(
    bootstrap_servers="localhost:9092", dlq_topic="onex.dlq.omnimarket.events.v1"
)


class _Bus:
    def __init__(self) -> None:
        self.publish = AsyncMock(return_value=None)


def _message(value: bytes) -> ModelEventMessage:
    return ModelEventMessage(
        topic=_COMMAND_TOPIC,
        key=None,
        value=value,
        headers=ModelEventHeaders(
            correlation_id=_HEADER_CORRELATION,
            timestamp=datetime.now(UTC),
            source="omn-21324-test",
            event_type=_COMMAND_TOPIC,
        ),
        offset="1",
        partition=0,
    )


async def _terminals_for(value: bytes) -> list[dict[str, object]]:
    bus = _Bus()
    engine = MessageDispatchEngine()
    engine.freeze()
    callback = _make_event_bus_callback(
        _COMMAND_TOPIC,
        engine,
        event_bus=bus,
        allowed_dispatcher_ids={"test-dispatcher"},
        failure_terminal_topics=(_FAILED_TOPIC,),
    )
    await callback(_message(value))
    terminals: list[dict[str, object]] = []
    for call in bus.publish.await_args_list:
        assert call.args[0] == _FAILED_TOPIC
        envelope = json.loads(call.args[2])
        terminals.append(
            {"correlation_id": envelope["correlation_id"], **envelope["payload"]}
        )
    return terminals


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("body", [b"", b"this is not json"])
async def test_delegate_skill_undecodable_body_terminalizes_with_typed_failure_code(
    body: bytes,
) -> None:
    terminals = await _terminals_for(body)
    assert len(terminals) == 1
    assert terminals[0]["failure_class"] == "JSONDecodeError"
    assert terminals[0]["failure_code"] is not None
    assert str(terminals[0]["failure_code"]).startswith("ONEX_")
    assert terminals[0]["correlation_id"] == str(_HEADER_CORRELATION)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_delegate_skill_array_body_correlation_is_the_header_correlation() -> (
    None
):
    terminals = await _terminals_for(b'["not", "an", "object"]')
    assert len(terminals) == 1
    assert terminals[0]["correlation_id"] == str(_HEADER_CORRELATION)


@pytest.mark.unit
def test_delegate_skill_non_retryable_refusal_is_not_replayed() -> None:
    exc = RuntimeError(
        "dispatch failed: ProtocolConfigurationError: "
        "[ONEX_CORE_345_ENVELOPE_VALIDATION_FAILED] envelope refused"
    )
    message = ModelDlqMessage(
        original_topic=_COMMAND_TOPIC,
        original_value='{"task": "x"}',
        failure_reason=sanitize_error_message(exc),
        correlation_id=_HEADER_CORRELATION,
        retry_count=0,
        error_type="HandlerDispatchFailureError",
        dlq_offset=1,
        dlq_partition=0,
    )
    eligible, reason = should_replay(message, _CONFIG)
    assert eligible is False, reason
    assert "non-retryable" in reason.lower()


@pytest.mark.unit
def test_delegate_skill_retryable_failure_is_still_replayed() -> None:
    message = ModelDlqMessage(
        original_topic=_COMMAND_TOPIC,
        original_value='{"task": "x"}',
        failure_reason="HandlerDispatchFailureError: upstream timed out",
        correlation_id=_HEADER_CORRELATION,
        retry_count=0,
        error_type="HandlerDispatchFailureError",
        dlq_offset=1,
        dlq_partition=0,
    )
    eligible, reason = should_replay(message, _CONFIG)
    assert eligible is True, reason


@pytest.mark.unit
@pytest.mark.asyncio
async def test_invalid_utf8_terminal_is_typed_and_non_retryable() -> None:
    terminals = await _terminals_for(b"\xff")
    assert len(terminals) == 1
    assert terminals[0]["failure_class"] == "UnicodeDecodeError"
    assert terminals[0]["failure_code"] == "ONEX_CORE_007_INVALID_INPUT"
    assert terminals[0]["retryable"] is False
    assert terminals[0]["correlation_id"] == str(_HEADER_CORRELATION)


@pytest.mark.unit
@pytest.mark.parametrize("link", ["__cause__", "__context__"])
def test_undecodable_body_code_walks_exception_links(link: str) -> None:
    outer = RuntimeError("wrapped")
    setattr(outer, link, json.JSONDecodeError("bad body", "", 0))
    assert (
        EnumNonRetryableErrorCategory.undecodable_body_code(outer)
        == "ONEX_CORE_007_INVALID_INPUT"
    )


@pytest.mark.unit
def test_undecodable_body_code_stops_on_exception_cycles() -> None:
    first, second = RuntimeError("first"), RuntimeError("second")
    first.__cause__, second.__context__ = second, first
    assert EnumNonRetryableErrorCategory.undecodable_body_code(first) is None
    first.__context__ = first
    first.__cause__ = None
    assert EnumNonRetryableErrorCategory.undecodable_body_code(first) is None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("explicit", "expected"), [(False, False), (None, True), (True, True)]
)
def test_explicit_retryable_attribute_can_refuse_retry(
    explicit: bool | None, expected: bool
) -> None:
    class FailureError(RuntimeError):
        retryable = explicit

    terminal = classify_boundary_failure(
        FailureError("transient"),
        topic=_COMMAND_TOPIC,
        correlation_id=_HEADER_CORRELATION,
        failure_reason="transient",
    )
    assert terminal.retryable is expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("wrapped: UnicodeDecodeError: invalid utf8", True),
        ("wrapped: ProtocolConfigurationError: refused", True),
        ("NotProtocolConfigurationError: transient", False),
        ("ProtocolConfigurationErrorSuffix: transient", False),
        ("_ProtocolConfigurationError: transient", False),
        ("TimeoutError: transient", False),
    ],
)
def test_non_retryable_text_requires_whole_class_tokens(
    text: str, expected: bool
) -> None:
    assert EnumNonRetryableErrorCategory.is_non_retryable_text(text) is expected
