# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A publisher-malformed command terminalizes on its contract's failure topic.

A command that fails its handler's input validation is refused by the engine
(NO_DISPATCHER, failure_class=publisher_malformed) and goes to the DLQ. Nothing
told the caller: the gateway status stayed ``published`` for ever. The engine
now carries an injected map of command topic -> declared failure topic and, for
a malformed command on a mapped topic, returns a typed failure terminal with the
envelope's correlation and workflow ids, keeping the DLQ address on the result.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
import yaml

from omnibase_core.models.dispatch.model_dispatch_route import ModelDispatchRoute
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums.enum_dispatch_status import EnumDispatchStatus
from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.models.dispatch.model_dispatch_refusal import (
    ModelMalformedCommandTerminal,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

_TOPIC = "onex.cmd.test-service.do-thing.v1"
_FAILURE_TOPIC = "onex.evt.test-service.do-thing-failed.v1"
_ALIAS = "test-service.do-thing"


def _matcher_requires_prompt(payload: object) -> bool:
    return isinstance(payload, dict) and "prompt" in payload


def _engine(failure_topic: str | None) -> tuple[MessageDispatchEngine, list[object]]:
    engine = MessageDispatchEngine()
    seen: list[object] = []

    async def handler(envelope: ModelEventEnvelope[object]) -> None:
        seen.append(envelope)

    engine.register_dispatcher(
        dispatcher_id="d1",
        dispatcher=handler,
        category=EnumMessageCategory.COMMAND,
        message_types={_ALIAS},
        payload_type_matcher=_matcher_requires_prompt,
    )
    engine.register_route(
        ModelDispatchRoute(
            route_id="r1",
            topic_pattern=_TOPIC,
            message_category=EnumMessageCategory.COMMAND,
            dispatcher_id="d1",
        )
    )
    if failure_topic is not None:
        engine.failure_terminals.bind(command_topic=_TOPIC, failure_topic=failure_topic)
    engine.freeze()
    return engine, seen


def _envelope(payload: object) -> ModelEventEnvelope[object]:
    return ModelEventEnvelope(
        payload=payload, correlation_id=uuid4(), event_type=_ALIAS
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_malformed_command_emits_typed_failure_terminal() -> None:
    engine, seen = _engine(_FAILURE_TOPIC)
    envelope = _envelope({"task_type": "bogus", "workflow_id": "wf-77"})

    result = await engine.dispatch(_TOPIC, envelope)

    assert seen == []
    assert result.status == EnumDispatchStatus.NO_DISPATCHER
    assert result.dlq_topic  # the DLQ row is kept
    assert len(result.output_events) == 1
    terminal = result.output_events[0]
    assert isinstance(terminal, ModelMalformedCommandTerminal)
    assert terminal.topic == _FAILURE_TOPIC
    assert terminal.status == "failed"
    assert terminal.failure_class == "publisher_malformed"
    assert terminal.retryable is False
    assert terminal.correlation_id == envelope.correlation_id
    assert terminal.workflow_id == "wf-77"
    assert terminal.origin_topic == _TOPIC
    assert terminal.failure_reason


@pytest.mark.unit
@pytest.mark.asyncio
async def test_valid_command_unchanged() -> None:
    engine, seen = _engine(_FAILURE_TOPIC)

    result = await engine.dispatch(_TOPIC, _envelope({"prompt": "hi"}))

    assert len(seen) == 1
    assert result.status == EnumDispatchStatus.SUCCESS
    assert result.output_events == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_contract_without_failure_topic_unchanged() -> None:
    engine, seen = _engine(None)

    result = await engine.dispatch(_TOPIC, _envelope({"task_type": "bogus"}))

    assert seen == []
    assert result.status == EnumDispatchStatus.NO_DISPATCHER
    assert result.dlq_topic
    assert result.output_events == []


@pytest.mark.unit
def test_failure_terminal_registration_is_refused_after_freeze() -> None:
    engine, _ = _engine(None)
    with pytest.raises(Exception, match=r"frozen"):
        engine.failure_terminals.bind(
            command_topic=_TOPIC, failure_topic=_FAILURE_TOPIC
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_applier_publishes_refusal_terminal_to_declared_topic() -> None:
    from unittest.mock import AsyncMock

    from omnibase_infra.protocols import ProtocolEventBusLike
    from omnibase_infra.runtime.service_dispatch_result_applier import (
        DispatchResultApplier,
    )

    engine, _ = _engine(_FAILURE_TOPIC)
    envelope = _envelope({"workflow_id": "wf-1"})
    result = await engine.dispatch(_TOPIC, envelope)
    bus = AsyncMock(spec=ProtocolEventBusLike)
    applier = DispatchResultApplier(
        event_bus=bus,
        output_topic="onex.evt.test-service.do-thing-completed.v1",
        allowed_output_topics={_FAILURE_TOPIC},
    )

    await applier.apply(result, envelope.correlation_id)

    assert bus.publish_envelope.await_count == 1
    assert bus.publish_envelope.await_args.kwargs["topic"] == _FAILURE_TOPIC


_COMPLETED = "onex.evt.test-service.do-thing-completed.v1"
_CONTRACT = f"""
name: node_test_service_orchestrator
version: 1.0.0
node_type: orchestrator
runtime_dispatch:
  terminal_events:
    success: {_COMPLETED}
    failure: {_FAILURE_TOPIC}
event_bus:
  version: {{major: 1, minor: 0, patch: 0}}
  subscribe_topics:
    - {_TOPIC}
  publish_topics:
    - {_COMPLETED}
    - {_FAILURE_TOPIC}
terminal_event: {_COMPLETED}
""".strip()


def _wired_engine(tmp_path: Path, text: str) -> MessageDispatchEngine:
    from types import SimpleNamespace

    from omnibase_infra.models.dispatch.model_dispatch_refusal import (
        bind_contract_failure_terminal,
    )

    path = tmp_path / "contract.yaml"
    path.write_text(text, encoding="utf-8")
    declared = yaml.safe_load(text)["event_bus"]
    contract = SimpleNamespace(
        contract_path=path,
        terminal_event=yaml.safe_load(text).get("terminal_event"),
        event_bus=SimpleNamespace(
            subscribe_topics=tuple(declared["subscribe_topics"]),
            publish_topics=tuple(declared["publish_topics"]),
        ),
    )
    engine = MessageDispatchEngine()
    from omnibase_infra.runtime.contract_terminal_events import (
        declared_failure_terminal_topics,
    )

    bind_contract_failure_terminal(
        engine,
        contract,
        declared_failure_terminal_topics(
            path,
            success_topic=contract.terminal_event,
            publishable_topics=contract.event_bus.publish_topics,
        ),
    )
    return engine


@pytest.mark.unit
def test_wiring_binds_the_contracts_declared_failure_topic(tmp_path: Path) -> None:
    engine = _wired_engine(tmp_path, _CONTRACT)
    assert engine.failure_terminals._topics == {_TOPIC: _FAILURE_TOPIC}


@pytest.mark.unit
def test_wiring_binds_nothing_for_a_contract_with_no_failure_topic(
    tmp_path: Path,
) -> None:
    no_failure = _CONTRACT.replace(f"    failure: {_FAILURE_TOPIC}\n", "").replace(
        f"    - {_FAILURE_TOPIC}\n", ""
    )
    assert _FAILURE_TOPIC not in no_failure
    engine = _wired_engine(tmp_path, no_failure)
    assert engine.failure_terminals._topics == {}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_boundary_raise_marks_terminal_already_emitted() -> None:
    from omnibase_infra.runtime.auto_wiring.handler_wiring import (
        HandlerDispatchFailureError,
        _raise_if_no_dispatcher_drop,
    )

    engine, _ = _engine(_FAILURE_TOPIC)
    answered = await engine.dispatch(_TOPIC, _envelope({"x": 1}))
    with pytest.raises(HandlerDispatchFailureError) as answered_exc:
        _raise_if_no_dispatcher_drop(answered, _TOPIC)
    assert answered_exc.value.terminal_emitted is True

    plain_engine, _ = _engine(None)
    plain = await plain_engine.dispatch(_TOPIC, _envelope({"x": 1}))
    with pytest.raises(HandlerDispatchFailureError) as plain_exc:
        _raise_if_no_dispatcher_drop(plain, _TOPIC)
    assert plain_exc.value.terminal_emitted is False
