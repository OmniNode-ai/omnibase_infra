# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A malformed command reaches the contract's declared failure topic end to end.

Wires the real contract file on disk through ``bind_contract_failure_terminal``
into a real ``MessageDispatchEngine``, dispatches a malformed command, and
applies the result through the real ``DispatchResultApplier`` onto a recording
bus. Known bad: one envelope on the declared failure topic. Known good: a valid
command publishes nothing on it.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
import yaml

from omnibase_core.models.dispatch.model_dispatch_route import ModelDispatchRoute
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums.enum_dispatch_status import EnumDispatchStatus
from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.models.dispatch.model_dispatch_refusal import (
    bind_contract_failure_terminal,
)
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.protocols import ProtocolEventBusLike
from omnibase_infra.runtime.contract_terminal_events import (
    declared_failure_terminal_topics,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
from omnibase_infra.runtime.service_dispatch_result_applier import (
    DispatchResultApplier,
)

_CMD = "onex.cmd.test-service.do-thing.v1"
_OK = "onex.evt.test-service.do-thing-completed.v1"
_FAIL = "onex.evt.test-service.do-thing-failed.v1"
_ALIAS = "test-service.do-thing"
_CONTRACT = f"""
name: node_test_service_orchestrator
version: 1.0.0
node_type: orchestrator
runtime_dispatch:
  terminal_events:
    success: {_OK}
    failure: {_FAIL}
event_bus:
  version: {{major: 1, minor: 0, patch: 0}}
  subscribe_topics:
    - {_CMD}
  publish_topics:
    - {_OK}
    - {_FAIL}
terminal_event: {_OK}
""".strip()


def _wired_engine(tmp_path: Path) -> MessageDispatchEngine:
    path = tmp_path / "contract.yaml"
    path.write_text(_CONTRACT, encoding="utf-8")
    declared = yaml.safe_load(_CONTRACT)
    contract = SimpleNamespace(
        contract_path=path,
        terminal_event=declared["terminal_event"],
        event_bus=SimpleNamespace(
            subscribe_topics=tuple(declared["event_bus"]["subscribe_topics"]),
            publish_topics=tuple(declared["event_bus"]["publish_topics"]),
        ),
    )
    engine = MessageDispatchEngine()
    bind_contract_failure_terminal(
        engine,
        contract,
        declared_failure_terminal_topics(
            path,
            success_topic=contract.terminal_event,
            publishable_topics=contract.event_bus.publish_topics,
        ),
    )

    async def handler(envelope: ModelEventEnvelope[object]) -> None:
        return None

    engine.register_dispatcher(
        dispatcher_id="d1",
        dispatcher=handler,
        category=EnumMessageCategory.COMMAND,
        message_types={_ALIAS},
        payload_type_matcher=lambda p: isinstance(p, dict) and "prompt" in p,
    )
    engine.register_route(
        ModelDispatchRoute(
            route_id="r1",
            topic_pattern=_CMD,
            message_category=EnumMessageCategory.COMMAND,
            dispatcher_id="d1",
        )
    )
    engine.freeze()
    return engine


async def _run(
    tmp_path: Path, payload: dict[str, object]
) -> tuple[ModelDispatchResult, list[str]]:
    engine = _wired_engine(tmp_path)
    envelope = ModelEventEnvelope(
        payload=payload, correlation_id=uuid4(), event_type=_ALIAS
    )
    result = await engine.dispatch(_CMD, envelope)
    bus = AsyncMock(spec=ProtocolEventBusLike)
    applier = DispatchResultApplier(
        event_bus=bus, output_topic=_OK, allowed_output_topics={_OK, _FAIL}
    )
    await applier.apply(result, envelope.correlation_id)
    return result, [c.kwargs["topic"] for c in bus.publish_envelope.await_args_list]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_malformed_command_publishes_on_declared_failure_topic(
    tmp_path: Path,
) -> None:
    result, topics = await _run(tmp_path, {"task_type": "bogus"})

    assert result.status == EnumDispatchStatus.NO_DISPATCHER
    assert topics == [_FAIL]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_valid_command_publishes_nothing_on_failure_topic(
    tmp_path: Path,
) -> None:
    result, topics = await _run(tmp_path, {"prompt": "hi"})

    assert result.status == EnumDispatchStatus.SUCCESS
    assert _FAIL not in topics
