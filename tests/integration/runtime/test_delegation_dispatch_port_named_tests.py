# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Named-test compatibility at the runtime bus boundary (OMN-17292)."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import TypedDict
from uuid import uuid4

import pytest
from pydantic import BaseModel

from omnibase_core.models.delegation.wire import ModelDelegationRequest
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.runtime.protocols.protocol_delegation_dispatch_port import (
    ProtocolDelegationDispatchPort,
)
from omnibase_infra.runtime.runtime_local_ingress import ModelRuntimeLocalIngressRoute
from omnibase_infra.runtime.service_delegation_dispatch_port import (
    RuntimeDelegationDispatchPort,
)

pytestmark = pytest.mark.integration


class ModelConsumerNamedTestFile(BaseModel):
    """Consumer-owned test file shape without importing a higher-layer package."""

    path: str
    content: str


class ModelConsumerNamedTestSpec(BaseModel):
    """Consumer-owned spec shape without importing a higher-layer package."""

    workspace_root: str | None = None
    answer_path: str
    test_ids: tuple[str, ...]
    test_files: tuple[ModelConsumerNamedTestFile, ...] = ()


class NamedTestsKwargs(TypedDict, total=False):
    named_tests: None


@pytest.mark.parametrize(
    "dispatch_method",
    [ProtocolDelegationDispatchPort.dispatch, RuntimeDelegationDispatchPort.dispatch],
)
def test_named_tests_is_optional_on_both_runtime_interfaces(
    dispatch_method: Callable[..., Awaitable[dict[str, object]]],
) -> None:
    parameter = inspect.signature(dispatch_method).parameters["named_tests"]
    assert parameter.default is None
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.asyncio
@pytest.mark.parametrize("pass_none", [False, True], ids=["omitted", "explicit-none"])
async def test_absent_named_tests_dispatch_through_the_real_runtime_bus(
    pass_none: bool,
) -> None:
    """Omission and explicit None preserve the consumer-decodable wire request."""
    route = ModelRuntimeLocalIngressRoute(
        node_name="node_delegation_orchestrator",
        contract_name="node_delegation_orchestrator",
        command_topic="onex.cmd.omnimarket.delegation-request.v1",
        event_type="omnimarket.delegation-request",
        terminal_event="onex.evt.omnimarket.delegation-completed.v1",
        terminal_events=(
            "onex.evt.omnimarket.delegation-completed.v1",
            "onex.evt.omnimarket.delegation-failed.v1",
        ),
        contract_path="/contracts/omnimarket/node_delegation_orchestrator/contract.yaml",
        package_name="omnimarket",
    )
    bus = EventBusInmemory(environment="test", group="named-tests-compat")
    await bus.start()
    payloads: list[dict[str, object]] = []

    async def consumer(message: ModelEventMessage) -> None:
        envelope = ModelEventEnvelope[dict[str, object]].model_validate_json(
            message.value
        )
        payloads.append(envelope.payload)
        request = ModelDelegationRequest.model_validate(envelope.payload)
        terminal = ModelEventEnvelope[dict[str, object]](
            payload={"content": "runtime-ok", "quality_passed": True},
            correlation_id=request.correlation_id,
            envelope_timestamp=datetime.now(UTC),
            event_type=route.terminal_events[0],
            source_tool="node_delegation_orchestrator",
        )
        await bus.publish(
            route.terminal_events[0], None, terminal.model_dump_json().encode(), None
        )

    unsubscribe = await bus.subscribe(
        route.command_topic, group_id="named-tests-consumer", on_message=consumer
    )
    port = RuntimeDelegationDispatchPort(
        event_bus=bus, routes={"delegation.orchestrate": route}
    )
    optional_kwargs: NamedTestsKwargs = {"named_tests": None} if pass_none else {}
    try:
        result = await port.dispatch(
            prompt="runtime probe",
            task_type="summarization",
            correlation_id=uuid4(),
            max_tokens=None,
            source_file_path=None,
            source_session_id=None,
            wait=True,
            execution_timeout_seconds=5,
            terminal_delivery_margin_seconds=1,
            **optional_kwargs,
        )
    finally:
        await unsubscribe()
        await bus.close()

    assert result["status"] == "completed"
    assert len(payloads) == 1
    assert "named_tests" not in payloads[0]


_ROUTE = ModelRuntimeLocalIngressRoute(
    node_name="node_delegation_orchestrator",
    contract_name="node_delegation_orchestrator",
    command_topic="onex.cmd.omnimarket.delegation-request.v1",
    event_type="omnimarket.delegation-request",
    terminal_event="onex.evt.omnimarket.delegation-completed.v1",
    terminal_events=(
        "onex.evt.omnimarket.delegation-completed.v1",
        "onex.evt.omnimarket.delegation-failed.v1",
    ),
    contract_path="/contracts/omnimarket/node_delegation_orchestrator/contract.yaml",
    package_name="omnimarket",
)


@pytest.mark.asyncio
async def test_inline_named_tests_ride_the_bus_command_payload() -> None:
    """OMN-17427: the deployed lane carries the tests a code answer is graded on."""
    bus = EventBusInmemory(environment="test", group="named-tests-carry")
    await bus.start()
    payloads: list[dict[str, object]] = []

    async def consumer(message: ModelEventMessage) -> None:
        envelope = ModelEventEnvelope[dict[str, object]].model_validate_json(
            message.value
        )
        payloads.append(envelope.payload)
        terminal = ModelEventEnvelope[dict[str, object]](
            payload={"content": "runtime-ok", "quality_passed": True},
            correlation_id=envelope.correlation_id,
            envelope_timestamp=datetime.now(UTC),
            event_type=_ROUTE.terminal_events[0],
            source_tool="node_delegation_orchestrator",
        )
        await bus.publish(
            _ROUTE.terminal_events[0], None, terminal.model_dump_json().encode(), None
        )

    unsubscribe = await bus.subscribe(
        _ROUTE.command_topic, group_id="named-tests-carry", on_message=consumer
    )
    port = RuntimeDelegationDispatchPort(
        event_bus=bus, routes={"delegation.orchestrate": _ROUTE}
    )
    try:
        result = await port.dispatch(
            prompt="Write add.",
            task_type="code_generation",
            correlation_id=uuid4(),
            max_tokens=None,
            source_file_path=None,
            source_session_id=None,
            wait=True,
            execution_timeout_seconds=5,
            terminal_delivery_margin_seconds=1,
            named_tests=ModelConsumerNamedTestSpec(
                answer_path="answer.py",
                test_ids=("test_answer.py::test_adds",),
                test_files=(
                    ModelConsumerNamedTestFile(
                        path="test_answer.py", content="def test_adds(): ...\n"
                    ),
                ),
            ),
        )
    finally:
        await unsubscribe()
        await bus.close()

    assert result["status"] == "completed"
    assert len(payloads) == 1
    assert payloads[0]["named_tests"] == {
        "test_ids": ["test_answer.py::test_adds"],
        "answer_path": "answer.py",
        "test_files": [{"path": "test_answer.py", "content": "def test_adds(): ...\n"}],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "spec",
    [
        ModelConsumerNamedTestSpec(
            workspace_root="/consumer-workspace",
            answer_path="answer.py",
            test_ids=("tests/test_answer.py::test_answer",),
            test_files=(ModelConsumerNamedTestFile(path="t.py", content="x"),),
        ),
        ModelConsumerNamedTestSpec(
            answer_path="answer.py", test_ids=("tests/test_answer.py::test_answer",)
        ),
    ],
    ids=["workspace-path", "no-inline-files"],
)
async def test_named_tests_the_bus_cannot_carry_are_refused_before_publication(
    spec: ModelConsumerNamedTestSpec,
) -> None:
    """The lane cannot see a caller's workspace: tests it cannot run are not sent."""
    # No routes and an unstarted transport: reaching route resolution would fail
    # before the expected explanation.
    bus = EventBusInmemory(environment="test", group="named-tests-refusal")
    port = RuntimeDelegationDispatchPort(event_bus=bus, routes={})
    with pytest.raises(ValueError, match=r"--test-file"):
        await port.dispatch(
            prompt="code probe",
            task_type="code_generation",
            correlation_id=uuid4(),
            max_tokens=None,
            source_file_path=None,
            source_session_id=None,
            wait=True,
            named_tests=spec,
        )
