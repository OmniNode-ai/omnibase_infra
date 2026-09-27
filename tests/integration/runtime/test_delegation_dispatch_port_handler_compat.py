# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Consumer-handler compatibility for the runtime-owned delegation port."""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from hashlib import sha256
from uuid import uuid4

import pytest

from omnibase_core.enums.enum_delegation_traffic_class import (
    EnumDelegationTrafficClass,
)
from omnibase_core.models.delegation.wire import (
    ModelDelegationProvenance,
    ModelDelegationRequest,
)
from omnibase_core.models.dispatch.model_dispatch_bus_command import (
    ModelDispatchBusCommand,
)
from omnibase_core.models.dispatch.model_dispatch_bus_terminal_result import (
    ModelDispatchBusTerminalResult,
)
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


def _delegation_route() -> ModelRuntimeLocalIngressRoute:
    return ModelRuntimeLocalIngressRoute(
        node_name="node_delegation_orchestrator",
        contract_name="node_delegation_orchestrator",
        command_topic="onex.cmd.omnibase-infra.delegation-request.v1",
        event_type="omnibase-infra.delegation-request",
        terminal_event="onex.evt.omnibase-infra.delegation-completed.v1",
        terminal_events=(
            "onex.evt.omnibase-infra.delegation-completed.v1",
            "onex.evt.omnibase-infra.delegation-failed.v1",
        ),
        contract_path="/contracts/omnimarket/node_delegation_orchestrator/contract.yaml",
        package_name="omnimarket",
    )


def test_runtime_port_exposes_consumer_handler_optional_parameters() -> None:
    """The injected implementation and its protocol evolve as one boundary."""
    for dispatch_method in (
        ProtocolDelegationDispatchPort.dispatch,
        RuntimeDelegationDispatchPort.dispatch,
    ):
        parameters = inspect.signature(dispatch_method).parameters
        assert parameters["max_tokens"].annotation in {"int | None", int | None}
        assert parameters["execution_timeout_seconds"].annotation in {"int", int}
        assert parameters["terminal_delivery_margin_seconds"].annotation in {
            "int",
            int,
        }
        assert parameters["backend_id"].default is None
        assert parameters["response_contract"].default is None
        assert parameters["system_prompt"].default is None
        assert parameters["temperature"].default is None
        assert parameters["response_format"].default is None
        # OMN-18321: added by OMN-18172 on the consumer side (omnimarket#2494,
        # squash 849fdae6) and not here, which took every dev-lane delegation to
        # a failed terminal with no FSM row for a day. The name is asserted
        # explicitly AND derived from the consumer's own declaration by
        # test_delegation_dispatch_port_consumer_kwarg_parity.py -- this line
        # pins the one that already cost an outage, that file catches the next.
        assert parameters["provenance"].default is None


async def _dispatch_with_captured_command(
    monkeypatch: pytest.MonkeyPatch,
    *,
    response_contract: dict[str, object] | None,
) -> tuple[dict[str, object], ModelDispatchBusCommand]:
    """Capture the exact command Pattern-B publishes through the runtime port."""
    route = _delegation_route()
    captured_commands: list[ModelDispatchBusCommand] = []

    class FakePatternBBroker:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def dispatch_request(
            self, command: ModelDispatchBusCommand
        ) -> tuple[ModelRuntimeLocalIngressRoute, ModelDispatchBusTerminalResult]:
            captured_commands.append(command)
            return route, ModelDispatchBusTerminalResult(
                correlation_id=command.correlation_id,
                status="completed",
                payload={"content": "workflow-ok"},
                completed_at=datetime.now(UTC),
            )

    monkeypatch.setattr(
        "omnibase_infra.runtime.service_delegation_dispatch_port.RuntimePatternBBroker",
        FakePatternBBroker,
    )
    port = RuntimeDelegationDispatchPort(
        event_bus=object(),  # type: ignore[arg-type]
        routes={"delegation.orchestrate": route},
    )
    result = await port.dispatch(
        prompt="workflow probe",
        task_type="reasoning",
        correlation_id=uuid4(),
        max_tokens=None,
        source_file_path=None,
        source_session_id=None,
        wait=True,
        execution_timeout_seconds=240,
        terminal_delivery_margin_seconds=60,
        quality_contract_mode="extend_task_class",
        acceptance_criteria=(),
        tenant_id=None,
        backend_id=None,
        response_contract=response_contract,
        system_prompt=None,
        temperature=None,
        response_format=None,
    )
    return result, captured_commands[0]


@pytest.mark.asyncio
async def test_absent_consumer_features_dispatch_through_runtime_bus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Always-supplied None kwargs must reach the existing Pattern-B route."""
    result, command = await _dispatch_with_captured_command(
        monkeypatch, response_contract=None
    )

    assert result["status"] == "completed"
    assert command.payload["prompt"] == "workflow probe"
    assert "max_tokens" not in command.payload
    assert "backend_id" not in command.payload
    assert "response_contract" not in command.payload
    assert "system_prompt" not in command.payload
    assert "temperature" not in command.payload
    assert "response_format" not in command.payload
    request = ModelDelegationRequest.model_validate(command.payload)
    assert request.requested_timeout_seconds == 240
    assert "terminal_delivery_margin_seconds" not in command.payload


@pytest.mark.asyncio
async def test_runtime_bus_publishes_declared_contract_to_the_core_wire_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A nested schema survives publication and validates in the consumer model."""
    response_contract: dict[str, object] = {
        "type": "object",
        "properties": {
            "result": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "sources": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["summary"],
                "additionalProperties": False,
            }
        },
        "required": ["result"],
        "additionalProperties": False,
    }
    result, command = await _dispatch_with_captured_command(
        monkeypatch, response_contract=response_contract
    )

    published_wire_request = ModelDelegationRequest.model_validate(command.payload)
    assert result["status"] == "completed"
    assert command.payload["response_contract"] == response_contract
    assert published_wire_request.response_contract == response_contract
    assert published_wire_request.requested_timeout_seconds == 240
    assert "terminal_delivery_margin_seconds" not in command.payload
    assert command.timeout_seconds == 300.0


@pytest.mark.asyncio
async def test_metered_terminal_cost_crosses_the_runtime_consumer_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A metered workflow terminal reaches the consumer as measured actual cost."""
    route = _delegation_route()

    class FakePatternBBroker:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def dispatch_request(
            self, command: ModelDispatchBusCommand
        ) -> tuple[ModelRuntimeLocalIngressRoute, ModelDispatchBusTerminalResult]:
            return route, ModelDispatchBusTerminalResult(
                correlation_id=command.correlation_id,
                status="completed",
                payload={
                    "model_used": "gemini-2.5-flash",
                    "prompt_tokens": 115,
                    "completion_tokens": 130,
                    "final_attempt_cost": 0.00137,
                    "cumulative_attempt_cost": 0.00182,
                },
                completed_at=datetime.now(UTC),
            )

    monkeypatch.setattr(
        "omnibase_infra.runtime.service_delegation_dispatch_port.RuntimePatternBBroker",
        FakePatternBBroker,
    )
    port = RuntimeDelegationDispatchPort(
        event_bus=object(),  # type: ignore[arg-type]
        routes={"delegation.orchestrate": route},
    )

    result = await port.dispatch(
        prompt="metered workflow probe",
        task_type="reasoning",
        correlation_id=uuid4(),
        max_tokens=None,
        source_file_path=None,
        source_session_id=None,
        wait=True,
        execution_timeout_seconds=240,
        terminal_delivery_margin_seconds=60,
        quality_contract_mode="extend_task_class",
        acceptance_criteria=(),
        tenant_id=None,
        backend_id=None,
        response_contract=None,
    )

    assert result["cost_usd"] == pytest.approx(0.00182)


@pytest.mark.asyncio
async def test_provenance_reaches_the_published_dispatch_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OMN-18321/OMN-18172: provenance is carried onto the wire, not absorbed.

    Accepting the keyword and dropping it would turn a loud TypeError into a
    silent classification hole -- the exact silent-drop defect OMN-18172 exists
    to close. The assertion is on the payload the bus path actually publishes.
    """
    route = _delegation_route()
    captured_commands: list[ModelDispatchBusCommand] = []

    class FakePatternBBroker:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def dispatch_request(
            self, command: ModelDispatchBusCommand
        ) -> tuple[ModelRuntimeLocalIngressRoute, ModelDispatchBusTerminalResult]:
            captured_commands.append(command)
            return route, ModelDispatchBusTerminalResult(
                correlation_id=command.correlation_id,
                status="completed",
                payload={"content": "alive"},
                completed_at=datetime.now(UTC),
            )

    monkeypatch.setattr(
        "omnibase_infra.runtime.service_delegation_dispatch_port.RuntimePatternBBroker",
        FakePatternBBroker,
    )
    port = RuntimeDelegationDispatchPort(
        event_bus=object(),  # type: ignore[arg-type]
        routes={"delegation.orchestrate": route},
    )

    provenance = ModelDelegationProvenance(
        source="external-client",
        traffic_class=EnumDelegationTrafficClass.SYNTHETIC,
        source_surface="scheduled-chain-canary",
        requested_by="chain-canary",
    )

    result = await port.dispatch(
        prompt="Reply with the single word: alive.",
        task_type="test",
        correlation_id=uuid4(),
        max_tokens=32,
        source_file_path=None,
        source_session_id=None,
        wait=True,
        execution_timeout_seconds=240,
        terminal_delivery_margin_seconds=60,
        quality_contract_mode="extend_task_class",
        acceptance_criteria=(),
        tenant_id=None,
        provenance=provenance,
        backend_id=None,
        response_contract=None,
        system_prompt=None,
        temperature=None,
        response_format=None,
    )

    assert result["status"] == "completed"
    published = captured_commands[0].payload["provenance"]
    assert published == provenance.model_dump(mode="json")
    assert published["traffic_class"] == EnumDelegationTrafficClass.SYNTHETIC.value
    assert published["source_surface"] == "scheduled-chain-canary"


@pytest.mark.asyncio
async def test_absent_provenance_leaves_no_key_on_the_dispatch_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """None stays absent rather than becoming a synthetic default.

    OMN-18172 is explicit that an absent provenance is unclassified, never
    synthetic. A null on the wire invites a consumer to read it as a value.
    """
    route = _delegation_route()
    captured_commands: list[ModelDispatchBusCommand] = []

    class FakePatternBBroker:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def dispatch_request(
            self, command: ModelDispatchBusCommand
        ) -> tuple[ModelRuntimeLocalIngressRoute, ModelDispatchBusTerminalResult]:
            captured_commands.append(command)
            return route, ModelDispatchBusTerminalResult(
                correlation_id=command.correlation_id,
                status="completed",
                payload={"content": "alive"},
                completed_at=datetime.now(UTC),
            )

    monkeypatch.setattr(
        "omnibase_infra.runtime.service_delegation_dispatch_port.RuntimePatternBBroker",
        FakePatternBBroker,
    )
    port = RuntimeDelegationDispatchPort(
        event_bus=object(),  # type: ignore[arg-type]
        routes={"delegation.orchestrate": route},
    )

    await port.dispatch(
        prompt="unclassified probe",
        task_type="test",
        correlation_id=uuid4(),
        max_tokens=None,
        source_file_path=None,
        source_session_id=None,
        wait=True,
        execution_timeout_seconds=240,
        terminal_delivery_margin_seconds=60,
        quality_contract_mode="extend_task_class",
        acceptance_criteria=(),
        tenant_id=None,
        provenance=None,
    )

    assert "provenance" not in captured_commands[0].payload


# ---------------------------------------------------------------------------
# OMN-18929 (K2): the response contract across the REAL Pattern-B broker.
#
# The tests above replace the broker with a fake and read the in-memory
# command, so they never see the bytes the bus carries. The tests below run
# the port through the real ``RuntimePatternBBroker`` on the in-memory bus.
# The stand-in consumer decodes the published BYTES into the Core wire model
# that the omnimarket delegation orchestrator validates, which is the decode
# boundary the deployed consumer applies. It answers on the route's own
# terminal topics. The consumer's own verdict logic (the quality gate that
# routes a contract violation to the failed terminal) is proved in omnimarket's
# seam golden. These tests prove what the dispatch seam does with the contract
# and with each kind of terminal.
# ---------------------------------------------------------------------------

_K2_RESPONSE_CONTRACT: dict[str, object] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "labels": {"type": "array", "items": {"enum": ["bug", "feature"]}},
    },
    "required": ["summary", "labels"],
    "additionalProperties": False,
}

_ConsumerReply = Callable[[ModelDelegationRequest], tuple[str, dict[str, object]]]


def _canonical_contract_sha256(contract: dict[str, object]) -> str:
    """Hash a contract as canonical JSON, the identity both boundaries compare."""
    return sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


async def _dispatch_through_real_broker(
    *,
    response_contract: dict[str, object] | None,
    reply: _ConsumerReply | None,
    execution_timeout_seconds: int = 5,
    terminal_delivery_margin_seconds: int = 1,
) -> tuple[dict[str, object], list[bytes], list[ModelDelegationRequest]]:
    """Dispatch through the port and the real broker, with a byte-level consumer.

    ``reply`` receives the consumer-decoded request and returns the terminal
    topic and payload to publish. ``None`` means no consumer answers, which is
    the absent-terminal-carrier case. The default budget is deliberately short:
    a consumer that cannot decode the bytes publishes nothing, and the test must
    then fail in seconds rather than wait out the production 300-second window.
    """
    route = _delegation_route()
    bus = EventBusInmemory(environment="test", group="k2-response-contract")
    await bus.start()
    published_bytes: list[bytes] = []
    decoded_requests: list[ModelDelegationRequest] = []

    async def consumer(message: ModelEventMessage) -> None:
        published_bytes.append(message.value)
        envelope = ModelEventEnvelope[object].model_validate_json(message.value)
        request = ModelDelegationRequest.model_validate(envelope.payload)
        decoded_requests.append(request)
        if reply is None:
            return
        topic, payload = reply(request)
        terminal = ModelEventEnvelope[object](
            payload=payload,
            correlation_id=envelope.correlation_id,
            envelope_timestamp=datetime.now(UTC),
            event_type=topic,
            source_tool="node_delegation_orchestrator",
        )
        await bus.publish(topic, None, terminal.model_dump_json().encode("utf-8"), None)

    unsubscribe: Callable[[], Awaitable[None]] = await bus.subscribe(
        route.command_topic, group_id="k2-consumer", on_message=consumer
    )
    port = RuntimeDelegationDispatchPort(
        event_bus=bus,
        routes={"delegation.orchestrate": route},
    )
    try:
        result = await port.dispatch(
            prompt="Classify the changelog entry.",
            task_type="summarization",
            correlation_id=uuid4(),
            max_tokens=None,
            source_file_path=None,
            source_session_id=None,
            wait=True,
            execution_timeout_seconds=execution_timeout_seconds,
            terminal_delivery_margin_seconds=terminal_delivery_margin_seconds,
            quality_contract_mode="extend_task_class",
            acceptance_criteria=(),
            tenant_id=None,
            backend_id=None,
            response_contract=response_contract,
            system_prompt=None,
            temperature=None,
            response_format=None,
        )
    finally:
        await unsubscribe()
        await bus.close()
    return result, published_bytes, decoded_requests


def _completed_reply(
    request: ModelDelegationRequest,
) -> tuple[str, dict[str, object]]:
    return _delegation_route().terminal_events[0], {
        "correlation_id": str(request.correlation_id),
        "content": '{"summary": "ok", "labels": ["bug"]}',
        "quality_passed": True,
    }


@pytest.mark.asyncio
async def test_declared_contract_hash_is_identical_at_dispatch_and_consumer_decode() -> (
    None
):
    """AC1: the contract the caller declared is the contract the consumer decodes.

    The comparison is made three ways, all against the caller's value: the raw
    published bytes, the Core wire model decoded from those bytes, and that
    model's re-serialization.
    """
    dispatch_hash = _canonical_contract_sha256(_K2_RESPONSE_CONTRACT)

    result, published_bytes, decoded = await _dispatch_through_real_broker(
        response_contract=_K2_RESPONSE_CONTRACT, reply=_completed_reply
    )

    assert result["status"] == "completed"
    assert len(published_bytes) == 1
    raw_envelope = json.loads(published_bytes[0])
    assert (
        _canonical_contract_sha256(raw_envelope["payload"]["response_contract"])
        == dispatch_hash
    )
    (consumer_request,) = decoded
    assert consumer_request.response_contract is not None
    assert (
        _canonical_contract_sha256(consumer_request.response_contract) == dispatch_hash
    )
    redumped = consumer_request.model_dump(mode="json")["response_contract"]
    assert _canonical_contract_sha256(redumped) == dispatch_hash


@pytest.mark.asyncio
async def test_no_contract_request_stays_accepted_through_real_broker() -> None:
    """Positive control: the supported no-contract request still round-trips."""
    result, published_bytes, decoded = await _dispatch_through_real_broker(
        response_contract=None, reply=_completed_reply
    )

    assert result["status"] == "completed"
    assert "response_contract" not in json.loads(published_bytes[0])["payload"]
    (consumer_request,) = decoded
    assert consumer_request.response_contract is None


@pytest.mark.asyncio
async def test_contract_violation_on_failed_terminal_reaches_caller_as_failed() -> None:
    """AC2: a malformed result the consumer refuses stays a failure at the caller."""
    violation = (
        "MALFORMED: response violates the declared response_contract: "
        "'labels' is a required property"
    )

    def refuse(request: ModelDelegationRequest) -> tuple[str, dict[str, object]]:
        return _delegation_route().terminal_events[1], {
            "correlation_id": str(request.correlation_id),
            "content": '{"summary": "ok"}',
            "quality_passed": False,
            "failure_reason": violation,
        }

    result, _published, decoded = await _dispatch_through_real_broker(
        response_contract=_K2_RESPONSE_CONTRACT, reply=refuse
    )

    assert decoded[0].response_contract == _K2_RESPONSE_CONTRACT
    assert result["status"] == "failed"
    assert result["quality_gate_passed"] is False
    assert result["error_message"] == violation


@pytest.mark.asyncio
async def test_failure_verdict_on_the_success_topic_is_never_reported_completed() -> (
    None
):
    """AC2 backstop: a failure verdict mis-routed to the success topic stays failed."""

    def misroute(request: ModelDelegationRequest) -> tuple[str, dict[str, object]]:
        return _delegation_route().terminal_events[0], {
            "correlation_id": str(request.correlation_id),
            "status": "failed",
            "content": '{"summary": "ok"}',
            "quality_passed": False,
            "failure_reason": "MALFORMED: response violates the declared contract",
        }

    result, _published, _decoded = await _dispatch_through_real_broker(
        response_contract=_K2_RESPONSE_CONTRACT, reply=misroute
    )

    assert result["status"] == "failed"
    assert result["quality_gate_passed"] is False


@pytest.mark.asyncio
async def test_absent_terminal_carrier_fails_visibly_not_as_success() -> None:
    """A request no consumer answers ends as a visible timeout, never success."""
    result, published_bytes, decoded = await _dispatch_through_real_broker(
        response_contract=_K2_RESPONSE_CONTRACT,
        reply=None,
        execution_timeout_seconds=1,
        terminal_delivery_margin_seconds=1,
    )

    assert len(published_bytes) == 1
    assert decoded[0].response_contract == _K2_RESPONSE_CONTRACT
    assert result["status"] == "timeout"
    assert result["quality_gate_passed"] is False
    assert "Timed out waiting" in str(result["error_message"])
