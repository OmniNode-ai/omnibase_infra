# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19972: the runtime dispatch port publishes a dispatched run's steps.

The deployed delegate-skill handler dispatches through this port, so a run that
goes through it must reach the run-trace projection: one event per step, on the
topic the delegate-skill contract declares as ``runtime_dispatch.run_step_event``,
numbered from one counter in the order the chain took the steps.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import UUID, uuid4

import pytest

from omnibase_core.models.dispatch.model_dispatch_bus_command import (
    ModelDispatchBusCommand,
)
from omnibase_core.models.dispatch.model_dispatch_bus_terminal_result import (
    ModelDispatchBusTerminalResult,
)
from omnibase_infra.models.delegation.model_delegation_run_step_event import (
    ModelDelegationRunStepEvent,
)
from omnibase_infra.protocols.protocol_pattern_b_broker_transport import (
    ProtocolPatternBBrokerTransport,
)
from omnibase_infra.runtime.runtime_local_ingress import (
    ModelRuntimeLocalIngressRoute,
    discover_runtime_local_ingress_routes,
)
from omnibase_infra.runtime.service_delegation_dispatch_port import (
    RuntimeDelegationDispatchPort,
)

pytestmark = pytest.mark.unit

_RUN_STEP_TOPIC = "onex.evt.omnimarket.delegation-run-step.v1"
_ENGINE_ALIAS = "omnimarket.node_delegation_orchestrator.delegation.orchestrate"


def _engine_route() -> ModelRuntimeLocalIngressRoute:
    return ModelRuntimeLocalIngressRoute(
        node_name="node_delegation_orchestrator",
        contract_name="node_delegation_orchestrator",
        command_topic="onex.cmd.omnibase-infra.delegation-request.v1",
        event_type="omnimarket.delegation-request",
        terminal_event="onex.evt.omnibase-infra.delegation-completed.v2",
        terminal_events=(
            "onex.evt.omnibase-infra.delegation-completed.v2",
            "onex.evt.omnibase-infra.delegation-failed-routed.v2",
        ),
        contract_path="/contracts/omnimarket/node_delegation_orchestrator.yaml",
        package_name="omnimarket",
    )


def _skill_route(run_step_event: str | None) -> ModelRuntimeLocalIngressRoute:
    return ModelRuntimeLocalIngressRoute(
        node_name="node_delegate_skill_orchestrator",
        contract_name="delegate_skill_orchestrator",
        command_topic="onex.cmd.omnimarket.delegate-skill.v1",
        event_type="omnimarket.delegate-skill",
        terminal_event="onex.evt.omnimarket.delegate-skill-completed.v1",
        terminal_events=("onex.evt.omnimarket.delegate-skill-completed.v1",),
        contract_path="/contracts/omnimarket/node_delegate_skill_orchestrator.yaml",
        package_name="omnimarket",
        run_step_event=run_step_event,
    )


class _RecordingBus:
    environment = "test"
    bootstrap_servers = "localhost:9092"

    def __init__(self) -> None:
        self.published: list[tuple[str, bytes | None, bytes]] = []

    async def publish(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: object = None,
    ) -> None:
        self.published.append((topic, key, value))


async def _dispatch(
    monkeypatch: pytest.MonkeyPatch,
    *,
    status: str,
    payload: dict[str, object] | None,
    tenant_id: str | None = "tenant-a",
    run_step_event: str | None = _RUN_STEP_TOPIC,
    wait: bool = True,
) -> tuple[_RecordingBus, UUID, dict[str, object]]:
    engine = _engine_route()

    class FakeBroker:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def dispatch_request(
            self, command: ModelDispatchBusCommand
        ) -> tuple[object, object]:
            return engine, ModelDispatchBusTerminalResult(
                correlation_id=command.correlation_id,
                status=status,
                payload=payload,
                completed_at=datetime.now(UTC),
            )

    monkeypatch.setattr(
        "omnibase_infra.runtime.service_delegation_dispatch_port.RuntimePatternBBroker",
        FakeBroker,
    )
    monkeypatch.setattr(
        "omnibase_infra.runtime.service_delegation_dispatch_port.resolve_bounded_delegation_route",
        lambda **_kwargs: None,
    )
    bus = _RecordingBus()
    port = RuntimeDelegationDispatchPort(
        cast("ProtocolPatternBBrokerTransport", bus),
        routes={
            _ENGINE_ALIAS: engine,
            "omnimarket.delegate_skill_orchestrator": _skill_route(run_step_event),
        },
    )
    correlation_id = uuid4()
    result = await port.dispatch(
        prompt="probe",
        task_type="document",
        correlation_id=correlation_id,
        max_tokens=None,
        source_file_path=None,
        source_session_id=None,
        wait=wait,
        tenant_id=tenant_id,
    )
    return bus, correlation_id, result


def _steps(bus: _RecordingBus) -> list[ModelDelegationRunStepEvent]:
    steps = []
    for topic, key, value in bus.published:
        assert topic == _RUN_STEP_TOPIC
        envelope = json.loads(value)
        assert envelope["event_type"] == "delegation.run_step"
        assert envelope["tenant_id"] == envelope["payload"]["tenant_id"]
        assert key == envelope["payload"]["correlation_id"].encode("utf-8")
        steps.append(ModelDelegationRunStepEvent.model_validate(envelope["payload"]))
    return steps


@pytest.mark.asyncio
async def test_an_escalated_run_publishes_every_step_in_the_order_the_chain_took_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus, correlation_id, result = await _dispatch(
        monkeypatch,
        status="completed",
        payload={
            "routing_disposition": "routed",
            "backend_ref": "backend-b",
            "model_used": "model-b",
            "cost_tier_name": "cheap_cloud",
            "latency_ms": 900,
            "escalation_history": [
                {
                    "backend_ref": "backend-a",
                    "model_used": "model-a",
                    "tier": "local",
                    "latency_ms": 400,
                    "acceptance_decision": "escalate",
                    "acceptance_reason": "quality_gate_rejection",
                    "failure_reasons": ["answer too short"],
                },
                {
                    "backend_ref": "backend-b",
                    "model_used": "model-b",
                    "tier": "cheap_cloud",
                    "latency_ms": 900,
                    "acceptance_decision": "accept",
                },
            ],
        },
    )

    assert result["status"] == "completed"
    steps = _steps(bus)
    assert [(s.step_sequence, s.step_kind, s.outcome) for s in steps] == [
        (1, "request", "succeeded"),
        (2, "routing_decision", "succeeded"),
        (3, "provider_call", "succeeded"),
        (4, "quality_gate", "failed"),
        (5, "routing_decision", "succeeded"),
        (6, "provider_call", "succeeded"),
        (7, "quality_gate", "succeeded"),
        (8, "terminal", "succeeded"),
    ]
    assert {s.tenant_id for s in steps} == {"tenant-a"}
    assert {s.correlation_id for s in steps} == {correlation_id}
    assert steps[0].reason == "document"
    assert (steps[1].backend_id, steps[1].model_id) == ("backend-a", "model-a")
    assert steps[1].reason == "first choice: tier local"
    assert steps[2].attempt_number == 1
    assert steps[2].duration_ms == 400
    assert steps[3].failure_cause == "quality_gate_rejection"
    assert steps[3].reason == "answer too short"
    assert steps[4].reason == (
        "tier cheap_cloud, after quality_gate_rejection on backend-a"
    )
    assert steps[5].attempt_number == 2
    assert steps[5].duration_ms == 900
    assert (steps[7].backend_id, steps[7].model_id) == ("backend-b", "model-b")
    assert steps[7].failing_step_sequence is None


@pytest.mark.asyncio
async def test_a_failed_run_terminal_names_the_step_that_failed_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus, _, _ = await _dispatch(
        monkeypatch,
        status="failed",
        payload={
            "routing_disposition": "routed",
            "backend_ref": "backend-a",
            "model_used": "model-a",
            "cost_tier_name": "local",
            "latency_ms": 120,
            "routed_failure_cause": {"kind": "provider"},
            "terminal_failure_reason": "connection refused",
        },
    )

    steps = _steps(bus)
    assert [(s.step_kind, s.outcome) for s in steps] == [
        ("request", "succeeded"),
        ("routing_decision", "succeeded"),
        ("provider_call", "failed"),
        ("terminal", "failed"),
    ]
    assert steps[2].failure_cause == "provider_call_failed"
    assert steps[2].reason == "connection refused"
    assert steps[3].failure_cause == "provider_call_failed"
    assert steps[3].failing_step_sequence == 3


@pytest.mark.asyncio
async def test_a_run_with_no_terminal_payload_publishes_its_request_and_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus, _, result = await _dispatch(monkeypatch, status="timeout", payload=None)

    assert result["status"] == "timeout"
    steps = _steps(bus)
    assert [(s.step_kind, s.outcome, s.failure_cause) for s in steps] == [
        ("request", "succeeded", None),
        ("terminal", "failed", "timeout"),
    ]


@pytest.mark.asyncio
async def test_an_unrouted_run_publishes_its_request_and_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus, _, _ = await _dispatch(
        monkeypatch,
        status="failed",
        payload={"routing_disposition": "unrouted", "unrouted_reason": "no_backend"},
    )

    steps = _steps(bus)
    assert [(s.step_kind, s.failure_cause) for s in steps] == [
        ("request", None),
        ("terminal", "no_backend"),
    ]


@pytest.mark.asyncio
async def test_the_terminal_tenant_is_used_when_the_request_carries_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus, _, _ = await _dispatch(
        monkeypatch,
        status="completed",
        payload={
            "tenant_id": "tenant-from-terminal",
            "routing_disposition": "routed",
            "backend_ref": "backend-a",
            "model_used": "model-a",
        },
        tenant_id=None,
    )

    assert {s.tenant_id for s in _steps(bus)} == {"tenant-from-terminal"}


@pytest.mark.asyncio
async def test_no_steps_are_published_without_a_tenant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus, _, result = await _dispatch(
        monkeypatch, status="timeout", payload=None, tenant_id=None
    )

    assert result["status"] == "timeout"
    assert bus.published == []


@pytest.mark.asyncio
async def test_no_steps_are_published_when_no_contract_declares_the_topic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus, _, result = await _dispatch(
        monkeypatch, status="timeout", payload=None, run_step_event=None
    )

    assert result["status"] == "timeout"
    assert bus.published == []


@pytest.mark.asyncio
async def test_a_fire_and_forget_dispatch_publishes_no_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus, _, _ = await _dispatch(monkeypatch, status="timeout", payload=None, wait=False)

    assert bus.published == []


@pytest.mark.asyncio
async def test_a_step_that_cannot_be_published_never_fails_the_delegation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def refuse(*_args: object, **_kwargs: object) -> None:
        raise ConnectionError("broker down")

    monkeypatch.setattr(_RecordingBus, "publish", refuse)
    _, _, result = await _dispatch(monkeypatch, status="timeout", payload=None)

    assert result["status"] == "timeout"


def test_route_discovery_reads_the_declared_run_step_topic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    node = tmp_path / "nodes" / "node_delegate_skill_orchestrator"
    node.mkdir(parents=True)
    (node / "contract.yaml").write_text(
        "name: delegate_skill_orchestrator\n"
        "event_bus:\n"
        "  subscribe_topics:\n"
        "    - onex.cmd.omnimarket.delegate-skill.v1\n"
        "runtime_dispatch:\n"
        "  command_topic: onex.cmd.omnimarket.delegate-skill.v1\n"
        "  terminal_events:\n"
        "    success: onex.evt.omnimarket.delegate-skill-completed.v1\n"
        f"  run_step_event: {_RUN_STEP_TOPIC}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "omnibase_infra.runtime.runtime_local_ingress._resolve_package_root",
        lambda _name: tmp_path,
    )

    routes = discover_runtime_local_ingress_routes(["omnimarket"])

    assert {route.run_step_event for route in routes.values()} == {_RUN_STEP_TOPIC}
