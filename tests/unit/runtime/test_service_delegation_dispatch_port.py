# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from omnibase_core.models.dispatch.model_dispatch_bus_command import (
    ModelDispatchBusCommand,
)
from omnibase_core.models.dispatch.model_dispatch_bus_terminal_result import (
    ModelDispatchBusTerminalResult,
)
from omnibase_infra.runtime.runtime_local_ingress import RuntimeLocalIngressRoute
from omnibase_infra.runtime.service_delegation_dispatch_port import (
    RuntimeDelegationDispatchPort,
    _normalize_result_payload,
    _select_delegation_route,
)

pytestmark = pytest.mark.unit


def _route(
    *,
    package_name: str,
    terminal_events: tuple[str, ...],
) -> RuntimeLocalIngressRoute:
    return RuntimeLocalIngressRoute(
        node_name="node_delegation_orchestrator",
        contract_name="node_delegation_orchestrator",
        command_topic=f"onex.cmd.{package_name}.delegation-request.v1",
        event_type=f"{package_name}.delegation-request",
        terminal_event=terminal_events[0] if terminal_events else None,
        terminal_events=terminal_events,
        contract_path=f"/contracts/{package_name}/node_delegation_orchestrator.yaml",
        package_name=package_name,
    )


def test_select_delegation_route_prefers_success_failure_terminal_interface() -> None:
    routes = {
        "omnimarket.node_delegation_orchestrator.delegation.orchestrate": _route(
            package_name="omnimarket",
            terminal_events=("onex.evt.omnimarket.delegation-completed.v1",),
        ),
        "omnibase_infra.node_delegation_orchestrator.delegation.orchestrate": _route(
            package_name="omnibase_infra",
            terminal_events=(
                "onex.evt.omnibase-infra.delegation-completed.v1",
                "onex.evt.omnibase-infra.delegation-failed.v1",
            ),
        ),
    }

    selected = _select_delegation_route(routes)

    assert (
        selected.alias
        == "omnibase_infra.node_delegation_orchestrator.delegation.orchestrate"
    )


def test_select_delegation_route_prefers_omnimarket_when_contracts_overlap() -> None:
    routes = {
        "omnibase_infra.node_delegation_orchestrator.delegation.orchestrate": _route(
            package_name="omnibase_infra",
            terminal_events=(
                "onex.evt.omnibase-infra.delegation-completed.v1",
                "onex.evt.omnibase-infra.delegation-failed.v1",
            ),
        ),
        "omnimarket.node_delegation_orchestrator.delegation.orchestrate": _route(
            package_name="omnimarket",
            terminal_events=(
                "onex.evt.omnibase-infra.delegation-completed.v1",
                "onex.evt.omnibase-infra.delegation-failed.v1",
            ),
        ),
    }

    selected = _select_delegation_route(routes)

    assert (
        selected.alias
        == "omnimarket.node_delegation_orchestrator.delegation.orchestrate"
    )


@pytest.mark.asyncio
async def test_runtime_delegation_dispatch_port_respects_dispatch_timeout_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = _route(
        package_name="omnimarket",
        terminal_events=(
            "onex.evt.omnibase-infra.delegation-completed.v1",
            "onex.evt.omnibase-infra.delegation-failed.v1",
        ),
    )
    captured_timeout_seconds: list[float] = []
    captured_payloads: list[dict[str, object]] = []

    class FakeBroker:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.args = _args
            self.kwargs = _kwargs

        async def dispatch_request(
            self, command: ModelDispatchBusCommand
        ) -> tuple[object, object]:
            await asyncio.sleep(0)
            captured_timeout_seconds.append(command.timeout_seconds)
            captured_payloads.append(dict(command.payload))
            return route, ModelDispatchBusTerminalResult(
                correlation_id=uuid4(),
                status="completed",
                payload={"content": "ok"},
                completed_at=datetime.now(UTC),
            )

    monkeypatch.setattr(
        "omnibase_infra.runtime.service_delegation_dispatch_port.RuntimePatternBBroker",
        FakeBroker,
    )
    port = RuntimeDelegationDispatchPort(
        event_bus=object(),  # type: ignore[arg-type]
        routes={
            "omnimarket.node_delegation_orchestrator.delegation.orchestrate": route
        },
    )

    await port.dispatch(
        prompt="probe",
        task_type="document",
        correlation_id=uuid4(),
        max_tokens=512,
        source_file_path=None,
        source_session_id=None,
        wait=True,
        output_schema_key=None,
    )

    assert captured_timeout_seconds == [600.0]
    assert captured_payloads[0]["prompt"] == "probe"
    assert captured_payloads[0]["task_type"] == "document"


def test_normalize_result_payload_flattens_delegation_event_shape() -> None:
    payload = {
        "topic": "onex.evt.omnibase-infra.delegation-completed.v1",
        "payload": {
            "model_used": "local-qwen-coder-30b",
            "content": "ok",
            "quality_passed": True,
            "prompt_tokens": 3,
            "completion_tokens": 4,
            "latency_ms": 125,
        },
    }

    normalized = _normalize_result_payload(
        status="completed",
        payload=payload,
        error_message=None,
    )

    assert normalized["status"] == "completed"
    assert normalized["content"] == "ok"
    assert normalized["model_name"] == "local-qwen-coder-30b"
    assert normalized["quality_gate_passed"] is True
    assert normalized["input_tokens"] == 3
    assert normalized["output_tokens"] == 4
    assert normalized["delegation_latency_ms"] == 125
