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
from omnibase_infra.errors import InfraUnavailableError
from omnibase_infra.runtime.runtime_local_ingress import ModelRuntimeLocalIngressRoute
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
    command_topic: str | None = None,
    contract_name: str = "node_delegation_orchestrator",
    contract_path: str | None = None,
) -> ModelRuntimeLocalIngressRoute:
    return ModelRuntimeLocalIngressRoute(
        node_name=contract_name,
        contract_name=contract_name,
        command_topic=(
            command_topic
            if command_topic is not None
            else f"onex.cmd.{package_name}.delegation-request.v1"
        ),
        event_type=f"{package_name}.delegation-request",
        terminal_event=terminal_events[0] if terminal_events else None,
        terminal_events=terminal_events,
        contract_path=(
            contract_path
            if contract_path is not None
            else f"/contracts/{package_name}/node_delegation_orchestrator.yaml"
        ),
        package_name=package_name,
    )


def test_select_delegation_route_binds_omnimarket_only() -> None:
    """Resolution binds the omnimarket route and ignores the infra surface.

    OMN-13547: the empty infra shell was deleted; resolution must bind
    omnimarket regardless of whether a (now-impossible) infra route would
    otherwise satisfy the terminal interface.
    """
    routes = {
        "omnimarket.node_delegation_orchestrator.delegation.orchestrate": _route(
            package_name="omnimarket",
            terminal_events=(
                "onex.evt.omnimarket.delegation-completed.v1",
                "onex.evt.omnimarket.delegation-failed.v1",
            ),
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
        == "omnimarket.node_delegation_orchestrator.delegation.orchestrate"
    )
    assert selected.route.package_name == "omnimarket"


def test_select_delegation_route_fails_closed_when_only_infra_route_present() -> None:
    """No omnimarket engine -> fail closed; never resolve a non-omnimarket route.

    OMN-13547: there is no infra-local fallback. A residual infra-shaped route
    must NOT be selected; the resolver raises a typed InfraUnavailableError.
    """
    routes = {
        "omnibase_infra.node_delegation_orchestrator.delegation.orchestrate": _route(
            package_name="omnibase_infra",
            terminal_events=(
                "onex.evt.omnibase-infra.delegation-completed.v1",
                "onex.evt.omnibase-infra.delegation-failed.v1",
            ),
        ),
    }

    with pytest.raises(InfraUnavailableError, match="No omnimarket delegation engine"):
        _select_delegation_route(routes)


def test_select_delegation_route_fails_closed_when_no_route_present() -> None:
    """Empty route map -> fail closed (omnimarket package absent)."""
    with pytest.raises(InfraUnavailableError, match="No omnimarket delegation engine"):
        _select_delegation_route({})


def test_select_delegation_route_resolves_single_omnimarket_route() -> None:
    routes = {
        "omnimarket.node_delegation_orchestrator.delegation.orchestrate": _route(
            package_name="omnimarket",
            terminal_events=(
                "onex.evt.omnimarket.delegation-completed.v1",
                "onex.evt.omnimarket.delegation-failed.v1",
            ),
        ),
    }

    selected = _select_delegation_route(routes)

    assert (
        selected.alias
        == "omnimarket.node_delegation_orchestrator.delegation.orchestrate"
    )


def test_select_delegation_route_rejects_invalid_omnimarket_route() -> None:
    """An omnimarket route without a success+failure terminal interface fails closed."""
    routes = {
        "delegation.orchestrate": _route(
            package_name="omnimarket",
            terminal_events=(),
            command_topic="onex.cmd.omnimarket.delegation-request.v1",
        ),
    }

    with pytest.raises(InfraUnavailableError, match="No omnimarket delegation engine"):
        _select_delegation_route(routes)


def test_select_delegation_route_accepts_valid_public_omnimarket_fallback() -> None:
    """The bare 'delegation.orchestrate' alias resolves only for omnimarket."""
    route = _route(
        package_name="omnimarket",
        terminal_events=(
            "onex.evt.omnimarket.delegation-completed.v1",
            "onex.evt.omnimarket.delegation-failed.v1",
        ),
    )

    selected = _select_delegation_route({"delegation.orchestrate": route})

    assert selected.alias == "delegation.orchestrate"
    assert selected.route is route


def test_select_delegation_route_rejects_bare_alias_for_non_omnimarket() -> None:
    """The bare alias must NOT resolve a non-omnimarket (e.g. infra) route."""
    route = _route(
        package_name="omnibase_infra",
        terminal_events=(
            "onex.evt.omnibase-infra.delegation-completed.v1",
            "onex.evt.omnibase-infra.delegation-failed.v1",
        ),
    )

    with pytest.raises(InfraUnavailableError, match="No omnimarket delegation engine"):
        _select_delegation_route({"delegation.orchestrate": route})


def test_select_delegation_route_ambiguous_omnimarket_routes_fail_closed() -> None:
    """Two distinct omnimarket routes with the interface -> ambiguous, fail closed."""
    routes = {
        "delegation.orchestrate": _route(
            package_name="omnimarket",
            terminal_events=(
                "onex.evt.omnimarket.delegation-completed.v1",
                "onex.evt.omnimarket.delegation-failed.v1",
            ),
            command_topic="onex.cmd.omnimarket.delegation-request.v1",
            contract_path="/contracts/omnimarket/node_delegation_orchestrator.yaml",
        ),
        "omnimarket.node_delegation_orchestrator.delegation.orchestrate": _route(
            package_name="omnimarket",
            terminal_events=(
                "onex.evt.omnimarket.delegation-completed-alt.v1",
                "onex.evt.omnimarket.delegation-failed-alt.v1",
            ),
            command_topic="onex.cmd.omnimarket.delegation-request-alt.v1",
            contract_path="/contracts/omnimarket/node_delegation_orchestrator_alt.yaml",
        ),
    }

    with pytest.raises(InfraUnavailableError, match="Ambiguous delegation dispatch"):
        _select_delegation_route(routes)


async def _dispatch_with_fake_broker(
    monkeypatch: pytest.MonkeyPatch,
    **dispatch_kwargs: object,
) -> tuple[
    ModelRuntimeLocalIngressRoute,
    list[dict[str, object]],
    list[float],
    list[dict[str, object]],
]:
    route = _route(
        package_name="omnimarket",
        terminal_events=(
            "onex.evt.omnibase-infra.delegation-completed.v1",
            "onex.evt.omnibase-infra.delegation-failed.v1",
        ),
    )
    captured_timeout_seconds: list[float] = []
    captured_payloads: list[dict[str, object]] = []
    captured_broker_kwargs: list[dict[str, object]] = []

    class FakeBroker:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.args = _args
            self.kwargs = _kwargs
            captured_broker_kwargs.append(dict(_kwargs))

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

    dispatch_args = {
        "prompt": "probe",
        "task_type": "document",
        "correlation_id": uuid4(),
        "max_tokens": 512,
        "source_file_path": None,
        "source_session_id": None,
        "wait": True,
        "output_schema_key": None,
    } | dispatch_kwargs
    await port.dispatch(
        **dispatch_args,  # type: ignore[arg-type]
    )
    return route, captured_payloads, captured_timeout_seconds, captured_broker_kwargs


@pytest.mark.asyncio
async def test_runtime_delegation_dispatch_port_respects_dispatch_timeout_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route, payloads, timeout_seconds, broker_kwargs = await _dispatch_with_fake_broker(
        monkeypatch
    )

    assert timeout_seconds == [600.0]
    assert payloads[0]["prompt"] == "probe"
    assert payloads[0]["task_type"] == "document"
    assert broker_kwargs[0]["command_topic"] == route.command_topic


@pytest.mark.asyncio
async def test_runtime_delegation_dispatch_port_forwards_quality_contract_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The port must accept and forward quality_contract_mode and acceptance_criteria."""
    _, payloads, _, _ = await _dispatch_with_fake_broker(
        monkeypatch,
        quality_contract_mode="replace_task_class",
        acceptance_criteria=("response_non_empty", "plain_text_only"),
    )

    assert payloads[0]["quality_contract_mode"] == "replace_task_class"
    assert payloads[0]["acceptance_criteria"] == [
        "response_non_empty",
        "plain_text_only",
    ]


@pytest.mark.asyncio
async def test_runtime_delegation_dispatch_port_defaults_quality_contract_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When kwargs omitted, payload carries wire-model defaults."""
    _, payloads, _, _ = await _dispatch_with_fake_broker(monkeypatch)

    assert payloads[0]["quality_contract_mode"] == "extend_task_class"
    assert payloads[0]["acceptance_criteria"] == []


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
