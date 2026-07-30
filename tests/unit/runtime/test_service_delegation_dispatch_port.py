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


@pytest.mark.asyncio
async def test_runtime_delegation_dispatch_port_accepts_absent_optional_bus_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Handler compatibility kwargs must not break the existing bus route."""
    _, payloads, _, _ = await _dispatch_with_fake_broker(
        monkeypatch,
        backend_id=None,
        response_contract=None,
    )

    assert "backend_id" not in payloads[0]
    assert "response_contract" not in payloads[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("dispatch_kwargs", "unsupported_feature"),
    [
        ({"backend_id": "local-coder-mlx"}, "backend_id"),
        (
            {"response_contract": {"type": "object"}},
            "response_contract",
        ),
    ],
)
async def test_runtime_delegation_dispatch_port_rejects_unsupported_bus_features(
    monkeypatch: pytest.MonkeyPatch,
    dispatch_kwargs: dict[str, object],
    unsupported_feature: str,
) -> None:
    """Explicit bus-only feature requests fail closed instead of being dropped."""
    with pytest.raises(NotImplementedError, match=unsupported_feature):
        await _dispatch_with_fake_broker(monkeypatch, **dispatch_kwargs)


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


# ---------------------------------------------------------------------------
# OMN-15471: provider provenance must never be the hardcoded literal "local".
#
# These drive the real normalizer (`_normalize_result_payload`) with the real
# wire shape of `onex.evt.omnibase-infra.delegation-completed.v1` as read back
# from the onex-dev lane, because the defect only appears for a payload that
# has NO `provider` key -- which is every genuine terminal. A fixture that
# supplies `provider` cannot reproduce it.
# ---------------------------------------------------------------------------

_GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
)

# Verbatim field set of the live delegation-completed payload for correlation
# a4000001-0000-4000-8000-000000000001 (onex-dev, 2026-07-30T04:49:52Z),
# trimmed to the fields this normalizer reads. Note: no "provider" key.
_LIVE_CLOUD_TERMINAL: dict[str, object] = {
    "correlation_id": "a4000001-0000-4000-8000-000000000001",
    "task_type": "test",
    "model_used": "gemini-2.5-flash",
    "endpoint_url": _GEMINI_ENDPOINT,
    "quality_passed": True,
    "quality_score": 1.0,
    "latency_ms": 4188,
    "prompt_tokens": 115,
    "completion_tokens": 130,
    "cost_tier_name": "cheap_cloud",
    "final_attempt_cost": 0.00049,
}


def test_cloud_routed_terminal_never_reports_local_provenance() -> None:
    """A Gemini-routed terminal must not be stamped as a local-provider run.

    RED before OMN-15471: `_normalize_result_payload` defaulted
    `delegated_to` to the literal "local" whenever the payload had no
    `provider` key, so this asserted "local" != "local" and failed.
    """
    normalized = _normalize_result_payload(
        status="completed",
        payload=dict(_LIVE_CLOUD_TERMINAL),
        error_message=None,
    )

    # The load-bearing assertion: the false claim is gone.
    assert normalized["delegated_to"] != "local"
    # And the value is the actual resolved endpoint, not a fabricated class.
    assert normalized["delegated_to"] == _GEMINI_ENDPOINT
    # The rest of the flattening is unchanged.
    assert normalized["model_name"] == "gemini-2.5-flash"
    assert normalized["quality_gate_passed"] is True


def test_cloud_routed_terminal_provenance_survives_the_nested_wire_shape() -> None:
    """The same guarantee holds through the legacy double-nested envelope."""
    normalized = _normalize_result_payload(
        status="completed",
        payload={
            "topic": "onex.evt.omnibase-infra.delegation-completed.v1",
            "payload": dict(_LIVE_CLOUD_TERMINAL),
        },
        error_message=None,
    )

    assert normalized["delegated_to"] != "local"
    assert normalized["delegated_to"] == _GEMINI_ENDPOINT


def test_local_routed_terminal_still_resolves_to_the_local_endpoint() -> None:
    """The fix is not "always say cloud" -- a local backend stays identifiable."""
    normalized = _normalize_result_payload(
        status="completed",
        payload={
            "model_used": "Qwen3.6-35B-A3B",
            "endpoint_url": "http://127.0.0.1:8001/v1/chat/completions",
            "cost_tier_name": "local",
            "quality_passed": True,
        },
        error_message=None,
    )

    assert normalized["delegated_to"] == "http://127.0.0.1:8001/v1/chat/completions"
    assert "generativelanguage.googleapis.com" not in str(normalized["delegated_to"])


def test_explicit_upstream_provider_stamp_wins() -> None:
    """An explicit producer-supplied `provider` is authoritative."""
    normalized = _normalize_result_payload(
        status="completed",
        payload={
            "provider": "cloud-gemini-pro",
            "endpoint_url": _GEMINI_ENDPOINT,
            "cost_tier_name": "cheap_cloud",
        },
        error_message=None,
    )

    assert normalized["delegated_to"] == "cloud-gemini-pro"


def test_provenance_falls_back_to_the_resolved_tier_when_endpoint_is_absent() -> None:
    """With no endpoint identity, the routing tier is the remaining real fact."""
    normalized = _normalize_result_payload(
        status="completed",
        payload={"model_used": "gemini-2.5-flash", "cost_tier_name": "cheap_cloud"},
        error_message=None,
    )

    assert normalized["delegated_to"] == "cheap_cloud"


def test_absent_provenance_stays_absent_rather_than_fabricated() -> None:
    """No provenance signal must yield an empty value, not a deployment class.

    An empty string is falsy, so
    `handler_delegate_skill._response_from_result` (`delegated_to or
    endpoint_url or ""`) still falls through its own chain. A fabricated
    "local" would short-circuit it -- that was the OMN-15471 mechanism.
    """
    normalized = _normalize_result_payload(
        status="failed",
        payload={"failure_reason": "no configured endpoint"},
        error_message="no configured endpoint",
    )

    assert normalized["delegated_to"] == ""
    assert normalized["delegated_to"] != "local"


def test_existing_delegated_to_is_never_overwritten() -> None:
    """The local dispatch port already stamps `delegated_to`; preserve it."""
    normalized = _normalize_result_payload(
        status="completed",
        payload={
            "delegated_to": "local-qwen-coder-30b",
            "endpoint_url": _GEMINI_ENDPOINT,
        },
        error_message=None,
    )

    assert normalized["delegated_to"] == "local-qwen-coder-30b"


def test_no_hardcoded_deployment_class_default_in_the_normalizer() -> None:
    """Mechanism guard: the literal default cannot be reintroduced silently.

    Per the rule-vs-mechanism standard, the behavioural tests above are
    necessary but not sufficient -- a future edit could reinstate the literal
    on a sibling field and every assertion above would still pass.

    This walks the module AST rather than grepping text, so it pins the
    executable construct (``<mapping>.get(<key>, "local"|"cloud")`` fed into a
    ``setdefault``) and is not satisfied or broken by prose that merely quotes
    the removed line -- the docstring of ``_resolve_delegation_provenance``
    deliberately does quote it.
    """
    import ast
    from pathlib import Path

    import omnibase_infra.runtime.service_delegation_dispatch_port as module

    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    fabricated = {"local", "cloud", "remote", "unknown"}
    offending: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "get":
            continue
        if len(node.args) != 2:
            continue
        default = node.args[1]
        if isinstance(default, ast.Constant) and default.value in fabricated:
            offending.append(ast.unparse(node))

    assert offending == [], (
        "OMN-15471: provenance must be derived from the terminal payload's own "
        "resolved fields, never defaulted to a fabricated deployment class; "
        f"found {offending}"
    )
