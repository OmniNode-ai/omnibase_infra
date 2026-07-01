# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime-owned dispatch port for consumer-facing delegation handlers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from omnibase_core.models.dispatch.model_dispatch_bus_command import (
    ModelDispatchBusCommand,
)
from omnibase_infra.errors import InfraUnavailableError
from omnibase_infra.protocols.protocol_pattern_b_broker_transport import (
    ProtocolPatternBBrokerTransport,
)
from omnibase_infra.runtime.models.model_pattern_b_broker_config import (
    ModelPatternBBrokerConfig,
)
from omnibase_infra.runtime.protocols.protocol_delegation_dispatch_port import (
    ProtocolDelegationDispatchPort,
)
from omnibase_infra.runtime.runtime_local_ingress import (
    ModelRuntimeLocalIngressRoute,
    discover_runtime_local_ingress_routes,
    parse_active_runtime_packages,
)
from omnibase_infra.runtime.service_pattern_b_broker import RuntimePatternBBroker

_DELEGATION_CONTRACT_NAME = "node_delegation_orchestrator"
_DELEGATION_OPERATION_ALIAS = "delegation.orchestrate"
_PREFERRED_DELEGATION_PACKAGE = "omnimarket"
_REQUESTER = "delegate_skill"
_DEFAULT_TIMEOUT_SECONDS = 600.0


@dataclass(
    frozen=True, slots=True
)  # internal-dataclass-ok: module-internal routing helper
class ModelSelectedDelegationRoute:
    alias: str
    route: ModelRuntimeLocalIngressRoute


def _has_delegation_terminal_interface(route: ModelRuntimeLocalIngressRoute) -> bool:
    return (
        route.contract_name == _DELEGATION_CONTRACT_NAME
        and bool(route.command_topic)
        and len(route.terminal_events) >= 2
    )


def _select_delegation_route(
    routes: Mapping[str, ModelRuntimeLocalIngressRoute],
) -> ModelSelectedDelegationRoute:
    """Resolve the omnimarket-backed delegation route, fail-closed otherwise.

    Delegation has exactly one real engine: the omnimarket
    ``node_delegation_orchestrator`` (routing -> inference -> quality-gate ->
    escalation FSM). The empty omnibase_infra shell was deleted in OMN-13547
    (OMN-12525 — no duplicate orchestrators; nodes live in omnimarket), so this
    resolver MUST bind the omnimarket package only. If no omnimarket route is
    present the runtime fails closed with a typed ``InfraUnavailableError`` —
    there is NO silent fallback to a local/infra route, because resolving a
    non-omnimarket "delegation" surface would route to a dead handler.
    """

    candidates: dict[str, tuple[str, ModelRuntimeLocalIngressRoute]] = {}
    for alias, route in routes.items():
        if route.contract_name != _DELEGATION_CONTRACT_NAME:
            continue
        if route.package_name != _PREFERRED_DELEGATION_PACKAGE:
            continue
        if alias != _DELEGATION_OPERATION_ALIAS and not alias.endswith(
            f".{_DELEGATION_CONTRACT_NAME}.{_DELEGATION_OPERATION_ALIAS}"
        ):
            continue
        if not _has_delegation_terminal_interface(route):
            continue
        candidates[route.contract_path] = (alias, route)

    if len(candidates) == 1:
        alias, route = next(iter(candidates.values()))
        return ModelSelectedDelegationRoute(alias=alias, route=route)

    if len(candidates) > 1:
        raise InfraUnavailableError(
            "Ambiguous delegation dispatch: multiple omnimarket "
            f"'{_DELEGATION_CONTRACT_NAME}' routes expose the "
            f"'{_DELEGATION_OPERATION_ALIAS}' interface "
            f"({sorted(candidates)})"
        )

    raise InfraUnavailableError(
        "No omnimarket delegation engine resolved: the "
        f"'{_PREFERRED_DELEGATION_PACKAGE}.{_DELEGATION_CONTRACT_NAME}' route "
        f"with the '{_DELEGATION_OPERATION_ALIAS}' interface is not installed. "
        "Delegation fails closed — there is no infra-local fallback engine "
        "(OMN-13547 / OMN-12525)."
    )


def _normalize_result_payload(
    *,
    status: str,
    payload: object,
    error_message: str | None,
) -> dict[str, object]:
    """Flatten delegation terminal payloads into the delegate-skill port shape."""

    if isinstance(payload, dict):
        normalized = dict(payload)
    else:
        normalized = {}

    nested_payload = normalized.get("payload")
    if isinstance(nested_payload, dict):
        normalized = dict(nested_payload) | {
            key: value for key, value in normalized.items() if key != "payload"
        }

    normalized["status"] = status
    if error_message:
        normalized["error_message"] = error_message
    normalized.setdefault("model_name", normalized.get("model_used", ""))
    normalized.setdefault("delegated_to", normalized.get("provider", "local"))
    normalized.setdefault(
        "quality_gate_passed", normalized.get("quality_passed", False)
    )
    normalized.setdefault("input_tokens", normalized.get("prompt_tokens", 0))
    normalized.setdefault("output_tokens", normalized.get("completion_tokens", 0))
    normalized.setdefault("delegation_latency_ms", normalized.get("latency_ms", 0))
    return normalized


class RuntimeDelegationDispatchPort:
    """Delegation dispatch port backed by runtime-owned Pattern B plumbing."""

    def __init__(
        self,
        event_bus: ProtocolPatternBBrokerTransport,
        *,
        package_names: Sequence[str] | None = None,
        routes: Mapping[str, ModelRuntimeLocalIngressRoute] | None = None,
        command_topic: str | None = None,
        response_topic: str | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._package_names = (
            tuple(package_names) if package_names is not None else None
        )
        self._routes = dict(routes) if routes is not None else None
        self._command_topic = command_topic
        self._response_topic = response_topic

    def _resolved_routes(self) -> dict[str, ModelRuntimeLocalIngressRoute]:
        if self._routes is not None:
            return dict(self._routes)
        package_names = parse_active_runtime_packages(
            self._package_names or ModelPatternBBrokerConfig().package_names
        )
        return discover_runtime_local_ingress_routes(package_names)

    async def dispatch(
        self,
        *,
        prompt: str,
        task_type: str,
        correlation_id: UUID,
        max_tokens: int,
        source_file_path: str | None,
        source_session_id: str | None,
        wait: bool,
        output_schema_key: str | None = None,
        quality_contract_mode: str = "extend_task_class",
        acceptance_criteria: tuple[str, ...] = (),
    ) -> dict[str, object]:
        """Dispatch a delegation request and return the terminal result payload."""

        routes = self._resolved_routes()
        selected = _select_delegation_route(routes)
        request_payload: dict[str, object] = {
            "prompt": prompt,
            "task_type": task_type,
            "source_session_id": source_session_id,
            "source_file_path": source_file_path,
            "correlation_id": str(correlation_id),
            "max_tokens": max_tokens,
            "emitted_at": datetime.now(UTC).isoformat(),
            "output_schema_key": output_schema_key,
            "quality_contract_mode": quality_contract_mode,
            "acceptance_criteria": list(acceptance_criteria),
        }

        command = ModelDispatchBusCommand(
            command_name=selected.alias,
            requester=_REQUESTER,
            payload={
                key: value
                for key, value in request_payload.items()
                if value is not None
            },
            correlation_id=correlation_id,
            response_topic=self._response_topic or selected.route.terminal_events[0],
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS if wait else 1.0,
        )
        broker = RuntimePatternBBroker(
            self._event_bus,
            command_topic=self._command_topic or selected.route.command_topic,
            routes=routes,
        )
        _route, result = await broker.dispatch_request(command)
        return _normalize_result_payload(
            status=result.status,
            payload=result.payload,
            error_message=result.error_message,
        )


__all__ = [
    "ProtocolDelegationDispatchPort",
    "RuntimeDelegationDispatchPort",
]
