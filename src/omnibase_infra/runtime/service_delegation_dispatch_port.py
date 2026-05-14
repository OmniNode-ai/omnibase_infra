# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime-owned dispatch port for consumer-facing delegation handlers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, cast
from uuid import UUID

from omnimarket.nodes.node_delegation_orchestrator.models.model_delegation_request import (
    ModelDelegationRequest,
)

from omnibase_core.models.dispatch.model_dispatch_bus_command import (
    ModelDispatchBusCommand,
)
from omnibase_infra.protocols.protocol_pattern_b_broker_transport import (
    ProtocolPatternBBrokerTransport,
)
from omnibase_infra.runtime.models.model_pattern_b_broker_config import (
    ModelPatternBBrokerConfig,
)
from omnibase_infra.runtime.runtime_local_ingress import (
    RuntimeLocalIngressRoute,
    discover_runtime_local_ingress_routes,
    parse_active_runtime_packages,
)
from omnibase_infra.runtime.service_pattern_b_broker import RuntimePatternBBroker

_DELEGATION_CONTRACT_NAME = "node_delegation_orchestrator"
_DELEGATION_OPERATION_ALIAS = "delegation.orchestrate"
_PREFERRED_DELEGATION_PACKAGE = "omnimarket"
_REQUESTER = "delegate_skill"
_DEFAULT_TIMEOUT_SECONDS = 600.0


@dataclass(frozen=True, slots=True)
class ModelSelectedDelegationRoute:
    alias: str
    route: RuntimeLocalIngressRoute


def _select_delegation_route(
    routes: Mapping[str, RuntimeLocalIngressRoute],
) -> ModelSelectedDelegationRoute:
    """Select the single route with the delegation terminal interface."""

    candidates: dict[str, tuple[str, RuntimeLocalIngressRoute]] = {}
    for alias, route in routes.items():
        if route.contract_name != _DELEGATION_CONTRACT_NAME:
            continue
        if alias != _DELEGATION_OPERATION_ALIAS and not alias.endswith(
            f".{_DELEGATION_CONTRACT_NAME}.{_DELEGATION_OPERATION_ALIAS}"
        ):
            continue
        if len(route.terminal_events) < 2:
            continue
        candidates[route.contract_path] = (alias, route)

    if len(candidates) == 1:
        alias, route = next(iter(candidates.values()))
        return ModelSelectedDelegationRoute(alias=alias, route=route)

    preferred = [
        candidate
        for candidate in candidates.values()
        if candidate[1].package_name == _PREFERRED_DELEGATION_PACKAGE
    ]
    if len(preferred) == 1:
        alias, route = preferred[0]
        return ModelSelectedDelegationRoute(alias=alias, route=route)

    fallback_route = routes.get(_DELEGATION_OPERATION_ALIAS)
    if not candidates and fallback_route is not None:
        return ModelSelectedDelegationRoute(
            alias=_DELEGATION_OPERATION_ALIAS,
            route=fallback_route,
        )

    raise RuntimeError(
        "Unable to resolve a single delegation dispatch route with success and "
        "failure terminal events"
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
        routes: Mapping[str, RuntimeLocalIngressRoute] | None = None,
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

    def _resolved_routes(self) -> dict[str, RuntimeLocalIngressRoute]:
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
    ) -> dict[str, object]:
        """Dispatch a delegation request and return the terminal result payload."""

        routes = self._resolved_routes()
        selected = _select_delegation_route(routes)
        request = ModelDelegationRequest(
            prompt=prompt,
            task_type=cast("Literal['test', 'document', 'research']", task_type),
            source_session_id=source_session_id,
            source_file_path=source_file_path,
            correlation_id=correlation_id,
            max_tokens=max_tokens,
            emitted_at=datetime.now(UTC),
            output_schema_key=output_schema_key,
        )

        command = ModelDispatchBusCommand(
            command_name=selected.alias,
            requester=_REQUESTER,
            payload=request.model_dump(mode="json", exclude_none=True),
            correlation_id=correlation_id,
            response_topic=self._response_topic or _default_response_topic(),
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS if wait else 1.0,
        )
        broker = RuntimePatternBBroker(
            self._event_bus,
            command_topic=self._command_topic or _default_command_topic(),
            routes=routes,
        )
        _route, result = await broker.dispatch_request(command)
        return _normalize_result_payload(
            status=result.status,
            payload=result.payload,
            error_message=result.error_message,
        )


def _default_command_topic() -> str:
    from omnimarket.adapters.codex.runtime_client import default_command_topic

    return default_command_topic()


def _default_response_topic() -> str:
    from omnimarket.adapters.codex.runtime_client import default_response_topic

    return default_response_topic()


__all__ = ["RuntimeDelegationDispatchPort"]
