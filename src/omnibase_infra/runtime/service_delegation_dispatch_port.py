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


def _resolve_delegation_provenance(normalized: Mapping[str, object]) -> str:
    """Resolve where the delegation actually ran, from the terminal's own fields.

    OMN-15471: this used to be ``normalized.get("provider", "local")``. No real
    ``delegation-completed.v1`` payload carries a ``provider`` key — that event
    models the resolved serving endpoint as ``endpoint_url`` plus
    ``cost_tier_name`` — so the literal default fired on EVERY bus-path
    delegation and stamped ``provider="local"`` on the durable terminal. A
    Gemini-routed result (``endpoint_url`` = the Google Generative Language API,
    ``cost_tier_name`` = ``cheap_cloud``) was recorded as a local-provider run:
    39/39 ``delegate-skill-completed.v1`` rows read ``local`` on the onex-dev
    lane and not one of them ran on a local model.

    Provenance is therefore derived ONLY from facts the terminal payload
    actually carries, in descending order of how directly they identify the
    serving endpoint:

    1. ``provider`` — an explicit upstream stamp, if a producer ever sets one.
    2. ``endpoint_url`` — the host that was really called. This is the strongest
       available provenance fact: it cannot read as local for a cloud call, and
       for a genuinely local backend it is the private/loopback address, so the
       local case stays identifiable.
    3. ``cost_tier_name`` / ``cost_tier_type`` — the resolved routing tier, used
       only when no endpoint identity survived into the terminal.

    When none of those resolve, the return is the empty string. That is
    deliberate: an absent provenance must stay absent so the consumer
    (``handler_delegate_skill._response_from_result``, which reads
    ``delegated_to or endpoint_url or ""``) falls through its own chain instead
    of inheriting a fabricated deployment class. Never invent one here.
    """

    for key in ("provider", "endpoint_url", "cost_tier_name", "cost_tier_type"):
        value = normalized.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _resolve_measured_actual_cost(normalized: Mapping[str, object]) -> float | None:
    """Resolve non-negative measured spend from the canonical terminal fields.

    The cumulative total is authoritative for a current terminal, while the
    final-attempt value keeps older/defaulted terminals compatible.  Taking the
    maximum also enforces the domain invariant that total spend cannot be less
    than the final attempt, without turning a genuine free-local ``0/0`` into an
    absent measurement.
    """

    costs: list[float] = []
    for key in ("cumulative_attempt_cost", "final_attempt_cost"):
        value = normalized.get(key)
        if (
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and value >= 0.0
        ):
            costs.append(float(value))
    return max(costs) if costs else None


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
    # OMN-15471: derive real provenance; never default to the literal "local".
    normalized.setdefault("delegated_to", _resolve_delegation_provenance(normalized))
    normalized.setdefault(
        "quality_gate_passed", normalized.get("quality_passed", False)
    )
    normalized.setdefault("input_tokens", normalized.get("prompt_tokens", 0))
    normalized.setdefault("output_tokens", normalized.get("completion_tokens", 0))
    normalized.setdefault("delegation_latency_ms", normalized.get("latency_ms", 0))
    # OMN-15520: the workflow terminal owns measured actual cost.  Total cost
    # across an escalation ladder is cumulative; single-attempt/legacy
    # terminals expose only the final attempt.  Preserve an explicit zero by
    # checking for None rather than truthiness, and overwrite any stale
    # consumer-shaped ``cost_usd`` with the upstream measurement when present.
    actual_cost = _resolve_measured_actual_cost(normalized)
    if actual_cost is not None:
        normalized["cost_usd"] = actual_cost
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
        max_tokens: int | None,
        source_file_path: str | None,
        source_session_id: str | None,
        wait: bool,
        output_schema_key: str | None = None,
        quality_contract_mode: str = "extend_task_class",
        acceptance_criteria: tuple[str, ...] = (),
        tenant_id: str | None = None,
        backend_id: str | None = None,
        response_contract: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Dispatch a delegation request and return the terminal result payload."""

        # OmniMarket's consumer-facing handler always supplies these optional
        # arguments.  The deployed bus model does not expose either field yet,
        # so None preserves the existing route while explicit requests fail
        # closed instead of being silently dropped at this boundary.
        if backend_id is not None:
            raise NotImplementedError(
                "backend_id pin is not yet supported on the deployed bus "
                "dispatch path (RuntimeDelegationDispatchPort)"
            )
        if response_contract is not None:
            raise NotImplementedError(
                "response_contract is not yet supported on the deployed bus "
                "dispatch path (RuntimeDelegationDispatchPort)"
            )

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
            "tenant_id": tenant_id,
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
