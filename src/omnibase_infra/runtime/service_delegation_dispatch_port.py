# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime-owned dispatch port for consumer-facing delegation handlers."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from omnibase_core.models.delegation.wire import ModelDelegationProvenance
from omnibase_core.models.dispatch.model_dispatch_bus_command import (
    ModelDispatchBusCommand,
)
from omnibase_infra.errors import InfraUnavailableError
from omnibase_infra.protocols.protocol_pattern_b_broker_transport import (
    ProtocolPatternBBrokerTransport,
)
from omnibase_infra.runtime.bounded_delegation_routes import (
    resolve_bounded_delegation_route,
)
from omnibase_infra.runtime.models.model_delegation_terminal_evidence import (
    ModelDelegationTerminalEvidence,
)
from omnibase_infra.runtime.models.model_pattern_b_broker_config import (
    ModelPatternBBrokerConfig,
)
from omnibase_infra.runtime.protocol_addressed_broker_transport import (
    ProtocolAddressedBrokerTransport,
)
from omnibase_infra.runtime.protocols.protocol_delegation_dispatch_port import (
    DEFAULT_EXECUTION_TIMEOUT_SECONDS,
    DEFAULT_TERMINAL_DELIVERY_MARGIN_SECONDS,
    ProtocolDelegationDispatchPort,
)
from omnibase_infra.runtime.protocols.protocol_delegation_terminal_evidence_sink import (
    ProtocolDelegationTerminalEvidenceSink,
)
from omnibase_infra.runtime.runtime_local_ingress import (
    ModelRuntimeLocalIngressRoute,
    discover_runtime_local_ingress_routes,
    parse_active_runtime_packages,
)
from omnibase_infra.runtime.service_pattern_b_broker import RuntimePatternBBroker

logger = logging.getLogger(__name__)

_DELEGATION_CONTRACT_NAME = "node_delegation_orchestrator"
_DELEGATION_OPERATION_ALIAS = "delegation.orchestrate"
_PREFERRED_DELEGATION_PACKAGE = "omnimarket"
_REQUESTER = "delegate_skill"


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
        terminal_evidence_sink: ProtocolDelegationTerminalEvidenceSink | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._package_names = (
            tuple(package_names) if package_names is not None else None
        )
        self._routes = dict(routes) if routes is not None else None
        self._command_topic = command_topic
        self._response_topic = response_topic
        self._terminal_evidence_sink = terminal_evidence_sink

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
        # OMN-18924. These arrived as REQUIRED keyword-only arguments under
        # OMN-15504 while the caller deployed on the lane passes neither, so
        # every delegation on the dev lane terminalized `provider_error` with
        # a missing-argument TypeError -- the same producer-before-consumer
        # inversion as the absent `execution_budgets` map, from the same
        # change. Defaulted rather than required until the caller passes them:
        # a new argument lands consumer-first or is excluded when unset.
        #
        # The values are the contract default, kept honest against
        # `DEFAULT_EXECUTION_BUDGET` by
        # `tests/unit/runtime/test_dispatch_port_budget_defaults_omn18924.py`
        # rather than by an import, because runtime does not depend on cli.
        execution_timeout_seconds: int = DEFAULT_EXECUTION_TIMEOUT_SECONDS,
        terminal_delivery_margin_seconds: int = DEFAULT_TERMINAL_DELIVERY_MARGIN_SECONDS,
        output_schema_key: str | None = None,
        quality_contract_mode: str = "extend_task_class",
        acceptance_criteria: tuple[str, ...] = (),
        tenant_id: str | None = None,
        provenance: ModelDelegationProvenance | None = None,
        backend_id: str | None = None,
        no_escalation: bool = False,
        response_contract: dict[str, object] | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        response_format: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Dispatch a delegation request and return the terminal result payload."""

        # OmniMarket's consumer-facing handler always supplies these optional
        # arguments. The deployed bus model does not expose the completion-shaping
        # fields yet, so None preserves the existing route while explicit requests
        # fail closed instead of being silently dropped at this boundary.
        for feature_name, feature_value in (
            ("system_prompt", system_prompt),
            ("temperature", temperature),
            ("response_format", response_format),
        ):
            if feature_value is not None:
                raise NotImplementedError(
                    f"{feature_name} is not yet supported on the deployed bus "
                    "dispatch path (RuntimeDelegationDispatchPort); threading it "
                    "requires the canonical delegation request wire to carry it "
                    "end to end (OMN-15482)"
                )
        if backend_id is None and no_escalation:
            raise ValueError("no_escalation requires backend_id")
        if self._terminal_evidence_sink is not None and tenant_id is None:
            raise ValueError(
                "terminal evidence capture requires the actual dispatch tenant_id"
            )
        if execution_timeout_seconds <= 0:
            raise ValueError("execution_timeout_seconds must be positive")
        if terminal_delivery_margin_seconds <= 0:
            raise ValueError("terminal_delivery_margin_seconds must be positive")
        routes = self._resolved_routes()
        selected = _select_delegation_route(routes)
        # OMN-18933 (K6): on a bounded lane the declared row, the runtime's broker
        # identity and the selected consumer contract must agree BEFORE the
        # broker exists. A refusal raises here, so no command is published and no
        # terminal or projection row can exist for this correlation.
        bounded_route = resolve_bounded_delegation_route(
            transport=self._event_bus,
            selected_route=selected.route,
        )
        if bounded_route is not None:
            logger.info(
                "bounded delegation route accepted before dispatch",
                extra={
                    "correlation_id": str(correlation_id),
                    **bounded_route.model_dump(mode="json"),
                },
            )
        if backend_id is not None:
            # OMN-18931 (K1): a backend pin is admitted only as a declared
            # dogfood fault route on the dogfood broker, with that route's exact
            # no-escalation and timeout policy. The consumer re-checks the same
            # declaration, so a raw broker record cannot bypass this producer gate.
            if not isinstance(self._event_bus, ProtocolAddressedBrokerTransport):
                raise InfraUnavailableError(
                    "a delegation backend pin requires a runtime bus that exposes "
                    "its configured broker and environment identity"
                )
            from omnibase_infra.runtime.dogfood_delegation_fault_routes import (
                resolve_dogfood_delegation_fault_route,
            )

            fault_route = resolve_dogfood_delegation_fault_route(
                environment=self._event_bus.environment,
                bootstrap_servers=self._event_bus.bootstrap_servers,
                backend_id=backend_id,
            )
            if no_escalation is not fault_route.no_escalation:
                raise InfraUnavailableError(
                    "declared dogfood fault backend pin requires no-escalation policy"
                )
            if execution_timeout_seconds != fault_route.requested_timeout_seconds:
                raise InfraUnavailableError(
                    "declared dogfood fault backend pin requires its exact timeout "
                    "policy"
                )
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
            "response_contract": response_contract,
            # The selected node_delegation_orchestrator contract validates this
            # payload as ModelDelegationRequest. Its request timeout is part of
            # that model; the terminal delivery margin only bounds this caller's
            # broker wait and is not a delegation request field.
            "requested_timeout_seconds": execution_timeout_seconds,
            "backend_id": backend_id,
            # OMN-18321 / OMN-18172: carried ONTO THE WIRE, not merely accepted.
            # Accepting the keyword and dropping it would trade a loud TypeError
            # for a silent classification hole -- precisely the silent-drop
            # defect OMN-18172 exists to close. `None` falls out of the
            # comprehension below, so an unclassified delegation publishes no
            # `provenance` key rather than a null a consumer could misread as a
            # value; OMN-18172 is explicit that absent means unclassified and
            # never synthetic.
            "provenance": (
                None if provenance is None else provenance.model_dump(mode="json")
            ),
        }
        if no_escalation:
            request_payload["no_escalation"] = True

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
            timeout_seconds=(
                float(execution_timeout_seconds + terminal_delivery_margin_seconds)
                if wait
                else 1.0
            ),
        )
        broker = RuntimePatternBBroker(
            self._event_bus,
            command_topic=self._command_topic or selected.route.command_topic,
            routes=routes,
        )
        terminal_evidence_sink = self._terminal_evidence_sink
        if terminal_evidence_sink is None:
            _route, result = await broker.dispatch_request(command)
        else:
            # The fail-closed guard above establishes the actual dispatch
            # tenant before this nested observer is created. Bind both values
            # locally: mypy cannot retain a mutable instance attribute's
            # narrowing across an async closure.
            if tenant_id is None:
                raise RuntimeError(
                    "terminal evidence sink reached without a dispatch tenant_id"
                )
            evidence_tenant_id = tenant_id

            async def observe_terminal(terminal: object) -> None:
                from omnibase_infra.runtime.service_pattern_b_broker import (
                    TerminalPayload,
                )

                if not isinstance(terminal, TerminalPayload):
                    raise TypeError(
                        "broker terminal observer received an invalid terminal"
                    )
                await terminal_evidence_sink(
                    ModelDelegationTerminalEvidence(
                        correlation_id=correlation_id,
                        tenant_id=evidence_tenant_id,
                        topic=terminal.topic,
                        raw_envelope=terminal.raw_envelope,
                        encoding="utf-8",
                        partition=terminal.partition,
                        offset=terminal.offset,
                    )
                )

            _route, result = await broker.dispatch_request(
                command,
                terminal_observer=observe_terminal,
            )
        return _normalize_result_payload(
            status=result.status,
            payload=result.payload,
            error_message=result.error_message,
        )


__all__ = [
    "ProtocolDelegationDispatchPort",
    "RuntimeDelegationDispatchPort",
]
