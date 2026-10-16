# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime-owned dispatch port for consumer-facing delegation handlers."""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

from pydantic import BaseModel

from omnibase_core.models.delegation.wire import (
    EnumDelegationTerminalFailureCause,
    ModelDelegationProvenance,
)
from omnibase_core.models.dispatch.model_dispatch_bus_command import (
    ModelDispatchBusCommand,
)
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.errors import InfraUnavailableError
from omnibase_infra.models.delegation.model_delegation_run_step_event import (
    ModelDelegationRunStepEvent,
    RunStepKind,
)
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
from omnibase_infra.runtime.protocols.protocol_delegation_dispatch_port import (
    DEFAULT_EXECUTION_TIMEOUT_SECONDS,
    DEFAULT_TERMINAL_DELIVERY_MARGIN_SECONDS,
    ProtocolDelegationDispatchPort,
)
from omnibase_infra.runtime.runtime_local_ingress import (
    ModelRuntimeLocalIngressRoute,
    discover_runtime_local_ingress_routes,
    parse_active_runtime_packages,
)
from omnibase_infra.runtime.service_pattern_b_broker import RuntimePatternBBroker

logger = logging.getLogger(__name__)

# Persists final-run terminal evidence once dispatch has received the terminal
# (OMN-18931). A plain callable, not a Protocol: omnibase_infra adds no protocols.
DelegationTerminalEvidenceSink = Callable[
    [ModelDelegationTerminalEvidence], Awaitable[None]
]

_DELEGATION_CONTRACT_NAME = "node_delegation_orchestrator"
_DELEGATION_OPERATION_ALIAS = "delegation.orchestrate"
_PREFERRED_DELEGATION_PACKAGE = "omnimarket"
_REQUESTER = "delegate_skill"
_RUN_STEP_EVENT_TYPE = "delegation.run_step"
_RUN_STEP_SOURCE_TOOL = "omnibase_infra.runtime_delegation_dispatch_port"
# A step's free text is a short explanation, not a transcript.
_RUN_STEP_REASON_LIMIT = 500


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
    # OMN-19558: the broker can exhaust its terminal wait without receiving a
    # payload. Carry that timeout as a typed cause so the delegate-skill handler
    # does not infer a provider error from the broker's diagnostic text.
    if status == "timeout" and normalized.get("terminal_failure_cause") is None:
        normalized["terminal_failure_cause"] = (
            EnumDelegationTerminalFailureCause.TIMEOUT.value
        )
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
        terminal_evidence_sink: DelegationTerminalEvidenceSink | None = None,
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
        # The port and its protocol are pinned to one default by
        # `tests/unit/runtime/test_dispatch_port_budget_defaults_omn18924.py`.
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
        attribution: Mapping[str, str] | None = None,
        model: str | None = None,
        named_tests: BaseModel | None = None,
    ) -> dict[str, object]:
        """Dispatch a request; guard OMN-18931 faults, pass OMN-19124 routing pins.

        Dev run 8c2b5dfc-3907-475d-aab6-126fade5f707 exposed the blanket
        fault-pin refusal on a routing pin.

        ``attribution`` (OMN-20606) is accepted and unused: this port publishes
        to a deployed lane whose own handler stamps the caller onto its
        terminal. Only omnimarket's in-process port writes an evidence terminal
        of its own and needs it.
        """
        del attribution

        # Refuse a spec the lane cannot run before any route is resolved or
        # anything is published.
        named_tests_wire = (
            None if named_tests is None else _named_tests_wire(named_tests)
        )

        # OmniMarket's consumer-facing handler always supplies these optional
        # arguments. The deployed bus model does not expose the completion-shaping
        # fields yet, so None preserves the existing route while explicit requests
        # fail closed instead of being silently dropped at this boundary.
        for feature_name, feature_value in (
            ("system_prompt", system_prompt),
            ("temperature", temperature),
            ("response_format", response_format),
            # OMN-20844: the customer's model choice runs only on omnimarket's
            # in-process BYOK route; the bus wire does not carry it.
            ("model", model),
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
                "bounded delegation route accepted before dispatch: "
                "correlation_id=%s route=%s",
                str(correlation_id),
                bounded_route.model_dump_json(),
                extra={
                    "correlation_id": str(correlation_id),
                    **bounded_route.model_dump(mode="json"),
                },
            )
        if backend_id is not None:
            # OMN-18931 (K1) guards fault pins only. OMN-19124 routing pins
            # pass to Market; dev run 8c2b5dfc-3907-475d-aab6-126fade5f707
            # exposed the former blanket refusal. The consumer re-checks faults.
            from omnibase_infra.runtime.dogfood_delegation_fault_routes import (
                is_dogfood_fault_pin,
                resolve_dogfood_delegation_fault_route,
            )

            if is_dogfood_fault_pin(
                backend_id=backend_id,
                no_escalation=no_escalation,
            ):
                bus_environment = getattr(self._event_bus, "environment", None)
                bus_broker = getattr(self._event_bus, "bootstrap_servers", None)
                if not isinstance(bus_environment, str) or not isinstance(
                    bus_broker, str
                ):
                    raise InfraUnavailableError(
                        "a delegation backend pin requires a runtime bus that exposes "
                        "its configured broker and environment identity"
                    )
                fault_route = resolve_dogfood_delegation_fault_route(
                    environment=bus_environment,
                    bootstrap_servers=bus_broker,
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
        if named_tests_wire is not None:
            # OMN-17427: the tests ride the command, inline, so the consumer
            # can run them in its sandbox against the answer it is about to gate.
            request_payload["named_tests"] = named_tests_wire

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
        # OMN-19972: the run began here. Its steps are numbered once the chain
        # answers, timed from this moment.
        run_started_at = datetime.now(UTC)
        run_started = time.monotonic()
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
        normalized = _normalize_result_payload(
            status=result.status,
            payload=result.payload,
            error_message=result.error_message,
        )
        if wait:
            await self._publish_run_steps(
                topic=_resolve_run_step_topic(routes, selected.route.package_name),
                correlation_id=correlation_id,
                task_type=task_type,
                tenant_id=tenant_id,
                started_at=run_started_at,
                started=run_started,
                status=result.status,
                flat=normalized if isinstance(result.payload, dict) else None,
            )
        return normalized

    async def _publish_run_steps(
        self,
        *,
        topic: str | None,
        correlation_id: UUID,
        task_type: str,
        tenant_id: str | None,
        started_at: datetime,
        started: float,
        status: str,
        flat: Mapping[str, object] | None,
    ) -> None:
        """Number a dispatched run's steps from its terminal and publish them.

        OMN-19972: the deployed delegate-skill handler dispatches through this
        port, so its runs reach the run-trace projection only if this port
        publishes their steps, on the topic the delegate-skill contract declares.
        Best-effort: a step that cannot be delivered is a gap in the trace, never
        a failed delegation.
        """
        if topic is None:
            logger.warning(
                "run steps of correlation_id=%s not published: no installed "
                "contract declares one runtime_dispatch.run_step_event",
                correlation_id,
            )
            return
        tenant = (None if flat is None else _text(flat.get("tenant_id"))) or _text(
            tenant_id
        )
        if tenant is None:
            logger.warning(
                "run steps of correlation_id=%s not published: no tenant on the "
                "request or the terminal",
                correlation_id,
            )
            return
        sequence = RunStepSequence(
            tenant_id=tenant,
            correlation_id=correlation_id,
            started_at=started_at,
            started=started,
        )
        try:
            _record_bus_run(sequence, task_type=task_type, status=status, flat=flat)
        except (AttributeError, KeyError, TypeError, ValueError):
            logger.warning(
                "run steps of correlation_id=%s could not be derived",
                correlation_id,
                exc_info=True,
            )
        key = str(correlation_id).encode("utf-8")
        for step in sequence.steps:
            try:
                envelope = ModelEventEnvelope[dict[str, object]](
                    payload=step.model_dump(mode="json"),
                    correlation_id=correlation_id,
                    envelope_timestamp=datetime.now(UTC),
                    event_type=_RUN_STEP_EVENT_TYPE,
                    source_tool=_RUN_STEP_SOURCE_TOOL,
                    tenant_id=step.tenant_id,
                )
                await self._event_bus.publish(
                    topic,
                    key,
                    envelope.model_dump_json(exclude_none=True).encode("utf-8"),
                    None,
                )
            except Exception:
                # A trace gap, never a failed delegation.
                logger.exception(
                    "run step %d of correlation_id=%s was not published",
                    step.step_sequence,
                    correlation_id,
                )


def _resolve_run_step_topic(
    routes: Mapping[str, ModelRuntimeLocalIngressRoute], package_name: str
) -> str | None:
    """The one run-step topic the delegation engine's package declares."""
    declared = {
        route.run_step_event
        for route in routes.values()
        if route.package_name == package_name and route.run_step_event
    }
    if len(declared) == 1:
        return next(iter(declared))
    if declared:
        logger.warning(
            "ambiguous runtime_dispatch.run_step_event declarations in %s: %s",
            package_name,
            sorted(declared),
        )
    return None


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


class RunStepSequence:
    """The step counter of one run; its number is the run's only ordering."""

    def __init__(
        self,
        *,
        tenant_id: str,
        correlation_id: UUID,
        started_at: datetime,
        started: float,
    ) -> None:
        self._tenant_id = tenant_id
        self._correlation_id = correlation_id
        self._started_at = started_at
        self._started = started
        self._next_sequence = 1
        self._calls = 0
        self._routed_backend_id: str | None = None
        self._last_failed_sequence: int | None = None
        self._last_failure_cause: str | None = None
        self._terminated = False
        self.steps: list[ModelDelegationRunStepEvent] = []

    def request(self, *, task_type: str) -> None:
        self._record("request", started_at=self._started_at, reason=task_type)

    def routing(self, *, backend_id: str, model_id: str, tier: str) -> None:
        """A backend was chosen. A retry on the backend already chosen is no decision."""
        if backend_id == self._routed_backend_id:
            return
        previous = self._routed_backend_id
        self._routed_backend_id = backend_id
        if previous is None:
            reason = f"first choice: tier {tier}"
        else:
            cause = self._last_failure_cause or "the previous attempt"
            reason = f"tier {tier}, after {cause} on {previous}"
        self._record(
            "routing_decision",
            started_at=datetime.now(UTC),
            backend_id=backend_id,
            model_id=model_id,
            reason=reason,
        )

    def provider_call(
        self,
        *,
        duration_ms: int | None,
        backend_id: str,
        model_id: str | None,
        failure_cause: str | None,
        reason: str | None,
    ) -> None:
        self._calls += 1
        self._record(
            "provider_call",
            started_at=self._started_at,
            duration_ms=duration_ms,
            failure_cause=failure_cause,
            backend_id=backend_id,
            model_id=model_id,
            attempt_number=self._calls,
            reason=reason,
        )

    def quality_gate(
        self, *, backend_id: str, failure_cause: str | None, reason: str | None
    ) -> None:
        self._record(
            "quality_gate",
            started_at=datetime.now(UTC),
            duration_ms=0,
            failure_cause=failure_cause,
            backend_id=backend_id,
            attempt_number=self._calls or None,
            reason=reason,
        )

    def terminal(
        self,
        *,
        succeeded: bool,
        backend_id: str | None = None,
        model_id: str | None = None,
        failure_cause: str | None = None,
    ) -> None:
        """The run ended. A failed run names the earlier step that failed it."""
        if self._terminated:
            return
        self._terminated = True
        cause = None if succeeded else (failure_cause or self._last_failure_cause)
        elapsed = timedelta(seconds=time.monotonic() - self._started)
        self._record(
            "terminal",
            started_at=datetime.now(UTC),
            duration_ms=max(0, int(elapsed / timedelta(milliseconds=1))),
            failure_cause=cause or (None if succeeded else "failed"),
            succeeded=succeeded,
            backend_id=backend_id,
            model_id=model_id,
            failing_step_sequence=None if succeeded else self._last_failed_sequence,
        )

    def _record(
        self,
        kind: RunStepKind,
        *,
        started_at: datetime,
        duration_ms: int | None = None,
        failure_cause: str | None = None,
        succeeded: bool | None = None,
        backend_id: str | None = None,
        model_id: str | None = None,
        attempt_number: int | None = None,
        reason: str | None = None,
        failing_step_sequence: int | None = None,
    ) -> None:
        # The number is taken before anything can fail, so a step that cannot be
        # built still holds its place in the run.
        sequence = self._next_sequence
        self._next_sequence += 1
        if succeeded is None:
            succeeded = failure_cause is None
        if not succeeded and kind != "terminal":
            self._last_failed_sequence = sequence
            self._last_failure_cause = failure_cause
        if reason is not None and len(reason) > _RUN_STEP_REASON_LIMIT:
            reason = reason[: _RUN_STEP_REASON_LIMIT - 1] + "…"
        try:
            self.steps.append(
                ModelDelegationRunStepEvent(
                    tenant_id=self._tenant_id,
                    correlation_id=self._correlation_id,
                    step_sequence=sequence,
                    step_kind=kind,
                    started_at=started_at,
                    duration_ms=duration_ms,
                    outcome="succeeded" if succeeded else "failed",
                    failure_cause=failure_cause,
                    backend_id=backend_id,
                    model_id=model_id,
                    attempt_number=attempt_number,
                    reason=reason or None,
                    failing_step_sequence=failing_step_sequence,
                )
            )
        except ValueError:
            logger.warning(
                "run step %d (%s) of correlation_id=%s could not be built",
                sequence,
                kind,
                self._correlation_id,
                exc_info=True,
            )


def _bus_attempts(
    flat: Mapping[str, object], *, status: str
) -> list[Mapping[str, object]]:
    """The chain's attempts, from its escalation ladder or, failing that, the terminal."""
    history = flat.get("escalation_history")
    if isinstance(history, list | tuple):
        ladder = [entry for entry in history if isinstance(entry, Mapping)]
        if ladder:
            return ladder
    cause = flat.get("routed_failure_cause")
    provider_failed = isinstance(cause, Mapping) and cause.get("kind") == "provider"
    failure = _text(flat.get("terminal_failure_reason")) or _text(
        flat.get("failure_reason")
    )
    attempt: dict[str, object] = {
        "backend_ref": flat.get("backend_ref") or flat.get("route"),
        "model_used": flat.get("model_used"),
        "tier": flat.get("cost_tier_name"),
        "latency_ms": flat.get("latency_ms"),
    }
    if status == "completed":
        attempt["acceptance_decision"] = "accept"
    elif provider_failed:
        attempt["acceptance_decision"] = "terminate"
        attempt["acceptance_reason"] = "provider_call_failed"
        attempt["failure_reasons"] = [failure] if failure else []
    else:
        attempt["acceptance_decision"] = "terminate"
        attempt["acceptance_reason"] = (
            _text(cause.get("kind")) if isinstance(cause, Mapping) else None
        ) or "quality_gate_rejection"
        attempt["failure_reasons"] = [failure] if failure else []
    return [attempt]


def _record_bus_run(
    sequence: RunStepSequence,
    *,
    task_type: str,
    status: str,
    flat: Mapping[str, object] | None,
) -> None:
    """Take a dispatched run's steps in the order the chain took them."""
    sequence.request(task_type=task_type)
    if flat is None:
        sequence.terminal(succeeded=False, failure_cause=status)
        return
    model = _text(flat.get("model_used"))
    backend = _text(flat.get("backend_ref")) or _text(flat.get("route"))
    if backend is None and flat.get("routing_disposition") == "unrouted":
        sequence.terminal(
            succeeded=False,
            model_id=model,
            failure_cause=_text(flat.get("unrouted_reason")) or "unrouted",
        )
        return
    tier = _text(flat.get("cost_tier_name")) or "unknown"
    attempts = _bus_attempts(flat, status=status)
    for index, attempt in enumerate(attempts):
        attempt_backend = _text(attempt.get("backend_ref")) or backend or "unknown"
        attempt_model = _text(attempt.get("model_used")) or model
        sequence.routing(
            backend_id=attempt_backend,
            model_id=attempt_model or "",
            tier=_text(attempt.get("tier")) or tier,
        )
        latency = attempt.get("latency_ms")
        if index == len(attempts) - 1 and not isinstance(latency, int):
            latency = flat.get("latency_ms")
        reason_code = _text(attempt.get("acceptance_reason"))
        reasons = attempt.get("failure_reasons")
        detail = (
            "; ".join(str(r) for r in reasons)
            if isinstance(reasons, list | tuple) and reasons
            else None
        )
        provider_failed = reason_code == "provider_call_failed"
        sequence.provider_call(
            duration_ms=(
                latency
                if isinstance(latency, int)
                and not isinstance(latency, bool)
                and latency >= 0
                else None
            ),
            backend_id=attempt_backend,
            model_id=attempt_model,
            failure_cause=reason_code if provider_failed else None,
            reason=detail if provider_failed else None,
        )
        if provider_failed:
            continue
        accepted = attempt.get("acceptance_decision") == "accept"
        sequence.quality_gate(
            backend_id=attempt_backend,
            failure_cause=None if accepted else (reason_code or "rejected"),
            reason=None if accepted else detail,
        )
    sequence.terminal(
        succeeded=status == "completed", backend_id=backend, model_id=model
    )


def _named_tests_wire(named_tests: BaseModel) -> dict[str, object]:
    """Build the bus request's ``named_tests`` field from the consumer's spec.

    The consumer owns the spec model (a higher layer), so it arrives as a
    ``BaseModel``. The deployed lane cannot see the caller's workspace: a spec
    that points at one, or that carries no test sources, would reach the lane
    and end as a failure inside it, so it is refused here before anything is
    published.
    """
    spec = named_tests.model_dump(mode="json")
    if spec.get("workspace_root") is not None:
        raise ValueError(
            "the deployed lane cannot read the caller's workspace; carry the "
            "tests inline with --test-file"
        )
    if not spec.get("test_files"):
        raise ValueError(
            "named tests reach the deployed lane only as inline sources; "
            "name them with --test-file"
        )
    return {
        "test_ids": list(spec["test_ids"]),
        "answer_path": spec["answer_path"],
        "test_files": [
            {"path": test_file["path"], "content": test_file["content"]}
            for test_file in spec["test_files"]
        ],
    }


__all__ = [
    "DelegationTerminalEvidenceSink",
    "ProtocolDelegationDispatchPort",
    "RuntimeDelegationDispatchPort",
]
