# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime-owned dispatch port for consumer-facing delegation handlers."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from pydantic import BaseModel

from omnibase_core.models.delegation.wire import ModelDelegationProvenance
from omnibase_core.models.dispatch.model_dispatch_bus_command import (
    ModelDispatchBusCommand,
)
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.errors import InfraUnavailableError
from omnibase_infra.handlers.handler_delegation_dispatch_terminal import (
    HandlerDelegationDispatchTerminal,
)
from omnibase_infra.models.delegation.model_delegation_dispatch_terminal_request import (
    ModelDelegationDispatchTerminalRequest,
)
from omnibase_infra.models.delegation.model_dispatch_budget import ModelDispatchBudget
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
    ProtocolDelegationDispatchPort,
)
from omnibase_infra.runtime.runtime_local_ingress import (
    ModelRuntimeLocalIngressRoute,
    discover_runtime_local_ingress_routes,
    local_ingress_contract_fingerprint,
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
        ModelDispatchBudget.refuse_past_dispatch_deadline(event_bus)
        self._event_bus = event_bus
        self._package_names = (
            tuple(package_names) if package_names is not None else None
        )
        self._routes = dict(routes) if routes is not None else None
        self._command_topic = command_topic
        self._response_topic = response_topic
        self._terminal_evidence_sink = terminal_evidence_sink
        self._terminal = HandlerDelegationDispatchTerminal()
        # OMN-21271: discovery parses every installed node contract. Run per
        # dispatch on the event loop it read 1014 contracts in 4.2-7.7 s and
        # stalled the loop for up to 7.7 s on the dev effects container. The
        # installed contract set changes only on an install (RULING
        # 2026-10-20T14:05:12Z, D2: hot-loaded contracts are limited to those
        # installed in the runtime image), so the stat-only fingerprint of that
        # set is the invalidation: (package_names, fingerprint, routes).
        self._route_cache: (
            tuple[
                tuple[str, ...],
                tuple[tuple[str, int, int], ...],
                dict[str, ModelRuntimeLocalIngressRoute],
            ]
            | None
        ) = None
        self._route_cache_lock = threading.Lock()

    async def _resolved_routes(self) -> dict[str, ModelRuntimeLocalIngressRoute]:
        if self._routes is not None:
            return dict(self._routes)
        package_names = parse_active_runtime_packages(
            self._package_names or ModelPatternBBrokerConfig().package_names
        )
        return await asyncio.to_thread(self._discover_routes_cached, package_names)

    def _discover_routes_cached(
        self, package_names: tuple[str, ...]
    ) -> dict[str, ModelRuntimeLocalIngressRoute]:
        """Resolve routes in a worker thread, re-reading contracts only on change.

        The fingerprint is taken before the parse, so a contract that changes in
        between costs one extra reload on the next call and never a stale hit.
        """
        fingerprint: tuple[tuple[str, int, int], ...] | None
        try:
            fingerprint = local_ingress_contract_fingerprint(package_names)
        except (ImportError, OSError, ValueError):
            # What resolving a package root or stat-ing a contract raises. Keep
            # no cache: discovery below then fails exactly as it did uncached.
            logger.debug(
                "local ingress contract fingerprint unavailable; resolving uncached",
                exc_info=True,
            )
            fingerprint = None
        with self._route_cache_lock:
            cache = self._route_cache
            if (
                fingerprint is not None
                and cache is not None
                and cache[0] == package_names
                and cache[1] == fingerprint
            ):
                return dict(cache[2])
            routes = discover_runtime_local_ingress_routes(package_names)
            if fingerprint is not None:
                self._route_cache = (package_names, fingerprint, dict(routes))
            return routes

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
        # Unset (None) means the task class's declared budget (OMN-21604). The
        # port and its protocol are pinned to that one unset default by
        # `tests/unit/runtime/test_dispatch_port_budget_defaults_omn18924.py`.
        execution_timeout_seconds: int | None = None,
        terminal_delivery_margin_seconds: int | None = None,
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
        trace_id: UUID | None = None,
        queue_wait_ms: int | None = None,
    ) -> dict[str, object]:
        """Dispatch a request; guard OMN-18931 faults, pass OMN-19124 routing pins.

        Dev run 8c2b5dfc-3907-475d-aab6-126fade5f707 exposed the blanket
        fault-pin refusal on a routing pin.

        ``attribution`` (OMN-20606) and ``queue_wait_ms`` (OMN-19448) are accepted
        and unused: this port publishes to a deployed lane whose own handler
        stamps the caller onto its terminal and measures its own queue wait.
        Only omnimarket's in-process port writes an evidence terminal of its
        own and needs these caller values. The Core wire request has no
        queue-wait field and forbids extras, so publishing it would be refused.
        """
        del attribution, queue_wait_ms

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
        # OMN-21604: an unset budget is the class's declared one, overlay included.
        timeout, margin = ModelDispatchBudget.resolve(
            task_type, execution_timeout_seconds, terminal_delivery_margin_seconds
        )
        routes = await self._resolved_routes()
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
                if timeout != fault_route.requested_timeout_seconds:
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
            "trace_id": None if trace_id is None else str(trace_id),
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
            "requested_timeout_seconds": timeout,
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
            timeout_seconds=(float(timeout + margin) if wait else 1.0),
        )
        broker = RuntimePatternBBroker(
            self._event_bus,
            command_topic=self._command_topic or selected.route.command_topic,
            routes=routes,
        )
        # OMN-19972: the run began here; its steps are timed from this moment.
        run_started_at, run_started = datetime.now(UTC), time.monotonic()
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
        finished = await self._terminal.handle(
            ModelDelegationDispatchTerminalRequest(
                terminal=result,
                correlation_id=correlation_id,
                task_type=task_type,
                tenant_id=tenant_id,
                started_at=run_started_at,
                started_monotonic=run_started,
                engine_package=selected.route.package_name,
                publish_steps=wait,
            ),
            routes=routes,
        )
        key = str(correlation_id).encode("utf-8")
        for step in finished.run_steps:
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
                    finished.run_step_topic or "",
                    key,
                    envelope.model_dump_json(exclude_none=True).encode("utf-8"),
                    None,
                )
            except Exception:  # a trace gap, never a failed delegation
                logger.exception(
                    "run step %d of %s was not published",
                    step.step_sequence,
                    correlation_id,
                )
        return dict(finished.result)


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
