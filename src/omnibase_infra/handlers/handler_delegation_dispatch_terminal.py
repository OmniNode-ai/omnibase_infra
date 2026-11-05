# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""What the delegation dispatch port does once the broker has answered a run.

OMN-19972. Bound by ``node_delegation_dispatch_terminal_effect``. The handler
turns the broker's terminal into the delegate-skill result shape the port
returns, and, for a run the caller waited on, numbers the run's steps and
builds one event envelope per step for the topic the delegation engine's
package declares as ``runtime_dispatch.run_step_event``. The port sends those
envelopes: a handler holds no event bus.

The step derivation is omnibase_infra#5041's, moved here because
the dispatch port's file may only shrink (canonical-file-shape, OMN-20304).
The terminal normaliser moved here unchanged from the dispatch port.

This is an EFFECT handler: it reads the installed node contracts to find the
declared run-step topic.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

import yaml

from omnibase_core.models.delegation.wire import EnumDelegationTerminalFailureCause
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.delegation.model_delegation_dispatch_terminal_request import (
    ModelDelegationDispatchTerminalRequest,
)
from omnibase_infra.models.delegation.model_delegation_dispatch_terminal_result import (
    ModelDelegationDispatchTerminalResult,
)
from omnibase_infra.models.delegation.model_delegation_run_step_event import (
    ModelDelegationRunStepEvent,
    RunStepKind,
)

if TYPE_CHECKING:
    from omnibase_infra.runtime.runtime_local_ingress import (
        ModelRuntimeLocalIngressRoute,
    )

logger = logging.getLogger(__name__)

_RUN_STEP_EVENT_TYPE = "delegation.run_step"
_RUN_STEP_SOURCE_TOOL = "omnibase_infra.runtime_delegation_dispatch_port"
# A step's free text is a short explanation, not a transcript.
_RUN_STEP_REASON_LIMIT = 500


def _resolve_delegation_provenance(normalized: Mapping[str, object]) -> str:
    """Resolve where the delegation actually ran, from the terminal's own fields.

    OMN-15471: this used to be ``normalized.get("provider", "local")``. No real
    ``delegation-completed.v1`` payload carries a ``provider`` key — that event
    models the resolved serving endpoint as ``endpoint_url`` plus
    ``cost_tier_name`` — so the literal default fired on EVERY bus-path
    delegation and stamped ``provider="local"`` on the durable terminal. A
    Gemini-routed result (``endpoint_url`` = the Google Generative Language API,
    ``cost_tier_name`` = ``cheap_cloud``) was recorded as a local-provider run:
    39/39 ``delegate-skill-completed.v1`` rows read ``local`` on the shared dev
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


@functools.lru_cache(maxsize=256)
def _declared_run_step_event(path: str, mtime_ns: int, size: int) -> str | None:
    """A contract file's ``runtime_dispatch.run_step_event``, read once per version.

    ``mtime_ns`` and ``size`` are the cache key's version stamp: a rewritten
    contract is read again, an unchanged one never is.
    """
    try:
        document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        logger.warning("contract %s could not be read for its run-step topic", path)
        return None
    runtime_dispatch = (
        document.get("runtime_dispatch") if isinstance(document, dict) else None
    )
    if not isinstance(runtime_dispatch, dict):
        return None
    return _text(runtime_dispatch.get("run_step_event"))


def _run_step_topic(
    routes: Mapping[str, ModelRuntimeLocalIngressRoute], package_name: str
) -> str | None:
    """The one run-step topic the delegation engine's package declares."""
    declared: set[str] = set()
    for contract_path in sorted(
        {
            route.contract_path
            for route in routes.values()
            if route.package_name == package_name and route.contract_path
        }
    ):
        try:
            stat = Path(contract_path).stat()
        except OSError:
            continue
        topic = _declared_run_step_event(contract_path, stat.st_mtime_ns, stat.st_size)
        if topic is not None:
            declared.add(topic)
    if len(declared) == 1:
        return next(iter(declared))
    if declared:
        logger.warning(
            "ambiguous runtime_dispatch.run_step_event declarations in %s: %s",
            package_name,
            sorted(declared),
        )
    return None


def _run_step_message(
    step: ModelDelegationRunStepEvent, correlation_id: UUID
) -> bytes | None:
    """One step as a serialized event envelope; None when it cannot be built."""
    try:
        envelope = ModelEventEnvelope[dict[str, object]](
            payload=step.model_dump(mode="json"),
            correlation_id=correlation_id,
            envelope_timestamp=datetime.now(UTC),
            event_type=_RUN_STEP_EVENT_TYPE,
            source_tool=_RUN_STEP_SOURCE_TOOL,
            tenant_id=step.tenant_id,
        )
        return envelope.model_dump_json(exclude_none=True).encode("utf-8")
    except ValueError:
        # A trace gap, never a failed delegation.
        logger.warning(
            "run step %d of correlation_id=%s could not be serialized",
            step.step_sequence,
            correlation_id,
            exc_info=True,
        )
        return None


class HandlerDelegationDispatchTerminal:
    """Turn a dispatched run's terminal into its result and its step messages."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self,
        request: ModelDelegationDispatchTerminalRequest,
        *,
        routes: Mapping[str, ModelRuntimeLocalIngressRoute],
    ) -> ModelDelegationDispatchTerminalResult:
        """Normalize the terminal; for a waited run, also build its step messages.

        Best-effort for the steps: no declared topic, no tenant or a step that
        cannot be built is a gap in the trace, never a failed delegation.
        """
        terminal = request.terminal
        normalized = _normalize_result_payload(
            status=terminal.status,
            payload=terminal.payload,
            error_message=terminal.error_message,
        )
        if not request.publish_steps:
            return ModelDelegationDispatchTerminalResult(result=normalized)
        topic = await asyncio.to_thread(_run_step_topic, routes, request.engine_package)
        if topic is None:
            logger.warning(
                "run steps of correlation_id=%s not published: no installed "
                "contract declares one runtime_dispatch.run_step_event",
                request.correlation_id,
            )
            return ModelDelegationDispatchTerminalResult(result=normalized)
        flat = normalized if isinstance(terminal.payload, dict) else None
        tenant = (None if flat is None else _text(flat.get("tenant_id"))) or _text(
            request.tenant_id
        )
        if tenant is None:
            logger.warning(
                "run steps of correlation_id=%s not published: no tenant on the "
                "request or the terminal",
                request.correlation_id,
            )
            return ModelDelegationDispatchTerminalResult(result=normalized)
        sequence = RunStepSequence(
            tenant_id=tenant,
            correlation_id=request.correlation_id,
            started_at=request.started_at,
            started=request.started_monotonic,
        )
        try:
            _record_bus_run(
                sequence,
                task_type=request.task_type,
                status=terminal.status,
                flat=flat,
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            logger.warning(
                "run steps of correlation_id=%s could not be derived",
                request.correlation_id,
                exc_info=True,
            )
        messages = tuple(
            message
            for step in sequence.steps
            if (message := _run_step_message(step, request.correlation_id)) is not None
        )
        return ModelDelegationDispatchTerminalResult(
            result=normalized, run_step_topic=topic, run_step_messages=messages
        )


__all__ = ["HandlerDelegationDispatchTerminal"]
