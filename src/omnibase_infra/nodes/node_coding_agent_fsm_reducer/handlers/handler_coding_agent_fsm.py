# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Coding-agent FSM REDUCER handler (OMN-13247, plan §5.2).

The deterministic-execution core. Pure fold: ``delta(state, event) ->
(state, intents[])``. No I/O, no bus, no DB. Owns:

  - the FSM: IDLE -> VALIDATING -> INVOKING -> CAPTURING ->
    COMPLETED | FAILED | REJECTED
  - retry policy: WORKSPACE_WRITE invocations are NEVER auto-retried; a duplicate
    correlation_id never re-runs (recognized from state); READ_ONLY may retry up
    to a bounded count.
  - the N-failure circuit breaker -> FAILED.
  - replay safety: replaying historical events recomputes state + projection but
    issues NO live intent (a pure reducer cannot itself perform I/O, so replay
    can never cause a subprocess).
  - the correlation-trace projection (plan §5.6).

The handler returns ``for_reducer`` carrying the advanced FSM state + the trace
projection. Intents are returned by the pure ``delta`` helper for the
orchestrator to translate into live bus commands; the reducer never publishes.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import TypeVar
from uuid import uuid4

from omnibase_core.models.dispatch.model_handler_output import ModelHandlerOutput
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.coding_agent.enum_agent_sandbox import EnumAgentSandbox
from omnibase_infra.models.coding_agent.enum_coding_agent_event_kind import (
    EnumCodingAgentEventKind,
)
from omnibase_infra.models.coding_agent.enum_coding_agent_fsm_state import (
    TERMINAL_STATES,
    EnumCodingAgentFsmState,
)
from omnibase_infra.models.coding_agent.enum_coding_agent_intent_kind import (
    EnumCodingAgentIntentKind,
)
from omnibase_infra.models.coding_agent.model_coding_agent_event import (
    ModelCodingAgentEvent,
)
from omnibase_infra.models.coding_agent.model_coding_agent_fsm_state import (
    ModelCodingAgentFsmState,
)
from omnibase_infra.models.coding_agent.model_coding_agent_intent import (
    ModelCodingAgentIntent,
)
from omnibase_infra.models.coding_agent.model_coding_agent_trace_projection import (
    ModelCodingAgentTraceProjection,
)

HANDLER_ID = "coding-agent-fsm-reducer"

# Dispatch payloads are coerced at runtime; the protocol entry is generic over the
# envelope payload type (ProtocolMessageHandler.handle(ModelEventEnvelope[T])).
T = TypeVar("T")


def _intent(
    state: ModelCodingAgentFsmState,
    kind: EnumCodingAgentIntentKind,
    *,
    is_replay: bool,
) -> tuple[ModelCodingAgentIntent, ...]:
    """Build a single next-step intent, marked non-live on replay.

    On replay the reducer recomputes state but must not request a live action,
    so the intent is emitted with ``is_live=False``. The orchestrator must not
    act on a non-live intent. (Replay safety, plan §5.2.)
    """
    return (
        ModelCodingAgentIntent(
            correlation_id=state.correlation_id,
            kind=kind,
            is_live=not is_replay,
        ),
    )


def delta(
    state: ModelCodingAgentFsmState,
    event: ModelCodingAgentEvent,
) -> tuple[ModelCodingAgentFsmState, tuple[ModelCodingAgentIntent, ...]]:
    """Pure FSM transition. Returns the next state and any next-step intents.

    Deterministic and side-effect free. Duplicate / out-of-order events that do
    not match the current state are folded as no-ops (the same state, no
    intents) — this is the same-correlation_id dedupe guard: an event for a run
    already past that point never re-runs the effect.
    """
    is_replay = event.is_replay
    kind = event.kind
    current = state.current_state

    # A run already in a terminal state never advances or re-runs anything.
    if current in TERMINAL_STATES:
        return state, ()

    if kind == EnumCodingAgentEventKind.INVOKE_REQUESTED:
        if current != EnumCodingAgentFsmState.IDLE:
            # Duplicate invoke-requested for an in-flight run: no re-run.
            return state, ()
        new_state = state.model_copy(
            update={"current_state": EnumCodingAgentFsmState.VALIDATING}
        )
        return new_state, _intent(
            new_state, EnumCodingAgentIntentKind.DISPATCH_VALIDATE, is_replay=is_replay
        )

    if kind == EnumCodingAgentEventKind.WORKSPACE_REJECTED:
        if current != EnumCodingAgentFsmState.VALIDATING:
            return state, ()
        new_state = state.model_copy(
            update={
                "current_state": EnumCodingAgentFsmState.REJECTED,
                "error_message": event.error_message or "workspace rejected",
            }
        )
        # Terminal — no subprocess ever runs.
        return new_state, _intent(
            new_state, EnumCodingAgentIntentKind.EMIT_TERMINAL, is_replay=is_replay
        )

    if kind == EnumCodingAgentEventKind.WORKSPACE_OK:
        if current != EnumCodingAgentFsmState.VALIDATING:
            return state, ()
        new_state = state.model_copy(
            update={
                "current_state": EnumCodingAgentFsmState.INVOKING,
                "invoke_attempts": state.invoke_attempts + 1,
            }
        )
        return new_state, _intent(
            new_state, EnumCodingAgentIntentKind.DISPATCH_INVOKE, is_replay=is_replay
        )

    if kind == EnumCodingAgentEventKind.INVOKE_COMPLETED:
        if current != EnumCodingAgentFsmState.INVOKING:
            return state, ()
        new_state = state.model_copy(
            update={
                "current_state": EnumCodingAgentFsmState.CAPTURING,
                "consecutive_failures": 0,
            }
        )
        return new_state, _intent(
            new_state, EnumCodingAgentIntentKind.DISPATCH_CAPTURE, is_replay=is_replay
        )

    if kind == EnumCodingAgentEventKind.INVOKE_FAILED:
        if current != EnumCodingAgentFsmState.INVOKING:
            return state, ()
        return _on_invoke_failed(state, event, is_replay=is_replay)

    if kind == EnumCodingAgentEventKind.DIFF_CAPTURED:
        if current != EnumCodingAgentFsmState.CAPTURING:
            return state, ()
        new_state = state.model_copy(
            update={"current_state": EnumCodingAgentFsmState.COMPLETED}
        )
        return new_state, _intent(
            new_state, EnumCodingAgentIntentKind.EMIT_TERMINAL, is_replay=is_replay
        )

    return state, ()


def _on_invoke_failed(
    state: ModelCodingAgentFsmState,
    event: ModelCodingAgentEvent,
    *,
    is_replay: bool,
) -> tuple[ModelCodingAgentFsmState, tuple[ModelCodingAgentIntent, ...]]:
    """Apply the retry policy + circuit breaker to a failed invocation."""
    new_failures = state.consecutive_failures + 1
    err = event.error_message or "invoke failed"

    # Circuit breaker: N consecutive failures -> FAILED, no further retries.
    if new_failures >= state.max_consecutive_failures:
        failed = state.model_copy(
            update={
                "current_state": EnumCodingAgentFsmState.FAILED,
                "consecutive_failures": new_failures,
                "error_message": (
                    f"circuit breaker: {new_failures} consecutive failures: {err}"
                ),
            }
        )
        return failed, _intent(
            failed, EnumCodingAgentIntentKind.EMIT_TERMINAL, is_replay=is_replay
        )

    # WORKSPACE_WRITE is NEVER auto-retried: a partial edit must not be blindly
    # re-run. Fail immediately (do not trip the breaker count toward a retry).
    if state.sandbox == EnumAgentSandbox.WORKSPACE_WRITE:
        failed = state.model_copy(
            update={
                "current_state": EnumCodingAgentFsmState.FAILED,
                "consecutive_failures": new_failures,
                "error_message": (
                    f"workspace-write invocation failed (never auto-retried): {err}"
                ),
            }
        )
        return failed, _intent(
            failed, EnumCodingAgentIntentKind.EMIT_TERMINAL, is_replay=is_replay
        )

    # READ_ONLY: bounded retry if budget remains; else FAILED.
    if state.invoke_attempts > state.max_read_only_retries:
        failed = state.model_copy(
            update={
                "current_state": EnumCodingAgentFsmState.FAILED,
                "consecutive_failures": new_failures,
                "error_message": f"read-only retry budget exhausted: {err}",
            }
        )
        return failed, _intent(
            failed, EnumCodingAgentIntentKind.EMIT_TERMINAL, is_replay=is_replay
        )

    retried = state.model_copy(
        update={
            "current_state": EnumCodingAgentFsmState.INVOKING,
            "consecutive_failures": new_failures,
            "invoke_attempts": state.invoke_attempts + 1,
            "error_message": err,
        }
    )
    return retried, _intent(
        retried, EnumCodingAgentIntentKind.DISPATCH_INVOKE, is_replay=is_replay
    )


def _hash(value: str) -> str:
    """Stable short hash for projection fields (empty in -> empty out)."""
    if not value:
        return ""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def project(state: ModelCodingAgentFsmState) -> ModelCodingAgentTraceProjection:
    """Materialize the correlation-trace projection from the FSM state."""
    return ModelCodingAgentTraceProjection(
        correlation_id=state.correlation_id,
        agent=state.agent,
        sandbox=state.sandbox,
        status=state.current_state,
    )


class HandlerCodingAgentFsm:
    """Pure reducer: fold one event into the FSM state + trace projection."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    async def handle(self, envelope: ModelEventEnvelope[T]) -> ModelHandlerOutput[None]:
        """Fold the event and emit the advanced state + projection.

        The handler returns the FSM state and the correlation-trace projection as
        reducer projections. The pure ``delta`` helper computes the next-step
        intents for the orchestrator; the reducer itself emits no events/intents.
        """
        state, event = _coerce(envelope.payload)
        new_state, _intents = delta(state, event)
        projection = project(new_state)
        return ModelHandlerOutput.for_reducer(
            input_envelope_id=envelope.envelope_id,
            correlation_id=(
                envelope.correlation_id or new_state.correlation_id or uuid4()
            ),
            handler_id=HANDLER_ID,
            projections=(new_state, projection),
        )


def _coerce(
    payload: object,
) -> tuple[ModelCodingAgentFsmState, ModelCodingAgentEvent]:
    """Coerce the dispatched payload into (prior FSM state, fold event)."""
    mapping = _as_mapping(payload)
    if mapping is None:
        raise TypeError(
            "coding-agent FSM payload must be a mapping/model carrying 'state' "
            f"and 'event'; got {type(payload).__name__}"
        )
    if "state" not in mapping or "event" not in mapping:
        raise ValueError(
            "coding-agent FSM payload must carry both 'state' and 'event' keys"
        )
    state = ModelCodingAgentFsmState.model_validate(_as_dict(mapping["state"]))
    event = ModelCodingAgentEvent.model_validate(_as_dict(mapping["event"]))
    return state, event


def _as_mapping(candidate: object) -> Mapping[str, object] | None:
    if isinstance(candidate, Mapping):
        return candidate
    model_dump = getattr(candidate, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json")
        if isinstance(dumped, Mapping):
            return dumped
    return None


def _as_dict(candidate: object) -> dict[str, object]:
    if isinstance(candidate, Mapping):
        return dict(candidate)
    model_dump = getattr(candidate, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json")
        if isinstance(dumped, Mapping):
            return dict(dumped)
    raise TypeError(f"expected a mapping/model; got {type(candidate).__name__}")


__all__: list[str] = [
    "HANDLER_ID",
    "HandlerCodingAgentFsm",
    "delta",
    "project",
]
