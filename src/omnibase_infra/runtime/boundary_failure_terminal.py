# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Terminalizing a consume-boundary handler failure (OMN-16812).

OMN-16798 made the auto-wired consume boundary safe for the RECORD: a handler
exception is DLQ'd, and an offset is never committed over a record that landed
nowhere durable. It deliberately did nothing for the CALLER. On the ``.201`` dev
lane (revision ``4529c3486a8b``, correlation ``7a300827-1000-…-000000000012``)
that produced the following, twice, deterministically::

    [ERROR] handler_wiring: Auto-wiring callback error:
      topic=onex.cmd.omnibase-infra.delegation-routing-request.v1
      error_type=HandlerDispatchFailureError
      error=... ProtocolConfigurationError: [ONEX_CORE_041_INVALID_CONFIGURATION]
            No tier has a configured endpoint for task_type='agent_delegation' ...
    [ERROR] handler_wiring: metric_name=boundary_swallow_prevented dlq_routed=true

...within milliseconds, followed by 120 seconds of nothing and then::

    {"ok": false, "error": {"code": "dispatch_timeout", "retryable": true}}

Two separate defects live in that answer, both independent of whatever made the
handler raise:

1. **Misattribution.** ``dispatch_timeout`` means "we do not know". The runtime
   knew the class — ``ONEX_CORE_041_INVALID_CONFIGURATION`` — before the caller
   had finished its first second of waiting, and then discarded it.
2. **A false ``retryable``.** A missing routing endpoint is not fixed by trying
   again. A caller that obeys ``retryable: true`` retries forever.

This module supplies the two things the boundary needs to answer honestly:

* :class:`ModelBoundaryFailureTerminal` — the typed terminal payload. It is not
  a new error channel: it is published onto the failing contract's OWN declared
  failure terminal topic, as a ``ModelEventEnvelope``, with the field names the
  existing terminal readers already consult. ``status="failed"`` is what
  :func:`~omnibase_infra.runtime.contract_terminal_events.resolve_terminal_verdict`
  reads, and ``failure_reason`` is what the Pattern B broker's
  ``_terminal_error_message`` reads. Nothing downstream needs to learn a new
  shape to benefit.
* :func:`classify_boundary_failure` — the attribution. It reads the class and
  the ONEX error code out of the exception the boundary is already holding, and
  derives ``retryable`` from :class:`EnumNonRetryableErrorCategory`, the enum
  DLQ replay has classified retry eligibility with since OMN-1032. Reusing that
  enum is the point: "is this worth retrying" must not have two answers in one
  runtime.

WHY THE ATTRIBUTION HAS TO READ THE MESSAGE TEXT. The exception the boundary
catches is ``HandlerDispatchFailureError``, raised by
``_raise_if_silent_dispatch_failure`` from a FAILED ``ModelDispatchResult``.
``MessageDispatchEngine.dispatch()`` caught the handler's real exception inside
itself and recorded it as ``error_message`` / ``error_code`` — the exception
OBJECT does not survive that hop, so there is no ``__cause__`` to walk to. The
chain walk here covers the directly-raised case, and the token scan covers the
engine-flattened case; the boundary hands over ``error_code`` explicitly where
it has it. All three read the same original failure.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums.enum_non_retryable_error_category import (
    EnumNonRetryableErrorCategory,
)

__all__ = [
    "ModelBoundaryFailureTerminal",
    "classify_boundary_failure",
]

# Exception-chain walk depth. A boundary failure is at most a handful of frames
# deep (handler -> engine wrapper -> boundary guard); the bound exists so a
# pathological self-referential ``__context__`` cannot spin here.
_MAX_CHAIN_DEPTH = 8

# ``SomethingError`` / ``SomethingException`` appearing as a whole token in a
# flattened message. Deliberately anchored on the conventional suffixes rather
# than "any CapWord": a bare CapWord scan would match ordinary prose in an error
# message and mint a fabricated failure class.
_ERROR_CLASS_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_])([A-Z][A-Za-z0-9_]*(?:Error|Exception))(?![A-Za-z0-9_])"
)
# The canonical ONEX code shape, e.g. ``ONEX_CORE_041_INVALID_CONFIGURATION``.
_ONEX_CODE_TOKEN = re.compile(r"(?<![A-Za-z0-9_])(ONEX_[A-Z0-9_]+)(?![A-Za-z0-9_])")

# Boundary-internal wrapper types. They describe WHERE the failure was noticed,
# never WHAT failed, so they must not win the ``failure_class`` attribution over
# a real cause — but see ``classify_boundary_failure``: they are still a valid
# last-resort answer when nothing more specific exists anywhere.
_BOUNDARY_WRAPPER_CLASS_NAMES: frozenset[str] = frozenset(
    {
        "BoundaryApplyPublishError",
        "BoundaryDlqNotPersistedError",
        "BoundaryPublishError",
        "HandlerDispatchFailureError",
        "UndeliverableDispatchOutputError",
    }
)


class ModelBoundaryFailureTerminal(BaseModel):
    """The terminal a consume boundary emits when it fails a record for good.

    Emitted at exactly one moment: the boundary is about to commit an offset
    over a handler failure, so no further attempt will be made and no other
    surface will ever produce a terminal for this correlation. Anything waiting
    on the contract's terminal topics is, at that instant, waiting on nothing —
    this is the record that says so.

    Field choices are dictated by the readers that already exist rather than by
    what would be tidy in isolation:

    * ``status`` is the ``completed``/``failed`` wire vocabulary
      ``resolve_terminal_verdict`` terminalizes on, so the Pattern B broker
      derives ``failed`` from this payload even on a contract whose single
      declared terminal is nominally the success topic.
    * ``failure_reason`` is the first key ``_terminal_error_message`` looks for,
      so the caller's error message is the attributed cause and not a
      placeholder.
    * ``retryable`` is stated explicitly because the ingress previously DERIVED
      it (``retryable = status == "timeout"``) and had no way to be told
      otherwise.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID = Field(
        ...,
        description=(
            "Correlation of the record that failed. Correlation-exact by "
            "construction: taken from the in-flight envelope, never minted here."
        ),
    )
    status: Literal["failed"] = Field(
        default="failed",
        description=(
            "Terminal verdict in the runtime's own wire vocabulary. Always "
            "'failed' — this model exists only for the failure moment."
        ),
    )
    failure_class: str = Field(
        ...,
        description=(
            "Originating error class, e.g. 'ProtocolConfigurationError'. The "
            "attributed cause the ticket's AC2 requires, never 'timeout'."
        ),
    )
    failure_code: str | None = Field(
        default=None,
        description=(
            "Canonical ONEX error code when one is recoverable from the "
            "failure, e.g. 'ONEX_CORE_041_INVALID_CONFIGURATION'. None when the "
            "failure carried no code — an absent code is reported as absent "
            "rather than guessed at."
        ),
    )
    retryable: bool = Field(
        ...,
        description=(
            "Whether retrying this record could plausibly succeed. Derived from "
            "EnumNonRetryableErrorCategory — the same classifier DLQ replay "
            "uses — so a configuration or validation failure is False."
        ),
    )
    failure_reason: str = Field(
        ...,
        description=(
            "Sanitized human-readable cause. Read by the Pattern B broker's "
            "_terminal_error_message and surfaced to the caller."
        ),
    )
    origin_topic: str = Field(
        ...,
        description="Topic whose consume boundary failed the record.",
    )
    terminalized_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC instant the boundary gave up on the record.",
    )


def _exception_chain(exc: BaseException) -> Iterator[BaseException]:
    """Yield ``exc`` and its ``__cause__``/``__context__`` ancestry, once each.

    ``__cause__`` (explicit ``raise ... from``) is preferred over ``__context__``
    (implicit during-handling) at each step, matching how a reader of the
    traceback would attribute the failure, but both are followed: several infra
    boundaries re-raise inside an ``except`` block without ``from``.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    depth = 0
    while current is not None and depth < _MAX_CHAIN_DEPTH:
        if id(current) in seen:
            return
        seen.add(id(current))
        yield current
        depth += 1
        current = current.__cause__ or current.__context__


def _candidate_class_names(exc: BaseException) -> tuple[str, ...]:
    """Every error class name observable from ``exc``, most-specific-first.

    Two sources, in order of trust: the actual types on the exception chain,
    then class-name tokens parsed out of the messages. The second source is what
    covers the live shape in this module's docstring, where the engine flattened
    the real ``ProtocolConfigurationError`` into a string before the boundary
    ever saw it.
    """
    ordered: list[str] = []
    messages: list[str] = []
    for item in _exception_chain(exc):
        name = type(item).__name__
        if name not in ordered:
            ordered.append(name)
        messages.append(str(item))
    for message in messages:
        for token in _ERROR_CLASS_TOKEN.findall(message):
            if token not in ordered:
                ordered.append(token)
    return tuple(ordered)


def _first_onex_code(exc: BaseException) -> str | None:
    """Return the first canonical ONEX error code found on the chain, if any."""
    for item in _exception_chain(exc):
        code = getattr(item, "error_code", None)
        code_value = getattr(code, "value", code)
        if isinstance(code_value, str) and _ONEX_CODE_TOKEN.fullmatch(code_value):
            return code_value
        match = _ONEX_CODE_TOKEN.search(str(item))
        if match is not None:
            return match.group(1)
    return None


def classify_boundary_failure(
    exc: BaseException,
    *,
    topic: str,
    correlation_id: UUID,
    failure_reason: str,
    failure_code: str | None = None,
) -> ModelBoundaryFailureTerminal:
    """Attribute ``exc`` and build the terminal the boundary will publish.

    ``failure_code`` is a FALLBACK, not an override. It is the FAILED dispatch
    result's typed ``error_code``, which the engine sets to the generic
    ``HANDLER_EXECUTION_ERROR`` for every dispatcher crash alike — it says a
    handler failed, never which failure. The specific code lives in the
    flattened message (``sanitize_error_message`` stamps
    ``ProtocolConfigurationError: [ONEX_CORE_041_INVALID_CONFIGURATION] …``), so
    a recovered code always wins over the passed one. When neither exists the
    field stays ``None`` rather than being invented.

    ``retryable`` is False when ANY class name observable from the failure is
    one :class:`EnumNonRetryableErrorCategory` names. "Any" rather than "the
    outermost" is deliberate: the boundary wrapper is always the outermost class
    and is never itself in that enum, so an outermost-only reading would call
    every engine-flattened configuration error retryable — precisely the live
    defect. The direction of the bias is also deliberate: this decides whether a
    caller should try again, and wrongly saying "retryable" costs an unbounded
    retry loop while wrongly saying "not retryable" costs one surfaced error.
    """
    candidates = _candidate_class_names(exc)
    retryable = not any(
        EnumNonRetryableErrorCategory.is_non_retryable(name) for name in candidates
    )
    specific = next(
        (name for name in candidates if name not in _BOUNDARY_WRAPPER_CLASS_NAMES),
        None,
    )
    return ModelBoundaryFailureTerminal(
        correlation_id=correlation_id,
        failure_class=specific or type(exc).__name__,
        failure_code=_first_onex_code(exc) or failure_code,
        retryable=retryable,
        failure_reason=failure_reason,
        origin_topic=topic,
    )
