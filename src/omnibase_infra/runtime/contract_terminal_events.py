# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract-declared terminal events: one reader, one on-the-wire shape (OMN-15468).

Two things must agree about a contract's terminal events, and before this module
they were written twice and agreed only by accident:

1. **Which topics are terminal.** ``runtime_local_ingress`` reads them to build
   the Pattern B broker's subscription set. The def-B auto-wiring needs the same
   answer to know which handler publishes are terminal emissions. A second
   hand-rolled reader in the wiring layer would be free to drift from the
   broker's view of the same contract — the exact class of seam mismatch that
   produced this ticket. :func:`extract_terminal_event_topics` is the single
   reader; ``runtime_local_ingress._extract_terminal_events`` delegates to it.

2. **What a terminal record looks like on the bus.** ``DispatchResultApplier``
   publishes the def-B return value as a full :class:`ModelEventEnvelope`. A
   handler that emits its own terminal through the wiring-injected
   ``event_publisher`` published the bytes it was handed — RAW, un-enveloped.
   Same contract, same terminal pair, two different wire shapes depending on
   which half of the wiring did the publishing.

   Live proof (``.201`` dev lane, 2026-07-30T17:13Z, merged ``5dc68190`` — the
   readback that reopened OMN-15468 after #2560): for correlation
   ``4a5e0730-…-000000000002`` the SUCCESS topic carried a full envelope
   (``event_type='omnimarket.node-generation-completed'``, 19 envelope keys,
   ``payload.contract_passed=False``) while the FAILURE topic carried
   ``{"correlation_id": …, "task_description": …, "attempts": [...]}`` with no
   ``event_type``, no ``envelope_id`` and no ``payload`` wrapper. #2560 had
   already made the broker subscribe to the failure topic (proven by the
   runtime's own ``Updating subscribed topics to: frozenset({'…-failed.v1'})``
   log line at both probe start times), so the subscription was live and the
   record was sitting on it — but the shape is not one the broker's
   envelope-decoding terminal path accepts, and the outer ``/skill`` response
   stayed ``ok=true`` / ``status=completed`` / ``error=null``, byte-identical to
   the success control.

:func:`envelope_terminal_payload` closes (2) at the ONE factory that hands every
def-B handler its publisher, so the fix lands for every contract that declares a
terminal event rather than per node. Field parity with the applier's success
envelope is by construction: the same ``derive_event_type_from_topic`` helper
stamps ``event_type``, the same ``uuid5``-from-correlation scheme mints
``envelope_id``, and the same ``correlation_id``/``envelope_timestamp`` fields
are populated.

3. **Whether a terminal record is telling the truth.** ``DispatchResultApplier``
   routed the def-B return value by class name alone: a model missing the
   contract's ``published_events`` map fell back to the SUCCESS terminal
   whatever verdict its payload carried, and the Pattern B broker then derived
   ``completed`` from the arrival topic. :func:`resolve_terminal_verdict` reads
   the verdict the payload already states and
   :func:`apply_failure_terminal_guard` re-routes it to the contract's declared
   failure terminal — fail-closed, and only when the model states a failure.
   This is OMN-15468 acceptance criterion 2, which the earlier revision of this
   docstring correctly recorded as still open; it is closed here.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Container, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid5

import yaml

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.utils import derive_event_type_from_topic

logger = logging.getLogger(__name__)

# A record carrying any of these keys is already an envelope; wrapping it again
# would nest a terminal inside a terminal and break the broker's single unwrap.
_ENVELOPE_MARKER_KEYS: frozenset[str] = frozenset(
    {"envelope_id", "envelope_version", "payload_type", "envelope_timestamp"}
)

__all__ = [
    "apply_failure_terminal_guard",
    "envelope_terminal_payload",
    "extract_terminal_event_topics",
    "load_terminal_event_topics",
    "resolve_terminal_verdict",
    "terminal_event_topics_from_declaration",
]

# Status strings that name a terminal outcome, in the vocabulary the runtime
# already uses on the wire (``ModelDispatchBusTerminalResult.status`` is
# completed/failed/timeout; producers additionally use error/cancelled).
_FAILED_STATUS_VALUES: frozenset[str] = frozenset(
    {"failed", "failure", "timeout", "timed_out", "error", "cancelled", "canceled"}
)
_SUCCEEDED_STATUS_VALUES: frozenset[str] = frozenset(
    {"completed", "complete", "succeeded", "success", "ok"}
)


def _safe_optional_string(value: object) -> str | None:
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    return None


def terminal_event_topics_from_declaration(declaration: object) -> tuple[str, ...]:
    """Normalize one ``terminal_events`` declaration into success-first topics.

    A mapping declaration is emitted with its ``success`` entry FIRST, regardless
    of YAML key order, because the Pattern B broker treats
    ``terminal_events[0]`` as the success topic whenever the contract has no
    top-level ``terminal_event`` (``_status_for_terminal_topic``). Leaving that
    to mapping order would make a terminal's completed-vs-failed meaning depend
    on how the contract author happened to sort two YAML keys.
    """

    if isinstance(declaration, dict):
        ordered: list[object] = []
        if "success" in declaration:
            ordered.append(declaration["success"])
        ordered.extend(value for key, value in declaration.items() if key != "success")
        values: Iterable[object] = ordered
    elif isinstance(declaration, list | tuple):
        values = declaration
    else:
        values = ()

    topics: list[str] = []
    for value in values:
        topic = _safe_optional_string(value)
        if topic is not None:
            topics.append(topic)
    return tuple(topics)


def extract_terminal_event_topics(raw: Mapping[object, object]) -> tuple[str, ...]:
    """Return all contract-declared terminal topics, success-first, de-duplicated.

    Reads three declaration sites, in success-first order:

    1. top-level ``terminal_event`` (single success topic),
    2. top-level ``terminal_events`` (mapping or sequence),
    3. ``runtime_dispatch.terminal_events`` (OMN-15468 / #2560).

    Site 3 is the address external clients — the dashboard included — dispatch
    through, and it is where **51** contracts declare their FAILURE terminal and
    nowhere else.

    PROVENANCE — every number here is a measurement, not a constant. Framing: the
    RAW corpus of 384 ``src/omnimarket/nodes/*/contract.yaml`` files at
    ``omnimarket@aea0c33dd89fb82fdca33aac7149992a21c46d43`` (``origin/dev``),
    measured 2026-07-30, no discovery filter. Re-derive rather than copy forward:
    51 = contracts whose ``runtime_dispatch.terminal_events`` normalizes
    non-empty; 30 of those 51 declare no top-level ``terminal_event`` *or*
    ``terminal_events``; 17 of those 30 also clear the route-discovery filter.
    These drift as contracts land.
    """

    terminal_events: list[str] = []
    terminal_event = _safe_optional_string(raw.get("terminal_event"))
    if terminal_event is not None:
        terminal_events.append(terminal_event)

    terminal_events.extend(
        terminal_event_topics_from_declaration(raw.get("terminal_events"))
    )

    runtime_dispatch = raw.get("runtime_dispatch")
    if isinstance(runtime_dispatch, dict):
        terminal_events.extend(
            terminal_event_topics_from_declaration(
                runtime_dispatch.get("terminal_events")
            )
        )

    return tuple(dict.fromkeys(terminal_events))


def load_terminal_event_topics(contract_path: Path | None) -> frozenset[str]:
    """Read a contract file and return its declared terminal topics.

    Fail-open by design: an unreadable or non-mapping contract yields the empty
    set, which restores the pre-OMN-15468 publish behavior (bytes forwarded
    verbatim) rather than failing a wiring that has nothing to do with terminals.
    The contract discovery path already validated this file; a failure here means
    the file moved or the runtime lost read access, and a handler must still be
    constructible in that state.
    """

    if contract_path is None:
        return frozenset()
    path = contract_path
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        logger.warning(
            "contract_terminal_events: could not read terminal declarations from "
            "%s (%s) — handler publishes will be forwarded un-enveloped",
            path,
            type(exc).__name__,
        )
        return frozenset()
    if not isinstance(raw, Mapping):
        return frozenset()
    return frozenset(extract_terminal_event_topics(raw))


def resolve_terminal_verdict(event: object) -> bool | None:
    """Read a returned model's OWN terminal verdict. ``None`` means unknown.

    OMN-15468 AC2. ``DispatchResultApplier`` routes a definition-B return value
    to the contract's SUCCESS terminal whenever the model's class name misses
    the ``published_events`` map — the payload's verdict is never consulted.
    Live on 2026-07-30 that republished a ``contract_passed=false`` benchmark
    onto ``node-generation-completed.v1``, and the Pattern B broker, which
    derives status purely from the arrival topic, reported ``ok=true`` for a run
    that had failed. This function is the missing read.

    Fields are consulted in decreasing order of explicitness. Each is a field
    ONEX producers already carry; nothing new is required of a handler to be
    covered:

    1. ``terminal_failure_cause`` — a typed non-``None`` cause is an
       unambiguous failure declaration (the delegate-skill seam, OMN-15469).
    2. ``status`` — the wire vocabulary the broker itself terminalizes on.
    3. ``ok`` / ``success`` — explicit booleans.
    4. ``contract_passed`` — the generation-benchmark verdict field from this
       ticket's original live reproduction.

    Returns ``None`` when the model declares no verdict at all, which is the
    common case and MUST leave routing exactly as it was: this is a fail-closed
    correction for models that state a failure, never a guess about models that
    state nothing.
    """
    cause = getattr(event, "terminal_failure_cause", None)
    if cause is not None:
        return False

    status = getattr(event, "status", None)
    if isinstance(status, str):
        normalized = status.strip().lower()
        if normalized in _FAILED_STATUS_VALUES:
            return False
        if normalized in _SUCCEEDED_STATUS_VALUES:
            return True

    for attribute in ("ok", "success", "contract_passed"):
        value = getattr(event, attribute, None)
        if isinstance(value, bool):
            return value

    return None


def apply_failure_terminal_guard(
    event: object,
    topic: str,
    *,
    success_topic: str,
    failure_terminal_topics: Sequence[str],
) -> str:
    """Re-route a failure-verdict return value off the SUCCESS terminal.

    OMN-15468 AC2 — the residual PR #2578 left open. Class-based routing (a
    handler returning a ``…Failed`` variant declared in ``published_events``) is
    the primary mechanism and this guard never overrides it: it fires ONLY when
    the resolution already landed on the contract's success terminal. It is the
    fail-closed backstop for the case that produced the live defect — a returned
    model whose class misses the map, which falls back to the success terminal
    no matter what verdict the payload carries.

    Three conditions must ALL hold, so the guard cannot fire speculatively:

    * the resolved topic is the contract's success terminal;
    * the contract declares exactly ONE failure terminal (two or more is
      ambiguous — there is no basis to choose, so routing is left alone and the
      ambiguity is logged rather than guessed at);
    * the model declares an explicit failure verdict
      (:func:`resolve_terminal_verdict` returns ``False``; ``None`` — no verdict
      field at all, the common case — changes nothing).

    Lives here rather than on ``DispatchResultApplier`` so that *which topics
    are terminal* and *what a failure verdict is* stay in the one module that
    already owns both questions for the broker's subscription set.
    """
    if topic != success_topic:
        return topic
    if len(failure_terminal_topics) != 1:
        if failure_terminal_topics and resolve_terminal_verdict(event) is False:
            logger.warning(
                "Failure-verdict output event left on the success terminal: "
                "contract declares %d failure terminals, cannot disambiguate "
                "(OMN-15468)",
                len(failure_terminal_topics),
                extra={
                    "output_event_type": type(event).__name__,
                    "success_topic": topic,
                    "failure_topics": list(failure_terminal_topics),
                },
            )
        return topic
    if resolve_terminal_verdict(event) is not False:
        return topic
    failure_topic = failure_terminal_topics[0]
    logger.warning(
        "Re-routing failure-verdict output event from success terminal %s to "
        "declared failure terminal %s (OMN-15468)",
        topic,
        failure_topic,
        extra={
            "output_event_type": type(event).__name__,
            "success_topic": topic,
            "failure_topic": failure_topic,
        },
    )
    return failure_topic


def _is_already_envelope(body: Mapping[str, object]) -> bool:
    return bool(_ENVELOPE_MARKER_KEYS & set(body)) and "payload" in body


def envelope_terminal_payload(
    *,
    topic: str,
    payload: bytes,
    terminal_topics: Container[str],
) -> bytes:
    """Wrap a terminal-topic publish in a ``ModelEventEnvelope``; else pass through.

    Field parity with the success path (``DispatchResultApplier.apply``) is the
    point, so each field is derived the same way that applier derives it:

    * ``payload`` — the publisher's own JSON body, unmodified.
    * ``correlation_id`` — the body's top-level ``correlation_id``.
    * ``event_type`` — ``derive_event_type_from_topic(topic)``, the shared
      helper the applier calls, so a failure terminal is stamped ``…-failed``
      exactly as the success terminal is stamped ``…-completed``.
    * ``envelope_id`` — ``uuid5`` in the correlation's namespace, so a
      redelivered terminal mints the SAME id and stays de-duplicable.
    * ``envelope_timestamp`` — emission time (UTC).

    Pass-through (byte-identical, zero behavior change) when any of these hold —
    each is a case where wrapping would either be wrong or unverifiable:

    * ``topic`` is not a contract-declared terminal topic (ordinary command /
      side-effect publishes keep their existing shape),
    * the body is not a JSON object,
    * the body carries no parseable ``correlation_id`` — an envelope whose
      correlation cannot be set is undeliverable to any waiting caller anyway,
    * the body is already an envelope (never double-wrap).
    """

    if topic not in terminal_topics:
        return payload

    try:
        decoded: object = json.loads(payload.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return payload
    if not isinstance(decoded, dict):
        return payload
    body: dict[str, object] = decoded
    if _is_already_envelope(body):
        return payload

    raw_correlation = body.get("correlation_id")
    if raw_correlation is None:
        return payload
    try:
        correlation_id = UUID(str(raw_correlation))
    except (TypeError, ValueError):
        return payload

    envelope: ModelEventEnvelope[object] = ModelEventEnvelope[object](
        envelope_id=uuid5(
            correlation_id,
            f"{topic}:{hashlib.sha256(payload).hexdigest()}",
        ),
        payload=body,
        correlation_id=correlation_id,
        envelope_timestamp=datetime.now(UTC),
    )
    event_type = derive_event_type_from_topic(topic)
    if event_type is not None:
        envelope = envelope.model_copy(update={"event_type": event_type})
    return envelope.model_dump_json().encode("utf-8")
