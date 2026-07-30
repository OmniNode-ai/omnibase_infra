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

Scope, stated so it is not mistaken for more than it is: this makes the failure
terminal *decodable and correctly typed* on the wire. It does NOT stop
``DispatchResultApplier`` from also republishing a negative-verdict return value
onto the contract's SUCCESS terminal — that verdict-blind republish is the
duplicate-producer defect tracked in OMN-15469, and OMN-15468 acceptance
criterion 2 stays open until it lands.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Container, Iterable, Mapping
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
    "envelope_terminal_payload",
    "extract_terminal_event_topics",
    "load_terminal_event_topics",
    "terminal_event_topics_from_declaration",
]


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
