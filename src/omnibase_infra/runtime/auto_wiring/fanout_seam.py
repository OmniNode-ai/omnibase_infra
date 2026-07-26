# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Def-B multi-event (fan-out) publish seam — OMN-14403 P3a (Kafka half).

The infra-boundary half of the seam: the seam-gated coercion, the wiring-time
``published_events`` coverage gate, and the census channel. The **pure** logic —
what counts as a fan-out sequence, carrier rejection, and class -> topic
injectivity — lives in ``omnibase_core.runtime.runtime_fanout_resolver`` and is
imported here so this path and the RuntimeLocal path (``LocalRuntimeBusAdapter``)
share ONE resolver and cannot drift again (Fable refinement 3;
``docs/plans/2026-07-14-p3a-sanctioned-direction-design.md`` §6ii). This module
adds only the infra-specific concerns: the seam on/off gating (passed in — the env
read lives in ``handler_wiring``, the OMN-11069-approved module), the warn-drop
census, and the contract-annotation coverage scan.

The defect this seam fixes is EXISTS-but-WRONG. Today a def-B handler that returns
``[event_a, event_b]`` matches neither the ``ModelHandlerOutput`` branch nor the
``BaseModel`` branch of ``_normalize_handler_result``, so it falls through to
``output_events=[]``, ``status=SUCCESS``: the dispatch reports success and
publishes NOTHING. Enabling the seam turns that sequence into the published batch
and makes a malformed element fatal instead of silently swallowed.
"""

from __future__ import annotations

import importlib
import logging
import re
from collections.abc import Sequence
from types import UnionType
from typing import TYPE_CHECKING, get_args, get_origin

from pydantic import BaseModel

from omnibase_core.models.errors import ModelOnexError
from omnibase_core.runtime.runtime_fanout_resolver import (
    assert_published_events_injective,
    is_fanout_sequence,
    normalize_fanout_elements,
)

if TYPE_CHECKING:
    from omnibase_infra.models.contracts.model_discovered_contract import (
        ModelDiscoveredContract,
    )

__all__ = [
    "check_fanout_publish_coverage",
    "fanout_element_class_names",
    "is_fanout_sequence",
    "normalize_fanout_sequence",
]

logger = logging.getLogger(__name__)

_FANOUT_ANNOTATION_RE = re.compile(
    r"^(?:list|List|typing\.List|Sequence|typing\.Sequence)\[(?P<elem>.+)\]$"
)

# Element types that name no single emittable class, so wiring-time coverage can
# neither prove nor disprove their published_events mapping (e.g. the
# ``list[BaseModel]`` of node_delegation_orchestrator). Never a hard boot failure
# — warned instead, so the unprovable case is visible rather than assumed safe.
# The publish-time backstop for these is OMN-14518.
_FANOUT_OPAQUE_ELEMENTS: frozenset[str] = frozenset({"BaseModel", "object", "Any"})


def _warn_dropped_fanout(
    result: Sequence[object], message_type: str | None, env_flag: str
) -> None:
    """Warn that a fan-out return is being dropped — the warn-first census channel.

    While the seam is OFF the runtime keeps today's drop, but it is no longer
    *silent*. This warning is what proves, from live logs, which wired handlers
    the flip would change.
    """
    if not result:
        # An empty sequence is the explicit "emit nothing" (§2), same as None. Not
        # a drop, so not a warning — it must not pollute the census channel.
        return
    logger.warning(
        "Handler returned a sequence of %d element(s) which the runtime is "
        "DROPPING (output_events=[], status=SUCCESS) — the events are NOT "
        "published. This is the OMN-14403 def-B fan-out publish seam; set %s=1 to "
        "publish them as a fan-out batch. message_type=%s element_types=%s",
        len(result),
        env_flag,
        message_type,
        sorted({type(element).__name__ for element in result}),
        extra={"ticket": "OMN-14403"},
    )


def normalize_fanout_sequence(
    result: Sequence[object],
    message_type: str | None,
    *,
    seam_enabled: bool,
    env_flag: str,
) -> list[BaseModel]:
    """Coerce a def-B fan-out sequence return into the applier's output_events.

    Seam OFF: warn-drop (census), publish nothing — today's behavior, no longer
    silent. Seam ON: delegate to the shared core validator
    (``normalize_fanout_elements`` — validate-then-RAISE, carrier rejection), so
    the Kafka path and the RuntimeLocal path validate a fan-out batch identically.
    The applier resolves each element's topic via ``published_events`` (short-name
    lookup), matching the core resolver's semantics; the boot coverage gate below
    keeps that resolution fail-closed.
    """
    if not seam_enabled:
        _warn_dropped_fanout(result, message_type, env_flag)
        return []
    return normalize_fanout_elements(result, message_type=message_type)


def fanout_element_class_names(annotation: object) -> tuple[str, ...]:
    """Return the element class names of a fan-out ``list[X]`` return annotation.

    Operates on the RAW annotation — string or resolved type. Under
    ``from __future__ import annotations`` a handler's return annotation is the
    STRING ``"list[ModelIntentReceipt]"``, so this must not depend on
    ``get_type_hints`` resolving, which can explode on a forward ref at boot.
    Returns ``()`` when the annotation is not a fan-out sequence.
    """
    if isinstance(annotation, str):
        match = _FANOUT_ANNOTATION_RE.match(annotation.strip())
        if match is None:
            return ()
        raw_members = match.group("elem").split("|")
    elif get_origin(annotation) in (list, Sequence):
        args = get_args(annotation)
        if not args:
            return ()
        element = args[0]
        members = get_args(element) if get_origin(element) is UnionType else (element,)
        raw_members = [getattr(member, "__name__", str(member)) for member in members]
    else:
        return ()
    return tuple(
        member.strip().rsplit(".", 1)[-1] for member in raw_members if member.strip()
    )


def _handler_fanout_elements(entry: object) -> tuple[str, ...]:
    """Return the classes a contract-declared handler can fan out, per its annotation."""
    handler_ref = getattr(entry, "handler", None)
    module_name = getattr(handler_ref, "module", None)
    class_name = getattr(handler_ref, "name", None)
    if not isinstance(module_name, str) or not isinstance(class_name, str):
        return ()
    try:
        handler_cls = getattr(importlib.import_module(module_name), class_name, None)
    except Exception:  # noqa: BLE001 — boundary: introspection must never break boot
        return ()
    if handler_cls is None:
        return ()
    for method_name in ("handle", "handle_async"):
        method = getattr(handler_cls, method_name, None)
        if method is None:
            continue
        elements = fanout_element_class_names(
            getattr(method, "__annotations__", {}).get("return")
        )
        if elements:
            return elements
    return ()


def check_fanout_publish_coverage(
    contract: ModelDiscoveredContract,
    *,
    seam_enabled: bool,
    env_flag: str,
) -> None:
    """Assert every fan-out handler's emittable classes are contract-declared.

    §2 Amendment C — coverage is proven at BOOT, not at event k.

    Without this gate an unmapped fan-out class falls back to the single
    ``_output_topic`` (``_resolve_mapped_output_topic``) and SILENTLY MISROUTES
    onto an unrelated declared topic — strictly WORSE than the drop the seam
    exists to fix. The published_events map is also asserted injective (one class
    -> exactly one topic) via the shared core check, so two classes cannot collide
    on one topic.

    Warn-only while the seam is OFF — the census channel that names the
    non-compliant handlers in live logs BEFORE the flip; fail-closed once ON.
    """
    if contract.handler_routing is None:
        return

    from omnibase_infra.runtime.event_bus_subcontract_wiring import (
        load_published_events_map,
    )

    published = load_published_events_map(contract.contract_path) or {}
    # A non-injective published_events map is a boot-time contract defect: an
    # unmapped fan-out element would misroute, and two classes on one topic makes
    # the (topic, model) reverse ambiguous. Delegated to the shared core assertion
    # so both runtimes reject the same shape. Seam OFF: warn-only.
    if published:
        try:
            assert_published_events_injective(published, context=contract.name)
        except ModelOnexError as exc:
            if seam_enabled:
                raise
            logger.warning(
                "%s (warn-only: %s is OFF)",
                exc,
                env_flag,
                extra={"ticket": "OMN-14403"},
            )
    # published_events keys are the applier's SHORT names (class name with the
    # "Model" prefix removed); accept either spelling so the gate matches the
    # resolver rather than a convention.
    declared: set[str] = set()
    for key in published:
        declared.add(key)
        declared.add(f"Model{key}")

    for entry in contract.handler_routing.handlers:
        for element in _handler_fanout_elements(entry):
            if element in _FANOUT_OPAQUE_ELEMENTS:
                logger.warning(
                    "Fan-out handler '%s' on contract '%s' declares an opaque "
                    "element type (%s), so its published_events coverage cannot be "
                    "proven at wiring time. Declare a concrete element type.",
                    entry.handler.name,
                    contract.name,
                    element,
                    extra={"ticket": "OMN-14403"},
                )
                continue
            if element in declared:
                continue
            message = (
                f"Fan-out handler '{entry.handler.name}' on contract "
                f"'{contract.name}' can emit {element!r}, which is NOT declared in "
                f"the contract's published_events {sorted(published)}. An unmapped "
                "fan-out element falls back to the single output_topic and silently "
                "misroutes. Declare it in published_events, or stop returning it as "
                "a fan-out element."
            )
            if seam_enabled:
                raise ModelOnexError(message)
            logger.warning(
                "%s (warn-only: %s is OFF)",
                message,
                env_flag,
                extra={"ticket": "OMN-14403"},
            )
