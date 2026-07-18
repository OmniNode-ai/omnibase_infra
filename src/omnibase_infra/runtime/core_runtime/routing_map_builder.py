# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Build the RuntimeDispatch routing map for the delegation command spine (S6, §a).

Net-new for the single-runtime unification (epic OMN-14717, ticket OMN-14758). This
module resolves each ALLOWLISTED subscribe topic to exactly one
``omnibase_core.runtime.runtime_dispatch.DispatchRoute`` from the owning contract's
``handler_routing`` + ``published_events`` (read verbatim from the contract YAML).

Structural phantom-avoidance (OMN-14755): the map is keyed PURELY by the contract's
literal subscribe-topic string. There is NO ``event_type`` / ``payload_type`` /
``type(payload).__name__`` inbound sub-key anywhere — ``RuntimeDispatch._route_for`` is
topic-only, so the "subscribed but never dispatched" class cannot exist for an
allowlisted topic.

Single-owner build gate (§c.3.2): the ``dict`` build RAISES on a duplicate topic key
rather than last-write-wins, so two contracts declaring the same subscribe topic is a
loud boot failure, not a silent mis-wire.

Handler-resolution parity (R-7): the builder does NOT instantiate handlers itself — it
takes a ``handler_resolver`` callable so the composition root can hand it the SAME
handler instances the legacy kernel wired (shared container / resolver). Model classes
(``event_model``) are plain DTOs and are resolved via importlib by default.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from omnibase_core.enums.enum_core_error_code import EnumCoreErrorCode
from omnibase_core.errors.model_onex_error import ModelOnexError
from omnibase_core.models.dispatch.model_handler_ref import ModelHandlerRef
from omnibase_core.runtime.runtime_dispatch import DispatchRoute
from omnibase_infra.runtime.auto_wiring.models.model_discovered_contract import (
    ModelDiscoveredContract,
)
from omnibase_infra.runtime.auto_wiring.models.model_handler_routing_entry import (
    ModelHandlerRoutingEntry,
)
from omnibase_infra.runtime.event_bus_subcontract_wiring import (
    load_published_events_map,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DefBTarget",
    "HandlerResolver",
    "ModelResolver",
    "PublishedEventsLoader",
    "build_routing_map",
    "import_model_cls",
]


class DefBTarget(Protocol):
    """Structural def-B handler: ``handle(request) -> response`` (sync or awaitable)."""

    handle: Callable[..., object]


# A handler resolver maps a contract handler ref to a live def-B handler instance
# exposing ``handle(request) -> response``. Injected so RuntimeDispatch shares the
# legacy kernel's handler-construction path (R-7), never re-instantiating.
HandlerResolver = Callable[[ModelHandlerRef], DefBTarget]
# A model resolver maps an ``event_model`` ref to its BaseModel subclass.
ModelResolver = Callable[[ModelHandlerRef], type[BaseModel]]
# Loads the ``published_events`` class-name -> topic map from a contract path.
PublishedEventsLoader = Callable[[Path], Mapping[str, str]]


def import_model_cls(ref: ModelHandlerRef) -> type[BaseModel]:
    """Import ``ref.name`` from ``ref.module`` and assert it is a BaseModel subclass."""
    module = importlib.import_module(ref.module)
    try:
        obj = getattr(module, ref.name)
    except AttributeError as exc:
        raise ModelOnexError(
            message=(
                f"S6 routing-map builder: event_model {ref.name!r} not found in module "
                f"{ref.module!r}."
            ),
            error_code=EnumCoreErrorCode.CONTRACT_VALIDATION_ERROR,
        ) from exc
    if not (isinstance(obj, type) and issubclass(obj, BaseModel)):
        raise ModelOnexError(
            message=(
                f"S6 routing-map builder: event_model {ref.module}.{ref.name} is not a "
                "pydantic BaseModel subclass; cannot use it as a def-B input model."
            ),
            error_code=EnumCoreErrorCode.CONTRACT_VALIDATION_ERROR,
        )
    return obj


def _resolve_owning_entry(
    contract: ModelDiscoveredContract, topic: str
) -> ModelHandlerRoutingEntry:
    """Resolve the ONE handler entry that owns ``topic`` for ``contract`` (§a.1).

    Fail-closed: a topic that resolves to zero or more-than-one handler entry is a
    build error (RuntimeDispatch is single-route-per-topic; a genuine multi-handler
    topic cannot be expressed as one route — §a notes / §c.3.2).
    """
    if contract.handler_routing is None or not contract.handler_routing.handlers:
        raise ModelOnexError(
            message=(
                f"S6 routing-map builder: contract {contract.name!r} subscribes topic "
                f"{topic!r} but declares no handler_routing.handlers — cannot resolve a "
                "DispatchRoute (default_handler-only contracts are out of scope for the "
                "S6 command spine)."
            ),
            error_code=EnumCoreErrorCode.CONTRACT_VALIDATION_ERROR,
        )
    entries = contract.handler_routing.handlers
    # An entry with an explicit ``topic`` owns exactly that topic; an entry with no
    # topic applies to the contract's subscribe topics generally.
    explicit = [e for e in entries if e.topic == topic]
    if len(explicit) == 1:
        return explicit[0]
    if len(explicit) > 1:
        raise ModelOnexError(
            message=(
                f"S6 routing-map builder: contract {contract.name!r} declares "
                f"{len(explicit)} handler entries all bound to topic {topic!r}. "
                "RuntimeDispatch is single-route-per-topic; a topic must resolve to "
                "exactly one handler. Refusing to silently pick one."
            ),
            error_code=EnumCoreErrorCode.CONTRACT_VALIDATION_ERROR,
        )
    # No explicit-topic entry. Fall back to topic-agnostic entries.
    generic = [e for e in entries if e.topic is None]
    if len(generic) == 1:
        return generic[0]
    raise ModelOnexError(
        message=(
            f"S6 routing-map builder: contract {contract.name!r} subscribes topic "
            f"{topic!r} but has {len(generic)} topic-agnostic handler entries "
            f"({[e.handler.name for e in generic]}). A multiplexed topic (many "
            "handlers, one topic, chosen by payload type) cannot be expressed as a "
            "single RuntimeDispatch route — fail closed rather than route ambiguously."
        ),
        error_code=EnumCoreErrorCode.CONTRACT_VALIDATION_ERROR,
    )


def build_routing_map(
    contracts: Sequence[ModelDiscoveredContract],
    allowlist: frozenset[str],
    *,
    handler_resolver: HandlerResolver,
    model_resolver: ModelResolver = import_model_cls,
    published_events_loader: PublishedEventsLoader = load_published_events_map,
) -> dict[str, DispatchRoute]:
    """Build the topic -> :class:`DispatchRoute` map for the allowlisted topics (§a).

    For every discovered contract that subscribes an allowlisted topic, resolve exactly
    one route. The build RAISES on a duplicate topic key (single-owner build gate) and
    on any allowlist topic that no discovered contract owns.
    """
    routing_map: dict[str, DispatchRoute] = {}
    for contract in contracts:
        if contract.event_bus is None:
            continue
        for topic in contract.event_bus.subscribe_topics:
            if topic not in allowlist:
                continue
            if topic in routing_map:
                raise ModelOnexError(
                    message=(
                        f"S6 routing-map builder: topic {topic!r} is declared by more "
                        f"than one contract (single-owner-per-topic violation). "
                        "RuntimeDispatch routes a topic to exactly one handler; two "
                        "owners is a wiring bug — resolve the duplicate consumer before "
                        "allowlisting."
                    ),
                    error_code=EnumCoreErrorCode.CONTRACT_VALIDATION_ERROR,
                )
            entry = _resolve_owning_entry(contract, topic)
            handler = handler_resolver(entry.handler)
            input_model_cls = (
                model_resolver(entry.event_model)
                if entry.event_model is not None
                else None
            )
            published_events = dict(published_events_loader(contract.contract_path))
            routing_map[topic] = DispatchRoute(
                name=entry.handler.name,
                handler=handler,
                published_events=published_events,
                input_model_cls=input_model_cls,
            )
            logger.info(
                "S6 routing-map: bound topic=%s -> handler=%s (input_model=%s) "
                "published_events=%s contract=%s",
                topic,
                entry.handler.name,
                None if input_model_cls is None else input_model_cls.__name__,
                sorted(published_events),
                contract.name,
            )

    missing = allowlist - set(routing_map)
    if missing:
        raise ModelOnexError(
            message=(
                f"S6 routing-map builder: allowlisted topics {sorted(missing)} are not "
                "declared as a subscribe_topic of any discovered contract. Every "
                "ONEX_CORE_RUNTIME_TOPICS entry must be owned by exactly one contract "
                "(fail closed at boot, never silently drop the topic)."
            ),
            error_code=EnumCoreErrorCode.CONTRACT_VALIDATION_ERROR,
        )
    return routing_map
