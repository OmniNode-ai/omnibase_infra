# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-12548 dispatch-selection parity harness (Mode A + determinism audit).

This is the v2 successor to ``docs/evidence/OMN-12545/baseline_selection_harness.py``.
It builds a REAL :class:`MessageDispatchEngine` from the live entry-point contract
corpus (the same ``discover_contracts()`` the kernel uses), then drives the
engine's own ``dispatch()`` selection over a probe taxonomy (§2 of the design
``docs/plans/2026-07-02-omn-12548-dispatch-parity-gate-design.md``) and records the
full selection tuple per probe.

Equivalence tuple (design D2)::

    (status, [dispatcher_id...] IN ORDER, message_category, message_type, dlq_topic)

Two entry points:

* :func:`build_snapshot` — canonical, sorted registration order (design D3). This is
  the committed Mode-A oracle. Any PR that drifts engine selection semantics fails
  the parity test against this fixture.
* :func:`build_determinism_audit` — re-runs selection under a permuted registration
  order and enumerates every probe whose tuple changes (real multi-match ties
  resolved by insertion order). Report-only at S0 (design D3), never gate-blocking.

Why this exercises the LIVE engine, not a reimplementation:
  Route/dispatcher metadata is derived by calling the *actual* library helpers
  (``_derive_dispatcher_id``, ``_topics_for_handler_entry``, ``_make_payload_type_matcher``,
  etc.). Selection is the engine's own ``dispatch()`` (async), which runs
  ``_find_matching_dispatchers`` including OMN-12416 payload type-scoping. The gate
  therefore pins what the engine *does*, warts included (design D1).

Payload construction (design D6/P4): for every type-scoped handler we import the
declared ``event_model`` and build an instance via ``model_construct()`` (no
validation, so a minimal instance always succeeds); if the model cannot even be
imported the probe is recorded ``construct-only=false`` with the import error —
never silently dropped.

Corpus completeness (design D6): the harness reads whatever ``discover_contracts()``
returns in the current venv and records the per-package inventory in the fixture
header. Packages absent from the venv are recorded as EXCLUSIONS (see
:data:`EXPECTED_ONEX_NODES_PACKAGES`) — zero silent exclusions.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from omnibase_core.enums.enum_execution_shape import EnumMessageCategory as _CoreCat
from omnibase_core.models.dispatch.model_dispatch_route import ModelDispatchRoute
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumMessageCategory
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _derive_dispatcher_id,
    _derive_event_type_alias_from_topic,
    _derive_handler_entry_key,
    _derive_message_category,
    _derive_route_id,
    _derive_topic_pattern_from_topic,
    _import_event_model_class,
    _literal_event_type_aliases_from_topics,
    _make_payload_type_matcher,
    _topics_for_handler_entry,
)
from omnibase_infra.runtime.message_dispatch_engine import (
    MessageDispatchEngine,
    _derive_dlq_topic,
)

# ---------------------------------------------------------------------------
# Fixture-header metadata: corpus completeness (design D6) and P0 outcomes (D5/D7).
# ---------------------------------------------------------------------------

#: Distributions that ship an ``onex.nodes`` entry-point group AND are inside the
#: ``omnibase_infra`` dependency closure, so they MUST be present in the CI venv
#: (``uv sync``). The parity test FAILS if any of these is missing — a missing
#: required package silently shrinks the corpus and hides regressions (design D6).
REQUIRED_ONEX_NODES_PACKAGES: tuple[str, ...] = (
    "omnibase-core",
    "omnibase-infra",
    "onex-change-control",
)

#: Distributions that ship ``onex.nodes`` entry points but are structurally OUTSIDE
#: the ``omnibase_infra`` dependency closure (siblings/downstreams — omnimarket
#: depends on infra, not the reverse). They cannot co-install in the infra CI venv
#: via ``uv sync`` and are therefore EXPECTED-EXCLUDED with a documented reason —
#: an explicit, non-silent exclusion (design D6). To cover their contracts, a
#: separate cross-repo corpus job would install them; that is out of scope for the
#: infra-hosted S0 gate and tracked under OMN-12525.
EXPECTED_EXCLUDED_ONEX_NODES_PACKAGES: tuple[str, ...] = (
    "omnimarket",
    "omniclaude",
    "omniintelligence",
    "omnimemory",
)

#: Full set for header reporting.
EXPECTED_ONEX_NODES_PACKAGES: tuple[str, ...] = (
    REQUIRED_ONEX_NODES_PACKAGES + EXPECTED_EXCLUDED_ONEX_NODES_PACKAGES
)

#: P0 live-trace outcomes folded into the fixture header (design D5/D7). These are
#: RUNTIME FACTS captured on the .201 stability-test lane on 2026-07-02, not the
#: harness's own inference. The static harness cannot observe the runtime DLQ
#: fall-through; it records the outcome so the fixture is self-describing and the
#: gate does not silently encode the (false) ticket-comment claim.
P0_OUTCOMES: dict[str, Any] = {
    "P0-1_guard_tripped_orchestrators": {
        "verdict": "NO_DISPATCHER_DLQ_REFINED",
        "confirmed_on": "stability-test lane (.201 via Tailscale 100.109.203.94)",
        "date": "2026-07-02",
        "summary": (
            "Routing outage CONFIRMED at runtime for node_rsd_orchestrator: "
            "live-probed end-to-end (synthetic publish to "
            "onex.evt.rsd.scores-calculated.v1, offset 19; runtime log at "
            "2026-07-02T18:12:20 shows 'No dispatcher found ... routing to DLQ topic "
            "onex.dlq.omnibase-infra.rsd.v1'). The static oracle (this fixture) "
            "REFINES the design-D5 blanket claim that all six generalize "
            "identically: 4 of the 6 are uniformly NO_DISPATCHER on every subscribe "
            "topic (the guard registers ZERO routes — "
            "_topics_for_handler_entry returns () at handler_wiring.py:2242-2281), "
            "but 2 (node_chain_orchestrator, node_registration_orchestrator) are "
            "MIXED — some handler entries dodge the guard and DO register routes, so "
            "those topics dispatch while their event_model-only multi-handler topics "
            "still fall through to NO_DISPATCHER. Secondary: the rsd DLQ topic did "
            "not exist at runtime, so the event was fully dropped."
        ),
        "uniform_no_dispatcher_orchestrators": [
            "node_rsd_orchestrator",
            "node_routing_orchestrator",
            "node_merge_sweep_workflow_orchestrator",
            "node_scope_workflow_orchestrator",
        ],
        "mixed_routing_orchestrators": [
            "node_chain_orchestrator",
            "node_registration_orchestrator",
        ],
        "live_probed_end_to_end": ["node_rsd_orchestrator"],
        "corpus_caveat": (
            "This slice excludes omnimarket (outside infra's dependency closure); "
            "a full cross-repo corpus could add sibling handlers that alter the "
            "success/no_dispatcher split. The gate pins THIS slice's behavior and a "
            "cross-repo corpus job is tracked under OMN-12525."
        ),
        "gate_posture": (
            "The gate PINS the observed per-topic behavior as-is (design D1). The "
            "rsd/routing/merge-sweep/scope outage is a High finding under "
            "OMN-12525, fixed BEHIND the gate — not in this parity PR. The D5 "
            "over-generalization is corrected here by evidence."
        ),
    },
    "P0-2_live_contract_source": {
        "verdict": "ENTRY_POINT_DISCOVERY",
        "confirmed_on": "stability-test lane (.201 via Tailscale 100.109.203.94)",
        "date": "2026-07-02",
        "summary": (
            "discover_contracts() -> wire_from_manifest() (entry-point scan) builds "
            "the initial/bulk dispatch route table on BOTH the main and effects "
            "runtimes. KafkaContractSource is constructed on the effects runtime "
            "ONLY, ~2.5 min after its own discovery burst, and is strictly a "
            "post-freeze incremental listener for dynamically-registered contracts "
            "(node-registration events) — it does NOT build the initial table and is "
            "absent entirely from the main runtime. The fixture's entry-point "
            "corpus is therefore the correct oracle source (design D7)."
        ),
    },
}


def _installed_onex_nodes_packages() -> dict[str, str | None]:
    """Return {canonical_dist_name: version or None-if-absent} for expected packages.

    Uses ``importlib.metadata`` (venv-truth), normalizing '-'/'_' so
    'omnibase_infra' and 'omnibase-infra' compare equal.
    """
    present: dict[str, str] = {}
    for dist in importlib.metadata.distributions():
        try:
            name = dist.metadata["Name"]
        except Exception:  # noqa: BLE001 — malformed metadata is not our concern
            continue
        if name is None:
            continue
        present[name.lower().replace("_", "-")] = dist.version
    out: dict[str, str | None] = {}
    for pkg in EXPECTED_ONEX_NODES_PACKAGES:
        key = pkg.lower().replace("_", "-")
        out[pkg] = present.get(key)
    return out


# ---------------------------------------------------------------------------
# Wiring-metadata derivation (verbatim from _prepare_handler_wiring).
# ---------------------------------------------------------------------------


def _derive_message_types(contract: Any, entry: Any) -> set[str] | None:
    """Reproduce ``_prepare_handler_wiring``'s message_types derivation verbatim."""
    message_types: set[str] | None = None
    if entry.event_model is not None:
        message_types = {entry.event_model.name}
    event_type_alias = entry.event_type.strip() if entry.event_type else ""
    subscribe_topics = contract.event_bus.subscribe_topics if contract.event_bus else ()
    literal_topic_aliases = _literal_event_type_aliases_from_topics(subscribe_topics)
    if literal_topic_aliases:
        message_types = (message_types or set()).union(literal_topic_aliases)
    if event_type_alias:
        message_types = (message_types or set()) | {event_type_alias}
    elif contract.event_bus:
        topic_aliases = {
            alias
            for topic in subscribe_topics
            if (alias := _derive_event_type_alias_from_topic(topic)) is not None
        }
        if topic_aliases:
            message_types = (message_types or set()).union(topic_aliases)
    return message_types


def _derive_category(contract: Any, entry: Any) -> EnumMessageCategory:
    """Reproduce ``_prepare_handler_wiring``'s ``_early_category`` derivation."""
    category_str = "event"
    if entry.message_category:
        category_str = entry.message_category.strip().lower()
    elif contract.event_bus and contract.event_bus.subscribe_topics:
        category_str = _derive_message_category(contract.event_bus.subscribe_topics[0])
    return EnumMessageCategory(category_str)


def _stub_dispatcher(envelope: object) -> None:  # pragma: no cover - never invoked
    """Inert dispatcher callable. Selection never executes it (payload=None probes)."""
    return


# ---------------------------------------------------------------------------
# Engine construction from the corpus (design D3: caller controls order).
# ---------------------------------------------------------------------------


def _build_engine(
    contracts: list[Any],
) -> tuple[
    MessageDispatchEngine, dict[str, dict[str, Any]], list[dict[str, Any]], set[str]
]:
    """Build + freeze a real engine from ``contracts`` in the given order.

    Returns (engine, dispatcher_meta, route_meta, subscribe_topics).
    The caller decides the registration order — :func:`build_snapshot` sorts by
    name (canonical), the determinism audit permutes it.
    """
    engine = MessageDispatchEngine()
    dispatcher_meta: dict[str, dict[str, Any]] = {}
    route_meta: list[dict[str, Any]] = []
    subscribe_topics: set[str] = set()
    registered_dispatcher_ids: set[str] = set()
    registered_route_ids: set[str] = set()

    for contract in contracts:
        if contract.event_bus is None or contract.handler_routing is None:
            continue
        if contract.event_bus.subscribe_topics:
            subscribe_topics.update(contract.event_bus.subscribe_topics)

        for entry in contract.handler_routing.handlers:
            handler_key = _derive_handler_entry_key(entry)
            dispatcher_id = _derive_dispatcher_id(contract.name, handler_key)
            category = _derive_category(contract, entry)
            message_types = _derive_message_types(contract, entry)
            type_scoped = entry.event_model is not None

            if dispatcher_id not in registered_dispatcher_ids:
                # OMN-12416: install the REAL payload matcher when event_model is
                # declared, so payload-driven probes (P4/P5/P8) exercise the live
                # type-scoping path — not a wiring-level approximation.
                matcher = (
                    _make_payload_type_matcher(entry.event_model)
                    if entry.event_model is not None
                    else None
                )
                engine.register_dispatcher(
                    dispatcher_id=dispatcher_id,
                    dispatcher=_stub_dispatcher,
                    category=category,
                    message_types=message_types,
                    payload_type_matcher=matcher,
                )
                registered_dispatcher_ids.add(dispatcher_id)
                dispatcher_meta[dispatcher_id] = {
                    "handler_id": dispatcher_id,
                    "contract_name": contract.name,
                    "package": contract.package_name,
                    "handler_name": entry.handler.name,
                    "handler_module": entry.handler.module,
                    "category": category.value,
                    "message_types": sorted(message_types)
                    if message_types is not None
                    else None,
                    "type_scoped": type_scoped,
                    "event_model_module": entry.event_model.module
                    if entry.event_model is not None
                    else None,
                    "event_model_name": entry.event_model.name
                    if entry.event_model is not None
                    else None,
                }

            for topic in _topics_for_handler_entry(contract, entry):
                route_id = _derive_route_id(contract.name, handler_key, topic)
                if route_id in registered_route_ids:
                    continue
                topic_pattern = _derive_topic_pattern_from_topic(topic)
                # Core ModelDispatchRoute renamed the field to ``handler_id``
                # (``dispatcher_id`` remains an input alias + read property). Pass
                # the canonical field name so mypy --strict is satisfied; the engine
                # reads it via _get_route_dispatcher_id.
                route = ModelDispatchRoute(
                    route_id=route_id,
                    topic_pattern=topic_pattern,
                    message_category=category,
                    handler_id=dispatcher_id,
                )
                engine.register_route(route)
                registered_route_ids.add(route_id)
                route_meta.append(
                    {
                        "route_id": route_id,
                        "contract_name": contract.name,
                        "topic": topic,
                        "topic_pattern": topic_pattern,
                        "message_category": category.value,
                        "dispatcher_id": dispatcher_id,
                        "handler_id": dispatcher_id,
                        "type_scoped": type_scoped,
                    }
                )

    engine.freeze()
    return engine, dispatcher_meta, route_meta, subscribe_topics


# ---------------------------------------------------------------------------
# Probe construction (design §2 P1-P8).
# ---------------------------------------------------------------------------


def _envelope(*, event_type: str | None, payload: object) -> ModelEventEnvelope[object]:
    """Build a real envelope. ``event_type`` is passed through the constructor as an
    extra field (the engine reads it via model_fields presence, else None)."""
    kwargs: dict[str, Any] = {"payload": payload}
    if event_type is not None:
        kwargs["event_type"] = event_type
    return ModelEventEnvelope(**kwargs)


class _DictPayload:
    """A payload object whose class name is a stable, meaningless token.

    Used for P1(b): event_type-absent + non-model payload. The engine falls back to
    ``type(payload).__name__`` for message_type; a plain object gives a deterministic
    class name that will not accidentally match any registered dispatcher.
    """


async def _selection_tuple(
    engine: MessageDispatchEngine,
    *,
    topic: str,
    event_type: str | None,
    payload: object,
) -> dict[str, Any]:
    """Drive the LIVE engine ``dispatch()`` and extract the equivalence tuple (D2).

    Uses the stub dispatcher so ``dispatch()`` executes selection + fan-out ordering
    without side effects. The returned dict is the serializable per-probe tuple.
    """
    envelope = _envelope(event_type=event_type, payload=payload)
    result = await engine.dispatch(topic=topic, envelope=envelope)
    # The engine records the ORDERED executed-dispatcher list as a comma-joined
    # string on ``result.dispatcher_id`` (message_dispatch_engine.py:1564,1640).
    # Since the harness uses an inert stub dispatcher (no errors, status=SUCCESS),
    # this string is exactly the ordered fan-out list (design D2). On
    # NO_DISPATCHER / INVALID_MESSAGE the string is empty -> empty list.
    raw = result.dispatcher_id or ""
    ordered_ids = [d for d in (p.strip() for p in raw.split(",")) if d]
    return {
        "status": result.status.value,
        "dispatcher_ids": ordered_ids,
        "message_category": result.message_category.value
        if result.message_category is not None
        else None,
        "message_type": result.message_type,
        "dlq_topic": result.dlq_topic,
    }


def _model_construct_instance(
    module: str, name: str
) -> tuple[object | None, str | None]:
    """Import ``module.name`` and build an unvalidated instance via model_construct.

    Returns (instance, error). On import/attr failure returns (None, error-string)
    so the probe is recorded ``construct-only=false`` — never dropped (design D6).
    """
    from omnibase_infra.runtime.auto_wiring.models.model_handler_ref import (
        ModelHandlerRef,
    )

    try:
        model_cls = _import_event_model_class(ModelHandlerRef(name=name, module=module))
    except Exception as exc:  # noqa: BLE001 — record, do not drop
        return None, f"{type(exc).__name__}: {exc}"
    try:
        return model_cls.model_construct(), None
    except Exception as exc:  # noqa: BLE001 — record, do not drop
        return None, f"model_construct failed: {type(exc).__name__}: {exc}"


async def _run_probes(
    engine: MessageDispatchEngine,
    *,
    dispatcher_meta: dict[str, dict[str, Any]],
    subscribe_topics: set[str],
) -> dict[str, Any]:
    """Run the P1-P8 probe taxonomy against the frozen engine.

    Each probe records: probe_id, family, topic, the input (event_type + payload
    descriptor), and the selection tuple. Probe ids are stable and sorted so the
    fixture is byte-deterministic.
    """
    probes: dict[str, dict[str, Any]] = {}

    # ---- P1 topic/event_type matrix + P2 category edges + P6 NO_DISPATCHER ----
    # For each distinct subscribe topic drive the two runtime-representative
    # envelope shapes:
    #   (a) event_type = topic-derived ONEX alias   -> the normal publisher path
    #   (b) event_type absent + opaque payload      -> message_type = class name
    # Together these cover P1(a)/P1(c)-alias, P2 category parse, P3 filters (the
    # engine applies them), and P6 (unroutable -> DLQ derivation).
    for topic in sorted(subscribe_topics):
        alias = _derive_event_type_alias_from_topic(topic)
        # (a) alias-driven
        pa = await _selection_tuple(
            engine, topic=topic, event_type=alias, payload=_DictPayload()
        )
        probes[f"P1a::{topic}"] = {
            "family": "P1_topic_event_type_matrix",
            "topic": topic,
            "input": {"event_type": alias, "payload": "_DictPayload"},
            "selection": pa,
        }
        # (b) no event_type, opaque payload
        pb = await _selection_tuple(
            engine, topic=topic, event_type=None, payload=_DictPayload()
        )
        probes[f"P1b::{topic}"] = {
            "family": "P1_topic_event_type_matrix",
            "topic": topic,
            "input": {"event_type": None, "payload": "_DictPayload"},
            "selection": pb,
        }

    # ---- P2 explicit category edges (synthesized topics) ----
    synthetic_p2 = [
        # unparseable category -> INVALID_MESSAGE, no DLQ
        ("onex.evt.nomatch", "P2_unparseable_category"),
        # projections suffix (unmapped in _SUFFIX_TO_CATEGORY per design) 5-seg
        ("onex.proj.omnibase-infra.some-projection.v1", "P2_projections_suffix"),
        # non-5-segment ONEX-ish topic
        ("weird.events.topic", "P2_non_5_segment"),
        # case variant of an events topic (IGNORECASE parity)
        ("onex.EVT.omnibase-infra.SOME-THING.v1", "P2_case_variant"),
    ]
    for topic, family in synthetic_p2:
        alias = _derive_event_type_alias_from_topic(topic)
        sel = await _selection_tuple(
            engine, topic=topic, event_type=alias, payload=_DictPayload()
        )
        probes[f"P2::{family}::{topic}"] = {
            "family": family,
            "topic": topic,
            "input": {"event_type": alias, "payload": "_DictPayload"},
            "selection": sel,
        }

    # ---- P4 payload type-scoping + P5 fan-out + P8 guard-tripped orchestrators ----
    # For every type-scoped dispatcher, build its declared event_model instance and
    # drive it on each of the dispatcher's subscribe topics. This exercises the LIVE
    # OMN-12416 narrowing: a matching payload keeps the dispatcher, a mismatched one
    # drops it. Guard-tripped orchestrators (P8) fall out here as NO_DISPATCHER
    # because they register zero routes (design D5 / P0-1) — the probe pins that.
    type_scoped = {
        did: meta for did, meta in dispatcher_meta.items() if meta.get("type_scoped")
    }
    construct_errors: list[dict[str, str]] = []
    for did, meta in sorted(type_scoped.items()):
        module = meta.get("event_model_module")
        name = meta.get("event_model_name")
        if module is None or name is None:
            continue
        instance, err = _model_construct_instance(module, name)
        construct_only = err is None
        if err is not None:
            construct_errors.append(
                {"dispatcher_id": did, "event_model": f"{module}.{name}", "error": err}
            )
        # Find this dispatcher's topics from the route table via meta linkage.
        # We drive the contract's own subscribe topics (recorded on the route rows);
        # here we recover them from subscribe_topics filtered by the alias route.
        for topic in sorted(_topics_for_dispatcher(did, meta, subscribe_topics)):
            alias = _derive_event_type_alias_from_topic(topic)
            payload: object = instance if instance is not None else _DictPayload()
            sel = await _selection_tuple(
                engine, topic=topic, event_type=alias, payload=payload
            )
            probes[f"P4::{did}::{topic}"] = {
                "family": "P4_payload_type_scoping",
                "topic": topic,
                "dispatcher_id": did,
                "event_model": f"{module}.{name}",
                "construct_only": construct_only,
                "input": {
                    "event_type": alias,
                    "payload": name if instance is not None else "_DictPayload",
                },
                "selection": sel,
            }

    return {
        "probes": dict(sorted(probes.items())),
        "construct_errors": construct_errors,
    }


def _topics_for_dispatcher(
    dispatcher_id: str, meta: dict[str, Any], subscribe_topics: set[str]
) -> set[str]:
    """Best-effort recovery of a dispatcher's subscribe topics for P4 driving.

    A type-scoped dispatcher's message_types include the topic-derived aliases; we
    match those back to the corpus subscribe topics. This keeps P4 probes anchored
    to real topics the dispatcher actually consumes.
    """
    msg_types = set(meta.get("message_types") or ())
    out: set[str] = set()
    for topic in subscribe_topics:
        alias = _derive_event_type_alias_from_topic(topic)
        if alias is not None and (alias in msg_types or topic in msg_types):
            out.add(topic)
    return out


# ---------------------------------------------------------------------------
# Public entry points.
# ---------------------------------------------------------------------------


def _corpus_from_discovery() -> tuple[
    list[Any], list[dict[str, str]], list[dict[str, str]]
]:
    """Discover contracts and return (contracts, discovery_errors, duplicate_winners).

    ``duplicate_winners`` records the first-seen-wins resolution for duplicate
    contract names (design D3) — every dropped duplicate is enumerated, not silent.
    """
    manifest = discover_contracts()
    discovery_errors = [
        {
            "package": err.package_name,
            "entry_point": err.entry_point_name,
            "error": str(err.error),
        }
        for err in manifest.errors
    ]
    # Duplicate winners: discover_contracts already dedups first-seen-wins; the
    # dropped duplicates surface as discovery errors mentioning DUPLICATE. Record
    # the winner (the contract that IS in manifest.contracts) for each dup error.
    by_name: dict[str, Any] = {c.name: c for c in manifest.contracts}
    duplicate_winners: list[dict[str, str]] = []
    for err in manifest.errors:
        msg = str(err.error)
        if "Duplicate contract name" in msg or "DUPLICATE" in msg:
            # extract the contract name in single quotes
            name = None
            if "'" in msg:
                name = msg.split("'")[1]
            winner = by_name.get(name) if name else None
            duplicate_winners.append(
                {
                    "contract_name": name or "<unparsed>",
                    "winner_package": winner.package_name if winner else "<unknown>",
                    "dropped_package": err.package_name,
                }
            )
    return list(manifest.contracts), discovery_errors, duplicate_winners


def _header(
    *,
    contracts: list[Any],
    discovery_errors: list[dict[str, str]],
    duplicate_winners: list[dict[str, str]],
    dispatcher_meta: dict[str, dict[str, Any]],
    route_meta: list[dict[str, Any]],
    subscribe_topics: set[str],
    probe_count: int,
    construct_errors: list[dict[str, str]],
) -> dict[str, Any]:
    installed = _installed_onex_nodes_packages()
    # Expected-excluded siblings: documented, not silent (design D6).
    exclusions = [
        {
            "package": pkg,
            "reason": (
                "outside omnibase_infra dependency closure (sibling/downstream); "
                "cannot co-install via infra `uv sync`. Contracts covered by a "
                "separate cross-repo corpus job, tracked under OMN-12525."
            ),
            "installed": installed.get(pkg) is not None,
        }
        for pkg in EXPECTED_EXCLUDED_ONEX_NODES_PACKAGES
    ]
    # Required-but-missing: a real corpus-shrink bug the test must catch.
    required_missing = [
        pkg for pkg in REQUIRED_ONEX_NODES_PACKAGES if installed.get(pkg) is None
    ]
    contracts_by_pkg: dict[str, int] = defaultdict(int)
    for c in contracts:
        contracts_by_pkg[c.package_name] += 1
    return {
        "fixture_version": "v2",
        "ticket": "OMN-12548",
        "epic": "OMN-12525",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "accuracy": "engine-dispatch (live MessageDispatchEngine.dispatch)",
        "accuracy_note": (
            "Selection is the engine's own async dispatch() over a probe taxonomy, "
            "including OMN-12416 payload type-scoping (real _make_payload_type_matcher "
            "installed per dispatcher). Registration order is canonical (contracts "
            "sorted by name, design D3). The equivalence tuple is "
            "(status, ordered dispatcher_ids, message_category, message_type, dlq_topic)."
        ),
        "registration_order": "contracts sorted by name (canonical, design D3)",
        "equivalence_tuple": [
            "status",
            "dispatcher_ids (ordered)",
            "message_category",
            "message_type",
            "dlq_topic",
        ],
        "core_message_category_from_topic": str(_CoreCat),
        "corpus": {
            "contracts_discovered": len(contracts),
            "contracts_by_package": dict(sorted(contracts_by_pkg.items())),
            "discovery_errors": len(discovery_errors),
            "distinct_subscribe_topics": len(subscribe_topics),
            "registered_dispatchers": len(dispatcher_meta),
            "registered_routes": len(route_meta),
            "type_scoped_dispatchers": sum(
                1 for m in dispatcher_meta.values() if m.get("type_scoped")
            ),
            "probe_count": probe_count,
        },
        "required_onex_nodes_packages": list(REQUIRED_ONEX_NODES_PACKAGES),
        "expected_excluded_onex_nodes_packages": list(
            EXPECTED_EXCLUDED_ONEX_NODES_PACKAGES
        ),
        "installed_package_versions": installed,
        "required_missing_packages": required_missing,
        "exclusions": exclusions,
        "duplicate_winners": duplicate_winners,
        "construct_only_failures": construct_errors,
        "p0_outcomes": P0_OUTCOMES,
    }


def build_snapshot() -> dict[str, Any]:
    """Build the canonical Mode-A fixture (sorted registration order, design D3)."""
    contracts, discovery_errors, duplicate_winners = _corpus_from_discovery()
    ordered = sorted(contracts, key=lambda c: c.name)
    engine, dispatcher_meta, route_meta, subscribe_topics = _build_engine(ordered)
    probe_result = asyncio.run(
        _run_probes(
            engine,
            dispatcher_meta=dispatcher_meta,
            subscribe_topics=subscribe_topics,
        )
    )
    probes = probe_result["probes"]
    construct_errors = probe_result["construct_errors"]
    header = _header(
        contracts=contracts,
        discovery_errors=discovery_errors,
        duplicate_winners=duplicate_winners,
        dispatcher_meta=dispatcher_meta,
        route_meta=route_meta,
        subscribe_topics=subscribe_topics,
        probe_count=len(probes),
        construct_errors=construct_errors,
    )
    return {
        "header": header,
        "discovery_errors": sorted(
            discovery_errors, key=lambda e: (e["package"], e["entry_point"])
        ),
        "dispatchers": dict(sorted(dispatcher_meta.items())),
        "routes": sorted(route_meta, key=lambda r: r["route_id"]),
        "probes": probes,
    }


def build_determinism_audit(seed: int = 1) -> dict[str, Any]:
    """Re-run selection under a permuted registration order; enumerate tuple diffs.

    Report-only at S0 (design D3). A probe whose tuple changes under permutation is
    a real multi-match tie resolved by insertion order — a latent production hazard,
    ticketed under OMN-12525, not gate-blocking here.
    """
    import random

    contracts, _errs, _dups = _corpus_from_discovery()
    canonical = sorted(contracts, key=lambda c: c.name)
    permuted = list(canonical)
    random.Random(seed).shuffle(permuted)

    eng_a, meta_a, _routes_a, topics_a = _build_engine(canonical)
    eng_b, meta_b, _routes_b, topics_b = _build_engine(permuted)
    probes_a = asyncio.run(
        _run_probes(eng_a, dispatcher_meta=meta_a, subscribe_topics=topics_a)
    )["probes"]
    probes_b = asyncio.run(
        _run_probes(eng_b, dispatcher_meta=meta_b, subscribe_topics=topics_b)
    )["probes"]

    order_sensitive: list[dict[str, Any]] = []
    all_ids = sorted(set(probes_a) | set(probes_b))
    for pid in all_ids:
        sel_a = probes_a.get(pid, {}).get("selection")
        sel_b = probes_b.get(pid, {}).get("selection")
        if sel_a != sel_b:
            order_sensitive.append(
                {
                    "probe_id": pid,
                    "canonical": sel_a,
                    "permuted": sel_b,
                }
            )

    return {
        "ticket": "OMN-12548",
        "epic": "OMN-12525",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "seed": seed,
        "probe_count": len(all_ids),
        "order_sensitive_count": len(order_sensitive),
        "gate_posture": (
            "Report-only at S0 (design D3). Each entry is a real multi-match tie "
            "resolved by registration/insertion order — a latent production hazard "
            "ticketed under OMN-12525, NOT gate-blocking in this parity PR."
        ),
        "order_sensitive_probes": order_sensitive,
    }


def _write_json(obj: dict[str, Any], path: str) -> None:
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
        fh.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Fixture JSON output path")
    parser.add_argument(
        "--audit-out",
        default=None,
        help="Optional determinism-audit JSON output path",
    )
    parser.add_argument(
        "--audit-seed", type=int, default=1, help="Permutation seed for the audit"
    )
    args = parser.parse_args()

    snapshot = build_snapshot()
    _write_json(snapshot, args.out)
    h = snapshot["header"]["corpus"]
    print(
        f"corpus: {h['contracts_discovered']} contracts, "
        f"{h['registered_dispatchers']} dispatchers, "
        f"{h['registered_routes']} routes, "
        f"{h['distinct_subscribe_topics']} topics, "
        f"{h['probe_count']} probes",
        file=sys.stderr,
    )
    excl = snapshot["header"]["exclusions"]
    print(
        f"exclusions: {len(excl)} ({', '.join(e['package'] for e in excl) or 'none'})",
        file=sys.stderr,
    )
    print(f"wrote {args.out}", file=sys.stderr)

    if args.audit_out:
        audit = build_determinism_audit(seed=args.audit_seed)
        _write_json(audit, args.audit_out)
        print(
            f"determinism audit: {audit['order_sensitive_count']} order-sensitive "
            f"of {audit['probe_count']} probes; wrote {args.audit_out}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
