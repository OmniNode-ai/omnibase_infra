# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Projection liveness as a runtime health dimension (OMN-16994).

This module is the deterministic half of OMN-16843's deferred AC6: *"19
contracts failing to wire should not leave the runtime reporting healthy."*

The two masks it closes
-----------------------
**Unattached.** ``ServiceRuntimeHealthMonitor`` derives its expected consumer
groups from the LIVE bus registry whenever one is available, because
``discover_contracts()`` also sees contracts this runtime deliberately does not
own. That override is correct for its original purpose and catastrophic here: a
projection that failed to prepare its handler never registers a subscription, so
it silently leaves the expectation set and ``topic_coverage`` reports "All N
expected consumer group(s) covered". Nineteen ``database_ref: application``
projections were unattached on every ``.201`` compose lane for months while
``/health`` stayed green (OMN-16843). The fix is to compare the *contract-
declared* projection set against the live registry instead of letting the
registry redefine what was expected.

**DLQ-saturated.** ``node_projection_session_replay`` on the stability lane
attached at zero lag, consumed every event, and routed 100% of them to the
platform quarantine sink on a Postgres auth failure. The DLQ route commits the
offset, so lag reads 0, the consumer group reads ``Stable``, and ``/health``
returned ``status: "healthy"`` with ``failed_handlers: {}`` over a total loss
(``docs/tracking/2026-08-29-hook-emission-ledger-trace.md``, hop 6). The ratio
is read from the OMN-16777 consumer-flow windows, which already count
``messages_in`` and ``messages_dlq`` per ``(consumer_group, topic)``.

Shape
-----
Pure functions over injected data: no clock, no bus, no database, no class with
a lifecycle. Nothing here is a ``Plugin*``, a manager, or a daemon — the caller
is the existing ``ServiceRuntimeHealthMonitor`` cycle, which already owns the
schedule. Both halves fail to UNKNOWN rather than to a fabricated failure when
their input is unobservable.

Related Tickets:
    - OMN-16994: this module (OMN-16843 AC6, deferred)
    - OMN-16843: compose-lane internal DSN wiring; source of the unattached mask
    - OMN-16690: read-under-write-declaration; source of the DLQ-saturated mask
    - OMN-16777: the consumer-flow counters the saturation ratio is read from
    - OMN-15217: the verdict -> ``/health`` fold this dimension rides
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from omnibase_infra.enums.enum_consumer_group_purpose import (
    EnumConsumerGroupPurpose,
)
from omnibase_infra.models.health.model_flow_attribution import (
    ModelFlowAttribution,
)
from omnibase_infra.models.health.model_projection_contract_ref import (
    ModelProjectionContractRef,
)
from omnibase_infra.models.health.model_projection_liveness_verdict import (
    ModelProjectionLivenessVerdict,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from omnibase_infra.models.observability import ModelNodeFlowWindow
    from omnibase_infra.protocols.protocol_auto_wiring_manifest_like import (
        ProtocolAutoWiringManifestLike,
    )

# Minimum envelopes a projection must have taken, summed across the retained
# windows, before a 100% DLQ ratio is a verdict rather than an artifact. Below
# this a single poison-pill event would degrade the whole runtime for a full
# check interval, which is noise that gets a real signal switched off. The real
# failure mode this exists to catch moves thousands of events per window (the
# session-replay projection took ~12k/session-day), so the floor costs nothing
# against a genuinely broken projection.
DLQ_SATURATION_MIN_MESSAGES: int = 10

# The ratio that counts as "fully DLQing". Exactly 1.0 by design: a projection
# that persists even one row in the observation window is a different problem
# (partial failure, poison pill, schema drift) and is NOT what this dimension
# claims. Narrow and true beats broad and arguable.
DLQ_SATURATION_RATIO: float = 1.0

# Cap on names rendered into a dimension detail. The detail is served on every
# ``/health`` response and shipped on every health event; a fleet-wide breakage
# must not turn it into a log dump. Mirrors
# ``service_runtime_health_monitor._MAX_NAMED_DISCOVERY_ERRORS``.
MAX_NAMED_PROJECTIONS: int = 8


def _declared_projection_refs(
    manifest: ProtocolAutoWiringManifestLike,
) -> tuple[ModelProjectionContractRef, ...]:
    """Every contract-declared projection in the manifest, ordered by name.

    The single discriminator both selectors below read, so the in-scope half
    and the excluded half can never disagree about what a projection is.

    In scope: a contract with ``subscribe_topics`` that declares
    ``db_io.db_tables``. That is the same discriminator the wiring seam uses to
    choose the projection dispatch arm (``handler_wiring._choose_dispatch_
    callback``: "Use projection callback when contract declares
    db_io.db_tables"), so the health surface and the wiring seam cannot
    disagree about what a projection is — a contract that one calls a
    projection and the other does not is how a gap hides.
    """
    contracts = getattr(manifest, "contracts", ())
    selected: list[ModelProjectionContractRef] = []
    for contract in contracts:
        event_bus = getattr(contract, "event_bus", None)
        if event_bus is None:
            continue
        topics = tuple(
            str(t) for t in (getattr(event_bus, "subscribe_topics", ()) or ())
        )
        if not topics:
            continue
        if bool(getattr(event_bus, "plugin_managed", False)):
            continue
        if bool(getattr(contract, "requires_cloud_gateway", False)):
            continue

        db_io = getattr(contract, "db_io", None)
        if not (db_io is not None and getattr(db_io, "db_tables", None)):
            continue

        name = str(getattr(contract, "name", "") or "")
        if not name:
            # A contract with no readable name cannot be named on the health
            # detail, and a placeholder ("unknown") would be worse than an
            # omission: it reports a projection that no operator can look up.
            continue

        selected.append(ModelProjectionContractRef(name=name, subscribe_topics=topics))
    return tuple(sorted(selected, key=lambda ref: ref.name))


def select_projection_contracts(
    manifest: ProtocolAutoWiringManifestLike,
    *,
    kernel_nonwriting: frozenset[str] = frozenset(),
) -> tuple[ModelProjectionContractRef, ...]:
    """Return the contract-declared projections this runtime must have attached.

    Deliberately OUT of scope, because being unattached is their correct state
    rather than a defect — including them would make this dimension fire on
    healthy lanes, which is how a health signal earns a permanent exclusion:

    * ``plugin_managed`` contracts — a domain plugin owns the subscription, so
      it never appears in the runtime's own bus registry (OMN-10864).
    * ``requires_cloud_gateway`` contracts — deliberately unwired on lanes with
      no cloud mirroring provisioned (OMN-13809).
    * **raw-event projection contracts** (``consumer_purpose: audit|projection``
      with no ``db_io.db_tables``). ``handler_wiring`` skips their Kafka
      subscription outright unless the kernel registered a result applier for
      that exact contract name (``_raw_event_projection_enabled``), and the
      registry is not visible from a health cycle. Being unattached is their
      documented default, so selecting them would report a permanent fleet-wide
      outage. Census 2026-08-29: the five contracts in this class
      (``node_build_loop_projection_compute``,
      ``node_gateway_link_health_projection_compute``,
      ``node_ledger_projection_compute``, ``node_pr_state_projection_compute``,
      ``node_validation_ledger_projection_compute``) all live in
      ``omnibase_infra`` and NONE of them declares ``db_io.db_tables``, while
      all 49 ``db_io.db_tables`` projections live in ``omnimarket`` and NONE of
      them declares a raw-event ``consumer_purpose`` — the two sets are
      disjoint, so nothing real is lost by narrowing to ``db_io``.
    * **kernel-nonwriting contracts** (OMN-17562) — every handler entry on the
      contract is wired with a no-op dispatch, so this kernel deliberately
      withholds the Kafka subscription rather than consuming, acking and
      discarding. Their topics correctly leave the live bus registry. This
      selector is manifest-derived while ``attached_topics`` is the live
      registry, so keeping them here would simply trade ``projection_write_path``
      DEGRADED for ``projection_attachment`` DEGRADED on every one of them —
      the same false outage reported by a different dimension. They are not
      dropped from the health surface: :func:`select_kernel_nonwriting_projections`
      returns exactly the excluded half, and the write-path leg names it.

    Args:
        manifest: A discovery manifest, already filtered to this runtime's
            profile by the caller. Read structurally so a manifest double that
            exposes only the protocol's guarantees degrades to an empty set
            instead of raising inside a health check.
        kernel_nonwriting: Contract names this process wires with NO live
            dispatcher, from
            ``projection_dispatch_ledger.projections_with_no_live_dispatcher``.

    Returns:
        The in-scope projections, ordered by contract name.
    """
    return tuple(
        ref
        for ref in _declared_projection_refs(manifest)
        if ref.name not in kernel_nonwriting
    )


def _topic_declarers(
    manifest: ProtocolAutoWiringManifestLike,
) -> dict[str, frozenset[str]]:
    """Map every subscribed topic in the manifest to the contracts declaring it.

    Read over EVERY contract, not only the ones the projection discriminator
    admits: ``onex.evt.omninode.node-introspection.v1`` is co-owned by
    ``node_ledger_projection_compute`` (a raw-event projection with no
    ``db_io.db_tables``) and ``node_registration_orchestrator`` (not a
    projection at all), and both of them put that topic in the live registry.
    A census narrowed to projections would miss exactly the co-owners that
    make an attachment unattributable.
    """
    declarers: dict[str, set[str]] = {}
    for contract in getattr(manifest, "contracts", ()):
        event_bus = getattr(contract, "event_bus", None)
        if event_bus is None:
            continue
        name = str(getattr(contract, "name", "") or "")
        if not name:
            continue
        for topic in getattr(event_bus, "subscribe_topics", ()) or ():
            declarers.setdefault(str(topic), set()).add(name)
    return {topic: frozenset(names) for topic, names in declarers.items()}


def select_kernel_nonwriting_projections(
    manifest: ProtocolAutoWiringManifestLike,
    kernel_nonwriting: frozenset[str],
) -> tuple[ModelProjectionContractRef, ...]:
    """Return the declared projections the fourth exclusion removed from scope.

    The complement of :func:`select_projection_contracts` over the same
    discriminator, so a contract can never be dropped by one and unclaimed by
    the other.

    Resolving names against the manifest here is also what keeps the write-path
    detail honest: a stale ledger entry, or one belonging to a contract this
    runtime profile does not own, resolves to nothing and is never rendered.
    An operator must be able to look up every name on a health detail.

    Each returned ref also carries ``attributable_subscribe_topics``: the
    topics whose presence in the live registry can only be this contract's own
    subscription, because every OTHER contract declaring them is itself in
    *kernel_nonwriting* and therefore had its subscription withheld too. The
    live registry is topic-keyed, so a topic shared with a contract that has a
    live in-process dispatcher carries no attribution at all — asking "is this
    projection's declared topic subscribed here?" instead of "did this process
    subscribe on behalf of this projection?" named ``projection_llm_cost``,
    ``projection_registration`` and ``projection_live_events`` as silent-loss
    sites on both ``.201`` lanes on 2026-09-04 while all 12 of their writers
    were healthy and consuming (OMN-17557).

    Args:
        manifest: The same profile-filtered discovery manifest.
        kernel_nonwriting: Contract names this process wires with NO live
            dispatcher.

    Returns:
        The excluded projections, ordered by contract name.
    """
    declarers = _topic_declarers(manifest)
    return tuple(
        ref.model_copy(
            update={
                "attributable_subscribe_topics": tuple(
                    topic
                    for topic in ref.subscribe_topics
                    if (declarers.get(topic, frozenset({ref.name})) - {ref.name})
                    <= kernel_nonwriting
                )
            }
        )
        for ref in _declared_projection_refs(manifest)
        if ref.name in kernel_nonwriting
    )


def _contract_group_infix(contract: object) -> str | None:
    """The ``.{package}.{node_name}.consume.`` infix of a contract's group id.

    The wiring seam's consumer identity is
    ``{env}.{service}.{node_name}.consume.{version}``; this is the middle of
    it, delimited by ``.`` at both ends so a component boundary is always
    matched. ``env`` and ``version`` are dropped for the reasons
    :func:`select_projection_group_suffixes` documents.

    Shared by the projection map and the non-projection map (OMN-16753) so the
    two can never derive an infix differently for the same contract -- a
    disagreement between them would decide whether a group is attributed or
    excluded, which is exactly the class of silent divergence this module keeps
    closing.

    Returns ``None`` for a contract no group id could have been minted from,
    which is dropped rather than guessed at.
    """
    from omnibase_infra.utils import normalize_kafka_identifier

    package = str(getattr(contract, "package_name", "") or "")
    name = str(getattr(contract, "name", "") or "")
    if not package or not name:
        return None
    try:
        parts = [
            normalize_kafka_identifier(package),
            normalize_kafka_identifier(name),
            normalize_kafka_identifier(EnumConsumerGroupPurpose.CONSUME.value),
        ]
    except ValueError:
        # A component that normalizes to nothing cannot be part of a group id
        # the wiring seam could have minted either, so there is nothing to
        # attribute. Skipped, never guessed at.
        return None
    return "." + ".".join(parts) + "."


def _unambiguous_group_infixes(
    manifest: ProtocolAutoWiringManifestLike,
    *,
    admit: Callable[[object, str], bool],
) -> dict[str, str]:
    """``{group_infix: contract_name}`` over the admitted contracts.

    Two guards, both of which drop rather than guess:

    * an infix two admitted contracts derive is dropped from BOTH;
    * an infix contained in another admitted infix is dropped, so a group
      matching the longer one cannot also match the shorter.
    """
    owners: dict[str, set[str]] = {}
    for contract in getattr(manifest, "contracts", ()):
        name = str(getattr(contract, "name", "") or "")
        if not admit(contract, name):
            continue
        infix = _contract_group_infix(contract)
        if infix is None:
            continue
        owners.setdefault(infix, set()).add(name)
    unique = {
        infix: next(iter(names)) for infix, names in owners.items() if len(names) == 1
    }
    return {
        infix: name
        for infix, name in unique.items()
        if not any(infix in other for other in unique if other != infix)
    }


def select_projection_group_suffixes(
    manifest: ProtocolAutoWiringManifestLike,
    projections: tuple[ModelProjectionContractRef, ...],
) -> dict[str, str]:
    """Map each in-scope projection's consumer-group INFIX to its contract name.

    OMN-16753. The saturation half reads OMN-16777 flow deltas, which are keyed
    by ``(consumer_group, topic)``. A topic is not an attribution -- three
    projections declare the same four ``onex.evt.omniclaude.*`` topics on the
    ``.201`` lanes -- but a consumer group is: the wiring seam binds the
    counters with ``compute_consumer_group_id(identity, CONSUME)`` for exactly
    one contract. This map is what turns the group back into the contract.

    **An infix of the group id, not the whole thing.** The wiring seam's
    identity is ``{env}.{service}.{node_name}.consume.{version}``, and the key
    kept here is ``.{service}.{node_name}.consume.`` -- the two ends are
    dropped deliberately:

    * ``env`` is a process-wide variable no contract declares. Binding it in
      would make attribution depend on this process and the wiring seam
      agreeing about ``ONEX_ENVIRONMENT``, and a disagreement fails SILENTLY
      back to "nobody is saturated" -- a false all-clear on the one dimension
      that exists to catch a total loss.
    * ``version`` is the one component that changes under a contract bump, and
      it buys no discrimination: ``(service, node_name)`` is already unique
      within a manifest. Measured 2026-09-08 against the live lab: the
      ``.201`` stability lane's ``projection_session_replay`` group carries
      ``consume.1.1.0`` while the dev-lane manifest declares a different
      version, so a version-bound key silently matched nothing.

    The bus appends ``.__i.<instance>`` and ``.__t.<topic>`` when it joins, but
    ``handler_wiring`` registers the flow counters BEFORE that, so a delta may
    carry either the bare base id or the full live name. A containment test
    accepts both.

    Two guards, both of which drop rather than guess -- the same posture as
    ``attributable_subscribe_topics``, because an arbitrary attribution here is
    the defect this function exists to remove:

    * a key two contracts derive is dropped from BOTH;
    * a key that is itself contained in another key is dropped, so a group
      matching the longer one cannot also match the shorter.

    Args:
        manifest: The profile-filtered discovery manifest.
        projections: The in-scope refs from :func:`select_projection_contracts`.
            Only these are mapped -- a name no dimension can report must not
            reach the totals.

    Returns:
        ``{group_infix: projection_name}``, each key delimited by ``.`` at both
        ends so a component boundary is always matched.
    """
    in_scope = {ref.name for ref in projections}
    return _unambiguous_group_infixes(
        manifest, admit=lambda contract, name: name in in_scope
    )


def select_nonprojection_group_infixes(
    manifest: ProtocolAutoWiringManifestLike,
    projection_group_suffixes: Mapping[str, str],
) -> dict[str, str]:
    """Map each NON-projection consumer's group INFIX to its contract name.

    OMN-16753, round 3. The saturation half reads flow deltas keyed by
    ``(consumer_group, topic)``, and a topic a projection declares is not a
    topic only projections consume. On the ``.201`` stability lane twelve
    consumer groups carry ``onex.evt.omniclaude.session-{started,ended}.v1``:
    eleven are the three co-declaring projections' subscriptions, and the
    twelfth pair belongs to ``node_session_phase_reducer`` -- a REDUCER.

    Its flow reached :func:`_attribute_delta`, matched no projection infix
    (there is none to derive: it is not a projection), and could not take the
    sole-declarer fallback either (three in-scope projections declare the
    topic), so it was recorded as UNATTRIBUTABLE. Round 2 had just made
    unattributable flow degrade the dimension, so the refresh gate refused a
    lane whose every other leg passed and rolled it back onto the old image
    (attempt 3 of 3, receipt ``20260908T235241Z-915a10446a8d.json``).

    **The discriminator is the contract, never a name list.** A contract is
    admitted here when it subscribes to at least one topic and declares NO
    ``db_io.db_tables`` -- the exact absence of the single positive signal
    :func:`select_projection_contracts` admits on, which is itself the signal
    the wiring seam uses to choose the projection dispatch arm. So a reducer,
    an effect and a forwarder are all recognised by what their contract
    declares, and a node added tomorrow needs no edit here.

    **Exclusion requires positive proof, and its absence fails closed.** A
    group that matches nothing in this map is NOT excluded: it falls through to
    the unchanged attribution path, where an unnameable group on a
    multi-declarer topic is still UNATTRIBUTABLE and still degrades the
    dimension. Only a group provably owned by a non-projection consumer is
    dropped from the arithmetic.

    Three guards, one more than the projection map has:

    * an infix two admitted contracts derive is dropped from both;
    * an infix contained in another admitted infix is dropped;
    * an infix that collides with, contains, or is contained in ANY projection
      infix is dropped. That direction matters most: excluding a group the
      projection map could have matched would silence a real projection's flow,
      which is strictly worse than the DEGRADED it would replace.

    Args:
        manifest: The profile-filtered discovery manifest.
        projection_group_suffixes: The in-scope projection map from
            :func:`select_projection_group_suffixes`, so the two maps cannot
            claim the same group.

    Returns:
        ``{group_infix: contract_name}`` for the non-projection consumers.
    """

    def _admit(contract: object, name: str) -> bool:
        event_bus = getattr(contract, "event_bus", None)
        if event_bus is None:
            return False
        if not (getattr(event_bus, "subscribe_topics", ()) or ()):
            return False
        db_io = getattr(contract, "db_io", None)
        return not (db_io is not None and getattr(db_io, "db_tables", None))

    candidates = _unambiguous_group_infixes(manifest, admit=_admit)
    return {
        infix: name
        for infix, name in candidates.items()
        if not any(
            infix in projection_infix or projection_infix in infix
            for projection_infix in projection_group_suffixes
        )
    }


#: The reason rendered on an EXCLUDED reading, and on the health detail beside
#: the group it excused. One string, so the prose an operator reads and the
#: rule the fold applied cannot describe different things.
NONPROJECTION_EXCLUSION_REASON: str = (
    "not a declared projection (declares no db_io.db_tables projection "
    "target), so its flow is not a projection's to account for"
)


def _attribute_delta(
    consumer_group: str,
    topic: str,
    *,
    group_suffixes: Mapping[str, str],
    topic_declarers: Mapping[str, frozenset[str]],
    nonprojection_infixes: Mapping[str, str],
) -> ModelFlowAttribution:
    """Read one flow delta as ATTRIBUTED, EXCLUDED, or UNATTRIBUTABLE.

    Three readings, in this order, and the order is the fix:

    1. **Group first.** A consumer group provably one in-scope projection's
       subscription attributes to it.
    2. **Non-projection owner second** (OMN-16753 round 3). A group provably
       owned by a contract that declares no projection target table is
       EXCLUDED: its flow was never a projection's to take, so it belongs in
       neither a ratio nor the unattributable set.
    3. **Sole-declarer topic last.** The fallback fires only for a topic
       exactly ONE in-scope projection declares, where it carries a real
       attribution; on a shared topic it carries none, and the pre-OMN-16753
       behaviour of handing it to the alphabetically last declarer is precisely
       the misattribution this function exists to remove.

    Step 2 must precede step 3, not follow it. On a topic exactly one
    projection declares, a reducer's delta would otherwise take the fallback
    and be folded into that projection's ratio -- the same misattribution class
    in its most damaging form, since a reducer DLQing everything would name a
    healthy projection as 100% DLQ-routed.

    Anything not positively resolved by one of the three is UNATTRIBUTABLE,
    which degrades the dimension. Exclusion is never the default.
    """
    declarers = topic_declarers.get(topic)
    if not declarers:
        return ModelFlowAttribution(
            consumer_group=consumer_group, outcome="UNATTRIBUTABLE"
        )
    # EVERY match, not the first. Iterating a dict and returning on the first
    # containment made the answer depend on manifest insertion order: the
    # guards in ``select_projection_group_suffixes`` remove infixes that are
    # ambiguous or nested WITHIN the map, but they cannot prove a real group id
    # contains no two of them -- a service name that embeds another
    # projection's normalised name would do it. Two matches is exactly the
    # "cannot tell" case, so it drops to unattributable, which now degrades the
    # dimension rather than vanishing.
    matched = {
        name
        for infix, name in group_suffixes.items()
        if infix in consumer_group and name in declarers
    }
    if len(matched) == 1:
        return ModelFlowAttribution(
            consumer_group=consumer_group,
            outcome="ATTRIBUTED",
            projection=next(iter(matched)),
        )
    if matched:
        return ModelFlowAttribution(
            consumer_group=consumer_group, outcome="UNATTRIBUTABLE"
        )
    # An exclusion has to NAME its owner: one nobody can look up is invisible
    # on the health detail, and an invisible exclusion is the same false
    # all-clear this ticket removes. Two owners is therefore not "doubly
    # excluded" but unnameable, and fails closed to UNATTRIBUTABLE.
    owners = {
        name for infix, name in nonprojection_infixes.items() if infix in consumer_group
    }
    if len(owners) == 1:
        return ModelFlowAttribution(
            consumer_group=consumer_group,
            outcome="EXCLUDED",
            owner_contract=next(iter(owners)),
            reason=NONPROJECTION_EXCLUSION_REASON,
        )
    if not owners and len(declarers) == 1:
        return ModelFlowAttribution(
            consumer_group=consumer_group,
            outcome="ATTRIBUTED",
            projection=next(iter(declarers)),
        )
    return ModelFlowAttribution(consumer_group=consumer_group, outcome="UNATTRIBUTABLE")


def evaluate_projection_liveness(
    *,
    projections: tuple[ModelProjectionContractRef, ...],
    attached_topics: frozenset[str],
    flow_windows: Iterable[ModelNodeFlowWindow],
    kernel_nonwriting: tuple[ModelProjectionContractRef, ...] = (),
    projection_group_suffixes: Mapping[str, str] | None = None,
    nonprojection_group_infixes: Mapping[str, str] | None = None,
) -> ModelProjectionLivenessVerdict:
    """Compute the projection liveness verdict from injected observations.

    Args:
        projections: The in-scope projections from
            :func:`select_projection_contracts`.
        attached_topics: Topics with a live subscription on this runtime's bus
            registry. **Empty means unobservable, not empty**: the attachment
            half reports UNKNOWN rather than flagging every projection, because
            a test double or a bus without ``get_consumer_groups()`` would
            otherwise manufacture a fleet-wide outage.
        flow_windows: Closed OMN-16777 flow windows. Empty means the saturation
            half reports UNKNOWN — a runtime whose heartbeat has not yet closed
            a window has not proven anything either way.
        kernel_nonwriting: OMN-17448/OMN-17562. The declared projections every
            one of whose handler entries this process wired with a no-op
            dispatch, from :func:`select_kernel_nonwriting_projections`. Unlike
            the two halves above, an empty tuple here is NOT ambiguous: the
            underlying ledger is written by the wiring seam on the same branches
            it describes, so "nothing recorded" means "every projection this
            process wired has a live dispatcher". Passed as resolved refs rather
            than bare names so a name that is not a declared projection of this
            runtime cannot reach a health detail, and so the subset that is
            STILL attached — the actual silent-loss state — can be computed.
        projection_group_suffixes: OMN-16753. ``{group_infix: projection}``
            from :func:`select_projection_group_suffixes`.
        nonprojection_group_infixes: OMN-16753. ``{group_infix: contract}``
            from :func:`select_nonprojection_group_infixes` — the consumers on
            this manifest that are NOT projections. Their deltas are dropped
            from the arithmetic entirely and recorded on
            ``excluded_nonprojection_flow`` instead. ``None`` and empty are the
            same conservative reading: nothing is excluded, so a reducer's flow
            falls through to UNATTRIBUTABLE exactly as it did before this
            argument existed.

    Returns:
        The verdict. Names, counts, and two UNKNOWN flags; no status word.
    """
    windows = tuple(flow_windows)

    attachment_evaluated = bool(attached_topics)
    unattached: list[str] = []
    if attachment_evaluated:
        unattached = [
            ref.name
            for ref in projections
            if any(topic not in attached_topics for topic in ref.subscribe_topics)
        ]

    saturation_evaluated = bool(windows)
    saturated: list[str] = []
    unattributable: set[str] = set()
    excluded: dict[str, ModelFlowAttribution] = {}
    if saturation_evaluated:
        # OMN-16753. Attribution is per SUBSCRIPTION, never per topic. The
        # previous fold was
        #
        #     {topic: ref.name for ref in projections
        #             for topic in ref.subscribe_topics}
        #
        # which collapses a topic to ONE projection while the deltas it reads
        # are keyed by (consumer_group, topic). ``projections`` is name-ordered,
        # so on a topic with several declarers the alphabetically LAST one
        # absorbed every sibling's counters. Both directions were live on the
        # .201 stability lane on 2026-09-08: a healthy projection named for a
        # peer's total loss, and -- where the peers also consume -- the failing
        # projection's own ratio diluted below the threshold so nothing is
        # reported at all.
        declarers: dict[str, set[str]] = {}
        for ref in projections:
            for declared_topic in ref.subscribe_topics:
                declarers.setdefault(declared_topic, set()).add(ref.name)
        topic_declarers = {
            topic: frozenset(names) for topic, names in declarers.items()
        }
        group_suffixes = dict(projection_group_suffixes or {})
        nonprojection_infixes = dict(nonprojection_group_infixes or {})

        totals_in: dict[str, int] = {}
        totals_dlq: dict[str, int] = {}
        for window in windows:
            for delta in window.consumer_deltas:
                attribution = _attribute_delta(
                    delta.consumer_group,
                    delta.topic,
                    group_suffixes=group_suffixes,
                    topic_declarers=topic_declarers,
                    nonprojection_infixes=nonprojection_infixes,
                )
                projection_name = attribution.projection
                if projection_name is not None:
                    totals_in[projection_name] = (
                        totals_in.get(projection_name, 0) + delta.messages_in
                    )
                    totals_dlq[projection_name] = (
                        totals_dlq.get(projection_name, 0) + delta.messages_dlq
                    )
                    continue
                if not (
                    delta.topic in topic_declarers
                    and (delta.messages_in or delta.messages_dlq)
                ):
                    # Either no in-scope projection declares the topic at all,
                    # or the window recorded no movement on it. Neither is a
                    # fact about projection liveness.
                    continue
                if attribution.outcome == "EXCLUDED":
                    # OMN-16753 round 3. Flow a NON-projection consumer took on
                    # a topic some projection also declares. It is not this
                    # dimension's to account for -- but it IS recorded, keyed
                    # by consumer group so the same group across topics and
                    # windows collapses to one entry, and it is rendered on the
                    # detail. An exclusion nobody can see would be the same
                    # false all-clear as a misattribution, in a third costume.
                    excluded.setdefault(delta.consumer_group, attribution)
                    continue
                # Traffic on a declared projection topic that no single
                # projection can be shown to have taken. Recorded rather than
                # dropped: silently skipping it would turn a case the gate
                # cannot answer into a clean bill of health, which is the same
                # failure class in a new costume.
                unattributable.add(delta.topic)
        saturated = sorted(
            name
            for name, taken in totals_in.items()
            if taken >= DLQ_SATURATION_MIN_MESSAGES
            and (min(totals_dlq.get(name, 0), taken) / taken) >= DLQ_SATURATION_RATIO
        )

    # OMN-17562. Two different facts, and collapsing them is what made the
    # OMN-17448 dimension unactionable:
    #
    #  * ``nonwriting`` — this kernel dispatches nothing for them. Expected and
    #    correct on every lane where a dedicated writer process owns the rows;
    #    whether that writer is DEPLOYED here is a corpus-level claim over the
    #    deployment manifests (OMN-17448 AC5), which a kernel process cannot
    #    see and therefore must not report on.
    #  * ``nonwriting_attached`` — dispatches nothing AND is still consuming.
    #    That is the silent-loss state itself: the consumer takes every message
    #    and commits every offset while no handler runs, so the events are
    #    destroyed rather than merely unwritten. After this ticket the kernel
    #    withholds the subscription, so this list is empty; a change that
    #    re-subscribes one reopens the loss and lights the dimension.
    #
    # The second is read off ``attributable_subscribe_topics``, never off the
    # ref's full declared set. ``attached_topics`` is topic-keyed: a topic is
    # in it when ANY contract in this process subscribed it, so a declared
    # topic shared with a live-dispatching co-owner proves nothing about this
    # projection. The residual is a loss of sensitivity, not of safety — on a
    # wholly shared topic no in-process fact can attribute the subscription,
    # and the wiring seam's withholding (which IS per contract) is what
    # actually prevents the loss.
    #
    # ``attachment_evaluated`` gates the second: with no readable registry the
    # subset is unknowable, and an empty list must not read as "none attached".
    nonwriting = tuple(sorted(ref.name for ref in kernel_nonwriting))
    nonwriting_attached: tuple[str, ...] = ()
    if attachment_evaluated:
        nonwriting_attached = tuple(
            sorted(
                ref.name
                for ref in kernel_nonwriting
                if any(
                    topic in attached_topics
                    for topic in ref.attributable_subscribe_topics
                )
            )
        )

    return ModelProjectionLivenessVerdict(
        projection_count=len(projections),
        attachment_evaluated=attachment_evaluated,
        unattached_projections=tuple(sorted(unattached)),
        saturation_evaluated=saturation_evaluated,
        dlq_saturated_projections=tuple(saturated),
        observed_window_count=len(windows),
        unattributable_flow_topics=tuple(sorted(unattributable)),
        excluded_nonprojection_flow=tuple(
            excluded[group] for group in sorted(excluded)
        ),
        nonwriting_projections=nonwriting,
        nonwriting_attached_projections=nonwriting_attached,
    )


def _name_list(names: tuple[str, ...]) -> str:
    """Render a capped, comma-joined name list with an explicit remainder."""
    listed = ", ".join(names[:MAX_NAMED_PROJECTIONS])
    remaining = len(names) - MAX_NAMED_PROJECTIONS
    if remaining > 0:
        listed = f"{listed}, +{remaining} more"
    return listed


def describe_projection_attachment(verdict: ModelProjectionLivenessVerdict) -> str:
    """Build the ``projection_attachment`` dimension detail."""
    if not verdict.attachment_evaluated:
        return (
            "no live subscription registry available — projection attachment "
            "UNKNOWN (not asserted healthy)"
        )
    if not verdict.unattached_projections:
        return f"All {verdict.projection_count} declared projection(s) attached"
    return (
        f"{len(verdict.unattached_projections)}/{verdict.projection_count} declared "
        f"projection(s) have no attached consumer and persist nothing: "
        f"{_name_list(verdict.unattached_projections)}"
    )


#: The runtime health-dimension vocabulary, declared here so the saturation
#: status is produced IN it rather than as a free string a call site has to
#: narrow. The narrowing step it replaces mapped every unrecognised value to
#: DEGRADED, which is fail-closed but silently regrades a typo; a Literal makes
#: the same mistake a type error at the point it is written.
EnumDlqSaturationStatus = Literal["HEALTHY", "DEGRADED", "CRITICAL"]


def dlq_saturation_status(
    verdict: ModelProjectionLivenessVerdict,
) -> EnumDlqSaturationStatus:
    """The ``projection_dlq_saturation`` dimension status for a verdict.

    OMN-16753. Lives beside :func:`describe_dlq_saturation` and is the sole
    source of the dimension's status, so the fact the prose reports and the
    fact the verdict is taken from cannot drift apart -- the first revision of
    this fix annotated the prose with the unattributable topics while the
    status was still computed from ``dlq_saturated_projections`` alone at the
    call site, which published a **HEALTHY** saturation dimension for a lane
    whose flow could not be attributed at all.

    ``DEGRADED`` when EITHER holds:

    * a projection is fully DLQ-routed -- the measured failure; or
    * a declared projection topic carried flow that no single projection can be
      shown to have taken. That is not a clean lane, it is an unanswered
      question, and this dimension exists precisely because a total loss reads
      green on every other signal. Excluding those topics from the ratios (the
      only honest arithmetic available) while still publishing HEALTHY would
      convert "the gate could not tell" into "nothing is wrong", which is the
      same false all-clear this ticket removes, in a new costume.

    There is no third status here: the runtime health vocabulary is
    HEALTHY/DEGRADED/CRITICAL, so an indeterminate saturation reading fails
    closed to DEGRADED rather than being rendered as an UNKNOWN nobody gates
    on.

    ``excluded_nonprojection_flow`` is deliberately NOT read here (OMN-16753
    round 3). An excluded group is a question that was answered -- the flow was
    a reducer's, an effect's or a forwarder's, and was never a projection's to
    account for -- so degrading on it would report a runtime defect over a lane
    doing exactly what its contracts declare. That is what failed stability
    refresh attempt 3 of 3 at ``915a10446``: two ``session-{started,ended}``
    topics went DEGRADED because ``node_session_phase_reducer`` also consumes
    them. The exclusion is still rendered by :func:`describe_dlq_saturation`,
    because unreported is not the same as ungated.
    """
    if verdict.dlq_saturated_projections or verdict.unattributable_flow_topics:
        return "DEGRADED"
    return "HEALTHY"


def describe_dlq_saturation(verdict: ModelProjectionLivenessVerdict) -> str:
    """Build the ``projection_dlq_saturation`` dimension detail."""
    if not verdict.saturation_evaluated:
        return (
            "no closed flow window observed yet — projection DLQ ratio UNKNOWN "
            "(not asserted healthy)"
        )
    # OMN-16753. Carried on BOTH the clean and the degraded rendering: a reader
    # who sees only "no projection is fully DLQ-routed" cannot tell a measured
    # zero from a topic the gate could not attribute, and those are different
    # facts.
    unattributed = ""
    if verdict.unattributable_flow_topics:
        unattributed = (
            f" ({len(verdict.unattributable_flow_topics)} topic(s) carried flow "
            "no single declaring projection could be attributed, and are "
            f"excluded from every ratio: "
            f"{_name_list(verdict.unattributable_flow_topics)})"
        )
    # OMN-16753 round 3. Rendered on BOTH the clean and the degraded case, and
    # kept SEPARATE from the clause above: an excluded group is a question that
    # was answered ("that flow is not a projection's"), while an unattributable
    # topic is one that was not. Collapsing the two prose clauses would hide
    # which of those an operator is looking at, and the whole cost of this
    # ticket was a lane rolled back over that exact confusion.
    excluded = ""
    if verdict.excluded_nonprojection_flow:
        owners = tuple(
            sorted(
                {
                    entry.owner_contract
                    for entry in verdict.excluded_nonprojection_flow
                    if entry.owner_contract is not None
                }
            )
        )
        excluded = (
            f" [{len(verdict.excluded_nonprojection_flow)} consumer group(s) "
            f"carried flow on a declared projection topic and were excluded "
            f"from projection attribution — {NONPROJECTION_EXCLUSION_REASON}: "
            f"{_name_list(owners)}]"
        )
    if not verdict.dlq_saturated_projections:
        return (
            f"No projection is fully DLQ-routed over {verdict.observed_window_count} "
            f"flow window(s){unattributed}{excluded}"
        )
    return (
        f"{len(verdict.dlq_saturated_projections)} projection(s) routed 100% of "
        f"consumed events to a DLQ/quarantine sink over "
        f"{verdict.observed_window_count} flow window(s) — offsets commit, so lag "
        f"reads 0 over a total loss: {_name_list(verdict.dlq_saturated_projections)}"
        f"{unattributed}{excluded}"
    )


def describe_projection_write_path(verdict: ModelProjectionLivenessVerdict) -> str:
    """Build the ``projection_write_path`` dimension detail (OMN-17448/OMN-17562)."""
    if not verdict.nonwriting_projections:
        return (
            f"All {verdict.projection_count} declared projection(s) dispatch in-process"
        )
    if verdict.nonwriting_attached_projections:
        return (
            f"{len(verdict.nonwriting_attached_projections)} projection(s) are "
            f"SUBSCRIBED here but dispatch NOTHING in this process "
            f"(standalone-runner shape): offsets commit and every event is "
            f"consumed, acked and destroyed rather than left replayable for the "
            f"dedicated writer that owns the rows: "
            f"{_name_list(verdict.nonwriting_attached_projections)}"
        )
    return (
        f"{len(verdict.nonwriting_projections)} declared projection(s) have no "
        f"in-process dispatcher here and are deliberately not subscribed "
        f"(OMN-17562), so their events stay replayable; their rows depend "
        f"entirely on a dedicated writer this process cannot see, whose "
        f"presence is asserted by the static lane writer-coverage gate rather "
        f"than by this runtime: {_name_list(verdict.nonwriting_projections)}"
    )


__all__: list[str] = [
    "DLQ_SATURATION_MIN_MESSAGES",
    "DLQ_SATURATION_RATIO",
    "MAX_NAMED_PROJECTIONS",
    "NONPROJECTION_EXCLUSION_REASON",
    "describe_dlq_saturation",
    "describe_projection_attachment",
    "describe_projection_write_path",
    "evaluate_projection_liveness",
    "select_kernel_nonwriting_projections",
    "select_nonprojection_group_infixes",
    "select_projection_contracts",
]
