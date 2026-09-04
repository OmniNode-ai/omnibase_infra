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

from typing import TYPE_CHECKING

from omnibase_infra.models.health.model_projection_contract_ref import (
    ModelProjectionContractRef,
)
from omnibase_infra.models.health.model_projection_liveness_verdict import (
    ModelProjectionLivenessVerdict,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

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


def evaluate_projection_liveness(
    *,
    projections: tuple[ModelProjectionContractRef, ...],
    attached_topics: frozenset[str],
    flow_windows: Iterable[ModelNodeFlowWindow],
    kernel_nonwriting: tuple[ModelProjectionContractRef, ...] = (),
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
    if saturation_evaluated:
        topic_to_projection = {
            topic: ref.name for ref in projections for topic in ref.subscribe_topics
        }
        totals_in: dict[str, int] = {}
        totals_dlq: dict[str, int] = {}
        for window in windows:
            for delta in window.consumer_deltas:
                projection_name = topic_to_projection.get(delta.topic)
                if projection_name is None:
                    continue
                totals_in[projection_name] = (
                    totals_in.get(projection_name, 0) + delta.messages_in
                )
                totals_dlq[projection_name] = (
                    totals_dlq.get(projection_name, 0) + delta.messages_dlq
                )
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


def describe_dlq_saturation(verdict: ModelProjectionLivenessVerdict) -> str:
    """Build the ``projection_dlq_saturation`` dimension detail."""
    if not verdict.saturation_evaluated:
        return (
            "no closed flow window observed yet — projection DLQ ratio UNKNOWN "
            "(not asserted healthy)"
        )
    if not verdict.dlq_saturated_projections:
        return (
            f"No projection is fully DLQ-routed over {verdict.observed_window_count} "
            "flow window(s)"
        )
    return (
        f"{len(verdict.dlq_saturated_projections)} projection(s) routed 100% of "
        f"consumed events to a DLQ/quarantine sink over "
        f"{verdict.observed_window_count} flow window(s) — offsets commit, so lag "
        f"reads 0 over a total loss: {_name_list(verdict.dlq_saturated_projections)}"
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
    "describe_dlq_saturation",
    "describe_projection_attachment",
    "describe_projection_write_path",
    "evaluate_projection_liveness",
    "select_kernel_nonwriting_projections",
    "select_projection_contracts",
]
