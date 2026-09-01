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


def select_projection_contracts(
    manifest: ProtocolAutoWiringManifestLike,
) -> tuple[ModelProjectionContractRef, ...]:
    """Return the contract-declared projections this runtime must have attached.

    In scope: a contract with ``subscribe_topics`` that declares
    ``db_io.db_tables``. That is the same discriminator the wiring seam uses to
    choose the projection dispatch arm (``handler_wiring._choose_dispatch_
    callback``: "Use projection callback when contract declares
    db_io.db_tables"), so the health surface and the wiring seam cannot disagree
    about what a projection is — a contract that one calls a projection and the
    other does not is how a gap hides.

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

    Args:
        manifest: A discovery manifest, already filtered to this runtime's
            profile by the caller. Read structurally so a manifest double that
            exposes only the protocol's guarantees degrades to an empty set
            instead of raising inside a health check.

    Returns:
        The in-scope projections, ordered by contract name.
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


def evaluate_projection_liveness(
    *,
    projections: tuple[ModelProjectionContractRef, ...],
    attached_topics: frozenset[str],
    flow_windows: Iterable[ModelNodeFlowWindow],
    dispatch_skipped: frozenset[str] = frozenset(),
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
        dispatch_skipped: OMN-17448. Contract names this process wired onto the
            standalone-runner branch, from
            ``omnibase_infra.runtime.projection_dispatch_ledger``. Unlike the
            two halves above, an empty set here is NOT ambiguous: the ledger is
            written by the wiring seam on the same branch it describes, so
            "nothing recorded" means "no projection took that branch in this
            process". Only projections that are BOTH in scope and actually
            subscribed are named — a contract this runtime does not own cannot
            be a non-writing projection of it.

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

    # OMN-17448. Narrowed to projections this health cycle already has in scope,
    # so a stale or foreign ledger entry can never manufacture a name the
    # operator cannot look up in the contract set this runtime wired.
    in_scope = {ref.name for ref in projections}
    nonwriting = sorted(in_scope & dispatch_skipped)

    return ModelProjectionLivenessVerdict(
        projection_count=len(projections),
        attachment_evaluated=attachment_evaluated,
        unattached_projections=tuple(sorted(unattached)),
        saturation_evaluated=saturation_evaluated,
        dlq_saturated_projections=tuple(saturated),
        observed_window_count=len(windows),
        nonwriting_projections=tuple(nonwriting),
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
    """Build the ``projection_write_path`` dimension detail (OMN-17448)."""
    if not verdict.nonwriting_projections:
        return (
            f"All {verdict.projection_count} declared projection(s) dispatch in-process"
        )
    return (
        f"{len(verdict.nonwriting_projections)}/{verdict.projection_count} declared "
        f"projection(s) are subscribed here but dispatch NOTHING in this process "
        f"(standalone-runner shape): offsets commit and no row is written unless a "
        f"dedicated writer is deployed for each on this lane: "
        f"{_name_list(verdict.nonwriting_projections)}"
    )


__all__: list[str] = [
    "DLQ_SATURATION_MIN_MESSAGES",
    "DLQ_SATURATION_RATIO",
    "MAX_NAMED_PROJECTIONS",
    "describe_dlq_saturation",
    "describe_projection_attachment",
    "describe_projection_write_path",
    "evaluate_projection_liveness",
    "select_projection_contracts",
]
