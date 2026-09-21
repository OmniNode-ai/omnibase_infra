# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Projection flow invariants as runtime health dimensions (OMN-18910).

The gap this closes
-------------------
Eight dimensions ship on this surface today and every one of them answers
whether a consumer is ATTACHED and MOVING. None answers whether it is moving
and WRITING. That distinction is not academic: a consumer that refuses a
message still commits its offset, a projection that returns without writing
still commits its offset, and a cache that discards a delta still consumed it.
Consumer-group lag was zero throughout OMN-18880 (nine hours of total refusal),
OMN-18905 (a snapshot cache frozen mid-replay while readiness read healthy) and
OMN-18769 (a writer that wrote nothing and raised nothing).

What this claims, and what it does not
--------------------------------------
It does not prevent any of those three. It shortens detection, and that is the
honest claim: in all three the time-to-detection was set by a person happening
to look.

Two dimensions
--------------
``projection_apply_divergence``
    Per projection over the retained windows: source events consumed against
    rows upserted. DEGRADED when consumed advanced by at least
    :data:`APPLY_DIVERGENCE_MIN_CONSUMED` while upserts stayed at zero.

``projection_delta_dropped``
    Grades the cumulative discarded-delta gauge on ACCUMULATION, never on
    magnitude. One drop is legitimate idempotence; a high flat gauge is a
    measured steady state. Only a RISING gauge is loss. Flat-state alarming is
    what gets a dimension muted, and a muted dimension is worse than none.

Why the two exemptions are exclusions and not thresholds
--------------------------------------------------------
``node_projection_session_replay`` and ``node_projection_work_events`` publish a
fixed source coordinate and are CORRECT to do so. Both key on a
content-addressed, immutable grain, so the serving cache only ever compares a
key against a delta derived from the same source event, and the drop is the
intended idempotence rather than lost data. They are enumerated and excluded by
name rather than thresholded over, because a threshold wide enough to cover
them is wide enough to hide the defect.

Shape
-----
Pure functions over injected data, in the shape ``projection_dlq_saturation``
already establishes: a ``describe_*`` and a ``*_status`` beside each other, with
the status as the SOLE source of the dimension's verdict so the prose and the
grade cannot drift apart. No clock, no bus, no database, no lifecycle.

Fail-closed
-----------
An indeterminate reading is DEGRADED, never HEALTHY, and says which of the two
it is:

* registered projections with no closed window — the process cannot tell yet;
* a cumulative gauge that went BACKWARDS — the reading did not come from one
  continuous process, so it cannot honestly be graded as flat.

A process that registered NO projection dispatch is a different case and is
HEALTHY: that is a measured zero, there is nothing here to diverge, and whether
a projection ought to be attached in this process is already owned by
``projection_attachment`` and ``projection_write_path``.

Related Tickets:
    - OMN-18910: this module (epic OMN-18906 AC-4)
    - OMN-18880 / OMN-18905 / OMN-18769: the three defects it shortens
    - OMN-18992: the guard-refusal split whose accumulation this grades
    - OMN-16994: the sibling liveness dimensions whose shape this follows
"""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING, Literal

from omnibase_infra.models.health.model_projection_apply_flow_verdict import (
    ModelProjectionApplyFlowVerdict,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from omnibase_infra.models.observability.model_projection_apply_delta import (
        ModelProjectionApplyDelta,
    )


#: Envelopes a projection must have taken, summed across the retained windows,
#: before "wrote nothing" is a verdict rather than an artifact. Below this, one
#: slow tick with a single in-flight event would degrade the whole runtime for a
#: check interval, which is the noise that gets a real signal switched off. The
#: failures this exists to catch moved thousands of events per window — the
#: runner-fleet projection in OMN-18880 refused continuously for nine hours — so
#: the floor costs nothing against a genuinely broken projection.
APPLY_DIVERGENCE_MIN_CONSUMED: int = 20

#: How many window-over-window RISES in the cumulative discarded-delta gauge
#: count as accumulation rather than as the single legitimate idempotent drop.
#: Two, not one: one rise is the idempotence case by construction, and requiring
#: three would take three health intervals to see a cache that is losing every
#: delta it takes.
DELTA_DROP_MIN_RISING_WINDOWS: int = 2

#: Total rise across the retained windows before accumulation is a verdict.
#: Paired with the window count above so that neither a slow trickle of
#: legitimate redeliveries nor two large idempotent drops alone can grade a
#: healthy projection degraded.
DELTA_DROP_MIN_ACCUMULATION: int = 10

#: The interim fallback exemption set, consulted ONLY for an exposure whose
#: contract resolves no key grain (OMN-19081, operator ruling 2026-09-21).
#:
#: The contract declaration is the source of truth and WINS wherever it
#: answers; this list is not a second opinion, it is what covers the window in
#: which a deployed runtime carries an omnimarket older than the one that
#: introduced ``key_grain``.
#:
#: THIS EXISTS BECAUSE A PRE-PR PROOF FAILED, and the measurement is worth
#: keeping next to the constant. Deleting the literal outright and reading the
#: contract alone was measured on dogfood-101 against a runtime carrying
#: omnimarket 0.4.178: the declaration landed in 0.4.181, none of that image's
#: 409 contracts carried the field, so both content-addressed exposures became
#: unresolved, their intended idempotence graded as loss, and
#: ``projection_delta_dropped`` went DEGRADED on healthy behaviour. On a lane
#: whose container probe runs with ``--degraded-policy fail`` that is not a
#: false alarm, it is an outage.
#:
#: The two alternatives were refused for stated reasons: refusing to grade
#: below a version floor is a gate blinding itself, and exempting every
#: unresolved exposure exempts ALL of them on a pre-0.4.181 runtime, so the
#: dimension would grade nothing for the whole window.
#:
#: Both identity forms are listed because the projection identity recorded at
#: dispatch is the handler class name while the contract and every ticket name
#: the node. Matching is EXACT: a substring rule would exempt the next handler
#: whose name happens to contain one of these.
#:
#: DELETION IS TRIGGERED BY A READING, not by a date: the deployed dev-lane
#: omnimarket at or above 0.4.181, read from inside the runtime container
#: rather than from a pin. Tracked in the follow-up on OMN-19081.
FALLBACK_IMMUTABLE_GRAIN_PROJECTIONS: tuple[str, ...] = (
    "HandlerProjectionSessionReplay",
    "HandlerProjectionWorkEvents",
    "node_projection_session_replay",
    "node_projection_work_events",
)

#: Cap on names rendered into a dimension detail. The detail rides every
#: ``/health`` response and every health event; a fleet-wide breakage must not
#: turn it into a log dump. Mirrors ``projection_liveness.MAX_NAMED_PROJECTIONS``.
MAX_NAMED_PROJECTIONS: int = 8

#: The outcome token for "registered projections, no closed window". Rendered
#: verbatim into both details so an operator and a grep can tell an unobserved
#: dimension from a measured-clean one. A dimension that renders the same prose
#: for both is how a gate that never ran reads as a gate that passed.
OUTCOME_APPLY_FLOW_UNOBSERVED: str = "apply_flow_window_unobserved"

#: The outcome token for a cumulative gauge that decreased.
OUTCOME_DROP_GAUGE_NONMONOTONIC: str = "apply_flow_drop_gauge_nonmonotonic"

#: The outcome token for a registered projection whose contract-declared key
#: grain could not be resolved (OMN-19081). Such a projection is never
#: exempted on the strength of the missing declaration itself -- that would
#: hide a real accumulation behind a missing field. It is graded unless the
#: interim FALLBACK_IMMUTABLE_GRAIN_PROJECTIONS list covers it, and it is
#: named under this token either way so a reader can tell it from
#: one that genuinely declares a mutable grain. The remedies differ: one is a
#: contract edit, the other is a real investigation.
OUTCOME_KEY_GRAIN_UNRESOLVED: str = "apply_flow_key_grain_unresolved"

#: The runtime health-dimension vocabulary, declared here so both statuses are
#: produced IN it rather than as free strings a call site has to narrow. Mirrors
#: ``projection_liveness.EnumDlqSaturationStatus`` and for the same reason: a
#: narrowing step maps an unrecognised value to DEGRADED, which is fail-closed
#: but silently regrades a typo, while a Literal makes it a type error where it
#: is written.
EnumProjectionApplyFlowStatus = Literal["HEALTHY", "DEGRADED", "CRITICAL"]


def evaluate_projection_apply_flow(
    *,
    windows: Sequence[tuple[ModelProjectionApplyDelta, ...]],
    registered_projections: Iterable[str],
    immutable_grain_projections: Iterable[str] = (),
    grain_unresolved_projections: Iterable[str] = (),
    fallback_immutable_projections: Iterable[str] = (),
) -> ModelProjectionApplyFlowVerdict:
    """Grade consume-versus-write and drop accumulation over the closed windows.

    Args:
        windows: Closed apply windows, OLDEST FIRST. Order is load-bearing:
            the drop dimension reads monotonicity across it, so a caller
            handing them newest-first would invert every rise into a fall and
            read the whole lane indeterminate.
        registered_projections: Every projection this process wired a dispatch
            for. Injected rather than derived from the windows so that a
            projection which registered and then took nothing is still in
            scope — deriving scope from observed traffic is how a consumer that
            stopped entirely disappears from its own health dimension.
        immutable_grain_projections: Projections whose declared key grain is
            immutable and content-addressed, for which a discarded delta is
            intended idempotence. Excluded from the drop dimension only; they
            are still graded for divergence, because an immutable grain says
            nothing about whether rows land.
        grain_unresolved_projections: Projections registered here whose
            contract-declared grain could not be resolved. Graded exactly like
            a mutable grain and reported under their own outcome token, so
            neither an exemption nor an alarm is taken silently on a fact
            nobody established.

    Returns:
        The verdict. It carries no status word; see the two ``*_status``
        functions below.
    """
    registered = tuple(sorted({p for p in registered_projections if p}))
    declared_immutable = frozenset(p for p in immutable_grain_projections if p)
    unresolved = frozenset(p for p in grain_unresolved_projections if p)
    fallback = frozenset(p for p in fallback_immutable_projections if p)
    # RESOLUTION ORDER, and the order is the whole design. The contract wins
    # wherever it answers -- a projection the contract declares MUTABLE is
    # graded even if it appears in the fallback, so the literal can never
    # override a live declaration. The fallback is reached only where the
    # contract resolved nothing, which is the deployed-version window.
    fallback_exempt = frozenset(p for p in unresolved if p in fallback)
    # .union() rather than the | operator: the repository non-optional-union
    # ratchet counts this line as a TYPE union and trips on it, which is a
    # miscount rather than a finding. Written the explicit way instead of
    # raising the ratchet, because raising a gate to fit a false positive
    # is how the gate stops meaning anything.
    exempt = declared_immutable.union(fallback_exempt)

    consumed_by: dict[str, int] = dict.fromkeys(registered, 0)
    upserted_by: dict[str, int] = dict.fromkeys(registered, 0)
    refused_by: dict[str, int] = dict.fromkeys(registered, 0)
    # Ordered gauge readings per projection, one per window it appeared in.
    gauge_series: dict[str, list[int]] = {p: [] for p in registered}

    for window in windows:
        for delta in window:
            name = delta.projection
            consumed_by[name] = consumed_by.get(name, 0) + delta.consumed
            upserted_by[name] = upserted_by.get(name, 0) + delta.upserted
            refused_by[name] = refused_by.get(name, 0) + delta.refused_by_guard
            gauge_series.setdefault(name, []).append(delta.deltas_dropped_total)

    in_scope = tuple(sorted(set(registered) | set(consumed_by)))
    evaluated = bool(windows) and bool(in_scope)

    diverging: list[str] = []
    accumulating: list[str] = []
    indeterminate: list[str] = []

    if evaluated:
        for name in in_scope:
            if (
                consumed_by.get(name, 0) >= APPLY_DIVERGENCE_MIN_CONSUMED
                and upserted_by.get(name, 0) == 0
            ):
                diverging.append(name)

            if name in exempt:
                continue
            readings = gauge_series.get(name, [])
            rises = 0
            total_rise = 0
            fell = False
            for earlier, later in pairwise(readings):
                if later > earlier:
                    rises += 1
                    total_rise += later - earlier
                elif later < earlier:
                    # A cumulative gauge cannot decrease within one continuous
                    # process. Either the writer restarted mid-series or this
                    # is not the reading it claims to be; both are unanswered
                    # questions, and grading an unanswered question flat would
                    # publish "nothing is wrong" over "the gate could not tell".
                    fell = True
            if fell:
                indeterminate.append(name)
            elif (
                rises >= DELTA_DROP_MIN_RISING_WINDOWS
                and total_rise >= DELTA_DROP_MIN_ACCUMULATION
            ):
                accumulating.append(name)

    return ModelProjectionApplyFlowVerdict(
        projection_count=len(in_scope),
        observed_window_count=len(windows),
        apply_flow_evaluated=evaluated,
        in_scope_registered=bool(in_scope),
        diverging_projections=tuple(diverging),
        drop_accumulating_projections=tuple(accumulating),
        indeterminate_drop_projections=tuple(indeterminate),
        excluded_immutable_grain=tuple(sorted(exempt & set(in_scope))),
        grain_unresolved_projections=tuple(sorted(unresolved & set(in_scope))),
        fallback_exempted_projections=tuple(sorted(fallback_exempt & set(in_scope))),
        total_consumed=sum(consumed_by.values()),
        total_upserted=sum(upserted_by.values()),
        total_refused_by_guard=sum(refused_by.values()),
    )


def _name_list(names: tuple[str, ...]) -> str:
    """Render a capped, comma-joined name list with an explicit remainder."""
    listed = ", ".join(names[:MAX_NAMED_PROJECTIONS])
    remaining = len(names) - MAX_NAMED_PROJECTIONS
    if remaining > 0:
        listed = f"{listed}, +{remaining} more"
    return listed


def _unobserved(verdict: ModelProjectionApplyFlowVerdict) -> bool:
    """True when projections are registered here but no window has closed."""
    return verdict.in_scope_registered and not verdict.apply_flow_evaluated


def projection_apply_divergence_status(
    verdict: ModelProjectionApplyFlowVerdict,
) -> EnumProjectionApplyFlowStatus:
    """The ``projection_apply_divergence`` dimension status for a verdict.

    Sole source of the dimension's grade, so the prose beside it cannot
    disagree with it — the mistake ``dlq_saturation_status`` was written to
    stop being repeated here.

    DEGRADED when EITHER holds:

    * a projection consumed past the floor and wrote nothing; or
    * projections are registered in this process and no window has closed, so
      the dimension has no measurement. That is UNKNOWN, and publishing HEALTHY
      over an UNKNOWN converts "the gate could not tell" into "nothing is
      wrong", which is the false all-clear this epic exists to remove.

    HEALTHY when this process registered no projection dispatch at all: that is
    a measured zero rather than an unanswered question, and it is not this
    dimension's job to assert that a projection ought to be running here.
    """
    if verdict.diverging_projections or _unobserved(verdict):
        return "DEGRADED"
    return "HEALTHY"


def describe_projection_apply_divergence(
    verdict: ModelProjectionApplyFlowVerdict,
) -> str:
    """Build the ``projection_apply_divergence`` dimension detail."""
    if not verdict.in_scope_registered:
        return (
            "no projection dispatch registered in this process — apply "
            "divergence not applicable here (attachment is graded by "
            "projection_attachment)"
        )
    if not verdict.apply_flow_evaluated:
        return (
            f"{OUTCOME_APPLY_FLOW_UNOBSERVED}: {verdict.projection_count} "
            "projection(s) registered and no closed apply window observed yet "
            "— apply divergence UNKNOWN (not asserted healthy)"
        )
    if not verdict.diverging_projections:
        return (
            f"All {verdict.projection_count} dispatching projection(s) that "
            f"consumed also wrote, over {verdict.observed_window_count} apply "
            f"window(s) ({verdict.total_consumed} consumed, "
            f"{verdict.total_upserted} row(s) upserted, "
            f"{verdict.total_refused_by_guard} refused by an ordering guard)"
        )
    return (
        f"{len(verdict.diverging_projections)}/{verdict.projection_count} "
        f"projection(s) consumed at least {APPLY_DIVERGENCE_MIN_CONSUMED} "
        f"event(s) and upserted ZERO rows over "
        f"{verdict.observed_window_count} apply window(s) — offsets commit on "
        f"a refusal and on an early return, so lag reads 0 over a total loss: "
        f"{_name_list(verdict.diverging_projections)}"
    )


def projection_delta_dropped_status(
    verdict: ModelProjectionApplyFlowVerdict,
) -> EnumProjectionApplyFlowStatus:
    """The ``projection_delta_dropped`` dimension status for a verdict.

    DEGRADED on an ACCUMULATING discarded-delta gauge, on a gauge that went
    backwards, or on an unobserved window. Never on magnitude: a high flat
    gauge is a measured steady state, and alarming on it is what gets a
    dimension muted — 6,210,195 lifetime drops sat on the consumer-flow
    exposure while it served fresh rows.
    """
    if (
        verdict.drop_accumulating_projections
        or verdict.indeterminate_drop_projections
        or _unobserved(verdict)
    ):
        return "DEGRADED"
    return "HEALTHY"


def describe_projection_delta_dropped(verdict: ModelProjectionApplyFlowVerdict) -> str:
    """Build the ``projection_delta_dropped`` dimension detail."""
    if not verdict.in_scope_registered:
        return (
            "no projection dispatch registered in this process — delta drops "
            "not applicable here"
        )
    if not verdict.apply_flow_evaluated:
        return (
            f"{OUTCOME_APPLY_FLOW_UNOBSERVED}: {verdict.projection_count} "
            "projection(s) registered and no closed apply window observed yet "
            "— delta accumulation UNKNOWN (not asserted healthy)"
        )
    # Rendered on BOTH the clean and the degraded case. A reader who sees only
    # "no projection is accumulating drops" cannot tell a measured zero from a
    # set of exposures the dimension was told to skip, and those are different
    # facts. Same reason describe_dlq_saturation carries its exclusions.
    exempted = ""
    if verdict.excluded_immutable_grain:
        exempted = (
            f" [{len(verdict.excluded_immutable_grain)} exposure(s) declare an "
            "immutable, content-addressed key grain, for which a discarded "
            "delta is intended idempotence, and are excluded from this "
            f"dimension: {_name_list(verdict.excluded_immutable_grain)}]"
        )
    unresolved = ""
    if verdict.grain_unresolved_projections:
        unresolved = (
            f" ({OUTCOME_KEY_GRAIN_UNRESOLVED}: "
            f"{len(verdict.grain_unresolved_projections)} projection(s) "
            "declare no resolvable key grain. Each is graded unless the "
            "interim fallback below covers it, and none is exempted on the "
            "strength of the missing declaration alone: "
            f"{_name_list(verdict.grain_unresolved_projections)})"
        )
    fallback_note = ""
    if verdict.fallback_exempted_projections:
        fallback_note = (
            f" [{len(verdict.fallback_exempted_projections)} of those "
            "exemptions came from the INTERIM fallback list rather than from "
            "a contract declaration, because this runtime carries an "
            "omnimarket predating key_grain: "
            f"{_name_list(verdict.fallback_exempted_projections)}]"
        )
    indeterminate = ""
    if verdict.indeterminate_drop_projections:
        indeterminate = (
            f" ({OUTCOME_DROP_GAUGE_NONMONOTONIC}: "
            f"{len(verdict.indeterminate_drop_projections)} projection(s) "
            "reported a cumulative drop gauge that DECREASED, so the series "
            "did not come from one continuous process and cannot be graded "
            f"flat: {_name_list(verdict.indeterminate_drop_projections)})"
        )
    if not verdict.drop_accumulating_projections:
        return (
            f"No projection's discarded-delta count rose across "
            f"{verdict.observed_window_count} apply window(s)"
            f"{indeterminate}{unresolved}{exempted}{fallback_note}"
        )
    return (
        f"{len(verdict.drop_accumulating_projections)} projection(s) discarded "
        f"deltas at a RISING rate over {verdict.observed_window_count} apply "
        f"window(s) — the serving cache consumed them and kept nothing, which "
        f"reads as fresh at zero lag: "
        f"{_name_list(verdict.drop_accumulating_projections)}"
        f"{indeterminate}{unresolved}{exempted}{fallback_note}"
    )


__all__: list[str] = [
    "APPLY_DIVERGENCE_MIN_CONSUMED",
    "FALLBACK_IMMUTABLE_GRAIN_PROJECTIONS",
    "DELTA_DROP_MIN_ACCUMULATION",
    "DELTA_DROP_MIN_RISING_WINDOWS",
    "MAX_NAMED_PROJECTIONS",
    "OUTCOME_APPLY_FLOW_UNOBSERVED",
    "OUTCOME_DROP_GAUGE_NONMONOTONIC",
    "OUTCOME_KEY_GRAIN_UNRESOLVED",
    "describe_projection_apply_divergence",
    "describe_projection_delta_dropped",
    "evaluate_projection_apply_flow",
    "projection_apply_divergence_status",
    "projection_delta_dropped_status",
]
