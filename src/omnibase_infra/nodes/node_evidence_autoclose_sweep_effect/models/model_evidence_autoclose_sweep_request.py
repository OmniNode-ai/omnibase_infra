# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Request model for the evidence autoclose sweep."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.enum_evidence_autoclose_trigger import (
    EnumEvidenceAutocloseTrigger,
)


class ModelEvidenceAutocloseSweepRequest(BaseModel):
    """Request to sweep recently-merged OCC companions for governed Done flips."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Sweep run correlation ID.")
    occ_repo: str = Field(
        default="OmniNode-ai/onex_change_control",
        description="owner/repo of the evidence-companion repository to scan.",
    )
    lookback_hours: int = Field(
        default=24,
        ge=1,
        le=24 * 30,
        description="Scan companions merged within this many hours of now.",
    )
    # ------------------------------------------------------------------
    # OMN-17342 — the BACKFILL arm.
    #
    # `lookback_hours` above is a FRESHNESS window: it sees each merged
    # companion once, in the ~6h after it merges, and then never again. There is
    # no cursor and no watermark, so a companion that is scanned twelve times
    # and reaches no verdict in any of them ages out permanently. Measured
    # 2026-09-05 across the two beta sprint projects: 118 of 238 open tickets
    # carry a merged OCC companion, 5 of them inside the live window and 113
    # outside it for good — 74 of those carrying the behaviour-proof receipt
    # that is the hardest flip conjunct to satisfy. Those are not withheld
    # flips. A withheld flip is a decision; this is the absence of one.
    #
    # Widening `lookback_hours` is NOT the fix and these fields are not a
    # disguised way to do it. `dod_verify` dominates the run budget (~15s per
    # ticket under sweep concurrency, ~34s standalone — OMN-16961's
    # measurement) against a 30-minute cadence, so a window wide enough to
    # reach the backlog would overrun the cadence, and would overrun it worse
    # every time the board grew. The arm therefore takes a bounded ROTATING
    # SLICE of the wider window: bounded per tick so the run fits the cadence,
    # rotating so the tail still drains.
    #
    # It is OFF unless asked for. `backfill_lookback_hours=0` means the run has
    # exactly the single-arm behaviour it had before this field existed —
    # same candidates, same I/O, same counters.
    #
    # ARMED, 2026-09-05 (OMN-17658 / OMN-17950), in the same commit as the
    # fences it was deliberately sequenced behind. The default was 0 — off
    # unless a dispatcher asked — precisely because a wide arm without the
    # recurring-companion refusal (OMN-17934) and the children conjunct
    # (OMN-17658) would have reproduced the OMN-17292 re-flip across the whole
    # backlog instead of once. The live enumeration proved that in one line:
    # OCC#8193, which binds OMN-17292, is in the backfill pool. Both fences are
    # now in this same binary, so the arm is armed HERE, in the contract's own
    # default, and not by an expression in a workflow — same authority split as
    # `scheduled_apply`. 168h = 7 days: measured 2026-09-05 (dispatch run
    # 33944132063) to yield a pool of 168 companions against a per-tick slice
    # of 5, i.e. ~34 ticks (~17h) to sweep the pool once.
    backfill_lookback_hours: int = Field(
        default=168,
        ge=0,
        le=24 * 90,
        description=(
            "Second, wider enumeration window in hours. 0 (the default) "
            "disables the backfill arm entirely. When > 0, companions merged "
            "between this bound and `lookback_hours` form a pool from which a "
            "bounded rotating slice is offered to the SAME downstream pipeline "
            "-- binding, Linear state, dod_verify, the OMN-16736 AC-coverage "
            "guard, the OMN-15911 behaviour conjunct, flip or gap. Nothing "
            "about what counts as proven changes; only which candidates are "
            "asked."
        ),
    )
    backfill_max_candidates: int = Field(
        default=5,
        ge=1,
        le=100,
        description=(
            "Hard per-run bound on how many backfill candidates reach "
            "dod_verify. This is the run-budget guard and the thing that must "
            "not track the size of the board: at ~15s per verifier call, 5 is "
            "~75s of added work against a 30-minute cadence. Raising it trades "
            "drain rate for run duration and nothing else -- it can never "
            "widen what the flip predicate accepts."
        ),
    )
    backfill_pool_size: int = Field(
        default=200,
        ge=1,
        le=1000,
        description=(
            "Cap on how many merged companions the wider window enumerates "
            "before slicing. Bounds the `gh api` PR-list pagination, which is "
            "cheap per page but not free; the per-PR file listing is bounded "
            "separately and much more tightly, because the Linear state "
            "short-circuit now discards completed tickets before it runs."
        ),
    )
    backfill_rotation_minutes: int = Field(
        default=30,
        ge=1,
        le=1440,
        description=(
            "Cadence, in minutes, that the rotating slice advances on. The "
            "slice index is derived from the run's own wall clock divided by "
            "this period, so two runs in the same period examine the same "
            "slice (a retry re-does its work rather than skipping a slice) and "
            "consecutive scheduled runs advance by exactly one. Set it to the "
            "workflow's real cron interval; the default matches the sweep's "
            "*/30 schedule."
        ),
    )

    # ------------------------------------------------------------------
    # OMN-17658 — the arming authority, and where it lives.
    #
    # Until this field existed, whether an UNATTENDED run wrote to Linear was
    # decided by one disjunct inside a GitHub Actions expression:
    # `github.event_name == 'schedule' || (...)`. That is an arming authority
    # for every write nobody is watching, and it was invisible to this
    # contract, untyped, and changeable by anyone who could edit a YAML file
    # without touching the node the contract governs.
    #
    # `trigger` carries the FACT (what launched me) and `scheduled_apply`
    # carries the POLICY (may a run of that class write). Splitting them is
    # what lets the workflow stop deciding: it now reports its own event and
    # passes no arming value at all, and the contract below is the single place
    # a scheduled write is authorised from.
    #
    # `apply` (kept, below) is unchanged and remains the DISPATCH-time request.
    # The effective write mode is `apply or (trigger is SCHEDULE and
    # scheduled_apply)`, which keeps a dispatch that leaves the box unticked a
    # dry run even while the schedule is armed — that is the rehearsal surface,
    # and it must survive arming.
    trigger: EnumEvidenceAutocloseTrigger = Field(
        default=EnumEvidenceAutocloseTrigger.DISPATCH,
        description=(
            "What launched this run. DISPATCH is the default because a caller "
            "that names nothing is not the schedule — an un-named "
            "construction must never pick up the unattended arming authority "
            "by omission."
        ),
    )
    scheduled_apply: bool = Field(
        default=True,
        description=(
            "THE arming authority for unattended runs (OMN-17658). When True, "
            "a run whose `trigger` is SCHEDULE writes to Linear without any "
            "operator input; when False, the same run reaches every decision "
            "and writes none. It does not affect a dispatch, which is governed "
            "by `apply`. Declared here rather than in the workflow so that "
            "disarming the closer is a contract change with a diff, a review "
            "and a test, instead of an edit to an expression string."
        ),
    )
    # ------------------------------------------------------------------
    # OMN-17658 — the per-run blast-radius bound.
    max_flips_per_run: int = Field(
        default=5,
        ge=0,
        le=100,
        description=(
            "Hard cap on how many tickets one run may move to Done. Candidates "
            "that clear every conjunct after the budget is spent are recorded "
            "SKIPPED_FLIP_BUDGET_EXHAUSTED and offered again next run — a "
            "truncation, never a verdict. 0 means no flip may be written at "
            "all, which is a fail-closed value rather than 'unbounded'. This "
            "is what makes a defect in the predicate cost 5 wrong Dones per "
            "tick instead of the whole board: the first applying scheduled run "
            "(33932169358) flipped four tickets and one of them was wrong."
        ),
    )
    # ------------------------------------------------------------------
    # OMN-17658 — the persisted half of the auto-disarm.
    disarmed_by_ticket: str = Field(
        default="",
        description=(
            "A ticket id the caller asserts previously received an UNSAFE "
            "closer flip. Non-empty disarms the whole run before its first "
            "candidate: every decision is still reached and reported, and none "
            "is written. Plumbed by the workflow from the repo variable "
            "ONEX_AUTOCLOSE_DISARMED, the same reachable-from-every-event "
            "shape the ONEX_AUTOCLOSE_DISABLED kill switch uses (OMN-16792), "
            "so a disarm survives the run that discovered the problem and "
            "binds the scheduled runs nobody is watching.\n\n"
            "It is weaker than the kill switch and deliberately so: a halted "
            "run does zero I/O and produces no receipt, whereas a disarmed run "
            "still enumerates, still verifies and still reports what it WOULD "
            "have done — which is the evidence an operator needs to decide "
            "whether to re-arm."
        ),
    )
    # ------------------------------------------------------------------
    # OMN-17658 / OMN-17934 — the Linear state-history read that both the
    # prior-revert fence and the bound flip readback are resolved from.
    history_page_size: int = Field(
        default=100,
        ge=1,
        le=250,
        description=(
            "Page size for the per-ticket `stateHistory` walk. Mirrors "
            "node_sync_revert_watchdog_effect, which reads the same connection "
            "for the same reason."
        ),
    )
    history_max_pages: int = Field(
        default=3,
        ge=1,
        le=20,
        description=(
            "Page cap for the per-ticket history walk. The walk is "
            "newest-first, so hitting the cap truncates the OLDEST end — the "
            "entries the prior-revert fence and the flip readback depend on "
            "are always present. A reopen older than the walk resolves to 'not "
            "seen', which is the only direction this read is allowed to be "
            "wrong in and is why the cap is small."
        ),
    )

    # ------------------------------------------------------------------
    # OMN-17658 follow-up — the readback reads a connection that LAGS.
    #
    # Measured on the first scheduled run under the fences (33958237006,
    # f8b623672, 2026-09-05T09:33Z): the sweep flipped OMN-17658, `issueUpdate`
    # returned success, the ticket's own history shows `In Progress -> Done` at
    # 09:34:43.990Z — and the immediate post-write read of that same connection
    # showed nothing, so the run recorded ERROR_READBACK_UNCONFIRMED on a write
    # that had landed.
    #
    # A single immediate read of an eventually consistent connection is not a
    # proof of absence, it is a race, and this one loses every time: without a
    # retry `tickets_flipped` can never leave 0 and the closer is silently
    # reduced to a mechanism that writes Done and reports that it did not.
    readback_max_attempts: int = Field(
        default=4,
        ge=1,
        le=20,
        description=(
            "How many times the post-write state-history read may be retried "
            "before the flip is recorded ERROR_READBACK_UNCONFIRMED. The first "
            "attempt is immediate and costs nothing extra on a connection that "
            "is already consistent; only a genuine lag pays the delay."
        ),
    )
    readback_delay_seconds: int = Field(
        default=3,
        ge=0,
        le=60,
        description=(
            "Delay between post-write readback attempts. 0 is a real value and "
            "is what tests use — the retry must be exercisable without waiting "
            "out a production backoff."
        ),
    )

    apply: bool = Field(
        default=False,
        description=(
            "DRY-RUN is the default (apply=False): the sweep logs every "
            "decision it WOULD make but never calls the Linear mutation "
            "API. Pass apply=True to actually flip Done / post comments."
        ),
    )
    max_companions: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Safety cap on the number of merged companions scanned per run.",
    )
    dispatch_cwd: str = Field(
        default="",
        description=(
            "Working directory to run `onex skill dod_verify <ticket>` from. "
            "Empty string means: inherit the sweep process's own cwd. "
            "OMN-16846: this no longer selects the verifier's ENVIRONMENT. "
            "The verifier is dispatched from the sweep interpreter's own "
            "`onex` (see `_dod_verify_argv`), so the venv carrying "
            "node_dod_verify is decided by how the sweep was composed rather "
            "than by where it stands -- which is what lets the product clone "
            "the behaviour checks run pytest in stay lock-exact. This field "
            "now only sets the cwd the verifier process inherits."
        ),
    )
    dod_verify_timeout_seconds: int = Field(
        default=300,
        ge=1,
        le=1800,
        description="Timeout for each `onex skill dod_verify` subprocess call.",
    )
    gh_timeout_seconds: int = Field(
        default=90,
        ge=1,
        le=1800,
        description=(
            "Timeout for each `gh api` subprocess call (PR-list enumeration "
            "and per-PR file listing). Raised from a prior hardcoded 30.0s "
            "default that timed out in CI (OMN-16106: duration_ms==30048 on "
            "a live self-hosted-runner GitHub enumeration)."
        ),
    )
    exclude_tickets: tuple[str, ...] = Field(
        default=(),
        description=(
            "Ticket ids this run must refuse before it reads anything about "
            "them (OMN-17891). Matched case-insensitively after stripping "
            "surrounding whitespace; each match is recorded as "
            "SKIPPED_EXCLUDED and costs zero Linear I/O.\n\n"
            "This is a CALLER ASSERTION, never a derived fact. The node reads "
            "no ledger, no assignee, and no ownership signal — before this "
            "field existed an apply run's only refusals were a Linear label "
            "set beforehand, an already-completed state, a binding-hygiene "
            "skip, and the global ONEX_AUTOCLOSE_DISABLED kill switch, none "
            "of which can decline ONE candidate. Whoever dispatches the run "
            "supplies the list and owns its accuracy; the enum value is "
            "distinct from SKIPPED_LABEL precisely so the audit record says "
            "which authority refused.\n\n"
            "It does not weaken the kill switch: a halted run does zero work "
            "regardless of what this names, so an exclusion list can never "
            "opt one ticket back into a halted sweep."
        ),
    )
    close_if_done_label: str = Field(
        default="close-if-done",
        description=(
            "Linear label name that routes a ticket to the manual "
            "decision-only close path instead of this sweep."
        ),
    )


__all__ = ["ModelEvidenceAutocloseSweepRequest"]
