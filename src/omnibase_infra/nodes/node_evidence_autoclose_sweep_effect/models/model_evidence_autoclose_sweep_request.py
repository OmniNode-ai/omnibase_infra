# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Request model for the evidence autoclose sweep."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


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
    backfill_lookback_hours: int = Field(
        default=0,
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
