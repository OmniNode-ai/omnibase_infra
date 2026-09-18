# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Per-ticket outcome record for the evidence autoclose sweep."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.enum_evidence_autoclose_arm import (
    EnumEvidenceAutocloseArm,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.enum_evidence_autoclose_decision import (
    EnumEvidenceAutocloseDecision,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.model_ac_binding_row import (
    ModelAcBindingRow,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.model_check_result_row import (
    ModelCheckResultRow,
)


class ModelEvidenceAutocloseOutcome(BaseModel):
    """One companion-PR / bound-ticket pair's terminal sweep decision."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    ticket_id: str = Field(
        default="", description="Bound Linear ticket id, e.g. OMN-16106."
    )
    companion_pr_number: int = Field(
        default=0, description="Merged OCC companion PR number."
    )
    companion_pr_url: str = Field(
        default="", description="Merged OCC companion PR URL."
    )
    decision: EnumEvidenceAutocloseDecision = Field(
        ..., description="Terminal classification for this pair."
    )
    reason: str = Field(default="", description="Human-readable explanation.")
    # OMN-17342. Which enumeration arm offered this pair. Defaults to FORWARD so
    # every pre-existing construction site keeps its meaning unchanged — the
    # forward window was the only arm that existed. It is recorded on EVERY
    # outcome rather than only on backfilled ones because the absence of a
    # marker is not a readable signal: "no arm field" and "the forward arm" have
    # to be distinguishable in a receipt read months later, and a default that
    # is also a real value only works if it is written out.
    enumeration_arm: EnumEvidenceAutocloseArm = Field(
        default=EnumEvidenceAutocloseArm.FORWARD,
        description="Enumeration arm that selected this (companion, ticket) pair.",
    )
    dod_verify_total_checks: int = Field(default=0, ge=0)
    dod_verify_verified_count: int = Field(default=0, ge=0)
    dod_verify_failed_count: int = Field(default=0, ge=0)
    # OMN-15911: how many of the passing checks actually executed the claimed
    # behavior, as reported by dod_verify's own verdict. Recorded on every
    # outcome that reached a verdict — including the flip — so the sweep
    # result says on what STRENGTH of evidence a ticket was closed, not merely
    # that a count matched.
    dod_verify_behavior_proving_count: int = Field(default=0, ge=0)
    # OMN-16821: how many checks executed, exited 0, and could not have exited
    # otherwise for a product reason (OMN-15391). Recorded because the flip
    # equality is `verified + non_probative == total`, so without this field a
    # flip reads as an unexplained "6/12 verified" in the structured record and
    # is auditable only from the free-text reason — the counts-without-detail
    # problem OMN-16788 already hit once.
    dod_verify_non_probative_count: int = Field(default=0, ge=0)
    # OMN-18490. THE CHECKS BEHIND THE COUNTERS.
    #
    # The five fields above are a tally. Run 35299192253 recorded OMN-18426 as
    # `30/94 ACs verified, 3 failed` and named none of the three, on the
    # outcome, in the comment it posted to the ticket, and in the job log. A
    # collaborator asked twice which three and the answer was not recoverable
    # from any surface the run left behind — so a failure nobody could name was
    # a failure nobody could fix, for two days, on a mechanism whose whole job
    # is to say what is unproven.
    #
    # One row per check the verdict carried. Nothing is newly collected: the
    # per-check records are already on the dod_verify terminal payload and this
    # node already walks them for two classifiers and the gap fingerprint. What
    # is new is that the walk is written down.
    #
    # Empty on every outcome that reached no verdict — an excluded candidate, a
    # refusal taken ahead of the verifier, an unparseable receipt — because
    # there is nothing to report, and empty on a verdict whose `checks` payload
    # could not be read, which is the same silence every other consumer of that
    # list already answers with. The two are told apart by `reason`.
    check_results: tuple[ModelCheckResultRow, ...] = Field(
        default=(),
        description=(
            "One row per check on this ticket's dod_verify verdict: id, "
            "status, proof class, declared bindings and a bounded message "
            "excerpt. Descriptive only — every counter the flip predicate "
            "reads still comes from the verdict's own count fields."
        ),
    )
    uncovered_acceptance_criteria: tuple[str, ...] = Field(
        default=(),
        description=(
            "Acceptance criteria found in the ticket's Linear description that "
            "dod_verify's checks do not cover (GAP_AC_COVERAGE only). Recorded "
            "on the outcome as well as in the comment so a DRY-RUN, which posts "
            "no comment, still names exactly what blocked the flip."
        ),
    )
    # OMN-18056. THE AC-BINDING TABLE. One row per (acceptance criterion,
    # declaring check), plus exactly one row for each criterion no check
    # declares. Recorded on the outcome as well as rendered into the comment
    # for the same reason `uncovered_acceptance_criteria` is: a DRY-RUN posts
    # nothing, and a preview that cannot say WHICH criterion is unbound is not
    # a preview of the decision — it is a restatement of the counters, which
    # is the thing this ticket exists to stop reading as proof.
    #
    # Empty on every path that reached no verdict, and on a FLIP it carries
    # the bindings that RELEASED it: a closed ticket then states which check
    # discharged which criterion instead of stating an arithmetic identity.
    ac_binding_rows: tuple[ModelAcBindingRow, ...] = Field(
        default=(),
        description=(
            "Acceptance criterion -> declaring check rows for this verdict. "
            "A row with an empty `check_id` is a criterion nothing declares."
        ),
    )
    # ------------------------------------------------------------------
    # OMN-17658 — the BOUND READBACK. Three ids that make one flip checkable
    # by somebody who was not there: which companion carried the evidence
    # (above), which dod_verify verdict released it (`verdict_fingerprint`),
    # and which Linear state-history entry the write actually produced
    # (`readback_entry_id`, which must differ from `pre_write_head_entry_id`).
    #
    # The last pair is the point. `issueUpdate` returning `success: true` is
    # the API agreeing to the request, not evidence the board changed —
    # exactly the class of claim the deterministic-truth doctrine refuses. A
    # flip is recorded as FLIPPED only when a post-write read of the ticket's
    # own history shows a completed segment the pre-write read did not have.
    pre_write_head_entry_id: str = Field(
        default="",
        description=(
            "Newest `stateHistory` entry id observed BEFORE the flip was "
            "written. Empty on every non-applying path."
        ),
    )
    readback_entry_id: str = Field(
        default="",
        description=(
            "`stateHistory` entry id of the completed segment the flip "
            "produced, read back AFTER the write. Empty on every non-applying "
            "path, and empty on an applying path whose readback did not "
            "confirm — which is ERROR_READBACK_UNCONFIRMED, never FLIPPED."
        ),
    )
    # OMN-16106. THE ROLLBACK FLAG. Set when a flip was written, its bound
    # readback could not confirm it, and this run therefore RESTORED the
    # ticket's pre-write state.
    #
    # The label and the flip cannot coexist. Before this field existed, an
    # unconfirmed readback left the Done standing on the board and posted a
    # comment saying so -- "Treat this ticket's state as written but
    # unverified, and check it by hand". That is a closed ticket carrying its
    # own admission that nothing verified it, and the board reads Done to
    # every downstream sweep, rollup and human that never opens the comment.
    # OMN-16025 sat in exactly that shape from 2026-09-06T21:42:17Z until a
    # person reverted it by hand twelve minutes later.
    #
    # A write nobody can read back is not a proven write, and the doctrine's
    # own direction of conservatism is stated in this handler: a false hold
    # costs a comment and a human glance, a false flip writes an unearned Done
    # onto the board. Rolling back is the hold.
    flip_rolled_back: bool = Field(
        default=False,
        description=(
            "True when this run wrote a Done, could not read it back, and "
            "restored the ticket's pre-write state. The board ends the run in "
            "the state it started it in."
        ),
    )
    # OMN-16106. The gate probe this ticket named for itself, and what its
    # newest completed run concluded. Both empty on every path that did not
    # consult one -- a ticket with no `Gate:` line declares no probe.
    gate_probe_workflow: str = Field(
        default="",
        description=(
            "`<owner>/<repo> <workflow-file>` parsed from the ticket "
            "description's `Gate:` line, empty when the ticket names none."
        ),
    )
    gate_probe_conclusion: str = Field(
        default="",
        description=(
            "GitHub `conclusion` of the newest completed run of "
            "`gate_probe_workflow` (`success`, `failure`, ...), or empty when "
            "no probe was declared or none could be resolved."
        ),
    )
    # OMN-16106 D3. THE RUN-DISARM SIGNAL, and the only one.
    #
    # Set only on a FLIPPED outcome whose own post-write readback found the
    # ticket moved back OUT of a completed state again — i.e. this run wrote a
    # Done and, inside its own readback window, somebody undid it. That is the
    # closer having overruled a person and been overruled back while the run
    # was still going: the fences did NOT hold, so the rest of the run has no
    # standing to keep writing under the same predicate.
    #
    # It is deliberately NOT set by SKIPPED_PRIOR_REVERT. A prior-revert skip
    # is the fence WORKING — the closer declined to re-assert a verdict a
    # person had reversed — and a mechanism that refused correctly on one
    # ticket has lost no standing on any other. Conflating "the closer
    # overrode a human" with "the closer correctly refused to override a
    # human" is what disarmed the whole fleet from 2026-09-06T03:36Z onward:
    # the OMN-17556 refusal was recomputed from unchanged evidence on every
    # tick, so the disarm recurred forever and no other candidate was ever
    # adjudicated.
    flip_reverted_during_run: bool = Field(
        default=False,
        description=(
            "True when this run flipped the ticket Done and its own readback "
            "then observed a completed -> non-completed transition newer than "
            "the segment the flip produced. The one condition that disarms "
            "the remainder of the run."
        ),
    )
    verdict_fingerprint: str = Field(
        default="",
        description=(
            "Stable digest of the dod_verify counters that released or "
            "withheld this decision (total/verified/failed/non-probative/"
            "behaviour-proving). Lets two receipts be compared for 'same "
            "verdict' without re-parsing free text."
        ),
    )
    # OMN-18106. Non-empty ONLY when the positive prior-revert fence stopped
    # applying because this ticket's evidence landed after the reversal. It
    # names the reversal's timestamp and each piece of evidence that postdates
    # it, so a close taken over a prior human disagreement carries, in the
    # receipt itself, the ordering that authorised it. Empty is the ordinary
    # case — either no reversal, or a fence that held.
    post_revert_evidence_release: str = Field(
        default="",
        description=(
            "Why the prior-revert fence did not apply: the reversal timestamp "
            "and the evidence landings that postdate it. Empty when the fence "
            "was not reached or held."
        ),
    )
    # OMN-18336. THE FALSE-POSITIVE RATE, MADE COUNTABLE.
    #
    # Four of forty-seven closer flips were reverted by hand — an 8.5% error
    # rate that was being ABSORBED rather than measured, because a revert left
    # no fact anywhere that a series could be built from. Prose in a hold
    # reason is not a series value; a boolean on the receipt is.
    #
    # True ONLY when this outcome was reached because a flip THIS CLOSER made
    # was moved back out of a completed state — established from the closer's
    # own flip comment being on the ticket, not from the reversal alone. A
    # reverted HAND flip sets it False: that is somebody else's judgement being
    # undone and counting it here would inflate this mechanism's error rate
    # with errors it did not make.
    closer_flip_reverted: bool = Field(
        default=False,
        description=(
            "True when this decision was reached because a flip this closer "
            "itself made was reverted. The countable form of the "
            "false-positive rate; False for a reverted hand flip."
        ),
    )
    # OMN-18336. WHAT THE CLOSER BELIEVED AT THE MOMENT OF THE WRONG FLIP.
    #
    # `label -> check id` for every criterion the flip counted as discharged.
    # All four reverted flips shared one failure class — a guard was proven and
    # its remediation was not — and establishing that took an archaeology
    # exercise across receipts. Recording the pairing makes the wrongly-judged
    # criterion a citable fact, so the next predicate change is argued from
    # cases instead of guesses.
    counted_ac_bindings: tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "`<label> -> <check id>` for each criterion the reverted flip "
            "counted as bound. Empty when no flip of this closer's was "
            "reverted, or when the flip counted no labelled criterion."
        ),
    )
    linear_comment_posted: bool = Field(
        default=False, description="Whether an audit/gap comment was posted."
    )
    applied: bool = Field(
        default=False,
        description="True only when a real Linear mutation was made (apply=True run).",
    )


__all__ = ["ModelEvidenceAutocloseOutcome"]
