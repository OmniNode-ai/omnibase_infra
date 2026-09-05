# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Per-ticket outcome classification for the evidence autoclose sweep."""

from __future__ import annotations

from enum import StrEnum


class EnumEvidenceAutocloseDecision(StrEnum):
    """Terminal decision recorded for one (companion PR, bound ticket) pair.

    Exactly one FLIPPED/GAP_POSTED/SKIPPED_*/ERROR_* value is recorded per
    ticket the sweep considered. FLIPPED is the only value that ever mutates
    a ticket's Linear state; every other value is fail-closed (comment-only
    or no-op).
    """

    # All ACs receipt-proven (dod_verify: zero failed) -> Done flip + audit comment.
    FLIPPED = "flipped"
    # dod_verify ran cleanly (exit 0) but reported >=1 failed/unverified check
    # -> honest gap posted as a comment, never flipped.
    GAP_POSTED = "gap_posted"
    # dod_verify was fully green, but the ticket's Linear DESCRIPTION carries
    # acceptance criteria dod_verify structurally cannot see -- unchecked
    # markdown checkboxes, or an acceptance-criteria section listing more items
    # than dod_verify had checks. The flip is withheld and the uncovered
    # criteria are named in the comment (OMN-16736; the OMN-14362 lesson: an AC
    # that lives only in the ticket body is invisible to a contract verifier, so
    # a clean 0-failed run is not evidence about it). Counted as a gap, not an
    # error: the mechanism worked, the evidence base was incomplete.
    GAP_AC_COVERAGE = "gap_ac_coverage"
    # dod_verify was clean AND its checks covered the ticket's criteria, but
    # not one passing check executed the claimed behavior -- every green leg
    # was a merge-state read or a surrogate. The tally says the code landed;
    # nothing says the system does the thing, so the flip is withheld
    # (OMN-15911).
    GAP_NO_BEHAVIOR_PROOF = "gap_no_behavior_proof"
    # The caller named this ticket in `exclude_tickets` on the request, so the
    # sweep refused it BEFORE its first Linear read -- no issue fetch, no
    # dod_verify subprocess, no verdict (OMN-17891). Ordering is the property:
    # an excluded candidate that had already been read could still surface as
    # ERROR_LINEAR_API on a transport failure, which reads as "the fence did
    # not apply" rather than "the fence applied".
    #
    # Deliberately NOT folded into SKIPPED_LABEL. A label is a fact the sweep
    # observed in Linear; this is an assertion the dispatcher made, derived
    # from nothing the node can see (an open ledger CLAIM, a red gate, a
    # concurrent controller's ownership). The audit record has to say which
    # authority refused, because only one of the two is falsifiable from the
    # ticket itself.
    SKIPPED_EXCLUDED = "skipped_excluded"
    # Ticket carries the close-if-done label -> decision-only path stays manual.
    SKIPPED_LABEL = "skipped_label"
    # Ticket is already in a completed/canceled state -> nothing to do.
    SKIPPED_ALREADY_DONE = "skipped_already_done"
    # No contracts/OMN-XXXXX.yaml file and no evidence(OMN-XXXXX) title match.
    SKIPPED_NO_BINDING = "skipped_no_binding"
    # More than one distinct ticket id bound to the same merged companion.
    SKIPPED_AMBIGUOUS_BINDING = "skipped_ambiguous_binding"
    # The sweep reached the same gap verdict it has ALREADY posted on this
    # ticket, and did not repeat itself (OMN-16808). Enumeration is a bare
    # `now - lookback_hours` window with no cursor, so one merged companion sits
    # inside several consecutive scheduled windows; without a read-before-write
    # check every one of them posts an identical comment. Counted as a skip, not
    # a gap: the gap is real and still open, but this run added no information.
    #
    # Keyed on (ticket, gap class, verdict fingerprint) and NOT on the companion
    # PR — the same verdict re-derived from a later companion is the same
    # statement. A CHANGED verdict has a different fingerprint and does comment.
    SKIPPED_DUPLICATE_COMMENT = "skipped_duplicate_comment"
    # OMN-17658. The ticket is a PARENT with at least one child that is not in
    # a completed/canceled state, read live from Linear on this tick. A parent
    # is not done while its own decomposition is open, whatever its contract's
    # checks say: dod_verify verifies the parent's OCC contract, which is
    # structurally incapable of seeing a child ticket that carries its own
    # separate acceptance criteria. Measured 2026-09-05: 30 of 238 open beta
    # tickets are parents with open children and 4 of those carry a
    # behaviour-proof receipt, i.e. four candidates that could clear every
    # existing conjunct today.
    #
    # A conjunct, not a heuristic: the refusal needs no threshold and no
    # judgement, and it is falsifiable from the ticket itself.
    SKIPPED_HAS_CHILDREN = "skipped_has_children"
    # OMN-17934. The binding companion is the evidence companion of a RECURRING
    # BOT PR — the standing pin-bump refresh — rather than of a PR that did the
    # ticket's work. The discriminator is a conjunction of the product PR's
    # author being a GitHub App/Bot AND its title matching the measured
    # pin-bump shape (see `_is_recurring_bot_product_pr`), derived from
    # omnibase_infra#3192 and #3199 and positive-controlled against 300 PRs of
    # that repo: 16 matches, every one bot-authored, zero human PRs.
    #
    # Why it is a distinct class rather than a predicate tightening: the flip
    # predicate was SATISFIED for OMN-17292 — terminal `verified`, 0 failed,
    # 4 verified, 4+26==30, 1 behaviour-proving, all description boxes ticked.
    # Nothing about that verdict is wrong; what is wrong is that a ticket which
    # accumulates recurring bot PRs re-clears it every time one merges,
    # indefinitely. The ordinary case is untouched: a ticket whose evidence
    # legitimately arrives across several PRs is refused only if the BOUND
    # companion's product PR is itself one of the recurring shapes.
    SKIPPED_RECURRING_COMPANION = "skipped_recurring_companion"
    # OMN-17934 shape 2. The ticket has already been Done and reopened: its
    # Linear state history carries a completed -> non-completed transition made
    # by a real actor. Somebody disagreed with a previous close, and re-closing
    # it from the same mechanism is exactly the disagreement being overruled by
    # a cron tick. Read live from `stateHistory`, never inferred.
    SKIPPED_PRIOR_REVERT = "skipped_prior_revert"
    # OMN-17658. `max_flips_per_run` was already spent by earlier candidates in
    # this run. The bound is a blast-radius cap, not a verdict: the candidate
    # reached no decision about its evidence and the next run will offer it
    # again. Recorded rather than silently dropped so a truncated run is
    # legible as truncated.
    SKIPPED_FLIP_BUDGET_EXHAUSTED = "skipped_flip_budget_exhausted"
    # OMN-17658 auto-disarm. An earlier candidate in this run (or the persisted
    # marker the workflow handed in) established that a closer flip was later
    # found unsafe, so this run refuses to apply from that point on. Every
    # remaining candidate is recorded with this value instead of a verdict —
    # the sweep is disarmed, not silent.
    SKIPPED_DISARMED = "skipped_disarmed"
    # OMN-17658 bound readback. `issueUpdate` reported success but the
    # post-write read of the ticket's own state history did not show a
    # completed segment that the pre-write read did not already have. Recorded
    # as an ERROR and never as FLIPPED: a write whose effect cannot be read
    # back is not a proven write, and counting it as one is how a closer's
    # receipt drifts from the board it claims to describe.
    ERROR_READBACK_UNCONFIRMED = "error_readback_unconfirmed"
    # `uv run onex skill dod_verify <ticket>` exited non-zero (dispatch/runtime
    # failure, not a normal verified/failed verdict) -> fail closed, never flip.
    ERROR_VERIFY_NONZERO_EXIT = "error_verify_nonzero_exit"
    # dod_verify stdout did not parse as the expected ModelSkillResult JSON.
    ERROR_VERIFY_UNPARSEABLE = "error_verify_unparseable"
    # Linear read (issue/state lookup) or write (issueUpdate/commentCreate)
    # call failed or returned GraphQL errors.
    ERROR_LINEAR_API = "error_linear_api"
    # The companion's changed-file list could not be fetched from GitHub
    # (transport/API failure). Never silently treated as "zero files" -- that
    # would let a transient fetch failure masquerade as SKIPPED_NO_BINDING or
    # fall back to a title-only match that a real file listing might have
    # disambiguated or contradicted.
    ERROR_GITHUB_API = "error_github_api"


__all__ = ["EnumEvidenceAutocloseDecision"]
