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
    # Ticket carries the close-if-done label -> decision-only path stays manual.
    SKIPPED_LABEL = "skipped_label"
    # Ticket is already in a completed/canceled state -> nothing to do.
    SKIPPED_ALREADY_DONE = "skipped_already_done"
    # No contracts/OMN-XXXXX.yaml file and no evidence(OMN-XXXXX) title match.
    SKIPPED_NO_BINDING = "skipped_no_binding"
    # More than one distinct ticket id bound to the same merged companion.
    SKIPPED_AMBIGUOUS_BINDING = "skipped_ambiguous_binding"
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
