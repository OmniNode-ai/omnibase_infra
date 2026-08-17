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


__all__ = ["EnumEvidenceAutocloseDecision"]
