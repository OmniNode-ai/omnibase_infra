# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Per-ticket decision of the In Progress demotion leg (OMN-18489)."""

from __future__ import annotations

from enum import StrEnum


class EnumInProgressDemotionDecision(StrEnum):
    """What the demotion leg decided for one In Progress ticket.

    The five demotion conditions each own a SKIPPED value, so a receipt says
    which condition withheld a ticket rather than only that it was withheld.
    Every ERROR value means a condition could not be resolved: an indeterminate
    answer is never a permission to demote.
    """

    # Every condition held and the ticket was moved to Backlog and commented.
    DEMOTED = "demoted"
    # Every condition held; the run was a dry run, so nothing was written.
    WOULD_DEMOTE = "would_demote"
    # Condition 1: the ticket is not in the In Progress state.
    SKIPPED_NOT_IN_PROGRESS = "skipped_not_in_progress"
    # The ticket entered In Progress inside the window, so it has not yet had
    # the whole window to be touched.
    SKIPPED_STARTED_WITHIN_WINDOW = "skipped_started_within_window"
    # Condition 2: a comment other than a sweep's own landed inside the window.
    SKIPPED_RECENT_COMMENT = "skipped_recent_comment"
    # Condition 3: an open pull request cites the ticket.
    SKIPPED_OPEN_PULL_REQUEST = "skipped_open_pull_request"
    # Condition 4: a live execution claim comment is on the ticket.
    SKIPPED_LIVE_CLAIM = "skipped_live_claim"
    # Condition 5: the ticket is on the caller-supplied exclusion fence.
    SKIPPED_EXCLUDED = "skipped_excluded"
    # Every condition held and max_demotions_per_run was already spent. A
    # truncation, never a verdict: the next run offers the ticket again.
    SKIPPED_BUDGET_EXHAUSTED = "skipped_budget_exhausted"
    # A Linear read or write failed, or its payload could not be interpreted.
    ERROR_LINEAR_API = "error_linear_api"
    # The open-pull-request resolution failed or was incomplete.
    ERROR_GITHUB_API = "error_github_api"
    # The state write reported success but the readback did not show Backlog.
    ERROR_READBACK_UNCONFIRMED = "error_readback_unconfirmed"
    # The state moved to Backlog and the explanatory comment was not posted.
    ERROR_COMMENT_UNPOSTED = "error_comment_unposted"


__all__ = ["EnumInProgressDemotionDecision"]
