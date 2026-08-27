# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Per-ticket outcome classification for the sync-revert watchdog."""

from __future__ import annotations

from enum import StrEnum


class EnumSyncRevertWatchdogDecision(StrEnum):
    """Terminal decision recorded for one scanned Linear ticket.

    Exactly one value is recorded per ticket the watchdog considered.
    REFLIPPED is the only value that ever mutates a ticket's Linear state;
    every other value is fail-closed (no-op).
    """

    # A completed->non-completed history transition was found whose actor
    # was null with a populated botActor (Linear's own automation/
    # integration signature), no human comment fell inside the detection
    # window, and no later human-driven state change has occurred since ->
    # re-flipped to the prior completed state + diagnosis comment posted.
    REFLIPPED = "reflipped"
    # No completed->non-completed transition exists anywhere in the
    # scanned history window -> nothing to do.
    SKIPPED_NO_REVERT_FOUND = "skipped_no_revert_found"
    # The most recent completed->non-completed transition's actorId was
    # set (a real Linear user made the change) -> not a silent automation
    # revert, out of scope for this watchdog.
    SKIPPED_HUMAN_ACTOR = "skipped_human_actor"
    # The transition was automation-driven (actorId null + botActor set)
    # but a human posted a comment inside the detection window -> treated
    # as an explained, deliberate revert; never overridden.
    SKIPPED_HUMAN_COMMENT_NEARBY = "skipped_human_comment_nearby"
    # A human-actored history entry exists AFTER the detected automation
    # revert -> a person has since looked at the ticket and made a further
    # deliberate choice; the watchdog never overrides a later human
    # decision, even if that choice left the ticket in a non-completed
    # state.
    SKIPPED_STATE_CHANGED_SINCE = "skipped_state_changed_since"
    # The ticket's CURRENT state is already completed-type -> already
    # resolved (manually or by an earlier watchdog run); nothing to do.
    SKIPPED_ALREADY_RESOLVED = "skipped_already_resolved"
    # OMN-16762: the completed state this revert would restore was NOT
    # set by a human -- either automation set it (actorId null +
    # botActor, i.e. the very signature this watchdog detects) or its
    # provenance could not be established in the history read. The
    # operator's restore rule requires the pre-revert Done to be
    # human-set or formally adjudicated, so re-flipping here would
    # reinstate an automation artifact rather than a human decision.
    # Fails closed on BOT and UNKNOWN alike; see
    # EnumPriorDoneActorKind.
    SKIPPED_PRIOR_DONE_NOT_HUMAN_SET = "skipped_prior_done_not_human_set"
    # Linear read (issues/history/comments) or write (issueUpdate/
    # commentCreate) call failed or returned GraphQL errors.
    ERROR_LINEAR_API = "error_linear_api"
    # The detected revert's prior completed-type state no longer resolves
    # to a live state on the team (renamed/deleted workflow state) ->
    # fail closed rather than guess a replacement.
    ERROR_STATE_NOT_RESOLVABLE = "error_state_not_resolvable"


__all__ = ["EnumSyncRevertWatchdogDecision"]
