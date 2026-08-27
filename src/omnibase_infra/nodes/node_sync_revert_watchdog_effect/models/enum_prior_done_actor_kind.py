# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Provenance of the completed state a detected revert would restore."""

from __future__ import annotations

from enum import StrEnum


class EnumPriorDoneActorKind(StrEnum):
    """Who set the pre-revert completed state the watchdog would restore.

    The operator's restore rule (OMN-16762, derived from the OMN-16536
    revert-backlog adjudication) is that only a HUMAN-set — or formally
    adjudicated — Done may be re-flipped. Restoring a completed state that
    automation itself set reinstates an automation artifact rather than a
    human decision, which is precisely the failure the watchdog exists to
    prevent.

    Only ``HUMAN`` clears the precondition. ``BOT`` and ``UNKNOWN`` both
    fail closed.
    """

    # The history entry that set the prior completed state carried a
    # non-null actorId — a real Linear user made the call.
    HUMAN = "human"
    # actorId was null (Linear's documented integration/automation/system
    # signature). Restoring this state reinstates an automation artifact.
    BOT = "bot"
    # No transition INTO the prior completed state exists in the history
    # the sweep could read — provenance is unproven, so it fails closed
    # exactly like BOT. Reachable when a ticket was created directly in a
    # completed state, or when history_max_pages capped the walk before
    # the setting transition.
    UNKNOWN = "unknown"


__all__ = ["EnumPriorDoneActorKind"]
