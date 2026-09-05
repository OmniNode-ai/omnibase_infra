# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Why one In-Progress ticket got the verdict it got (OMN-17942).

Every ticket the sweep reaches terminates in exactly one of these. The sweep
NEVER writes a state: the whole taxonomy is about whether a ticket carries an
executable close probe and, if not, whether this run said so.
"""

from __future__ import annotations

from enum import StrEnum, unique


@unique
class EnumProbeHygieneDecision(StrEnum):
    """Terminal verdict for one In-Progress ticket."""

    #: The ticket declares at least one executable check. Nothing to say.
    HAS_PROBE = "has_probe"
    #: No executable probe anywhere, and this run posted the hygiene comment
    #: naming exactly what is missing.
    COMMENTED = "commented"
    #: No executable probe, and a previous run already said so on this ticket.
    #: Identified by the marker line, so the sweep says a thing ONCE — the
    #: OMN-16808 rule. Reported every run so the standing list stays visible
    #: even though only the first run wrote to the board.
    SKIPPED_ALREADY_COMMENTED = "skipped_already_commented"
    #: Would have commented, but the run is a dry run.
    SKIPPED_DRY_RUN = "skipped_dry_run"
    #: On the caller-supplied fence: another lane holds this ticket. Refused
    #: before any read about it, matching the closer's OMN-17891 fence.
    SKIPPED_EXCLUDED = "skipped_excluded"
    #: The comment budget for this run is spent. A hygiene sweep that can
    #: comment on 200 tickets in one tick is a notification storm, so the cap
    #: is a first-class outcome rather than a silent truncation.
    SKIPPED_COMMENT_BUDGET_EXHAUSTED = "skipped_comment_budget_exhausted"
    #: The ticket's comment history could not be read, so the sweep cannot
    #: establish it has not already said this here. Fails closed rather than
    #: risk a duplicate (OMN-16808).
    ERROR_LINEAR_API = "error_linear_api"
    #: The OCC contract for this ticket could not be read — the governance
    #: clone was missing or unreadable. "No contract" and "could not look"
    #: are different facts and must not collapse: the first is the finding,
    #: the second is a broken runner.
    ERROR_CONTRACT_UNREADABLE = "error_contract_unreadable"


__all__ = ["EnumProbeHygieneDecision"]
