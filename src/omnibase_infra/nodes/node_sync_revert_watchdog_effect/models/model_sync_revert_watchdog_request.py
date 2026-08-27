# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Request model for the sync-revert watchdog sweep."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelSyncRevertWatchdogRequest(BaseModel):
    """Request to sweep a team's recently-updated tickets for silent Done-reverts."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Sweep run correlation ID.")
    team_key: str = Field(
        default="OMN",
        description="Linear team key (e.g. 'OMN' for Omninode) to scan.",
    )
    lookback_hours: int = Field(
        default=24,
        ge=1,
        le=24 * 30,
        description="Scan tickets whose updatedAt falls within this many hours of now.",
    )
    apply: bool = Field(
        default=False,
        description=(
            "DRY-RUN is the default (apply=False): the watchdog logs every "
            "revert it WOULD correct but never calls the Linear mutation "
            "API. Pass apply=True to actually flip state / post comments."
        ),
    )
    max_issues: int = Field(
        default=200,
        ge=1,
        le=2000,
        description="Safety cap on the number of tickets scanned per run.",
    )
    history_page_size: int = Field(
        default=250,
        ge=1,
        le=250,
        description=(
            "Page size for the per-ticket history walk. The handler "
            "paginates FORWARD (GraphQL `first`/`after`) from the newest "
            "entry toward the oldest, following pageInfo.hasNextPage "
            "until the history is exhausted or history_max_pages is hit. "
            "250 is Linear's per-page ceiling, so the default costs the "
            "fewest round trips. "
            "OMN-16762 — this field previously fed GraphQL `last`, "
            "documented here as 'the N MOST RECENT entries'. That was "
            "the exact opposite of the API's behavior: Linear's "
            "`orderBy: createdAt` sorts DESCENDING, so `last: N` returns "
            "the tail of that list — the N OLDEST entries. Measured live "
            "on OMN-14888 (553 entries) 2026-08-27, `last: 50` returned "
            "2026-07-21..2026-07-26 and nothing newer, which left both "
            "safety guards scanning a stale window (0 fires across 126 "
            "detected reverts). The handler still never trusts the API's "
            "return order: every entry is re-sorted by createdAt "
            "client-side before use."
        ),
    )
    history_max_pages: int = Field(
        default=20,
        ge=1,
        le=100,
        description=(
            "Safety cap on history pages walked per ticket. At the "
            "default page size this bounds a ticket at 5000 entries — "
            "roughly 9x the largest history observed in the OMN-16536 "
            "audit (OMN-14888, 553). Because the walk runs newest-first, "
            "hitting the cap truncates the OLDEST end only: revert "
            "detection and the later-human-state-change guard stay "
            "sound, and a pre-revert Done that falls outside the cap "
            "resolves to EnumPriorDoneActorKind.UNKNOWN, which fails "
            "closed."
        ),
    )
    human_comment_window_seconds: int = Field(
        default=3600,
        ge=0,
        le=24 * 3600,
        description=(
            "A candidate revert is treated as a deliberate, explained "
            "human action (skipped) when a human-authored comment exists "
            "within this many seconds before or after the transition "
            "timestamp. Default 1h comfortably covers every observed "
            "automation-fire latency in the OMN-16536 incident family "
            "(2.56s-53s) while staying tight enough that an unrelated "
            "later comment does not mask a real silent revert."
        ),
    )
    linear_timeout_seconds: int = Field(
        default=15,
        ge=1,
        le=120,
        description="Timeout for each Linear GraphQL call.",
    )
    watchdog_comment_marker: str = Field(
        default="sync-revert-watchdog (OMN-16536)",
        description=(
            "Substring stamped into every diagnosis comment this watchdog "
            "posts and excluded when scanning for a 'human comment "
            "nearby' — otherwise a later run would see the watchdog's own "
            "prior comment and mistake it for a human explanation."
        ),
    )


__all__ = ["ModelSyncRevertWatchdogRequest"]
