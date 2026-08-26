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
        default=50,
        ge=1,
        le=250,
        description=(
            "Per-ticket history entries fetched via backward pagination "
            "(GraphQL `last`), which returns the N MOST RECENT entries "
            "regardless of total history length — correct for this "
            "watchdog's 'act on the latest revert' semantics without "
            "needing to paginate a ticket's full history. The handler "
            "never trusts the API's return order regardless: every entry "
            "is re-sorted by createdAt client-side before use."
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
