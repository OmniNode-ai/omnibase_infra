# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Request for the in-progress probe hygiene sweep (OMN-17942)."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelProbeHygieneRequest(BaseModel):
    """Sweep configuration: scope, write flag, safety caps."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID = Field(..., description="Run correlation id.")
    occ_repo_dir: str = Field(
        default="",
        description=(
            "Filesystem path to an onex_change_control clone whose "
            "contracts/ directory the sweep reads. Empty means the sweep "
            "cannot resolve contracts and every ticket terminates "
            "ERROR_CONTRACT_UNREADABLE — a broken runner is reported as one, "
            "never as 'no ticket has a probe', which is the same number and "
            "the opposite meaning."
        ),
    )
    project: str = Field(
        default="",
        description=(
            "Linear project id to scope the sweep to. Empty sweeps every "
            "In-Progress ticket the credential can see."
        ),
    )
    max_tickets: int = Field(
        default=200,
        ge=1,
        le=1000,
        description="Cap on In-Progress tickets enumerated per run.",
    )
    max_comments_per_run: int = Field(
        default=10,
        ge=0,
        le=200,
        description=(
            "Cap on hygiene comments written per run. The first run after "
            "this lands has a standing backlog to work through, and writing "
            "all of it at once is a notification storm; the rotation costs a "
            "few ticks and nothing is lost, because a ticket not commented "
            "on is reported in `outcomes` either way. 0 makes the run "
            "report-only without making it a dry run."
        ),
    )
    apply: bool = Field(
        default=False,
        description=(
            "DRY-RUN by default: every decision is reached and reported, and "
            "no comment is written."
        ),
    )
    exclude_tickets: tuple[str, ...] = Field(
        default=(),
        description=(
            "Ticket ids this run must refuse before it reads anything about "
            "them — the same per-candidate fence the evidence closer takes "
            "(OMN-17891), so a lane holding a ticket is not commented at by "
            "a second process."
        ),
    )
    linear_retry_max_attempts: int = Field(
        default=4,
        ge=1,
        le=10,
        description=(
            "Total attempts per Linear GraphQL call whose failure is "
            "retryable. Mirrors the closer's OMN-16106 policy for the same "
            "measured reason: an un-retried transient read drops the "
            "candidate from the run entirely."
        ),
    )
    linear_retry_base_delay_seconds: float = Field(
        default=1.0,
        ge=0.0,
        le=30.0,
        description=(
            "First-retry backoff window, doubled and jittered per attempt. "
            "0.0 is a real value and is what tests use."
        ),
    )


__all__ = ["ModelProbeHygieneRequest"]
