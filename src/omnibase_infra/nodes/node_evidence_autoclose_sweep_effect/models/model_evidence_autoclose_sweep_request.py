# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Request model for the evidence autoclose sweep."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelEvidenceAutocloseSweepRequest(BaseModel):
    """Request to sweep recently-merged OCC companions for governed Done flips."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Sweep run correlation ID.")
    occ_repo: str = Field(
        default="OmniNode-ai/onex_change_control",
        description="owner/repo of the evidence-companion repository to scan.",
    )
    lookback_hours: int = Field(
        default=24,
        ge=1,
        le=24 * 30,
        description="Scan companions merged within this many hours of now.",
    )
    apply: bool = Field(
        default=False,
        description=(
            "DRY-RUN is the default (apply=False): the sweep logs every "
            "decision it WOULD make but never calls the Linear mutation "
            "API. Pass apply=True to actually flip Done / post comments."
        ),
    )
    max_companions: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Safety cap on the number of merged companions scanned per run.",
    )
    dispatch_cwd: str = Field(
        default="",
        description=(
            "Working directory to run `uv run onex skill dod_verify <ticket>` "
            "from. Empty string means: inherit the sweep process's own cwd "
            "(the caller is expected to invoke the sweep from the "
            "omnibase_infra clone, the same locality the controller uses for "
            "governed flips -- dod_verify dispatches through omnimarket "
            "co-installed into THAT venv, see omnimarket_drift_guard)."
        ),
    )
    dod_verify_timeout_seconds: int = Field(
        default=300,
        ge=1,
        le=1800,
        description="Timeout for each `onex skill dod_verify` subprocess call.",
    )
    gh_timeout_seconds: int = Field(
        default=90,
        ge=1,
        le=1800,
        description=(
            "Timeout for each `gh api` subprocess call (PR-list enumeration "
            "and per-PR file listing). Raised from a prior hardcoded 30.0s "
            "default that timed out in CI (OMN-16106: duration_ms==30048 on "
            "a live self-hosted-runner GitHub enumeration)."
        ),
    )
    close_if_done_label: str = Field(
        default="close-if-done",
        description=(
            "Linear label name that routes a ticket to the manual "
            "decision-only close path instead of this sweep."
        ),
    )


__all__ = ["ModelEvidenceAutocloseSweepRequest"]
