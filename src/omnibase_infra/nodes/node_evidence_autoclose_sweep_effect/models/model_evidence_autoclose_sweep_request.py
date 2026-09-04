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
            "Working directory to run `onex skill dod_verify <ticket>` from. "
            "Empty string means: inherit the sweep process's own cwd. "
            "OMN-16846: this no longer selects the verifier's ENVIRONMENT. "
            "The verifier is dispatched from the sweep interpreter's own "
            "`onex` (see `_dod_verify_argv`), so the venv carrying "
            "node_dod_verify is decided by how the sweep was composed rather "
            "than by where it stands -- which is what lets the product clone "
            "the behaviour checks run pytest in stay lock-exact. This field "
            "now only sets the cwd the verifier process inherits."
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
    exclude_tickets: tuple[str, ...] = Field(
        default=(),
        description=(
            "Ticket ids this run must refuse before it reads anything about "
            "them (OMN-17891). Matched case-insensitively after stripping "
            "surrounding whitespace; each match is recorded as "
            "SKIPPED_EXCLUDED and costs zero Linear I/O.\n\n"
            "This is a CALLER ASSERTION, never a derived fact. The node reads "
            "no ledger, no assignee, and no ownership signal — before this "
            "field existed an apply run's only refusals were a Linear label "
            "set beforehand, an already-completed state, a binding-hygiene "
            "skip, and the global ONEX_AUTOCLOSE_DISABLED kill switch, none "
            "of which can decline ONE candidate. Whoever dispatches the run "
            "supplies the list and owns its accuracy; the enum value is "
            "distinct from SKIPPED_LABEL precisely so the audit record says "
            "which authority refused.\n\n"
            "It does not weaken the kill switch: a halted run does zero work "
            "regardless of what this names, so an exclusion list can never "
            "opt one ticket back into a halted sweep."
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
