# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Result model for the evidence autoclose sweep."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.model_evidence_autoclose_outcome import (
    ModelEvidenceAutocloseOutcome,
)


class ModelEvidenceAutocloseSweepResult(BaseModel):
    """Result of one evidence autoclose sweep run."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Sweep run correlation ID.")
    dry_run: bool = Field(..., description="True unless the request set apply=True.")
    kill_switch_engaged: bool = Field(
        default=False,
        description=(
            "True when ONEX_AUTOCLOSE_DISABLED was set — the sweep performed "
            "zero GitHub/Linear I/O and returned immediately."
        ),
    )
    companions_scanned: int = Field(default=0, ge=0)
    bindings_extracted: int = Field(default=0, ge=0)
    tickets_flipped: int = Field(default=0, ge=0)
    tickets_gap_posted: int = Field(default=0, ge=0)
    tickets_skipped: int = Field(default=0, ge=0)
    tickets_errored: int = Field(default=0, ge=0)
    outcomes: tuple[ModelEvidenceAutocloseOutcome, ...] = Field(
        default_factory=tuple,
        description="One entry per (companion, ticket) pair considered.",
    )
    success: bool = Field(
        default=True,
        description="False only on a sweep-level failure (GitHub enumeration failed).",
    )
    error_message: str = Field(default="", description="Sweep-level error, if any.")


__all__ = ["ModelEvidenceAutocloseSweepResult"]
