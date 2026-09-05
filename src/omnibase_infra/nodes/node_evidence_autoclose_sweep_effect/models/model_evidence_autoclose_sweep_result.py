# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Result model for the evidence autoclose sweep."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.enum_evidence_autoclose_mode import (
    EnumEvidenceAutocloseMode,
)
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
    # OMN-17658. `dry_run` above is a boolean and cannot distinguish the four
    # ways a run reaches "wrote nothing" — halted, disarmed, unarmed schedule,
    # dispatched preview. Those are different facts about the closer and only
    # one of them is a fault. See EnumEvidenceAutocloseMode.
    mode: EnumEvidenceAutocloseMode = Field(
        default=EnumEvidenceAutocloseMode.DRY_RUN,
        description="How this run resolved its authority to write.",
    )
    # OMN-17658 auto-disarm. Non-empty means this run refused to apply because
    # a closer flip was found unsafe — either handed in by the caller from the
    # persisted marker, or discovered mid-run from a candidate's own Linear
    # state history. The ticket named here is the trigger, and it is what an
    # operator re-arms against.
    disarm_triggered_by: str = Field(
        default="",
        description="Ticket whose unsafe closer flip disarmed this run.",
    )
    disarm_reason: str = Field(
        default="",
        description="Why the run disarmed. Empty unless `mode` is DISARMED.",
    )
    # OMN-17658. How much of `max_flips_per_run` this run did NOT spend. A run
    # that finishes with 0 remaining was TRUNCATED, and the difference between
    # "nothing else qualified" and "the budget ran out" has to be readable
    # without recounting the outcomes.
    flip_budget_remaining: int = Field(default=0, ge=0)
    companions_scanned: int = Field(default=0, ge=0)
    # OMN-17342. The two numbers that make the backfill arm's coverage claim
    # checkable rather than assertable. `backfill_pool_size` is how many merged
    # companions the wider window held after the forward window's own
    # candidates were removed; `backfill_candidates_selected` is how many of
    # them this tick's rotating slice actually offered to the pipeline. The
    # ratio is the drain rate, and `selected` is the run-budget bound — the
    # thing that must NOT grow with the board. Both stay 0 on a run that did
    # not ask for the arm, which is what a receipt should say about an arm that
    # did not run.
    backfill_pool_size: int = Field(default=0, ge=0)
    backfill_candidates_selected: int = Field(default=0, ge=0)
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
