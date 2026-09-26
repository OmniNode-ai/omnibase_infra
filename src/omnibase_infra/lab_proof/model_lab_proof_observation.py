# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""What happened when one planned step ran (or why it did not).

Ticket: OMN-19572
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.lab_proof.enum_lab_proof_attribution import (
    EnumLabProofAttribution,
)
from omnibase_infra.lab_proof.enum_lab_proof_step_id import EnumLabProofStepId
from omnibase_infra.lab_proof.enum_lab_proof_step_phase import EnumLabProofStepPhase


class ModelLabProofObservation(BaseModel):
    """One step's record. ``ok`` = ran, exit 0, and every expectation held."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step_id: EnumLabProofStepId
    phase: EnumLabProofStepPhase
    attribution: EnumLabProofAttribution
    ran: bool
    skip_reason: str = ""
    exit_code: int | None = None
    timed_out: bool = False
    attempts: int = Field(default=0, ge=0)
    duration_seconds: float = Field(default=0.0, ge=0.0)
    stdout_tail: str = ""
    stderr_tail: str = ""
    expectation_met: bool = False
    ok: bool = False
    pattern_counts: dict[str, int] = Field(default_factory=dict)
    extracted: tuple[str, ...] = ()
    log_path: str = ""


__all__ = ["ModelLabProofObservation"]
