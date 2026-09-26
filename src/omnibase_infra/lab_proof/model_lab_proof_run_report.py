# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Everything one proof run observed, in plan order.

Ticket: OMN-19572
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from omnibase_infra.lab_proof.enum_lab_proof_step_id import EnumLabProofStepId
from omnibase_infra.lab_proof.model_lab_proof_observation import (
    ModelLabProofObservation,
)


class ModelLabProofRunReport(BaseModel):
    """The run effect's output. ``aborted_at`` names the must-succeed step that failed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_key: str
    host: str
    started_at: str
    finished_at: str
    aborted_at: EnumLabProofStepId | None = None
    observations: tuple[ModelLabProofObservation, ...]

    def get(self, step_id: EnumLabProofStepId) -> ModelLabProofObservation | None:
        """Return the observation for ``step_id``, or None when the plan had no such step."""
        for observation in self.observations:
            if observation.step_id is step_id:
                return observation
        return None


__all__ = ["ModelLabProofRunReport"]
