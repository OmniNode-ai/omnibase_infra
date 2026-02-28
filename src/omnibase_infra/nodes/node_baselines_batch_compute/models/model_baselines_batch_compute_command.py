# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Command model for NodeBaselinesBatchCompute.

D1: correlation_id is REQUIRED — no default; callers must generate UUID4.

Ticket: OMN-3043
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class ModelBaselinesBatchComputeCommand(BaseModel):
    """Command to trigger a baselines batch computation run.

    Attributes:
        operation: Fixed operation discriminator for handler routing.
        correlation_id: Required correlation ID for tracing (no default —
            callers must generate via ``uuid.uuid4()``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    operation: Literal["baselines.batch_compute"] = "baselines.batch_compute"
    correlation_id: UUID  # required — no default; callers must generate UUID4


__all__: list[str] = ["ModelBaselinesBatchComputeCommand"]
