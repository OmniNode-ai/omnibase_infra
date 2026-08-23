# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Command model for the savings-correlation periodic batch step.

Ticket: OMN-16293
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class ModelSavingsCorrelationBatchCommand(BaseModel):
    """Command to trigger one savings-correlation batch tick.

    Attributes:
        operation: Fixed operation discriminator for handler routing.
        correlation_id: Required correlation ID for tracing (no default —
            callers must generate via ``uuid.uuid4()``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    operation: Literal["savings.correlation_batch_compute"] = (
        "savings.correlation_batch_compute"
    )
    correlation_id: UUID  # required — no default; callers must generate UUID4


__all__: list[str] = ["ModelSavingsCorrelationBatchCommand"]
