# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Output model for the savings-correlation periodic batch step.

Ticket: OMN-16293
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelSavingsCorrelationBatchOutput(BaseModel):
    """Result of one savings-correlation batch tick.

    Attributes:
        sessions_finalized: Number of sessions for which a
            ``ModelSavingsEstimate`` was computed and published.
        sessions_skipped_incomplete: Number of candidate sessions that were
            NOT ready to finalize this tick (no ``session_outcomes`` row yet
            and not past the timeout window).
        errors: Non-fatal per-session failures encountered this tick.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    sessions_finalized: int = Field(default=0, ge=0)
    sessions_skipped_incomplete: int = Field(default=0, ge=0)
    errors: tuple[str, ...] = Field(default=())


__all__: list[str] = ["ModelSavingsCorrelationBatchOutput"]
