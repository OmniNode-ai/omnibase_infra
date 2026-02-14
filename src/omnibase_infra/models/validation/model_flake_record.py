# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Flake record model for tracking individual check flake analysis.

Ticket: OMN-2151
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field


class ModelFlakeRecord(BaseModel):
    """Record of a flake detection analysis for a single check.

    Attributes:
        check_code: Check identifier.
        first_passed: Whether the first run passed.
        rerun_passed: Whether the rerun passed (None if not rerun).
        is_flake_suspected: Whether a flake was suspected.
        rerun_count: Number of reruns performed.
        detected_at: Timestamp of flake detection.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    check_code: str = Field(..., description="Check identifier.")
    first_passed: bool = Field(..., description="Whether the first run passed.")
    rerun_passed: bool | None = Field(
        default=None, description="Whether the rerun passed."
    )
    is_flake_suspected: bool = Field(
        default=False, description="Whether a flake was suspected."
    )
    rerun_count: int = Field(default=0, ge=0, description="Number of reruns performed.")
    detected_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=UTC),
        description="Timestamp of flake detection.",
    )


__all__: list[str] = ["ModelFlakeRecord"]
