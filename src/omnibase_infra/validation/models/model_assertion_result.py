# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Result model for a single demo loop assertion (OMN-2297)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.validation.enums.enum_assertion_status import EnumAssertionStatus


class ModelAssertionResult(BaseModel):
    """Result of a single demo loop assertion check.

    Attributes:
        name: Short identifier for the assertion (e.g., "canonical_pipeline").
        status: Whether the assertion passed, failed, or was skipped.
        message: Human-readable summary of the result.
        details: Optional list of detail lines (e.g., missing event types).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(..., description="Short assertion identifier")
    status: EnumAssertionStatus = Field(..., description="Assertion outcome")
    message: str = Field(..., description="Human-readable summary")
    details: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Optional detail lines for failure diagnostics",
    )
