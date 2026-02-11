# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Shared validation pipeline model — individual check result."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumCheckSeverity


class ModelCheckResult(BaseModel):
    """Result of a single validation check execution.

    Attributes:
        check_code: Check identifier (e.g., CHECK-PY-001).
        label: Human-readable check label.
        severity: Check severity classification.
        passed: Whether the check passed.
        skipped: Whether the check was skipped.
        message: Human-readable result message.
        error_output: Captured stderr/stdout on failure.
        duration_ms: Check execution duration in milliseconds.
        executed_at: Timestamp when the check was executed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    check_code: str = Field(..., description="Check identifier (e.g., CHECK-PY-001).")
    label: str = Field(..., description="Human-readable check name.")
    severity: EnumCheckSeverity = Field(
        ..., description="Check severity classification."
    )
    passed: bool = Field(..., description="Whether the check passed.")
    skipped: bool = Field(default=False, description="Whether the check was skipped.")
    message: str = Field(default="", description="Human-readable result message.")
    error_output: str = Field(
        default="", description="Captured stderr/stdout on failure."
    )
    duration_ms: float = Field(
        default=0.0, ge=0.0, description="Execution duration in ms."
    )
    executed_at: datetime | None = Field(
        default=None, description="Timestamp when the check was executed."
    )

    def is_blocking_failure(self) -> bool:
        """Return True if this is a required check that failed."""
        return not self.passed and not self.skipped and self.severity.blocks_verdict()


__all__: list[str] = ["ModelCheckResult"]
