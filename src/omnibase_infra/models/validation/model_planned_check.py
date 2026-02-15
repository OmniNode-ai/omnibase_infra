# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Shared validation pipeline model — planned check.

A single check to be executed as part of the validation plan.

Ticket: OMN-2147
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumCheckSeverity


class ModelPlannedCheck(BaseModel):
    """A single check scheduled for execution as part of a validation plan.

    Attributes:
        check_code: Unique check identifier (e.g., CHECK-PY-001).
        label: Human-readable check label (e.g., "mypy typecheck").
        command: Shell command to execute (e.g., "uv run mypy src/").
        severity: Whether this check is required, recommended, or informational.
        enabled: Whether the check should be executed.
        timeout_ms: Maximum execution time in milliseconds (0 = no limit).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    check_code: str = Field(
        ..., description="Unique check identifier (e.g., CHECK-PY-001)."
    )
    label: str = Field(..., description="Human-readable check name.")
    command: str = Field(default="", description="Shell command to execute.")
    severity: EnumCheckSeverity = Field(
        default=EnumCheckSeverity.REQUIRED,
        description="Severity classification for this check.",
    )
    enabled: bool = Field(
        default=True, description="Whether the check should be executed."
    )
    timeout_ms: float = Field(
        default=0.0, ge=0.0, description="Max execution time in ms (0 = no limit)."
    )


__all__: list[str] = ["ModelPlannedCheck"]
