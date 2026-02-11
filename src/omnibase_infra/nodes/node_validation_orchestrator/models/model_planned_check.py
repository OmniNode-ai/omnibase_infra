# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Planned check model for the validation orchestrator.

A single check to be executed as part of the validation plan.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumCheckSeverity


class ModelPlannedCheck(BaseModel):
    """A single check to be executed as part of the validation plan.

    Attributes:
        check_code: Check identifier (e.g., CHECK-PY-001).
        label: Human-readable check label.
        severity: Check severity level.
        enabled: Whether this check is enabled.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    check_code: str = Field(..., description="Check identifier (e.g., CHECK-PY-001).")
    label: str = Field(..., description="Human-readable check name.")
    severity: EnumCheckSeverity = Field(..., description="Check severity level.")
    enabled: bool = Field(default=True, description="Whether this check is enabled.")


__all__: list[str] = ["ModelPlannedCheck"]
