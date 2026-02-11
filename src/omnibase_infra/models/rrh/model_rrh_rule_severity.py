# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Per-rule severity configuration for RRH profiles."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.diagnostics.enum_verdict import EnumVerdict


class ModelRRHRuleSeverity(BaseModel):
    """Per-rule severity configuration.

    Attributes:
        rule_id: Rule identifier (e.g. ``"RRH-1001"``).
        enabled: Whether the rule is active in this profile.
        severity: Verdict level when the rule fails (WARN or FAIL).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    rule_id: str = Field(..., description="Rule identifier.")
    enabled: bool = Field(default=True, description="Whether the rule is active.")
    severity: EnumVerdict = Field(
        default=EnumVerdict.FAIL,
        description="Verdict when rule fails (WARN or FAIL).",
    )


__all__: list[str] = ["ModelRRHRuleSeverity"]
