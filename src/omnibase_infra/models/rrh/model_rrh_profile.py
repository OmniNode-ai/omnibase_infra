# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""RRH validation profile model.

A profile defines which rules are active and their severity.  Profiles
set the *baseline* — contracts can only **tighten** (add rules or raise
severity), never loosen.

Profile Precedence:
    PROFILE baseline -> CONTRACT can only TIGHTEN -> Final rule set
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.diagnostics.enum_verdict import EnumVerdict
from omnibase_infra.models.rrh.model_rrh_rule_severity import ModelRRHRuleSeverity


class ModelRRHProfile(BaseModel):
    """RRH validation profile.

    Attributes:
        name: Profile name (e.g. ``"default"``, ``"ticket-pipeline"``).
        description: Human-readable profile description.
        rules: Per-rule severity configuration.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    name: str = Field(..., description="Profile name.")
    description: str = Field(default="", description="Profile description.")
    rules: tuple[ModelRRHRuleSeverity, ...] = Field(
        default_factory=tuple, description="Per-rule configurations."
    )

    def is_rule_enabled(self, rule_id: str) -> bool:
        """Check if a rule is enabled in this profile."""
        for rule in self.rules:
            if rule.rule_id == rule_id:
                return rule.enabled
        return False

    def get_severity(self, rule_id: str) -> EnumVerdict:
        """Get the severity for a rule. Defaults to FAIL if not found."""
        for rule in self.rules:
            if rule.rule_id == rule_id:
                return rule.severity
        return EnumVerdict.FAIL


__all__: list[str] = ["ModelRRHProfile"]
