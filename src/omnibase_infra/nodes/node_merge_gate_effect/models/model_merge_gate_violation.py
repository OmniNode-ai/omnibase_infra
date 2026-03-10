# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Model for a single merge gate violation.

Related Tickets:
    - OMN-3140: NodeMergeGateEffect + migration
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelMergeGateViolation(BaseModel):
    """A single violation detected by the merge gate.

    Attributes:
        rule_code: Identifier of the violated rule (e.g. "RRH-1001").
        severity: Severity level of the violation.
        message: Human-readable description of the violation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    rule_code: str = Field(..., description="Identifier of the violated rule.")
    severity: str = Field(..., description="Severity level (FAIL, WARN).")
    message: str = Field(..., description="Human-readable violation description.")


__all__: list[str] = ["ModelMergeGateViolation"]
