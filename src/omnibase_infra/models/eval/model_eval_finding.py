# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Eval finding model for autonomous off-peak evaluations.

Related:
    - OMN-6795: Define eval task models and enums

.. versionadded:: 0.29.0
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums.enum_eval_finding_category import EnumEvalFindingCategory
from omnibase_infra.enums.enum_eval_finding_severity import EnumEvalFindingSeverity


class ModelEvalFinding(BaseModel):
    """A single finding from an evaluation task.

    Attributes:
        severity: Finding severity level.
        category: Finding category.
        file_path: File path where the finding was detected.
        line_number: Line number (0 if not applicable).
        description: Human-readable description of the finding.
        suggestion: Suggested fix or action.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    severity: EnumEvalFindingSeverity = Field(..., description="Severity level.")
    category: EnumEvalFindingCategory = Field(..., description="Finding category.")
    file_path: str = Field(default="", description="File path of the finding.")
    line_number: int = Field(default=0, ge=0, description="Line number.")
    description: str = Field(..., description="Description of the finding.")
    suggestion: str = Field(default="", description="Suggested fix.")


__all__: list[str] = ["ModelEvalFinding"]
