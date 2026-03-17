# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Model for a single validation violation record.

Field contract pinned to omnidash ``ViolationSchema`` in
``shared/validation-types.ts``.

Reference: OMN-5184 (Dashboard Data Pipeline Gaps), Batch B
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelViolation(BaseModel):
    """A single validation violation record."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    rule_id: str = Field(min_length=1)
    severity: Literal["error", "warning", "info"] = Field(
        description="Violation severity level"
    )
    message: str = Field(min_length=1)
    repo: str = Field(min_length=1)
    file_path: str | None = Field(default=None)
    line: int | None = Field(default=None, ge=0)
    validator: str = Field(min_length=1)


__all__ = ["ModelViolation"]
