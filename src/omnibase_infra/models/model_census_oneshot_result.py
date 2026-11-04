# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Results of census sampling and completion grading."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelCensusOneshotResult(BaseModel):
    """One final stdout document and all sample diagnostics."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)
    exit_code: int
    stdout: str = ""
    stderr: str = ""


class ModelPendingOneshot(BaseModel):
    """A particular execution whose completion deadline has not elapsed."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)
    lane: str
    container: str
    started_at: str
    deadline: str
    remaining_seconds: float


class ModelOneshotGrade(BaseModel):
    """A finding, pending execution or completed/absent observation."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)
    kind: str = ""
    detail: str = ""
    pending: ModelPendingOneshot | None = None
