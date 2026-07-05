# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Zombie-run candidate fact (OMN-13942) -- a queued/running job that has aged past
the wedge threshold. Facts only; the health COMPUTE node decides whether this
warrants a CANCEL_RUN recommendation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelZombieRunCandidate(BaseModel):
    """A workflow run observed as queued/in-progress well past a normal wait."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    repo: str = Field(..., description="org/repo the run belongs to.")
    run_id: int = Field(..., description="GitHub Actions run (databaseId).")
    workflow_name: str = Field(default="", description="Workflow display name.")
    status: Literal["queued", "in_progress"] = Field(
        ..., description="Run status at observation time."
    )
    age_seconds: float = Field(
        ..., ge=0.0, description="Seconds since the run entered its current status."
    )


__all__ = ["ModelZombieRunCandidate"]
