# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Active GitHub Actions job fact for runner-fleet alerts (OMN-17427)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelRunnerJobFact(BaseModel):
    """A queued or in-progress job observed in a watched workflow run."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    repo: str = Field(..., description="org/repo the job belongs to.")
    run_id: int = Field(..., description="GitHub Actions workflow run ID.")
    job_id: int = Field(..., description="GitHub Actions job ID.")
    workflow: str = Field(default="", description="Workflow display name.")
    job: str = Field(default="", description="Job display name.")
    event: str = Field(default="", description="Event that triggered the workflow run.")
    status: Literal["queued", "in_progress"] = Field(
        ..., description="Job status at observation time."
    )
    labels: tuple[str, ...] = Field(default=(), description="Requested runner labels.")
    runner: str = Field(
        default="",
        description="Assigned runner name; empty before a runner picks it up.",
    )
    age_seconds: float = Field(
        ...,
        ge=0.0,
        description=(
            "Seconds since start for in-progress jobs (creation if start is missing), "
            "or since creation for queued jobs."
        ),
    )


__all__ = ["ModelRunnerJobFact"]
