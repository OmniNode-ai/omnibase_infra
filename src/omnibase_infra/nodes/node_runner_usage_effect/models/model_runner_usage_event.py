# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runner usage event consumed by the runner usage effect."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ModelRunnerUsageEvent(BaseModel):
    """Self-hosted GitHub Actions runner usage evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    workflow_run_id: str = Field(..., min_length=1)
    job_id: str = Field(..., min_length=1)
    runner_minutes: float = Field(..., ge=0.0)
    repo_name: str | None = Field(default=None)
    machine_id: str | None = Field(default=None)
    runner_name: str | None = Field(default=None)
    workflow_name: str | None = Field(default=None)
    event_timestamp: datetime = Field(
        ..., description="Completion timestamp for the runner job."
    )

    @property
    def idempotency_key(self) -> tuple[str, str]:
        """Deduplication key for runner usage events."""
        return self.workflow_run_id, self.job_id

    @property
    def session_id(self) -> str:
        """Stable savings session ID derived from the idempotency key."""
        return f"runner:{self.workflow_run_id}:{self.job_id}"


__all__: list[str] = ["ModelRunnerUsageEvent"]
