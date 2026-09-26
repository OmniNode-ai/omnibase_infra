# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""One read of the deploy agent's HTTP surface (OMN-19811)."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelDeployAgentSnapshot(BaseModel):
    """What the lane's deploy agent reported at ``observed_at``.

    Read from the agent's own ``/health`` (``active_job``, ``last_result``,
    ``state``) and ``/job/{correlation_id}`` (the last job's ``accepted_at``)
    -- the same HTTP surface the ``verify-lane-converged`` job in
    ``runtime-rebuild-trigger.yml`` polls through
    ``scripts/ci/check_dev_lane_staleness.py``. No second source of deploy
    truth is introduced.

    ``readable=False`` means nothing below it is evidence. A reader must never
    treat an unreadable snapshot as "no deploy".
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    observed_at: datetime = Field(..., description="When this read was taken (UTC).")
    readable: bool = Field(
        ...,
        description="False when the agent could not be read; nothing else is then evidence.",
    )
    error: str = Field(default="", description="Sanitized read error when unreadable.")
    state: str = Field(
        default="",
        description="The agent's own state: idle, deploying or settling.",
    )
    active_correlation_id: UUID | None = Field(
        default=None, description="Correlation id of the job in flight, None when none."
    )
    active_accepted_at: datetime | None = Field(
        default=None, description="When the in-flight job was accepted."
    )
    last_correlation_id: UUID | None = Field(
        default=None, description="Correlation id of the most recently completed job."
    )
    last_accepted_at: datetime | None = Field(
        default=None,
        description=(
            "When the most recently completed job was accepted, from "
            "/job/{correlation_id}. None when that read failed; the job then "
            "counts as overlapping any window its completion falls in."
        ),
    )
    last_completed_at: datetime | None = Field(
        default=None, description="When the most recently completed job completed."
    )
    last_settling: bool = Field(
        default=False,
        description=(
            "The agent's post-terminal host work for the last job is still "
            "running (OMN-18636 AC5). A settling lane is not converged."
        ),
    )

    queued_commands: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Deploy commands waiting behind the in-flight job, from the agent's "
            "/queue commands_ahead (the surface check_dev_lane_staleness.py "
            "reads). None when /queue was unreadable, reported its depth "
            "unknown, or served a lag sample older than the 120 s bound -- an "
            "unread queue is not an empty one, and is not counted either way."
        ),
    )

    @property
    def busy(self) -> bool:
        """The agent is doing, or is about to do, something to the lane."""
        return bool(
            self.active_correlation_id is not None
            or self.last_settling
            or (self.state and self.state != "idle")
            or (self.queued_commands or 0) > 0
        )


__all__ = ["ModelDeployAgentSnapshot"]
