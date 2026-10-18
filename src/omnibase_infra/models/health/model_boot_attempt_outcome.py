# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Outcome of one runtime boot attempt — OMN-18550.

The request the boot-failure ledger handler consumes: what a runtime process
did between start and exit, judged against the bounds its identity declares.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.health.model_boot_identity import ModelBootIdentity


class ModelBootAttemptOutcome(BaseModel):
    """A finished boot attempt."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    identity: ModelBootIdentity = Field(
        ..., description="What the streak is counted for."
    )
    exit_code: int = Field(..., description="Exit code the process ended with.")
    cause: str = Field(default="", description="Sanitized cause of a failing exit.")
    uptime_seconds: float = Field(..., ge=0, description="Seconds from start to exit.")
    healthy_after_seconds: float = Field(
        ...,
        ge=0,
        description=(
            "A process that ran at least this long before failing was a healthy "
            "run, so its failure starts a new streak instead of extending one."
        ),
    )
    observed_at: datetime = Field(..., description="UTC time of the exit.")


__all__: list[str] = ["ModelBootAttemptOutcome"]
