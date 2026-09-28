# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Abandoned consume-loop dispatches this runtime still holds (OMN-19355).

When a subscriber callback outlives the per-dispatch deadline, the consume loop
quarantines the record and moves on, but it cannot stop the handler. A Python
thread cannot be killed, and a projection handler runs its blocking work in a
``to_thread`` worker, so the abandoned dispatch keeps its worker thread and its
slot in the runtime-wide projection gate until it returns or the process exits.

This model is what the bus reports about those dispatches. ``status`` is
derived, not stored, for the same reason ``ModelConsumerSyncStatus.ready`` is:
a health surface that states a verdict it did not compute from its own numbers
is the failure this family of tickets keeps finding.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelDispatchDeadlineStatus(BaseModel):
    """Per-dispatch deadline state of one event bus."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    deadline_seconds: float = Field(
        ...,
        gt=0.0,
        description="The per-dispatch deadline the consume loop applies.",
    )
    orphan_limit: int = Field(
        ...,
        ge=1,
        description="Abandoned dispatches at which the bus reports UNHEALTHY.",
    )
    orphaned_dispatches: int = Field(
        ...,
        ge=0,
        description=(
            "Dispatches abandoned at their deadline that have not returned "
            "yet. Each one still holds whatever the handler held: for a "
            "projection, a worker thread and a projection gate slot."
        ),
    )
    deadline_expiries_total: int = Field(
        ...,
        ge=0,
        description="Dispatches abandoned at their deadline since the bus started.",
    )
    oldest_orphan_age_seconds: float = Field(
        ...,
        ge=0.0,
        description="Seconds since the oldest still-running dispatch started.",
    )
    orphans: tuple[str, ...] = Field(
        ...,
        description=(
            "One line per still-running abandoned dispatch, oldest first: "
            "topic, partition, offset, subscription and age."
        ),
    )

    @property
    def status(self) -> Literal["healthy", "degraded", "unhealthy"]:
        """``unhealthy`` at the limit, ``degraded`` below it, else ``healthy``."""
        if self.orphaned_dispatches >= self.orphan_limit:
            return "unhealthy"
        if self.orphaned_dispatches > 0:
            return "degraded"
        return "healthy"


__all__ = ["ModelDispatchDeadlineStatus"]
