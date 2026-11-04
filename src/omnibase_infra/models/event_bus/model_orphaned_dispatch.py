# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""What an abandoned subscriber dispatch was doing at its deadline (OMN-19355)."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict

__all__ = ["ModelOrphanedDispatch"]


class ModelOrphanedDispatch(BaseModel):
    """What an abandoned dispatch was doing when its deadline expired."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    topic: str
    consumer_group: str
    subscription_key: str
    correlation_id: UUID
    record_coordinate: tuple[int, int] | None
    started_at: float
    #: Loop time the dispatch was abandoned; the OMN-20464 degraded grace runs
    #: from here, not from ``started_at``.
    abandoned_at: float

    def describe(self, now: float) -> str:
        partition, offset = self.record_coordinate or (None, None)
        return (
            f"{self.topic} partition={partition} offset={offset} "
            f"subscription={self.subscription_key} age={now - self.started_at:.0f}s"
        )
