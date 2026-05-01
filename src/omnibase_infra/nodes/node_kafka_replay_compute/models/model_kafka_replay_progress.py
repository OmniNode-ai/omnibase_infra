# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Progress model for Kafka replay compute."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelKafkaReplayProgress(BaseModel):
    """Replay progress for one topic partition."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    topic: str = Field(..., min_length=1, description="Kafka topic being replayed.")
    partition: int = Field(..., ge=0, description="Kafka partition being replayed.")
    current_offset: int = Field(..., ge=0, description="Current Kafka offset.")
    events_replayed_so_far: int = Field(
        ..., ge=0, description="Number of events replayed by this run."
    )
    eta_seconds: float | None = Field(
        default=None,
        ge=0,
        description="Optional estimated seconds remaining for the run.",
    )


__all__ = ["ModelKafkaReplayProgress"]
