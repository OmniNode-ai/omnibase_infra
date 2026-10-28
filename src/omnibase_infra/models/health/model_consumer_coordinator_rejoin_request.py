# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Inputs to the pure coordinator rejoin decision (OMN-21337)."""

from pydantic import BaseModel, ConfigDict, Field


class ModelConsumerCoordinatorRejoinRequest(BaseModel):
    """Measured coordinator episode and declared recovery bounds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    topic: str = Field(..., description="Topic being consumed.")
    consumer_group: str = Field(..., description="Effective Kafka consumer group id.")
    broker: str = Field(..., description="Sanitized bootstrap servers; may be empty.")
    coordinator_known: bool = Field(
        ..., description="The client knows its group coordinator."
    )
    broker_reachable: bool = Field(
        ..., description="Partition leaders answer offset probes."
    )
    coordinator_unavailable_seconds: float = Field(
        ...,
        ge=0,
        description="Seconds without a coordinator while leaders answer; restarts when the broker is unreachable.",
    )
    seconds_since_last_rejoin: float | None = Field(
        ...,
        ge=0,
        description="Seconds since the last client rebuild, or None before any rebuild.",
    )
    failed_rejoins: int = Field(
        ...,
        ge=0,
        description="Attempts this episode that failed to start or required another rejoin.",
    )
    alarm_raised: bool = Field(
        ..., description="An alarm was already raised this episode."
    )
    unavailable_bound_seconds: float = Field(
        ..., gt=0, description="Allowed seconds without a coordinator."
    )
    rejoin_cooldown_seconds: float = Field(
        ..., ge=0, description="Minimum seconds between coordinator rejoins."
    )


__all__ = ["ModelConsumerCoordinatorRejoinRequest"]
