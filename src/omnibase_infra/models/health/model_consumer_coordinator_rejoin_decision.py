# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pure coordinator rejoin decision (OMN-21337)."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.enums.enum_consumer_coordinator_rejoin_reason import (
    EnumConsumerCoordinatorRejoinReason,
)


class ModelConsumerCoordinatorRejoinDecision(BaseModel):
    """Recovery action and independent once-per-episode alarm."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reason: EnumConsumerCoordinatorRejoinReason = Field(
        ..., description="Why this evaluation reached its decision."
    )
    rejoin: bool = Field(..., description="Rebuild the consumer client now.")
    alarm: bool = Field(..., description="Raise the episode's one alarm now.")
    alarm_message: str = Field(
        default="",
        description="Non-empty exactly when alarm is True; identifies the group, broker, topic, duration and failed attempts.",
    )

    @model_validator(mode="after")
    def validate_alarm_message(self) -> Self:
        """Keep the message consistent with the alarm action."""
        if bool(self.alarm_message) != self.alarm:
            raise ValueError(
                "alarm_message must be non-empty exactly when alarm is True"
            )
        return self


__all__ = ["ModelConsumerCoordinatorRejoinDecision"]
