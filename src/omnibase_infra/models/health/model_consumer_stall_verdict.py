# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The verdict of one consumer stall evaluation (OMN-18640)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.health.enum_consumer_stall_reason import (
    EnumConsumerStallReason,
)


class ModelConsumerStallVerdict(BaseModel):
    """Outcome of evaluating one :class:`ModelConsumerPollObservation`.

    ``should_rejoin`` is DERIVED from ``reason`` and ``consecutive_stalls`` by
    the validator below rather than set by the caller, so a verdict cannot
    claim a rejoin its own reason does not support. This mirrors the derived
    verdict on ``ModelLabPassReceipt``: a field an emitter can state wrongly is
    a field that will eventually be stated wrongly.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    reason: EnumConsumerStallReason = Field(
        ..., description="Why this evaluation reached its conclusion."
    )
    consecutive_stalls: int = Field(
        ...,
        description=(
            "How many consecutive evaluations, including this one, have found "
            "a stall signature. Reset to zero by any non-stalled evaluation."
        ),
        ge=0,
    )
    required_consecutive_stalls: int = Field(
        ...,
        description=(
            "Confirmations required before a rejoin is ordered. One transient "
            "reading is not a wedge."
        ),
        ge=1,
    )

    @property
    def is_stalled(self) -> bool:
        """True when this single evaluation found a stall signature."""
        return self.reason in {
            EnumConsumerStallReason.STALLED_BACKLOG_NOT_DRAINING,
            EnumConsumerStallReason.STALLED_NO_ASSIGNMENT,
        }

    @property
    def should_rejoin(self) -> bool:
        """True when the stall has been confirmed enough times to act on."""
        return self.is_stalled and (
            self.consecutive_stalls >= self.required_consecutive_stalls
        )


__all__ = ["ModelConsumerStallVerdict"]
