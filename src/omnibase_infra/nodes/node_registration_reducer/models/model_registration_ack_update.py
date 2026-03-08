# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Typed column set model for ACK state-transition UPDATE."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ModelRegistrationAckUpdate(BaseModel):
    """Typed column set for ACK state-transition UPDATE.

    Produced by ``RegistrationReducerService.decide_ack()`` when transitioning
    a registration projection from AWAITING_ACK to ACTIVE.

    Field names match SQL column names in ``registration_projections`` exactly.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    current_state: str = Field(..., description="New FSM state value.")
    liveness_deadline: datetime = Field(..., description="Initial liveness deadline.")
    updated_at: datetime = Field(..., description="Timestamp of this state change.")


__all__: list[str] = [
    "ModelRegistrationAckUpdate",
]
