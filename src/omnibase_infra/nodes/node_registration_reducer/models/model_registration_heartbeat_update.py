# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Typed column set model for heartbeat liveness-extension UPDATE."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ModelRegistrationHeartbeatUpdate(BaseModel):
    """Typed column set for heartbeat liveness-extension UPDATE.

    Produced by ``RegistrationReducerService.decide_heartbeat()`` when extending
    a node's liveness deadline after receiving a heartbeat.

    Field names match SQL column names in ``registration_projections`` exactly.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    last_heartbeat_at: datetime = Field(..., description="Heartbeat event timestamp.")
    liveness_deadline: datetime = Field(..., description="Extended liveness deadline.")
    updated_at: datetime = Field(..., description="Timestamp of this update.")


__all__: list[str] = [
    "ModelRegistrationHeartbeatUpdate",
]
