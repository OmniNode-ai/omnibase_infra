# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Session-window input model for waste detection."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.nodes.node_waste_detection_compute.models.model_waste_call import (
    ModelWasteCall,
)


class ModelWasteDetectionInput(BaseModel):
    """Session-window input for waste detection."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    session_id: str = Field(min_length=1)
    calls: tuple[ModelWasteCall, ...] = Field(default_factory=tuple)
    detected_at: datetime

    @field_validator("detected_at")
    @classmethod
    def validate_detected_at_tz_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("detected_at must be timezone-aware")
        return value


__all__ = ["ModelWasteDetectionInput"]
