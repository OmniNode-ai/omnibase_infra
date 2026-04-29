# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Waste finding model for projection and Kafka emission."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

WasteSeverity = Literal["LOW", "MEDIUM", "HIGH"]


class ModelWasteFinding(BaseModel):
    """Deterministic finding projected to waste_findings and Kafka."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    session_id: str = Field(min_length=1)
    rule_id: str = Field(min_length=1, max_length=64)
    severity: WasteSeverity
    waste_tokens: int = Field(ge=0)
    waste_cost_usd: float = Field(ge=0.0)
    evidence: dict[str, object] = Field(default_factory=dict)
    evidence_hash: str = Field(min_length=64, max_length=64)
    dedup_key: str = Field(min_length=1)
    recommendation: str | None = None
    repo_name: str | None = None
    machine_id: str | None = None
    detected_at: datetime

    @field_validator("detected_at")
    @classmethod
    def validate_detected_at_tz_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("detected_at must be timezone-aware")
        return value

    def to_db_row(self) -> dict[str, object]:
        """Serialize to a waste_findings row."""
        return self.model_dump(mode="json")


__all__ = ["ModelWasteFinding", "WasteSeverity"]
