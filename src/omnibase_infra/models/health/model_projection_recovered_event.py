# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""ModelProjectionRecoveredEvent — emitted when a degraded projection returns within SLA."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ModelProjectionRecoveredEvent(BaseModel):
    """Emitted when a previously-degraded projection returns within its SLA."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    projection_name: str = Field(..., min_length=1)
    sla_seconds: int = Field(..., gt=0)
    actual_staleness_seconds: float = Field(..., ge=0)
    observed_at: datetime = Field(...)
    source_contract_hash: str = Field(..., min_length=1)


__all__: list[str] = ["ModelProjectionRecoveredEvent"]
