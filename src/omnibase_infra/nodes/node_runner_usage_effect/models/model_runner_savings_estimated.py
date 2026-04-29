# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Savings estimate emitted for runner cost avoidance."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class ModelRunnerSavingsEstimated(BaseModel):
    """Runner savings estimate payload for SAVINGS_ESTIMATED."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    schema_version: str = Field(default="1.0")
    source_event_id: str = Field(..., min_length=1)
    event_timestamp: datetime
    session_id: str = Field(..., min_length=1)
    workflow_run_id: str = Field(..., min_length=1)
    job_id: str = Field(..., min_length=1)
    runner_minutes: Decimal = Field(..., ge=Decimal("0"))
    model_local: str = Field(default="self_hosted_runner")
    model_cloud_baseline: str = Field(default="github_runner")
    local_cost_usd: Decimal = Field(..., ge=Decimal("0"))
    cloud_cost_usd: Decimal = Field(..., ge=Decimal("0"))
    savings_usd: Decimal = Field(..., ge=Decimal("0"))
    repo_name: str | None = Field(default=None)
    machine_id: str | None = Field(default=None)
    runner_name: str | None = Field(default=None)
    workflow_name: str | None = Field(default=None)
    pricing_manifest_version: str = Field(..., min_length=1)
    estimation_method: str = Field(default="runner_cost_manifest_v1")

    def to_kafka_payload(self) -> dict[str, object]:
        """Serialize to JSON-compatible field values."""
        return self.model_dump(mode="json")


__all__: list[str] = ["ModelRunnerSavingsEstimated"]
