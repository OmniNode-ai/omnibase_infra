# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Payload model for the upsert-metrics intent.

This payload drives HandlerUpsertMetrics which performs an upsert into
delta_metrics_by_model ON CONFLICT DO UPDATE with incremented counters.

Related Tickets:
    - OMN-3142: NodeDeltaMetricsEffect implementation
    - Migration 040: delta_metrics_by_model table
"""

from __future__ import annotations

from datetime import date
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelPayloadUpsertMetrics(BaseModel):
    """Payload for the delta-metrics upsert intent.

    Carries the fields needed to increment a single rollup row in
    delta_metrics_by_model. Each payload represents one completed bundle
    contributing to the rollup counters.

    Attributes:
        intent_type: Routing key -- always "delta_metrics.upsert_metrics".
        correlation_id: Correlation UUID from the originating context.
        coding_model: LLM model identifier for this bundle.
        subsystem: Subsystem classification for this bundle.
        outcome: Final PR outcome (merged, reverted, closed).
        gate_decision: Merge-gate verdict (PASS, WARN, QUARANTINE).
        is_fix_pr: Whether this bundle is a fix-PR.
        gate_violation_count: Number of gate violations in this bundle.
        period_start: Start date (inclusive) of the rollup period.
        period_end: End date (inclusive) of the rollup period.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    intent_type: Literal["delta_metrics.upsert_metrics"] = Field(
        default="delta_metrics.upsert_metrics",
        description="Routing key for this intent.",
    )
    correlation_id: UUID = Field(
        ...,
        description="Correlation UUID from the originating pipeline/session.",
    )
    coding_model: str = Field(
        ...,
        description="LLM model identifier (e.g. 'claude-opus-4-20250514').",
    )
    subsystem: str = Field(
        ...,
        description="Subsystem classification (e.g. 'omnibase_infra').",
    )
    outcome: Literal["merged", "reverted", "closed"] = Field(
        ...,
        description="Final PR outcome.",
    )
    gate_decision: Literal["PASS", "WARN", "QUARANTINE"] = Field(
        ...,
        description="Merge-gate verdict.",
    )
    is_fix_pr: bool = Field(
        default=False,
        description="Whether this bundle is a fix-PR (stabilization tax).",
    )
    gate_violation_count: int = Field(
        default=0,
        description="Number of gate violations in this bundle.",
    )
    period_start: date = Field(
        ...,
        description="Start date (inclusive) of the rollup period.",
    )
    period_end: date = Field(
        ...,
        description="End date (inclusive) of the rollup period.",
    )


__all__: list[str] = ["ModelPayloadUpsertMetrics"]
