# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""One evaluated DLQ topic — the materializable projection row (OMN-16769 AC1)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_dlq_depth_evaluate_compute.models.enum_dlq_depth_verdict import (
    EnumDlqDepthVerdict,
)


class ModelDlqTopicVerdict(BaseModel):
    """One evaluated DLQ topic — the projection row shape (AC1, depth/rate half)."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    topic: str = Field(..., min_length=1)
    partition_count: int = Field(..., ge=1)

    # --- Broker facts, carried through unmodified so the row is auditable. ---
    log_start_offset: int = Field(..., ge=0)
    high_watermark: int = Field(..., ge=0)
    window_start_offset: int = Field(..., ge=0)

    # --- Derived metrics. ---
    retained_depth: int = Field(
        ...,
        ge=0,
        description="high_watermark - log_start_offset. Records still on disk.",
    )
    arrivals_in_window: int = Field(
        ...,
        ge=0,
        description="high_watermark - window_start_offset. The primary signal.",
    )
    arrivals_per_minute: float = Field(
        ...,
        ge=0.0,
        description="arrivals_in_window normalized to a per-minute rate, for comparability across window sizes.",
    )

    # --- Judgement. ---
    max_arrivals_per_window: int = Field(
        ...,
        ge=0,
        description="The bound this topic was judged against (default or override).",
    )
    override_reason: str = Field(
        default="",
        description="Justification, when this topic was judged against an override.",
    )
    verdict: EnumDlqDepthVerdict = Field(...)

    window_seconds: int = Field(..., ge=60)
    evaluated_at: datetime = Field(...)


__all__ = ["ModelDlqTopicVerdict"]
