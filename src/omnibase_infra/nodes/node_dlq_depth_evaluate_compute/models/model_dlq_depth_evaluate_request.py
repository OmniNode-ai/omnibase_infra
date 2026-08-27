# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Request model for the pure DLQ depth/arrival evaluation (OMN-16769)."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.nodes.node_dlq_depth_evaluate_compute.models.model_dlq_threshold_policy import (
    ModelDlqThresholdPolicy,
)
from omnibase_infra.nodes.node_dlq_depth_evaluate_compute.models.model_dlq_topic_observation import (
    ModelDlqTopicObservation,
)


class ModelDlqDepthEvaluateRequest(BaseModel):
    """Observations + bounds + clock. Everything the verdict depends on."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Evaluation run correlation ID.")
    observations: tuple[ModelDlqTopicObservation, ...] = Field(
        default=(),
        description="Read-only broker observations, one row per DLQ topic.",
    )
    policy: ModelDlqThresholdPolicy = Field(
        default_factory=ModelDlqThresholdPolicy,
        description="Contract-pinned alert bounds.",
    )
    evaluated_at: datetime = Field(
        ...,
        description=(
            "Wall-clock instant of the probe, INJECTED BY THE CALLER — the "
            "handler never reads a clock. This is what makes the evaluation a "
            "pure function of its input and therefore replayable: the same "
            "request always yields the same result. Must be timezone-aware."
        ),
    )

    @field_validator("evaluated_at")
    @classmethod
    def _require_timezone_aware(cls, value: datetime) -> datetime:
        """A naive timestamp on a projection row is a silent correctness bug."""
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError(
                "evaluated_at must be timezone-aware — a naive timestamp would "
                "be materialized into the projection with an unknowable offset."
            )
        return value

    @field_validator("observations")
    @classmethod
    def _reject_duplicate_topics(
        cls, value: tuple[ModelDlqTopicObservation, ...]
    ) -> tuple[ModelDlqTopicObservation, ...]:
        """One row per topic — a duplicate means the probe double-counted."""
        topics = [observation.topic for observation in value]
        duplicates = sorted({topic for topic in topics if topics.count(topic) > 1})
        if duplicates:
            raise ValueError(
                f"Duplicate observations for topic(s): {', '.join(duplicates)} — "
                "the probe enumerated the same topic twice."
            )
        return value


__all__ = ["ModelDlqDepthEvaluateRequest"]
