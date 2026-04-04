# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Savings estimate output model for Kafka emission.

Related Tickets:
    - OMN-6964: Token savings emitter
    - OMN-7494: Heuristic savings and counterfactual model
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_savings_estimation_compute.models.enum_model_tier import (
    PRICING_MANIFEST_VERSION,
)
from omnibase_infra.nodes.node_savings_estimation_compute.models.model_savings_category import (
    ModelSavingsCategory,
)


class ModelSavingsEstimate(BaseModel):
    """Computed savings estimate for Kafka emission.

    Every field maps to a column in the ``savings_estimates`` table
    consumed by the omnidash read-model projection. The field names
    use snake_case to match the Kafka event payload that the omnidash
    consumer expects (see ``projectSavingsEstimated`` in
    ``omnibase-infra-projections.ts``).

    Note: session_id uses ``str`` because it is a free-form session
    identifier (not a UUID). source_event_id and correlation_id use
    ``UUID`` and are serialized to string via ``model_dump(mode="json")``.

    Attributes:
        source_event_id: Unique event ID for idempotent upsert.
        session_id: Session this estimate covers.
        correlation_id: Correlation ID for tracing.
        schema_version: Schema version string.
        actual_total_tokens: Tokens actually consumed.
        actual_cost_usd: Actual cost in USD.
        actual_model_id: Model that was actually used.
        counterfactual_model_id: Model used for counterfactual pricing.
        direct_savings_usd: Directly measured savings.
        direct_tokens_saved: Directly measured tokens saved.
        estimated_total_savings_usd: Total estimated savings including heuristics.
        estimated_total_tokens_saved: Total estimated tokens saved.
        categories: Breakdown by category.
        direct_confidence: Confidence in direct measurements (0.0-1.0).
        heuristic_confidence_avg: Average confidence across heuristic estimates.
        estimation_method: Algorithm identifier.
        treatment_group: A/B test group if applicable.
        is_measured: Whether savings are directly measured vs estimated.
        completeness_status: Data completeness ('complete', 'partial').
        pricing_manifest_version: Pricing table version used.
        timestamp_iso: Event timestamp in ISO format.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    source_event_id: UUID = Field(
        default_factory=uuid4,
        description="Unique event ID for idempotent upsert",
    )
    session_id: str = Field(
        ..., min_length=1, description="Free-form session identifier (not a UUID)"
    )
    correlation_id: UUID = Field(
        default_factory=uuid4,
        description="Correlation ID for tracing",
    )
    schema_version: str = Field(default="1.0", description="Schema version")
    actual_total_tokens: int = Field(
        default=0, ge=0, description="Total tokens consumed"
    )
    actual_cost_usd: float = Field(
        default=0.0, ge=0.0, description="Actual cost in USD"
    )
    actual_model_id: str | None = Field(default=None, description="Model actually used")
    counterfactual_model_id: str | None = Field(
        default=None, description="Counterfactual model for pricing"
    )
    direct_savings_usd: float = Field(
        default=0.0, ge=0.0, description="Directly measured savings"
    )
    heuristic_savings_usd: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Estimated heuristic savings from validator catches and avoided rework. "
            "This is an estimate, not measured cost accounting."
        ),
    )
    direct_tokens_saved: int = Field(
        default=0, ge=0, description="Directly measured tokens saved"
    )
    estimated_total_savings_usd: float = Field(
        default=0.0, ge=0.0, description="Total estimated savings (direct + heuristic)"
    )
    estimated_total_tokens_saved: int = Field(
        default=0, ge=0, description="Total estimated tokens saved"
    )
    categories: tuple[ModelSavingsCategory, ...] = Field(
        default_factory=tuple, description="Savings by category"
    )
    direct_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Direct measurement confidence"
    )
    heuristic_confidence_avg: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Average heuristic confidence"
    )
    estimation_method: str = Field(
        default="tiered_attribution_v1", description="Estimation algorithm"
    )
    treatment_group: str | None = Field(default=None, description="A/B test group")
    is_measured: bool = Field(default=False, description="True if directly measured")
    completeness_status: str = Field(
        default="complete", description="Data completeness"
    )
    pricing_manifest_version: str = Field(
        default=PRICING_MANIFEST_VERSION, description="Pricing version"
    )
    timestamp_iso: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="Event timestamp ISO",
    )

    def to_kafka_payload(self) -> dict[str, object]:
        """Serialize to a dict suitable for Kafka JSON production.

        Converts categories from Pydantic models to plain dicts so the
        payload is JSON-serializable without custom encoders.

        Returns:
            Dict with all fields, categories as list of dicts.
        """
        data = self.model_dump(mode="json")
        data["categories"] = [c.model_dump(mode="json") for c in self.categories]
        return data


__all__: list[str] = ["ModelSavingsEstimate"]
