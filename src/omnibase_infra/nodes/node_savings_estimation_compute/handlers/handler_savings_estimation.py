# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Handler for computing token and cost savings from injection effectiveness.

Takes injection effectiveness data (utilization scores, pattern counts,
tokens saved) and computes dollar savings using tiered model pricing.
Produces a :class:`ModelSavingsEstimate` ready for Kafka emission to
``onex.evt.omnibase-infra.savings-estimated.v1``.

Savings Formula:
    For each effectiveness entry:
        cost_saved = tokens_saved * price_per_token
    where price_per_token depends on model tier and token type (input/output):
        - Opus input: $15 / 1M tokens
        - Opus output: $75 / 1M tokens
        - Sonnet input: $3 / 1M tokens
        - Sonnet output: $15 / 1M tokens

    direct_confidence = weighted average of utilization_scores

Related Tickets:
    - OMN-6964: Token savings emitter
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from uuid import uuid4

from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
)
from omnibase_infra.nodes.node_savings_estimation_compute.models.enum_model_tier import (
    MODEL_PRICING_INPUT,
    MODEL_PRICING_OUTPUT,
    PRICING_MANIFEST_VERSION,
    TOKENS_PER_MILLION,
    EnumModelTier,
)
from omnibase_infra.nodes.node_savings_estimation_compute.models.enum_savings_category import (
    EnumSavingsCategory,
)
from omnibase_infra.nodes.node_savings_estimation_compute.models.model_effectiveness_entry import (
    ModelEffectivenessEntry,
)
from omnibase_infra.nodes.node_savings_estimation_compute.models.model_savings_category import (
    ModelSavingsCategory,
)
from omnibase_infra.nodes.node_savings_estimation_compute.models.model_savings_estimate import (
    ModelSavingsEstimate,
)
from omnibase_infra.nodes.node_savings_estimation_compute.models.model_savings_estimation_input import (
    ModelSavingsEstimationInput,
)

logger = logging.getLogger(__name__)


# Category mapping: patterns_count ranges to category names
# This provides a basic breakdown; can be refined with richer metadata later.
class CategoryAccumulator:
    """Mutable accumulator for per-category savings data."""

    __slots__ = ("confidence_sum", "count", "savings_usd", "tokens_saved")

    def __init__(self) -> None:
        self.savings_usd: float = 0.0
        self.tokens_saved: int = 0
        self.confidence_sum: float = 0.0
        self.count: int = 0


CATEGORY_THRESHOLDS: list[tuple[EnumSavingsCategory, int]] = [
    (EnumSavingsCategory.ARCHITECTURE, 10),  # >= 10 patterns = architecture-level
    (EnumSavingsCategory.FILE, 3),  # >= 3 patterns = file-level
    (EnumSavingsCategory.TOOL, 0),  # < 3 patterns = tool-level
]


def _classify_category(patterns_count: int) -> EnumSavingsCategory:
    """Classify an effectiveness entry into a savings category.

    Args:
        patterns_count: Number of patterns injected.

    Returns:
        Savings category enum value.
    """
    for cat, threshold in CATEGORY_THRESHOLDS:
        if patterns_count >= threshold:
            return cat
    return EnumSavingsCategory.TOOL


def _price_per_token(tier: EnumModelTier, is_output: bool) -> float:
    """Get the price per token for a given tier and token type.

    Args:
        tier: Model tier (opus or sonnet).
        is_output: Whether these are output tokens.

    Returns:
        Price per single token in USD.
    """
    pricing = MODEL_PRICING_OUTPUT if is_output else MODEL_PRICING_INPUT
    return pricing[tier] / TOKENS_PER_MILLION


class HandlerSavingsEstimation:
    """Compute token and cost savings from injection effectiveness data.

    This handler is stateless and pure. It receives effectiveness data,
    computes savings using the tiered pricing model, and returns a
    ``ModelSavingsEstimate`` ready for Kafka emission.

    The handler does NOT publish events itself (handlers must not have
    direct event bus access per ONEX conventions). The caller is
    responsible for publishing the result to Kafka.

    Note:
        This is an infrastructure handler (``INFRA_HANDLER``) with
        ``NONDETERMINISTIC_COMPUTE`` category because it uses
        ``uuid4()`` and ``datetime.now()``.
    """

    # ------------------------------------------------------------------
    # Handler classification
    # ------------------------------------------------------------------

    @property
    def handler_id(self) -> str:
        """Unique handler identifier."""
        return "handler-savings-estimation"

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role: infrastructure handler.

        Returns:
            EnumHandlerType.INFRA_HANDLER
        """
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Behavioral classification: non-deterministic computation.

        Returns:
            EnumHandlerTypeCategory.NONDETERMINISTIC_COMPUTE
        """
        return EnumHandlerTypeCategory.NONDETERMINISTIC_COMPUTE

    # ------------------------------------------------------------------
    # Core handle method
    # ------------------------------------------------------------------

    async def handle(
        self,
        estimation_input: ModelSavingsEstimationInput,
    ) -> ModelSavingsEstimate:
        """Compute savings from injection effectiveness data.

        Args:
            estimation_input: Effectiveness measurements from one or
                more sessions.

        Returns:
            A ``ModelSavingsEstimate`` with computed savings, ready
            for Kafka emission.
        """
        return self.compute_savings(estimation_input)

    def compute_savings(
        self,
        estimation_input: ModelSavingsEstimationInput,
    ) -> ModelSavingsEstimate:
        """Synchronous savings computation.

        Exposed as a separate method for testing convenience and for
        callers that don't need async.

        Args:
            estimation_input: Effectiveness measurements.

        Returns:
            Computed savings estimate.
        """
        entries = estimation_input.effectiveness_entries

        # Accumulate per-category savings
        category_accum: dict[EnumSavingsCategory, CategoryAccumulator] = {}
        total_direct_tokens = 0
        total_direct_usd = 0.0
        weighted_confidence_sum = 0.0
        total_tokens_for_weight = 0

        for entry in entries:
            savings_usd = self._entry_savings_usd(entry)
            cat_name = _classify_category(entry.patterns_count)

            if cat_name not in category_accum:
                category_accum[cat_name] = CategoryAccumulator()

            acc = category_accum[cat_name]
            acc.savings_usd += savings_usd
            acc.tokens_saved += entry.tokens_saved
            acc.confidence_sum += entry.utilization_score
            acc.count += 1

            total_direct_tokens += entry.tokens_saved
            total_direct_usd += savings_usd

            # Weight confidence by tokens_saved (more tokens = more signal)
            weighted_confidence_sum += entry.utilization_score * entry.tokens_saved
            total_tokens_for_weight += entry.tokens_saved

        # Build category models
        categories = tuple(
            ModelSavingsCategory(
                category=cat_name,
                savings_usd=round(acc.savings_usd, 10),
                tokens_saved=acc.tokens_saved,
                confidence=round(acc.confidence_sum / acc.count, 4)
                if acc.count > 0
                else 0.0,
            )
            for cat_name, acc in sorted(category_accum.items())
        )

        # Compute confidence scores
        direct_confidence = (
            round(weighted_confidence_sum / total_tokens_for_weight, 4)
            if total_tokens_for_weight > 0
            else 0.0
        )
        heuristic_confidence = (
            round(sum(e.utilization_score for e in entries) / len(entries), 4)
            if entries
            else 0.0
        )

        # Actual cost: tokens consumed * price for the dominant model tier
        dominant_tier = self._dominant_tier(entries)
        actual_cost = estimation_input.actual_total_tokens * _price_per_token(
            dominant_tier, is_output=False
        )

        estimate = ModelSavingsEstimate(
            source_event_id=uuid4(),
            session_id=estimation_input.session_id,
            correlation_id=estimation_input.correlation_id,
            actual_total_tokens=estimation_input.actual_total_tokens,
            actual_cost_usd=round(actual_cost, 10),
            actual_model_id=estimation_input.actual_model_id,
            counterfactual_model_id=None,
            direct_savings_usd=round(total_direct_usd, 10),
            direct_tokens_saved=total_direct_tokens,
            estimated_total_savings_usd=round(total_direct_usd, 10),
            estimated_total_tokens_saved=total_direct_tokens,
            categories=categories,
            direct_confidence=direct_confidence,
            heuristic_confidence_avg=heuristic_confidence,
            estimation_method="tiered_attribution_v1",
            is_measured=True,
            completeness_status="complete",
            pricing_manifest_version=PRICING_MANIFEST_VERSION,
            timestamp_iso=datetime.now(UTC).isoformat(),
        )

        logger.info(
            "Savings estimation completed: session=%s, "
            "direct_savings=$%.6f, tokens_saved=%d, "
            "confidence=%.4f (cid=%s)",
            estimation_input.session_id,
            total_direct_usd,
            total_direct_tokens,
            direct_confidence,
            estimation_input.correlation_id,
        )

        return estimate

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _entry_savings_usd(entry: ModelEffectivenessEntry) -> float:
        """Compute USD savings for a single effectiveness entry.

        Args:
            entry: Single effectiveness measurement.

        Returns:
            Savings in USD.
        """
        ppt = _price_per_token(entry.model_tier, entry.is_output_tokens)
        return entry.tokens_saved * ppt

    @staticmethod
    def _dominant_tier(
        entries: tuple[ModelEffectivenessEntry, ...],
    ) -> EnumModelTier:
        """Determine the dominant model tier from entries.

        Returns the tier with the most tokens_saved. Defaults to OPUS
        if entries are empty.

        Args:
            entries: Effectiveness entries.

        Returns:
            Dominant model tier.
        """
        tier_tokens: dict[EnumModelTier, int] = {}
        for entry in entries:
            tier_tokens[entry.model_tier] = (
                tier_tokens.get(entry.model_tier, 0) + entry.tokens_saved
            )
        if not tier_tokens:
            return EnumModelTier.OPUS
        return max(tier_tokens, key=lambda t: tier_tokens[t])


__all__: list[str] = ["HandlerSavingsEstimation"]
