# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for HandlerSavingsEstimation.

Tests cover:
    - Basic savings computation with Opus pricing
    - Sonnet pricing tier
    - Output token pricing (higher rate)
    - Mixed tier entries
    - Category classification (architecture/file/tool)
    - Confidence calculation (weighted by tokens)
    - Single entry edge case
    - Kafka payload serialization

Related Tickets:
    - OMN-6964: Token savings emitter
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_savings_estimation_compute.handlers.handler_savings_estimation import (
    HandlerSavingsEstimation,
    _classify_category,
)
from omnibase_infra.nodes.node_savings_estimation_compute.models import (
    MODEL_PRICING_INPUT,
    MODEL_PRICING_OUTPUT,
    TOKENS_PER_MILLION,
    EnumModelTier,
    EnumSavingsCategory,
    ModelEffectivenessEntry,
    ModelSavingsEstimationInput,
)


@pytest.fixture
def handler() -> HandlerSavingsEstimation:
    """Create a fresh handler instance."""
    return HandlerSavingsEstimation()


@pytest.fixture
def opus_entries() -> tuple[ModelEffectivenessEntry, ...]:
    """Two Opus input-token effectiveness entries."""
    return (
        ModelEffectivenessEntry(
            utilization_score=0.85,
            patterns_count=12,
            tokens_saved=1500,
            model_tier=EnumModelTier.OPUS,
        ),
        ModelEffectivenessEntry(
            utilization_score=0.72,
            patterns_count=8,
            tokens_saved=900,
            model_tier=EnumModelTier.OPUS,
        ),
    )


@pytest.fixture
def estimation_input(
    opus_entries: tuple[ModelEffectivenessEntry, ...],
) -> ModelSavingsEstimationInput:
    """Standard estimation input with Opus entries."""
    return ModelSavingsEstimationInput(
        session_id="test-session-001",
        correlation_id=uuid4(),
        effectiveness_entries=opus_entries,
        actual_total_tokens=50000,
        actual_model_id="claude-opus-4-6",
    )


class TestHandlerClassification:
    """Verify handler metadata properties."""

    def test_handler_id(self, handler: HandlerSavingsEstimation) -> None:
        assert handler.handler_id == "handler-savings-estimation"

    def test_handler_type(self, handler: HandlerSavingsEstimation) -> None:
        from omnibase_infra.enums import EnumHandlerType

        assert handler.handler_type == EnumHandlerType.INFRA_HANDLER

    def test_handler_category(self, handler: HandlerSavingsEstimation) -> None:
        from omnibase_infra.enums import EnumHandlerTypeCategory

        assert (
            handler.handler_category == EnumHandlerTypeCategory.NONDETERMINISTIC_COMPUTE
        )


class TestSavingsComputation:
    """Core savings computation tests."""

    def test_opus_input_savings(
        self,
        handler: HandlerSavingsEstimation,
        estimation_input: ModelSavingsEstimationInput,
    ) -> None:
        """Verify Opus input token pricing: $15/M tokens."""
        result = handler.compute_savings(estimation_input)

        # 1500 + 900 = 2400 tokens saved
        expected_tokens = 2400
        assert result.direct_tokens_saved == expected_tokens
        assert result.estimated_total_tokens_saved == expected_tokens

        # 2400 * (15 / 1_000_000) = $0.036
        expected_usd = (
            expected_tokens
            * MODEL_PRICING_INPUT[EnumModelTier.OPUS]
            / TOKENS_PER_MILLION
        )
        assert abs(result.direct_savings_usd - expected_usd) < 1e-8
        assert abs(result.estimated_total_savings_usd - expected_usd) < 1e-8

    def test_sonnet_input_savings(self, handler: HandlerSavingsEstimation) -> None:
        """Verify Sonnet input token pricing: $3/M tokens."""
        entries = (
            ModelEffectivenessEntry(
                utilization_score=0.90,
                patterns_count=5,
                tokens_saved=10000,
                model_tier=EnumModelTier.SONNET,
            ),
        )
        inp = ModelSavingsEstimationInput(
            session_id="sonnet-session",
            effectiveness_entries=entries,
            actual_total_tokens=20000,
        )
        result = handler.compute_savings(inp)

        expected_usd = (
            10000 * MODEL_PRICING_INPUT[EnumModelTier.SONNET] / TOKENS_PER_MILLION
        )
        assert abs(result.direct_savings_usd - expected_usd) < 1e-8

    def test_output_token_pricing(self, handler: HandlerSavingsEstimation) -> None:
        """Verify output token pricing is higher: Opus $75/M."""
        entries = (
            ModelEffectivenessEntry(
                utilization_score=0.80,
                patterns_count=5,
                tokens_saved=1000,
                model_tier=EnumModelTier.OPUS,
                is_output_tokens=True,
            ),
        )
        inp = ModelSavingsEstimationInput(
            session_id="output-session",
            effectiveness_entries=entries,
        )
        result = handler.compute_savings(inp)

        expected_usd = (
            1000 * MODEL_PRICING_OUTPUT[EnumModelTier.OPUS] / TOKENS_PER_MILLION
        )
        assert abs(result.direct_savings_usd - expected_usd) < 1e-8

    def test_mixed_tiers(self, handler: HandlerSavingsEstimation) -> None:
        """Entries with different tiers produce correct aggregate savings."""
        entries = (
            ModelEffectivenessEntry(
                utilization_score=0.90,
                patterns_count=5,
                tokens_saved=1000,
                model_tier=EnumModelTier.OPUS,
            ),
            ModelEffectivenessEntry(
                utilization_score=0.80,
                patterns_count=3,
                tokens_saved=2000,
                model_tier=EnumModelTier.SONNET,
            ),
        )
        inp = ModelSavingsEstimationInput(
            session_id="mixed-session",
            effectiveness_entries=entries,
        )
        result = handler.compute_savings(inp)

        opus_usd = 1000 * MODEL_PRICING_INPUT[EnumModelTier.OPUS] / TOKENS_PER_MILLION
        sonnet_usd = (
            2000 * MODEL_PRICING_INPUT[EnumModelTier.SONNET] / TOKENS_PER_MILLION
        )
        expected = opus_usd + sonnet_usd
        assert abs(result.direct_savings_usd - expected) < 1e-8
        assert result.direct_tokens_saved == 3000

    def test_positive_savings(
        self,
        handler: HandlerSavingsEstimation,
        estimation_input: ModelSavingsEstimationInput,
    ) -> None:
        """Savings must be positive when tokens are saved."""
        result = handler.compute_savings(estimation_input)
        assert result.direct_savings_usd > 0
        assert result.actual_total_tokens > 0
        assert result.direct_confidence > 0.5


class TestConfidence:
    """Confidence score calculation tests."""

    def test_weighted_confidence(self, handler: HandlerSavingsEstimation) -> None:
        """Confidence is weighted by tokens_saved."""
        entries = (
            ModelEffectivenessEntry(
                utilization_score=1.0,
                patterns_count=5,
                tokens_saved=900,
                model_tier=EnumModelTier.OPUS,
            ),
            ModelEffectivenessEntry(
                utilization_score=0.0,
                patterns_count=5,
                tokens_saved=100,
                model_tier=EnumModelTier.OPUS,
            ),
        )
        inp = ModelSavingsEstimationInput(
            session_id="weighted-session",
            effectiveness_entries=entries,
        )
        result = handler.compute_savings(inp)

        # Weighted: (1.0*900 + 0.0*100) / (900+100) = 0.9
        assert abs(result.direct_confidence - 0.9) < 1e-4

    def test_heuristic_confidence_is_unweighted(
        self, handler: HandlerSavingsEstimation
    ) -> None:
        """Heuristic confidence is simple average."""
        entries = (
            ModelEffectivenessEntry(
                utilization_score=1.0,
                patterns_count=5,
                tokens_saved=900,
                model_tier=EnumModelTier.OPUS,
            ),
            ModelEffectivenessEntry(
                utilization_score=0.0,
                patterns_count=5,
                tokens_saved=100,
                model_tier=EnumModelTier.OPUS,
            ),
        )
        inp = ModelSavingsEstimationInput(
            session_id="heuristic-session",
            effectiveness_entries=entries,
        )
        result = handler.compute_savings(inp)

        # Simple average: (1.0 + 0.0) / 2 = 0.5
        assert abs(result.heuristic_confidence_avg - 0.5) < 1e-4


class TestCategoryClassification:
    """Category classification tests."""

    def test_architecture_category(self) -> None:
        assert _classify_category(10) == EnumSavingsCategory.ARCHITECTURE
        assert _classify_category(15) == EnumSavingsCategory.ARCHITECTURE

    def test_file_category(self) -> None:
        assert _classify_category(3) == EnumSavingsCategory.FILE
        assert _classify_category(9) == EnumSavingsCategory.FILE

    def test_tool_category(self) -> None:
        assert _classify_category(0) == EnumSavingsCategory.TOOL
        assert _classify_category(2) == EnumSavingsCategory.TOOL

    def test_categories_in_output(
        self,
        handler: HandlerSavingsEstimation,
        estimation_input: ModelSavingsEstimationInput,
    ) -> None:
        """Output should contain category breakdowns."""
        result = handler.compute_savings(estimation_input)
        assert len(result.categories) > 0
        for cat in result.categories:
            assert cat.savings_usd >= 0
            assert cat.tokens_saved >= 0
            assert 0.0 <= cat.confidence <= 1.0


class TestOutputFields:
    """Verify output model field correctness."""

    def test_session_id_propagated(
        self,
        handler: HandlerSavingsEstimation,
        estimation_input: ModelSavingsEstimationInput,
    ) -> None:
        result = handler.compute_savings(estimation_input)
        assert result.session_id == "test-session-001"

    def test_correlation_id_propagated(
        self,
        handler: HandlerSavingsEstimation,
        estimation_input: ModelSavingsEstimationInput,
    ) -> None:
        result = handler.compute_savings(estimation_input)
        assert result.correlation_id == estimation_input.correlation_id

    def test_estimation_method(
        self,
        handler: HandlerSavingsEstimation,
        estimation_input: ModelSavingsEstimationInput,
    ) -> None:
        result = handler.compute_savings(estimation_input)
        assert result.estimation_method == "tiered_attribution_v1"

    def test_is_measured_flag(
        self,
        handler: HandlerSavingsEstimation,
        estimation_input: ModelSavingsEstimationInput,
    ) -> None:
        result = handler.compute_savings(estimation_input)
        assert result.is_measured is True

    def test_completeness_status(
        self,
        handler: HandlerSavingsEstimation,
        estimation_input: ModelSavingsEstimationInput,
    ) -> None:
        result = handler.compute_savings(estimation_input)
        assert result.completeness_status == "complete"

    def test_timestamp_iso_present(
        self,
        handler: HandlerSavingsEstimation,
        estimation_input: ModelSavingsEstimationInput,
    ) -> None:
        result = handler.compute_savings(estimation_input)
        assert result.timestamp_iso is not None
        assert len(result.timestamp_iso) > 0

    def test_pricing_manifest_version(
        self,
        handler: HandlerSavingsEstimation,
        estimation_input: ModelSavingsEstimationInput,
    ) -> None:
        result = handler.compute_savings(estimation_input)
        assert result.pricing_manifest_version == "anthropic-2026-03"


class TestKafkaPayload:
    """Verify Kafka payload serialization."""

    def test_to_kafka_payload_is_dict(
        self,
        handler: HandlerSavingsEstimation,
        estimation_input: ModelSavingsEstimationInput,
    ) -> None:
        result = handler.compute_savings(estimation_input)
        payload = result.to_kafka_payload()
        assert isinstance(payload, dict)

    def test_kafka_payload_categories_are_dicts(
        self,
        handler: HandlerSavingsEstimation,
        estimation_input: ModelSavingsEstimationInput,
    ) -> None:
        result = handler.compute_savings(estimation_input)
        payload = result.to_kafka_payload()
        for cat in payload["categories"]:
            assert isinstance(cat, dict)
            assert "category" in cat
            assert "savings_usd" in cat

    def test_kafka_payload_has_required_fields(
        self,
        handler: HandlerSavingsEstimation,
        estimation_input: ModelSavingsEstimationInput,
    ) -> None:
        """All fields the omnidash consumer expects must be present."""
        result = handler.compute_savings(estimation_input)
        payload = result.to_kafka_payload()
        required_fields = [
            "source_event_id",
            "session_id",
            "correlation_id",
            "actual_total_tokens",
            "actual_cost_usd",
            "direct_savings_usd",
            "direct_tokens_saved",
            "estimated_total_savings_usd",
            "estimated_total_tokens_saved",
            "categories",
            "direct_confidence",
            "heuristic_confidence_avg",
            "estimation_method",
            "is_measured",
            "completeness_status",
            "timestamp_iso",
        ]
        for field in required_fields:
            assert field in payload, f"Missing required field: {field}"


class TestAsyncHandle:
    """Verify the async handle method works."""

    @pytest.mark.asyncio
    async def test_async_handle(
        self,
        handler: HandlerSavingsEstimation,
        estimation_input: ModelSavingsEstimationInput,
    ) -> None:
        result = await handler.handle(estimation_input)
        assert result.direct_savings_usd > 0
        assert result.session_id == estimation_input.session_id
