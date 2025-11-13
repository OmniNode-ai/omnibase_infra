#!/usr/bin/env python3
"""
Unit tests for VariantSelector.

Tests variant selection logic, scoring, and fallback behavior.
"""

import pytest

from src.metadata_stamping.code_gen.patterns.models import EnumNodeType
from src.metadata_stamping.code_gen.templates.variant_metadata import (
    EnumTemplateVariant,
)
from src.metadata_stamping.code_gen.templates.variant_selector import VariantSelector


class TestVariantSelector:
    """Test suite for VariantSelector class."""

    @pytest.fixture
    def selector(self):
        """Create VariantSelector instance."""
        return VariantSelector()

    def test_selector_initialization(self, selector):
        """Test selector initializes correctly."""
        assert selector is not None
        assert len(selector._registry) == 9

    def test_list_variants_all(self, selector):
        """Test listing all variants."""
        variants = selector.list_variants()
        assert len(variants) == 9
        assert EnumTemplateVariant.MINIMAL in variants
        assert EnumTemplateVariant.PRODUCTION in variants

    def test_list_variants_by_node_type(self, selector):
        """Test listing variants filtered by node type."""
        effect_variants = selector.list_variants(node_type=EnumNodeType.EFFECT)
        assert EnumTemplateVariant.DATABASE_HEAVY in effect_variants
        assert EnumTemplateVariant.API_HEAVY in effect_variants

        orchestrator_variants = selector.list_variants(
            node_type=EnumNodeType.ORCHESTRATOR
        )
        assert EnumTemplateVariant.WORKFLOW in orchestrator_variants


class TestVariantSelectionSimple:
    """Test simple variant selection scenarios."""

    @pytest.fixture
    def selector(self):
        return VariantSelector()

    def test_select_minimal_variant(self, selector):
        """Test selection of minimal variant for simple requirements."""
        selection = selector.select_variant(
            node_type=EnumNodeType.EFFECT,
            operation_count=1,
            required_features=set(),
        )

        assert selection.variant == EnumTemplateVariant.MINIMAL
        assert selection.confidence >= 0.5
        assert selection.selection_time_ms < 5.0
        assert not selection.fallback_used

    def test_select_standard_variant(self, selector):
        """Test selection of standard variant for common requirements."""
        selection = selector.select_variant(
            node_type=EnumNodeType.EFFECT,
            operation_count=3,
            required_features={"logging", "error_handling"},
        )

        assert selection.variant == EnumTemplateVariant.STANDARD
        assert selection.confidence >= 0.5
        assert selection.selection_time_ms < 5.0

    def test_select_production_variant(self, selector):
        """Test selection of production variant for complex requirements."""
        selection = selector.select_variant(
            node_type=EnumNodeType.EFFECT,
            operation_count=8,
            required_features={
                "logging",
                "metrics",
                "event_publishing",
                "circuit_breaker",
                "retry",
            },
        )

        assert selection.variant == EnumTemplateVariant.PRODUCTION
        assert selection.confidence >= 0.6
        assert selection.selection_time_ms < 5.0


class TestVariantSelectionSpecialized:
    """Test specialized variant selection scenarios."""

    @pytest.fixture
    def selector(self):
        return VariantSelector()

    def test_select_database_heavy(self, selector):
        """Test selection of database-heavy variant."""
        selection = selector.select_variant(
            node_type=EnumNodeType.EFFECT,
            operation_count=5,
            required_features={
                "database",
                "connection_pooling",
                "transaction_management",
                "query_optimization",
            },
        )

        assert selection.variant == EnumTemplateVariant.DATABASE_HEAVY
        assert selection.confidence >= 0.7
        assert "connection_pooling" in selection.matched_features

    def test_select_api_heavy(self, selector):
        """Test selection of API-heavy variant."""
        selection = selector.select_variant(
            node_type=EnumNodeType.EFFECT,
            operation_count=4,
            required_features={
                "http_client",
                "api",
                "circuit_breaker",
                "retry_logic",
            },
        )

        assert selection.variant == EnumTemplateVariant.API_HEAVY
        assert selection.confidence >= 0.7

    def test_select_kafka_heavy(self, selector):
        """Test selection of Kafka-heavy variant."""
        selection = selector.select_variant(
            node_type=EnumNodeType.EFFECT,
            operation_count=3,
            required_features={
                "kafka",
                "event_publishing",
                "producer",
                "consumer",
            },
        )

        assert selection.variant == EnumTemplateVariant.KAFKA_HEAVY
        assert selection.confidence >= 0.7

    def test_select_ml_inference(self, selector):
        """Test selection of ML inference variant."""
        selection = selector.select_variant(
            node_type=EnumNodeType.COMPUTE,
            operation_count=3,
            required_features={
                "ml",
                "model_loading",
                "inference",
                "batch_processing",
            },
        )

        assert selection.variant == EnumTemplateVariant.ML_INFERENCE
        assert selection.confidence >= 0.7

    def test_select_analytics(self, selector):
        """Test selection of analytics variant."""
        selection = selector.select_variant(
            node_type=EnumNodeType.REDUCER,
            operation_count=3,
            required_features={
                "metrics",
                "aggregation",
                "percentile",
                "histogram",
            },
        )

        assert selection.variant == EnumTemplateVariant.ANALYTICS
        assert selection.confidence >= 0.7

    def test_select_workflow(self, selector):
        """Test selection of workflow variant."""
        selection = selector.select_variant(
            node_type=EnumNodeType.ORCHESTRATOR,
            operation_count=5,
            required_features={
                "workflow",
                "orchestration",
                "fsm",
                "state_management",
                "retry",
            },
        )

        assert selection.variant == EnumTemplateVariant.WORKFLOW
        assert selection.confidence >= 0.7


class TestVariantSelectionFallback:
    """Test fallback scenarios in variant selection."""

    @pytest.fixture
    def selector(self):
        return VariantSelector()

    def test_fallback_to_minimal(self, selector):
        """Test fallback to minimal for very simple requirements."""
        selection = selector.select_variant(
            node_type=EnumNodeType.COMPUTE,
            operation_count=0,
            required_features=set(),
        )

        # Should select minimal or standard
        assert selection.variant in [
            EnumTemplateVariant.MINIMAL,
            EnumTemplateVariant.STANDARD,
        ]

    def test_fallback_to_standard(self, selector):
        """Test fallback to standard when no specialized variant matches."""
        selection = selector.select_variant(
            node_type=EnumNodeType.EFFECT,
            operation_count=3,
            required_features={"custom_feature_1", "custom_feature_2"},
        )

        # Should select standard as fallback
        assert selection.variant == EnumTemplateVariant.STANDARD


class TestVariantSelectionPerformance:
    """Test performance characteristics of variant selection."""

    @pytest.fixture
    def selector(self):
        return VariantSelector()

    def test_selection_speed(self, selector):
        """Test variant selection completes within performance target."""
        selection = selector.select_variant(
            node_type=EnumNodeType.EFFECT,
            operation_count=5,
            required_features={"database", "connection_pooling", "transactions"},
        )

        # Performance target: <5ms
        assert selection.selection_time_ms < 5.0

    def test_selection_caching(self, selector):
        """Test selection caching improves performance."""
        # First selection (cache miss)
        selection1 = selector.select_variant(
            node_type=EnumNodeType.EFFECT,
            operation_count=5,
            required_features={"database", "connection_pooling"},
        )

        # Second selection (cache hit)
        selection2 = selector.select_variant(
            node_type=EnumNodeType.EFFECT,
            operation_count=5,
            required_features={"database", "connection_pooling"},
        )

        # Both should return same variant
        assert selection1.variant == selection2.variant

    def test_clear_cache(self, selector):
        """Test cache clearing works correctly."""
        # Select variant to populate cache
        selector.select_variant(
            node_type=EnumNodeType.EFFECT,
            operation_count=5,
            required_features={"database"},
        )

        # Clear cache
        selector.clear_cache()

        # Cache should be empty
        assert len(selector._selection_cache) == 0


class TestRequirementsAnalysis:
    """Test requirements analysis logic."""

    @pytest.fixture
    def selector(self):
        return VariantSelector()

    def test_analyze_database_requirements(self, selector):
        """Test analysis of database-focused requirements."""
        analysis = selector._analyze_requirements(
            node_type=EnumNodeType.EFFECT,
            operation_count=5,
            required_features={
                "database",
                "connection_pool",
                "query",
                "transaction",
            },
        )

        assert analysis.node_type == EnumNodeType.EFFECT
        assert analysis.operation_count == 5
        assert analysis.feature_categories["database"] >= 2

    def test_analyze_api_requirements(self, selector):
        """Test analysis of API-focused requirements."""
        analysis = selector._analyze_requirements(
            node_type=EnumNodeType.EFFECT,
            operation_count=4,
            required_features={
                "http_client",
                "api_call",
                "rest",
            },
        )

        assert analysis.feature_categories["api"] >= 2

    def test_complexity_calculation(self, selector):
        """Test complexity score calculation."""
        # Simple requirements
        simple_analysis = selector._analyze_requirements(
            node_type=EnumNodeType.COMPUTE,
            operation_count=1,
            required_features={"simple"},
        )
        assert simple_analysis.complexity_score < 0.3

        # Complex requirements
        complex_analysis = selector._analyze_requirements(
            node_type=EnumNodeType.EFFECT,
            operation_count=10,
            required_features={
                "database",
                "api",
                "kafka",
                "metrics",
                "circuit_breaker",
                "retry",
            },
        )
        assert complex_analysis.complexity_score > 0.7


class TestVariantScoring:
    """Test variant scoring algorithms."""

    @pytest.fixture
    def selector(self):
        return VariantSelector()

    def test_score_variants(self, selector):
        """Test variant scoring returns valid scores."""
        analysis = selector._analyze_requirements(
            node_type=EnumNodeType.EFFECT,
            operation_count=5,
            required_features={"database", "connection_pooling"},
        )

        scores = selector._score_variants(analysis)

        # Should have scores for compatible variants
        assert len(scores) > 0

        # All scores should be in valid range
        for score in scores.values():
            assert 0.0 <= score <= 1.0

    def test_specialization_bonus(self, selector):
        """Test specialization bonus calculation."""
        # Database-heavy analysis
        db_analysis = selector._analyze_requirements(
            node_type=EnumNodeType.EFFECT,
            operation_count=5,
            required_features={
                "database",
                "db_query",
                "connection_pool",
                "transaction",
            },
        )

        # Database-heavy should get bonus
        bonus = selector._calculate_specialization_bonus(
            EnumTemplateVariant.DATABASE_HEAVY,
            db_analysis,
        )
        assert bonus > 0.0


class TestVariantMetadataIntegration:
    """Test integration between selector and variant metadata."""

    @pytest.fixture
    def selector(self):
        return VariantSelector()

    def test_get_variant_metadata(self, selector):
        """Test retrieving variant metadata."""
        metadata = selector.get_variant_metadata(EnumTemplateVariant.DATABASE_HEAVY)

        assert metadata is not None
        assert metadata.variant == EnumTemplateVariant.DATABASE_HEAVY
        assert len(metadata.features) > 0

    def test_metadata_matches_requirements(self, selector):
        """Test metadata matching logic."""
        metadata = selector.get_variant_metadata(EnumTemplateVariant.DATABASE_HEAVY)

        # Should match database requirements
        assert metadata.matches_requirements(
            node_type=EnumNodeType.EFFECT,
            operation_count=5,
            required_features={"database", "connection_pooling"},
        )

        # Should not match incompatible requirements
        assert not metadata.matches_requirements(
            node_type=EnumNodeType.COMPUTE,  # Wrong node type
            operation_count=5,
            required_features={"database"},
        )


# Performance benchmarks
@pytest.mark.benchmark
class TestVariantSelectorBenchmarks:
    """Benchmark tests for variant selection performance."""

    @pytest.fixture
    def selector(self):
        return VariantSelector()

    def test_benchmark_simple_selection(self, selector, benchmark):
        """Benchmark simple variant selection."""

        def select():
            return selector.select_variant(
                node_type=EnumNodeType.EFFECT,
                operation_count=3,
                required_features={"logging"},
            )

        result = benchmark(select)
        assert result.selection_time_ms < 5.0

    def test_benchmark_complex_selection(self, selector, benchmark):
        """Benchmark complex variant selection."""

        def select():
            return selector.select_variant(
                node_type=EnumNodeType.EFFECT,
                operation_count=8,
                required_features={
                    "database",
                    "api",
                    "kafka",
                    "metrics",
                    "logging",
                    "circuit_breaker",
                },
            )

        result = benchmark(select)
        assert result.selection_time_ms < 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
