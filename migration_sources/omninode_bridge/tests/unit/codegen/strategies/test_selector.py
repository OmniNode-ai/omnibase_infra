#!/usr/bin/env python3
"""
Unit tests for StrategySelector.

Tests intelligent strategy selection based on requirements.
"""

import pytest

from omninode_bridge.codegen.strategies.base import EnumStrategyType
from omninode_bridge.codegen.strategies.selector import StrategySelector


class TestStrategySelector:
    """Test StrategySelector functionality."""

    @pytest.fixture
    def selector(self):
        """Create StrategySelector instance."""
        return StrategySelector(
            enable_llm=True,
            enable_validation=True,
        )

    @pytest.fixture
    def selector_no_llm(self):
        """Create StrategySelector with LLM disabled."""
        return StrategySelector(
            enable_llm=False,
            enable_validation=True,
        )

    def test_selector_initialization(self, selector):
        """Test selector initializes correctly."""
        assert selector.enable_llm is True
        assert selector.enable_validation is True

    def test_select_strategy_simple_requirements_chooses_jinja2(
        self, selector, simple_crud_requirements
    ):
        """Test that simple requirements select Jinja2Strategy."""
        result = selector.select_strategy(simple_crud_requirements)

        # Simple CRUD should select Jinja2 (fastest)
        assert result.selected_strategy == EnumStrategyType.JINJA2
        assert result.confidence >= 0.6

        # Verify reasoning
        assert len(result.reasoning) > 0
        assert any("simple" in r.lower() for r in result.reasoning)

    def test_select_strategy_moderate_complexity_chooses_template_load(
        self, selector, moderate_complexity_requirements
    ):
        """Test that moderate complexity chooses TemplateLoadStrategy."""
        result = selector.select_strategy(moderate_complexity_requirements)

        # Moderate complexity should select TemplateLoad (LLM enhancement)
        assert result.selected_strategy == EnumStrategyType.TEMPLATE_LOADING
        assert result.confidence >= 0.7

        # Verify reasoning includes complexity factor
        assert len(result.reasoning) > 0
        assert any("complexity" in r.lower() for r in result.reasoning)

    def test_select_strategy_complex_requirements_chooses_hybrid(
        self, selector, complex_orchestration_requirements
    ):
        """Test that complex requirements choose HybridStrategy."""
        result = selector.select_strategy(complex_orchestration_requirements)

        # Complex orchestration should select Hybrid (best quality)
        assert result.selected_strategy == EnumStrategyType.HYBRID
        assert result.confidence >= 0.8

        # Verify reasoning mentions production/critical
        assert len(result.reasoning) > 0

    def test_select_strategy_override(self, selector, moderate_complexity_requirements):
        """Test that override strategy is respected."""
        result = selector.select_strategy(
            moderate_complexity_requirements, override_strategy=EnumStrategyType.JINJA2
        )

        # Should use override regardless of scoring
        assert result.selected_strategy == EnumStrategyType.JINJA2
        assert result.confidence == 1.0
        assert "override" in result.reasoning[0].lower()

    def test_select_strategy_no_llm_chooses_jinja2_only(
        self, selector_no_llm, moderate_complexity_requirements
    ):
        """Test that selector without LLM only selects Jinja2."""
        result = selector_no_llm.select_strategy(moderate_complexity_requirements)

        # Without LLM, can only choose Jinja2
        assert result.selected_strategy == EnumStrategyType.JINJA2

    def test_select_strategy_returns_all_scores(
        self, selector, moderate_complexity_requirements
    ):
        """Test that selection result includes all scores."""
        result = selector.select_strategy(moderate_complexity_requirements)

        # Should have scores for all strategies
        assert len(result.all_scores) >= 2  # At least Jinja2 and TemplateLoad

        # Scores should be sorted (highest first)
        for i in range(len(result.all_scores) - 1):
            assert (
                result.all_scores[i].total_score >= result.all_scores[i + 1].total_score
            )

    def test_select_strategy_includes_fallback_strategies(
        self, selector, moderate_complexity_requirements
    ):
        """Test that selection result includes fallback strategies."""
        result = selector.select_strategy(moderate_complexity_requirements)

        # Should have fallback strategies (within 10 points of best)
        # Moderate complexity might have Hybrid as fallback to TemplateLoad
        assert isinstance(result.fallback_strategies, list)

    def test_calculate_complexity_simple_operations(self, selector):
        """Test complexity calculation for simple operations."""
        from omninode_bridge.codegen.prd_analyzer import ModelPRDRequirements

        simple_reqs = ModelPRDRequirements(
            service_name="test",
            node_type="effect",
            domain="database",
            business_description="Simple CRUD",
            operations=["create", "read"],  # Simple operations
            features=[],  # No features
            input_schema={},
            output_schema={},
            performance_requirements={},
            error_handling_strategy="simple",
            dependencies={},
        )

        complexity = selector._calculate_complexity(simple_reqs)

        # Should be low complexity (create(1) + read(1) + explicit threshold(10) = 12)
        assert complexity["total_complexity"] == 12
        assert complexity["operation_complexity"] == 2  # create(1) + read(1)

    def test_calculate_complexity_moderate_operations(self, selector):
        """Test complexity calculation for moderate operations."""
        from omninode_bridge.codegen.prd_analyzer import ModelPRDRequirements

        moderate_reqs = ModelPRDRequirements(
            service_name="test",
            node_type="compute",
            domain="data_processing",
            business_description="Transform and validate data",
            operations=["transform", "validate", "filter"],  # Moderate operations
            features=["caching", "retry_logic"],  # Medium features
            input_schema={},
            output_schema={},
            performance_requirements={},
            error_handling_strategy="retry",
            dependencies={"cache_service": "*"},
            complexity_threshold=7,
        )

        complexity = selector._calculate_complexity(moderate_reqs)

        # Should be moderate complexity (ops:6 + features:3 + deps:2 + threshold:7 = 18)
        assert complexity["total_complexity"] == 18
        assert (
            complexity["operation_complexity"] == 6
        )  # transform(2) + validate(2) + filter(2)
        assert complexity["feature_complexity"] == 3  # caching(1) + retry_logic(2)

    def test_calculate_complexity_complex_operations(self, selector):
        """Test complexity calculation for complex operations."""
        complexity = selector._calculate_complexity(
            complex_orchestration_requirements()  # From conftest
        )

        # Should be high complexity
        assert complexity["total_complexity"] >= 15
        assert complexity["custom_logic_score"] > 0  # Should detect keywords

    def test_calculate_complexity_with_performance_requirements(self, selector):
        """Test that performance requirements affect complexity."""
        from omninode_bridge.codegen.prd_analyzer import ModelPRDRequirements

        reqs_with_strict_perf = ModelPRDRequirements(
            service_name="test",
            node_type="effect",
            domain="api",
            business_description="High-performance API",
            operations=["process"],
            features=[],
            input_schema={},
            output_schema={},
            performance_requirements={
                "latency_ms": 50,  # Very strict
                "throughput_per_sec": 2000,  # High throughput
            },
            error_handling_strategy="retry",
            dependencies={},
        )

        complexity = selector._calculate_complexity(reqs_with_strict_perf)

        # Performance requirements should add complexity
        assert (
            complexity["performance_complexity"] >= 6
        )  # 3 for latency + 3 for throughput

    def test_score_jinja2_strategy_simple_requirements(
        self, selector, simple_crud_requirements
    ):
        """Test scoring Jinja2Strategy for simple requirements."""
        complexity_factors = selector._calculate_complexity(simple_crud_requirements)
        score = selector._score_jinja2_strategy(
            simple_crud_requirements, complexity_factors
        )

        # Jinja2 should score high for simple requirements
        assert score.strategy == EnumStrategyType.JINJA2
        assert score.total_score >= 70  # Should be high score
        assert score.confidence >= 0.8
        assert len(score.reasoning) > 0

        # Check component scores
        assert "complexity" in score.component_scores
        assert "performance" in score.component_scores
        assert "quality" in score.component_scores

    def test_score_template_load_strategy_complex_requirements(
        self, selector, complex_orchestration_requirements
    ):
        """Test scoring TemplateLoadStrategy for complex requirements."""
        complexity_factors = selector._calculate_complexity(
            complex_orchestration_requirements
        )
        score = selector._score_template_load_strategy(
            complex_orchestration_requirements, complexity_factors
        )

        # TemplateLoad should score high for complex requirements
        assert score.strategy == EnumStrategyType.TEMPLATE_LOADING
        assert score.total_score >= 70
        assert score.confidence >= 0.8

    def test_score_hybrid_strategy_production_requirements(
        self, selector, complex_orchestration_requirements
    ):
        """Test scoring HybridStrategy for production-critical requirements."""
        complexity_factors = selector._calculate_complexity(
            complex_orchestration_requirements
        )
        score = selector._score_hybrid_strategy(
            complex_orchestration_requirements, complexity_factors
        )

        # Hybrid should score very high for production-critical
        assert score.strategy == EnumStrategyType.HYBRID
        assert score.total_score >= 75  # High quality score
        assert score.confidence >= 0.85

        # Reasoning should mention production/critical
        assert any(
            "production" in r.lower() or "critical" in r.lower()
            for r in score.reasoning
        )

    def test_get_fallback_order_jinja2(self, selector):
        """Test fallback order for Jinja2 strategy."""
        fallbacks = selector.get_fallback_order(EnumStrategyType.JINJA2)

        # Jinja2 → TemplateLoad → Hybrid
        assert EnumStrategyType.TEMPLATE_LOADING in fallbacks
        assert EnumStrategyType.HYBRID in fallbacks

    def test_get_fallback_order_template_load(self, selector):
        """Test fallback order for TemplateLoad strategy."""
        fallbacks = selector.get_fallback_order(EnumStrategyType.TEMPLATE_LOADING)

        # TemplateLoad → Hybrid → Jinja2
        assert EnumStrategyType.HYBRID in fallbacks
        assert EnumStrategyType.JINJA2 in fallbacks

    def test_get_fallback_order_hybrid(self, selector):
        """Test fallback order for Hybrid strategy."""
        fallbacks = selector.get_fallback_order(EnumStrategyType.HYBRID)

        # Hybrid → TemplateLoad → Jinja2
        assert EnumStrategyType.TEMPLATE_LOADING in fallbacks
        assert EnumStrategyType.JINJA2 in fallbacks

    def test_get_fallback_order_no_llm(self, selector_no_llm):
        """Test fallback order when LLM is disabled."""
        fallbacks = selector_no_llm.get_fallback_order(EnumStrategyType.JINJA2)

        # Should only include Jinja2 when LLM disabled
        assert EnumStrategyType.TEMPLATE_LOADING not in fallbacks
        assert EnumStrategyType.HYBRID not in fallbacks

    def test_selection_factors_included_in_result(
        self, selector, moderate_complexity_requirements
    ):
        """Test that selection factors are included in result."""
        result = selector.select_strategy(moderate_complexity_requirements)

        # Should include selection factors
        assert "complexity" in result.selection_factors
        assert "operation_count" in result.selection_factors
        assert "feature_count" in result.selection_factors
        assert "domain" in result.selection_factors
        assert "node_type" in result.selection_factors

    def test_custom_logic_keywords_detected(self, selector):
        """Test that custom logic keywords are detected in description."""
        from omninode_bridge.codegen.prd_analyzer import ModelPRDRequirements

        reqs_with_keywords = ModelPRDRequirements(
            service_name="test",
            node_type="orchestrator",
            domain="workflow",
            business_description="Complex multi-step orchestration with dynamic routing and adaptive behavior",
            operations=["orchestrate"],
            features=[],
            input_schema={},
            output_schema={},
            performance_requirements={},
            error_handling_strategy="circuit_breaker",
            dependencies={},
        )

        complexity = selector._calculate_complexity(reqs_with_keywords)

        # Should detect keywords: complex, multi-step, dynamic, adaptive (4 keywords * 2 points = 8)
        # Note: "orchestration" doesn't match keyword "orchestrate"
        assert complexity["custom_logic_score"] == 8  # 4 keywords * 2 points


# Helper to import requirements from conftest
def complex_orchestration_requirements():
    """Get complex orchestration requirements."""
    from tests.fixtures.codegen.sample_requirements import (
        get_complex_orchestration_requirements,
    )

    return get_complex_orchestration_requirements()
