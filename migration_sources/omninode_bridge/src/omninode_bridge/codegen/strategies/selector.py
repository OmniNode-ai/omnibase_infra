#!/usr/bin/env python3
"""
Intelligent Strategy Selector for Auto-Mode Code Generation.

Automatically selects the best code generation strategy based on NodeRequirements
using a multi-factor scoring algorithm that considers:
- Node complexity (simple CRUD vs. complex business logic)
- Custom requirements and business logic needs
- Performance requirements (speed vs. quality trade-offs)
- Quality requirements (best quality vs. fastest generation)

Strategy Selection:
- Jinja2Strategy: Fast template-only generation for simple CRUD nodes
- TemplateLoadStrategy: Flexible LLM-enhanced generation for complex logic
- HybridStrategy: Best quality with multi-phase validation for production-critical nodes

ONEX v2.0 Compliance:
- Scoring-based intelligent selection (0-100 per strategy)
- Fallback logic for strategy failures
- Comprehensive logging for decision transparency
- Type-safe with Pydantic models

Example Usage:
    >>> selector = StrategySelector()
    >>> requirements = ModelPRDRequirements(
    ...     node_type="effect",
    ...     service_name="postgres_crud",
    ...     domain="database",
    ...     operations=["create", "read", "update", "delete"],
    ...     business_description="Simple PostgreSQL CRUD operations",
    ...     features=["connection_pooling"],
    ...     complexity_threshold=5,
    ... )
    >>> result = selector.select_strategy(requirements)
    >>> print(f"Selected: {result.selected_strategy.value}")
    >>> print(f"Confidence: {result.confidence:.2%}")
    >>> print(f"Reasoning: {result.reasoning}")
"""

import logging
from typing import Any, ClassVar, Optional

from pydantic import BaseModel, Field

from ..node_classifier import ModelClassificationResult
from ..prd_analyzer import ModelPRDRequirements
from .base import EnumStrategyType

logger = logging.getLogger(__name__)


class ModelStrategyScore(BaseModel):
    """
    Scoring result for a single strategy.

    Contains overall score, component scores, and reasoning.
    """

    strategy: EnumStrategyType = Field(..., description="Strategy being scored")
    total_score: float = Field(..., ge=0.0, le=100.0, description="Total score (0-100)")
    component_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Individual component scores (complexity, performance, quality)",
    )
    reasoning: list[str] = Field(
        default_factory=list, description="Reasoning for score components"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in this strategy selection"
    )


class ModelStrategySelectionResult(BaseModel):
    """
    Result of strategy selection with scoring details.

    Contains selected strategy, all scored strategies, and selection reasoning.
    """

    selected_strategy: EnumStrategyType = Field(
        ..., description="Selected strategy for code generation"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in selection"
    )
    reasoning: list[str] = Field(
        default_factory=list, description="Primary reasoning for selection"
    )

    # All scored strategies
    all_scores: list[ModelStrategyScore] = Field(
        default_factory=list, description="Scores for all evaluated strategies"
    )

    # Alternative strategies (if close scores)
    fallback_strategies: list[EnumStrategyType] = Field(
        default_factory=list,
        description="Fallback strategies to try if primary fails",
    )

    # Selection metadata
    selection_factors: dict[str, Any] = Field(
        default_factory=dict, description="Key factors influencing selection"
    )


class StrategySelector:
    """
    Intelligent strategy selector for auto-mode code generation.

    Uses multi-factor scoring algorithm to select optimal strategy:
    1. Complexity Score (40%): Evaluates node complexity and custom logic needs
    2. Performance Score (30%): Balances generation speed vs. quality
    3. Quality Score (30%): Considers quality requirements and validation needs

    Scoring Criteria:
    - Simple CRUD nodes (complexity < 10, standard operations) → Jinja2Strategy (fast, 60-80 score)
    - Complex business logic (complexity ≥ 10, custom operations) → TemplateLoadStrategy (70-90 score)
    - Production-critical (high quality requirements) → HybridStrategy (80-95 score)

    Fallback Logic:
    - If primary strategy fails → Try first fallback
    - If all strategies fail → Raise comprehensive error with attempted strategies
    """

    # Complexity thresholds for strategy selection
    COMPLEXITY_THRESHOLDS: ClassVar[dict[str, int]] = {
        "simple": 5,  # < 5: Very simple (basic CRUD)
        "moderate": 10,  # 5-10: Moderate (standard business logic)
        "complex": 20,  # 10-20: Complex (advanced logic)
        # > 20: Very complex (multi-step orchestration)
    }

    # Operation complexity weights
    OPERATION_COMPLEXITY: ClassVar[dict[str, int]] = {
        # Simple CRUD operations
        "create": 1,
        "read": 1,
        "update": 1,
        "delete": 1,
        "list": 1,
        # Moderate operations
        "search": 2,
        "filter": 2,
        "validate": 2,
        "transform": 2,
        # Complex operations
        "aggregate": 3,
        "orchestrate": 4,
        "coordinate": 4,
        "analyze": 3,
        # Very complex operations
        "optimize": 5,
        "predict": 5,
        "learn": 5,
    }

    # Feature complexity weights
    FEATURE_COMPLEXITY: ClassVar[dict[str, int]] = {
        # Low complexity features
        "logging": 1,
        "metrics": 1,
        "caching": 1,
        # Medium complexity features
        "connection_pooling": 2,
        "retry_logic": 2,
        "validation": 2,
        # High complexity features
        "circuit_breaker": 3,
        "rate_limiting": 3,
        "authentication": 3,
        "distributed_tracing": 3,
    }

    def __init__(
        self,
        enable_llm: bool = True,
        enable_validation: bool = True,
    ):
        """
        Initialize strategy selector.

        Args:
            enable_llm: Enable LLM-based strategies (TemplateLoadStrategy, HybridStrategy)
            enable_validation: Enable validation-based scoring adjustments
        """
        self.enable_llm = enable_llm
        self.enable_validation = enable_validation
        logger.info(
            f"StrategySelector initialized (llm={enable_llm}, validation={enable_validation})"
        )

    def select_strategy(
        self,
        requirements: ModelPRDRequirements,
        classification: Optional[ModelClassificationResult] = None,
        override_strategy: Optional[EnumStrategyType] = None,
    ) -> ModelStrategySelectionResult:
        """
        Select optimal code generation strategy for given requirements.

        Selection Algorithm:
        1. Calculate complexity score for requirements
        2. Score each strategy (0-100) based on:
           - Complexity fit (40% weight)
           - Performance needs (30% weight)
           - Quality needs (30% weight)
        3. Select strategy with highest total score
        4. Identify fallback strategies (scores within 10 points)

        Args:
            requirements: Extracted PRD requirements
            classification: Optional node classification result
            override_strategy: Optional strategy override (bypasses scoring)

        Returns:
            ModelStrategySelectionResult with selected strategy and reasoning

        Example:
            >>> selector = StrategySelector()
            >>> reqs = ModelPRDRequirements(
            ...     node_type="effect",
            ...     service_name="postgres_crud",
            ...     domain="database",
            ...     operations=["create", "read"],
            ...     business_description="Simple CRUD",
            ...     complexity_threshold=5,
            ... )
            >>> result = selector.select_strategy(reqs)
            >>> assert result.selected_strategy == EnumStrategyType.JINJA2
        """
        # Handle override
        if override_strategy and override_strategy != EnumStrategyType.AUTO:
            logger.info(f"Strategy override: {override_strategy.value}")
            return ModelStrategySelectionResult(
                selected_strategy=override_strategy,
                confidence=1.0,
                reasoning=[f"Explicit strategy override: {override_strategy.value}"],
                fallback_strategies=[],
                selection_factors={"override": True},
            )

        # Calculate requirement complexity
        complexity_factors = self._calculate_complexity(requirements)
        total_complexity = complexity_factors["total_complexity"]

        logger.info(
            f"Analyzing requirements: complexity={total_complexity}, "
            f"operations={len(requirements.operations)}, "
            f"features={len(requirements.features)}"
        )

        # Score all strategies
        all_scores = self._score_all_strategies(requirements, complexity_factors)

        # Sort by score (highest first)
        all_scores.sort(key=lambda s: s.total_score, reverse=True)

        # Select best strategy
        best_score = all_scores[0]
        selected_strategy = best_score.strategy

        # Identify fallback strategies (within 10 points of best)
        fallback_strategies = [
            score.strategy
            for score in all_scores[1:]
            if score.total_score >= best_score.total_score - 10.0
        ]

        # Build selection result
        result = ModelStrategySelectionResult(
            selected_strategy=selected_strategy,
            confidence=best_score.confidence,
            reasoning=best_score.reasoning,
            all_scores=all_scores,
            fallback_strategies=fallback_strategies,
            selection_factors={
                "complexity": total_complexity,
                "operation_count": len(requirements.operations),
                "feature_count": len(requirements.features),
                "domain": requirements.domain,
                "node_type": requirements.node_type,
                **complexity_factors,
            },
        )

        logger.info(
            f"Selected strategy: {selected_strategy.value} "
            f"(score={best_score.total_score:.1f}, confidence={best_score.confidence:.2%})"
        )
        logger.debug(
            f"All scores: {[(s.strategy.value, s.total_score) for s in all_scores]}"
        )
        logger.debug(f"Fallback strategies: {[s.value for s in fallback_strategies]}")

        return result

    def _calculate_complexity(
        self, requirements: ModelPRDRequirements
    ) -> dict[str, Any]:
        """
        Calculate complexity metrics for requirements.

        Complexity Factors:
        1. Operations (weighted by complexity)
        2. Features (weighted by complexity)
        3. Custom logic indicators (keywords in description)
        4. Dependencies (external service count)
        5. Performance requirements (strict requirements increase complexity)

        Args:
            requirements: PRD requirements

        Returns:
            Dictionary with complexity factors and total complexity score
        """
        # Operation complexity
        operation_complexity = sum(
            self.OPERATION_COMPLEXITY.get(op.lower(), 2)  # Default: moderate complexity
            for op in requirements.operations
        )

        # Feature complexity
        feature_complexity = sum(
            self.FEATURE_COMPLEXITY.get(feat.lower(), 1)  # Default: low complexity
            for feat in requirements.features
        )

        # Custom logic indicators (keywords in business description)
        custom_logic_keywords = [
            "custom",
            "complex",
            "advanced",
            "sophisticated",
            "intelligent",
            "adaptive",
            "dynamic",
            "multi-step",
            "orchestrate",
            "coordinate",
        ]
        custom_logic_score = sum(
            2
            for keyword in custom_logic_keywords
            if keyword in requirements.business_description.lower()
        )

        # Dependency complexity (external services)
        dependency_complexity = len(requirements.dependencies) * 2

        # Performance requirement complexity (strict requirements)
        perf_complexity = 0
        if requirements.performance_requirements:
            # Strict latency requirements increase complexity
            if "latency_ms" in requirements.performance_requirements:
                latency_target = requirements.performance_requirements["latency_ms"]
                if latency_target < 100:
                    perf_complexity += 3  # Very strict
                elif latency_target < 500:
                    perf_complexity += 2  # Strict
                else:
                    perf_complexity += 1  # Moderate

            # Throughput requirements
            if "throughput_per_sec" in requirements.performance_requirements:
                throughput = requirements.performance_requirements["throughput_per_sec"]
                if throughput > 1000:
                    perf_complexity += 3  # High throughput
                elif throughput > 100:
                    perf_complexity += 2  # Medium throughput
                else:
                    perf_complexity += 1  # Low throughput

        # Use explicit complexity threshold if provided
        if hasattr(requirements, "complexity_threshold"):
            explicit_complexity = requirements.complexity_threshold
        else:
            explicit_complexity = 0

        # Total complexity (weighted sum)
        total_complexity = (
            operation_complexity  # 1x weight
            + feature_complexity  # 1x weight
            + custom_logic_score  # 1x weight
            + dependency_complexity  # 1x weight
            + perf_complexity  # 1x weight
            + explicit_complexity  # Direct contribution
        )

        return {
            "total_complexity": total_complexity,
            "operation_complexity": operation_complexity,
            "feature_complexity": feature_complexity,
            "custom_logic_score": custom_logic_score,
            "dependency_complexity": dependency_complexity,
            "performance_complexity": perf_complexity,
            "explicit_complexity": explicit_complexity,
        }

    def _score_all_strategies(
        self,
        requirements: ModelPRDRequirements,
        complexity_factors: dict[str, Any],
    ) -> list[ModelStrategyScore]:
        """
        Score all available strategies for given requirements.

        Args:
            requirements: PRD requirements
            complexity_factors: Pre-calculated complexity factors

        Returns:
            List of ModelStrategyScore for all strategies
        """
        scores = []

        # Score Jinja2Strategy (template-only, fast)
        scores.append(self._score_jinja2_strategy(requirements, complexity_factors))

        # Score TemplateLoadStrategy (template + LLM, flexible)
        if self.enable_llm:
            scores.append(
                self._score_template_load_strategy(requirements, complexity_factors)
            )

        # Score HybridStrategy (best quality, validated)
        if self.enable_llm and self.enable_validation:
            scores.append(self._score_hybrid_strategy(requirements, complexity_factors))

        return scores

    def _score_jinja2_strategy(
        self,
        requirements: ModelPRDRequirements,
        complexity_factors: dict[str, Any],
    ) -> ModelStrategyScore:
        """
        Score Jinja2Strategy (template-only, fast).

        Ideal for:
        - Simple CRUD operations
        - Standard patterns
        - Low complexity (< 10)
        - Fast generation needs

        Scoring:
        - High score for simple operations (60-80)
        - Medium score for moderate complexity (40-60)
        - Low score for complex logic (20-40)
        """
        total_complexity = complexity_factors["total_complexity"]
        component_scores = {}
        reasoning = []

        # Complexity score (40% weight)
        # Lower complexity = higher score for Jinja2
        if total_complexity <= self.COMPLEXITY_THRESHOLDS["simple"]:
            complexity_score = 90.0
            reasoning.append(
                f"Very simple node (complexity={total_complexity}) - perfect for template-only"
            )
        elif total_complexity <= self.COMPLEXITY_THRESHOLDS["moderate"]:
            complexity_score = 60.0
            reasoning.append(
                f"Moderate complexity (complexity={total_complexity}) - template may be sufficient"
            )
        else:
            complexity_score = 30.0
            reasoning.append(
                f"High complexity (complexity={total_complexity}) - template-only may be insufficient"
            )
        component_scores["complexity"] = complexity_score

        # Performance score (30% weight)
        # Jinja2 is fastest, always high performance score
        performance_score = 95.0
        reasoning.append("Fastest generation strategy (template-only, no LLM)")
        component_scores["performance"] = performance_score

        # Quality score (30% weight)
        # Lower quality for complex requirements
        if total_complexity <= self.COMPLEXITY_THRESHOLDS["simple"]:
            quality_score = 85.0
            reasoning.append("High quality for simple patterns (well-tested templates)")
        elif total_complexity <= self.COMPLEXITY_THRESHOLDS["moderate"]:
            quality_score = 60.0
            reasoning.append("Moderate quality for moderate complexity")
        else:
            quality_score = 35.0
            reasoning.append("Lower quality for complex logic (no LLM customization)")
        component_scores["quality"] = quality_score

        # Calculate total score (weighted average)
        total_score = (
            complexity_score * 0.4 + performance_score * 0.3 + quality_score * 0.3
        )

        # Confidence based on complexity fit
        if total_complexity <= self.COMPLEXITY_THRESHOLDS["simple"]:
            confidence = 0.9
        elif total_complexity <= self.COMPLEXITY_THRESHOLDS["moderate"]:
            confidence = 0.6
        else:
            confidence = 0.3

        return ModelStrategyScore(
            strategy=EnumStrategyType.JINJA2,
            total_score=total_score,
            component_scores=component_scores,
            reasoning=reasoning,
            confidence=confidence,
        )

    def _score_template_load_strategy(
        self,
        requirements: ModelPRDRequirements,
        complexity_factors: dict[str, Any],
    ) -> ModelStrategyScore:
        """
        Score TemplateLoadStrategy (template + LLM enhancement).

        Ideal for:
        - Complex business logic
        - Custom requirements
        - Moderate to high complexity (10+)
        - Flexible quality needs

        Scoring:
        - Low score for simple operations (40-60)
        - High score for moderate complexity (70-85)
        - Very high score for complex logic (80-95)
        """
        total_complexity = complexity_factors["total_complexity"]
        component_scores = {}
        reasoning = []

        # Complexity score (40% weight)
        # Higher complexity = higher score for TemplateLoad
        if total_complexity <= self.COMPLEXITY_THRESHOLDS["simple"]:
            complexity_score = 45.0
            reasoning.append(
                f"Low complexity (complexity={total_complexity}) - LLM may be overkill"
            )
        elif total_complexity <= self.COMPLEXITY_THRESHOLDS["moderate"]:
            complexity_score = 80.0
            reasoning.append(
                f"Moderate complexity (complexity={total_complexity}) - ideal for LLM enhancement"
            )
        else:
            complexity_score = 95.0
            reasoning.append(
                f"High complexity (complexity={total_complexity}) - requires LLM for custom logic"
            )
        component_scores["complexity"] = complexity_score

        # Performance score (30% weight)
        # Slower than Jinja2, but acceptable
        performance_score = 65.0
        reasoning.append("Moderate speed (LLM enhancement adds latency)")
        component_scores["performance"] = performance_score

        # Quality score (30% weight)
        # High quality for complex requirements
        if total_complexity <= self.COMPLEXITY_THRESHOLDS["simple"]:
            quality_score = 70.0
            reasoning.append(
                "Good quality, but potentially over-engineered for simple cases"
            )
        elif total_complexity <= self.COMPLEXITY_THRESHOLDS["moderate"]:
            quality_score = 85.0
            reasoning.append("High quality with LLM-enhanced business logic")
        else:
            quality_score = 90.0
            reasoning.append("Excellent quality for complex custom logic")
        component_scores["quality"] = quality_score

        # Calculate total score (weighted average)
        total_score = (
            complexity_score * 0.4 + performance_score * 0.3 + quality_score * 0.3
        )

        # Confidence based on complexity fit
        if total_complexity <= self.COMPLEXITY_THRESHOLDS["simple"]:
            confidence = 0.5
        elif total_complexity <= self.COMPLEXITY_THRESHOLDS["moderate"]:
            confidence = 0.85
        else:
            confidence = 0.95

        return ModelStrategyScore(
            strategy=EnumStrategyType.TEMPLATE_LOADING,
            total_score=total_score,
            component_scores=component_scores,
            reasoning=reasoning,
            confidence=confidence,
        )

    def _score_hybrid_strategy(
        self,
        requirements: ModelPRDRequirements,
        complexity_factors: dict[str, Any],
    ) -> ModelStrategyScore:
        """
        Score HybridStrategy (best quality with multi-phase validation).

        Ideal for:
        - Production-critical nodes
        - High quality requirements (test coverage > 90%)
        - Complex orchestration
        - Maximum reliability needed

        Scoring:
        - Low score for simple operations (50-65)
        - High score for moderate complexity (75-85)
        - Very high score for production-critical (85-98)
        """
        total_complexity = complexity_factors["total_complexity"]
        component_scores = {}
        reasoning = []

        # Complexity score (40% weight)
        # Best for moderate to high complexity with quality needs
        if total_complexity <= self.COMPLEXITY_THRESHOLDS["simple"]:
            complexity_score = 55.0
            reasoning.append(
                f"Low complexity (complexity={total_complexity}) - multi-phase validation may be excessive"
            )
        elif total_complexity <= self.COMPLEXITY_THRESHOLDS["moderate"]:
            complexity_score = 80.0
            reasoning.append(
                f"Moderate complexity (complexity={total_complexity}) - benefits from validation"
            )
        elif total_complexity <= self.COMPLEXITY_THRESHOLDS["complex"]:
            complexity_score = 90.0
            reasoning.append(
                f"High complexity (complexity={total_complexity}) - validation critical"
            )
        else:
            # Very high complexity (> 20) with validation requirements
            complexity_score = 99.0
            reasoning.append(
                f"Very high complexity (complexity={total_complexity}) - multi-phase validation essential"
            )
        component_scores["complexity"] = complexity_score

        # Performance score (30% weight)
        # Slowest strategy (LLM + validation)
        performance_score = 50.0
        reasoning.append("Slower generation (LLM + multi-phase validation)")
        component_scores["performance"] = performance_score

        # Quality score (30% weight)
        # Highest quality with validation
        quality_boost = 0.0
        if requirements.min_test_coverage >= 0.9:
            quality_boost = 15.0
            reasoning.append(
                f"High test coverage requirement ({requirements.min_test_coverage:.0%}) - validates thoroughly"
            )
        elif requirements.min_test_coverage >= 0.8:
            quality_boost = 10.0
            reasoning.append("Standard test coverage - validation ensures quality")

        # Production-critical indicator
        if (
            "production" in requirements.business_description.lower()
            or "critical" in requirements.business_description.lower()
        ):
            quality_boost += 10.0
            reasoning.append("Production-critical node - maximum quality needed")

        quality_score = min(95.0 + quality_boost, 100.0)
        component_scores["quality"] = quality_score

        # Calculate total score (weighted average)
        total_score = (
            complexity_score * 0.4 + performance_score * 0.3 + quality_score * 0.3
        )

        # Confidence based on quality requirements
        if (
            requirements.min_test_coverage >= 0.9
            or "production" in requirements.business_description.lower()
        ):
            confidence = 0.95
        elif total_complexity > self.COMPLEXITY_THRESHOLDS["moderate"]:
            confidence = 0.85
        else:
            confidence = 0.65

        return ModelStrategyScore(
            strategy=EnumStrategyType.HYBRID,
            total_score=total_score,
            component_scores=component_scores,
            reasoning=reasoning,
            confidence=confidence,
        )

    def get_fallback_order(
        self, primary_strategy: EnumStrategyType
    ) -> list[EnumStrategyType]:
        """
        Get fallback strategy order for given primary strategy.

        Fallback Logic:
        - Jinja2 → TemplateLoad → Hybrid (simple to complex)
        - TemplateLoad → Hybrid → Jinja2 (balanced fallback)
        - Hybrid → TemplateLoad → Jinja2 (quality to speed)

        Args:
            primary_strategy: Primary strategy that failed

        Returns:
            List of fallback strategies in order of preference
        """
        fallback_map = {
            EnumStrategyType.JINJA2: [
                EnumStrategyType.TEMPLATE_LOADING,
                EnumStrategyType.HYBRID,
            ],
            EnumStrategyType.TEMPLATE_LOADING: [
                EnumStrategyType.HYBRID,
                EnumStrategyType.JINJA2,
            ],
            EnumStrategyType.HYBRID: [
                EnumStrategyType.TEMPLATE_LOADING,
                EnumStrategyType.JINJA2,
            ],
        }

        fallbacks = fallback_map.get(primary_strategy, [])

        # Filter out strategies not available
        available_fallbacks: list[EnumStrategyType] = []
        for strategy in fallbacks:
            if strategy == EnumStrategyType.JINJA2:
                available_fallbacks.append(strategy)
            elif strategy in (
                EnumStrategyType.TEMPLATE_LOADING,
                EnumStrategyType.HYBRID,
            ):
                if self.enable_llm:
                    available_fallbacks.append(strategy)

        logger.debug(
            f"Fallback order for {primary_strategy.value}: "
            f"{[s.value for s in available_fallbacks]}"
        )

        return available_fallbacks


# Export
__all__ = [
    "StrategySelector",
    "ModelStrategyScore",
    "ModelStrategySelectionResult",
]
