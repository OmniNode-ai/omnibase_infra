"""
Template variant selector for intelligent template selection.

This module provides the VariantSelector class that analyzes requirements
and selects the optimal template variant based on:
- Operation count and complexity
- Required features and capabilities
- Node type compatibility
- Performance characteristics

Performance Target: <5ms per selection
Accuracy Target: >95% correct variant selection
"""

import logging
import time
from typing import Optional

from pydantic import BaseModel, Field

from ..patterns.models import EnumNodeType
from .variant_metadata import (
    VARIANT_METADATA_REGISTRY,
    EnumTemplateVariant,
    ModelTemplateSelection,
    ModelVariantMetadata,
)

logger = logging.getLogger(__name__)


class ModelRequirementsAnalysis(BaseModel):
    """
    Analysis of requirements for variant selection.

    Attributes:
        node_type: Target node type
        operation_count: Number of operations in requirements
        required_features: Set of required feature tags
        complexity_score: Calculated complexity (0.0-1.0)
        feature_categories: Categorized features (database, api, kafka, etc.)
    """

    node_type: EnumNodeType = Field(..., description="Target node type")
    operation_count: int = Field(..., ge=0, description="Number of operations")
    required_features: set[str] = Field(
        default_factory=set, description="Required feature tags"
    )
    complexity_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Calculated complexity"
    )
    feature_categories: dict[str, int] = Field(
        default_factory=dict, description="Feature counts by category"
    )

    class Config:
        frozen = False


class VariantSelector:
    """
    Intelligent template variant selector.

    Uses a multi-factor scoring algorithm to select the optimal template
    variant based on requirements analysis.

    Scoring Factors:
    1. Node type compatibility (20%)
    2. Operation count fit (30%)
    3. Feature overlap (50%)

    Performance:
    - Selection time: <5ms (target)
    - Accuracy: >95% (target)
    - Memory: <10MB
    """

    def __init__(self):
        """Initialize variant selector with registry."""
        self._registry = VARIANT_METADATA_REGISTRY
        self._selection_cache: dict[str, ModelTemplateSelection] = {}
        self._cache_max_size = 100

        logger.info("VariantSelector initialized with %d variants", len(self._registry))

    def select_variant(
        self,
        node_type: EnumNodeType,
        operation_count: int,
        required_features: Optional[set[str]] = None,
        use_cache: bool = True,
    ) -> ModelTemplateSelection:
        """
        Select optimal template variant for given requirements.

        Args:
            node_type: Target node type
            operation_count: Number of operations in requirements
            required_features: Set of required feature tags
            use_cache: Whether to use cached selections

        Returns:
            ModelTemplateSelection with selected variant and confidence

        Example:
            ```python
            selector = VariantSelector()
            selection = selector.select_variant(
                node_type=EnumNodeType.EFFECT,
                operation_count=8,
                required_features={"database", "connection_pooling", "transactions"},
            )
            print(f"Selected: {selection.variant} (confidence: {selection.confidence})")
            # Output: Selected: database_heavy (confidence: 0.92)
            ```
        """
        start_time = time.perf_counter()

        if required_features is None:
            required_features = set()

        # Check cache
        cache_key = self._make_cache_key(node_type, operation_count, required_features)
        if use_cache and cache_key in self._selection_cache:
            cached_selection = self._selection_cache[cache_key]
            logger.debug("Cache hit for variant selection: %s", cache_key)
            return cached_selection

        # Analyze requirements
        analysis = self._analyze_requirements(
            node_type=node_type,
            operation_count=operation_count,
            required_features=required_features,
        )

        # Score all variants
        variant_scores = self._score_variants(analysis)

        # Select best variant
        if not variant_scores:
            # No variants match - use fallback
            selection = self._fallback_selection(analysis)
        else:
            # Select variant with highest score
            best_variant, best_score = max(variant_scores.items(), key=lambda x: x[1])
            metadata = self._registry[best_variant]

            # Calculate matched features
            matched_features = list(
                set(metadata.features).intersection(required_features)
            )

            # Generate rationale
            rationale = self._generate_rationale(
                metadata=metadata,
                analysis=analysis,
                score=best_score,
            )

            selection = ModelTemplateSelection(
                variant=best_variant,
                confidence=best_score,
                rationale=rationale,
                matched_features=matched_features,
                fallback_used=False,
                selection_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        # Cache selection
        if use_cache:
            self._cache_selection(cache_key, selection)

        logger.info(
            "Variant selected: %s (confidence: %.2f, time: %.2fms)",
            selection.variant.value,
            selection.confidence,
            selection.selection_time_ms,
        )

        return selection

    def _analyze_requirements(
        self,
        node_type: EnumNodeType,
        operation_count: int,
        required_features: set[str],
    ) -> ModelRequirementsAnalysis:
        """
        Analyze requirements to extract selection criteria.

        Args:
            node_type: Target node type
            operation_count: Number of operations
            required_features: Required feature tags

        Returns:
            Requirements analysis with extracted features
        """
        # Categorize features
        feature_categories = {
            "database": sum(
                1
                for f in required_features
                if any(
                    kw in f.lower()
                    for kw in [
                        "database",
                        "db",
                        "sql",
                        "query",
                        "connection",
                        "transaction",
                    ]
                )
            ),
            "api": sum(
                1
                for f in required_features
                if any(
                    kw in f.lower()
                    for kw in ["api", "http", "rest", "client", "request"]
                )
            ),
            "kafka": sum(
                1
                for f in required_features
                if any(
                    kw in f.lower()
                    for kw in ["kafka", "event", "producer", "consumer", "message"]
                )
            ),
            "ml": sum(
                1
                for f in required_features
                if any(
                    kw in f.lower()
                    for kw in ["ml", "model", "inference", "predict", "train"]
                )
            ),
            "analytics": sum(
                1
                for f in required_features
                if any(
                    kw in f.lower()
                    for kw in [
                        "metric",
                        "aggregate",
                        "analytics",
                        "percentile",
                        "histogram",
                    ]
                )
            ),
            "workflow": sum(
                1
                for f in required_features
                if any(
                    kw in f.lower()
                    for kw in ["workflow", "orchestrat", "fsm", "state", "step"]
                )
            ),
        }

        # Calculate complexity score (0.0-1.0)
        # Based on operation count and feature diversity
        operation_factor = min(operation_count / 10.0, 1.0)  # Normalize to 10 ops
        feature_diversity = len([c for c in feature_categories.values() if c > 0]) / 6.0
        complexity_score = (operation_factor * 0.6) + (feature_diversity * 0.4)

        return ModelRequirementsAnalysis(
            node_type=node_type,
            operation_count=operation_count,
            required_features=required_features,
            complexity_score=complexity_score,
            feature_categories=feature_categories,
        )

    def _score_variants(
        self,
        analysis: ModelRequirementsAnalysis,
    ) -> dict[EnumTemplateVariant, float]:
        """
        Score all variants against requirements.

        Args:
            analysis: Requirements analysis

        Returns:
            Dictionary mapping variants to scores (0.0-1.0)
        """
        scores: dict[EnumTemplateVariant, float] = {}

        for variant, metadata in self._registry.items():
            # Check if variant matches requirements
            if not metadata.matches_requirements(
                node_type=analysis.node_type,
                operation_count=analysis.operation_count,
                required_features=analysis.required_features,
            ):
                continue

            # Calculate match score
            score = metadata.calculate_match_score(
                node_type=analysis.node_type,
                operation_count=analysis.operation_count,
                required_features=analysis.required_features,
            )

            # Apply specialization bonus
            specialization_bonus = self._calculate_specialization_bonus(
                variant=variant,
                analysis=analysis,
            )
            score = min(score + specialization_bonus, 1.0)

            scores[variant] = score

        return scores

    def _calculate_specialization_bonus(
        self,
        variant: EnumTemplateVariant,
        analysis: ModelRequirementsAnalysis,
    ) -> float:
        """
        Calculate specialization bonus for specific variants.

        Specialized variants (database_heavy, api_heavy, etc.) get a bonus
        when their domain is prominently featured.

        Args:
            variant: Template variant
            analysis: Requirements analysis

        Returns:
            Bonus score (0.0-0.2)
        """
        bonus = 0.0
        categories = analysis.feature_categories

        # Database-heavy bonus
        if variant == EnumTemplateVariant.DATABASE_HEAVY:
            if categories.get("database", 0) >= 3:
                bonus += 0.15

        # API-heavy bonus
        elif variant == EnumTemplateVariant.API_HEAVY:
            if categories.get("api", 0) >= 2:
                bonus += 0.15

        # Kafka-heavy bonus
        elif variant == EnumTemplateVariant.KAFKA_HEAVY:
            if categories.get("kafka", 0) >= 2:
                bonus += 0.15

        # ML inference bonus
        elif variant == EnumTemplateVariant.ML_INFERENCE:
            if categories.get("ml", 0) >= 2:
                bonus += 0.15

        # Analytics bonus
        elif variant == EnumTemplateVariant.ANALYTICS:
            if categories.get("analytics", 0) >= 2:
                bonus += 0.15

        # Workflow bonus
        elif variant == EnumTemplateVariant.WORKFLOW:
            if categories.get("workflow", 0) >= 2:
                bonus += 0.15

        # Complexity-based bonuses
        if analysis.complexity_score < 0.3:
            # Simple requirements → bonus for MINIMAL
            if variant == EnumTemplateVariant.MINIMAL:
                bonus += 0.1
        elif analysis.complexity_score > 0.7:
            # Complex requirements → bonus for PRODUCTION
            if variant == EnumTemplateVariant.PRODUCTION:
                bonus += 0.1

        return bonus

    def _fallback_selection(
        self,
        analysis: ModelRequirementsAnalysis,
    ) -> ModelTemplateSelection:
        """
        Fallback selection when no variants match.

        Uses STANDARD variant as default fallback for most cases,
        MINIMAL for simple requirements, PRODUCTION for complex.

        Args:
            analysis: Requirements analysis

        Returns:
            Fallback template selection
        """
        # Select fallback based on complexity
        if analysis.operation_count <= 2:
            fallback_variant = EnumTemplateVariant.MINIMAL
            rationale = "Fallback to MINIMAL: Simple requirements with ≤2 operations"
        elif analysis.operation_count >= 8 or analysis.complexity_score > 0.7:
            fallback_variant = EnumTemplateVariant.PRODUCTION
            rationale = (
                "Fallback to PRODUCTION: Complex requirements with many operations"
            )
        else:
            fallback_variant = EnumTemplateVariant.STANDARD
            rationale = "Fallback to STANDARD: No specialized variant matched"

        # Check node type compatibility
        metadata = self._registry[fallback_variant]
        if analysis.node_type not in metadata.node_types:
            # Fallback's fallback - use STANDARD (supports all node types)
            fallback_variant = EnumTemplateVariant.STANDARD
            rationale = f"Fallback to STANDARD: Node type {analysis.node_type.value} not supported by primary fallback"

        logger.warning("Using fallback variant selection: %s", fallback_variant.value)

        return ModelTemplateSelection(
            variant=fallback_variant,
            confidence=0.5,  # Low confidence for fallback
            rationale=rationale,
            matched_features=[],
            fallback_used=True,
            selection_time_ms=0.0,
        )

    def _generate_rationale(
        self,
        metadata: ModelVariantMetadata,
        analysis: ModelRequirementsAnalysis,
        score: float,
    ) -> str:
        """
        Generate human-readable rationale for selection.

        Args:
            metadata: Selected variant metadata
            analysis: Requirements analysis
            score: Match score

        Returns:
            Rationale string
        """
        reasons = []

        # Node type match
        reasons.append(f"Node type {analysis.node_type.value} is supported")

        # Operation count fit
        if metadata.max_operations is None:
            reasons.append(
                f"{analysis.operation_count} operations fits unlimited range"
            )
        else:
            reasons.append(
                f"{analysis.operation_count} operations fits range "
                f"{metadata.min_operations}-{metadata.max_operations}"
            )

        # Feature matches
        matched_features = set(metadata.features).intersection(
            analysis.required_features
        )
        if matched_features:
            reasons.append(
                f"{len(matched_features)} features matched: "
                f"{', '.join(list(matched_features)[:3])}"
            )

        # Specialization
        if metadata.variant in [
            EnumTemplateVariant.DATABASE_HEAVY,
            EnumTemplateVariant.API_HEAVY,
            EnumTemplateVariant.KAFKA_HEAVY,
            EnumTemplateVariant.ML_INFERENCE,
            EnumTemplateVariant.ANALYTICS,
            EnumTemplateVariant.WORKFLOW,
        ]:
            reasons.append(
                f"Specialized for {metadata.variant.value.replace('_', ' ')}"
            )

        rationale = (
            f"Selected {metadata.variant.value} (score: {score:.2f}). "
            + "; ".join(reasons)
        )
        return rationale

    def _make_cache_key(
        self,
        node_type: EnumNodeType,
        operation_count: int,
        required_features: set[str],
    ) -> str:
        """Create cache key from requirements."""
        features_str = ",".join(sorted(required_features))
        return f"{node_type.value}:{operation_count}:{features_str}"

    def _cache_selection(
        self,
        cache_key: str,
        selection: ModelTemplateSelection,
    ) -> None:
        """Cache selection result with LRU eviction."""
        if len(self._selection_cache) >= self._cache_max_size:
            # Remove oldest entry (first key)
            oldest_key = next(iter(self._selection_cache))
            del self._selection_cache[oldest_key]

        self._selection_cache[cache_key] = selection

    def get_variant_metadata(
        self,
        variant: EnumTemplateVariant,
    ) -> Optional[ModelVariantMetadata]:
        """
        Get metadata for a specific variant.

        Args:
            variant: Template variant

        Returns:
            Variant metadata or None if not found
        """
        return self._registry.get(variant)

    def list_variants(
        self,
        node_type: Optional[EnumNodeType] = None,
    ) -> list[EnumTemplateVariant]:
        """
        List available variants, optionally filtered by node type.

        Args:
            node_type: Optional node type filter

        Returns:
            List of available variants
        """
        if node_type is None:
            return list(self._registry.keys())

        return [
            variant
            for variant, metadata in self._registry.items()
            if node_type in metadata.node_types
        ]

    def clear_cache(self) -> None:
        """Clear selection cache."""
        self._selection_cache.clear()
        logger.info("Variant selection cache cleared")
