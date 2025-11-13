"""
Pattern Matcher for Code Generation.

This module implements hybrid semantic + structural pattern matching
to find the most relevant patterns for given requirements.

Matching Algorithm:
- Feature overlap: 70% weight (Jaccard similarity on tags)
- Node type compatibility: 30% weight (binary match)

Performance Target: <10ms to match against 21 patterns
"""

import logging
from typing import Optional

from .models import (
    EnumNodeType,
    EnumPatternCategory,
    ModelPatternMatch,
    ModelPatternMetadata,
    ModelPatternQuery,
)
from .pattern_registry import PatternRegistry

logger = logging.getLogger(__name__)


class PatternMatcher:
    """
    Match node requirements to relevant patterns.

    This class implements a hybrid matching algorithm that combines
    semantic (feature-based) and structural (node type) matching
    to find the most relevant patterns.

    Attributes:
        registry: Pattern registry for accessing patterns
        _feature_cache: Cache of computed feature vectors

    Performance:
        - Matching against 21 patterns: <10ms
        - Feature overlap calculation: <0.5ms per pattern
        - Node type compatibility: <0.1ms per pattern
    """

    def __init__(self, registry: Optional[PatternRegistry] = None):
        """
        Initialize pattern matcher.

        Args:
            registry: Optional PatternRegistry instance
        """
        self.registry = registry or PatternRegistry()
        self._feature_cache: dict[str, set[str]] = {}

        logger.debug("PatternMatcher initialized")

    def match_patterns(
        self,
        node_type: EnumNodeType,
        required_features: set[str],
        top_k: int = 5,
        min_score: float = 0.3,
        categories: Optional[list[EnumPatternCategory]] = None,
    ) -> list[ModelPatternMatch]:
        """
        Find matching patterns for given requirements.

        This is the primary matching method that combines feature overlap
        and node type compatibility to rank patterns.

        Args:
            node_type: Target node type
            required_features: Set of required feature tags
            top_k: Maximum number of patterns to return
            min_score: Minimum match score threshold (0.0-1.0)
            categories: Optional filter by categories

        Returns:
            List of ModelPatternMatch objects, sorted by score descending

        Performance: <10ms for 21 patterns

        Example:
            >>> matcher = PatternMatcher()
            >>> matches = matcher.match_patterns(
            ...     node_type=EnumNodeType.EFFECT,
            ...     required_features={"async", "database", "error-handling"},
            ...     top_k=3
            ... )
            >>> for match in matches:
            ...     print(f"{match.pattern.name}: {match.score:.2f}")
        """
        # Ensure patterns are loaded
        self.registry.load_patterns()

        # Get all patterns
        all_patterns = list(self.registry._patterns.values())

        # Filter by category if specified
        if categories:
            all_patterns = [p for p in all_patterns if p.category in categories]

        # Calculate match scores
        matches = []
        for pattern in all_patterns:
            # Check node type compatibility first (fast filter)
            if node_type not in pattern.applicable_to:
                continue

            # Calculate match score
            score = self._calculate_match_score(pattern, node_type, required_features)

            # Filter by minimum score
            if score < min_score:
                continue

            # Calculate feature overlap for rationale
            matched_features = self._get_matched_features(pattern, required_features)

            # Generate rationale
            rationale = self._generate_rationale(
                pattern, node_type, matched_features, score
            )

            # Create match object
            match = ModelPatternMatch(
                pattern=pattern,
                score=score,
                rationale=rationale,
                matched_features=list(matched_features),
            )

            matches.append(match)

        # Sort by score descending
        matches.sort(key=lambda m: m.score, reverse=True)

        # Take top-k
        top_matches = matches[:top_k]

        logger.debug(
            f"Matched {len(matches)} patterns, returning top {len(top_matches)}"
        )

        return top_matches

    def match_from_query(self, query: ModelPatternQuery) -> list[ModelPatternMatch]:
        """
        Match patterns using a query object.

        Args:
            query: Pattern query with all parameters

        Returns:
            List of matching patterns

        Performance: <10ms
        """
        matches = self.match_patterns(
            node_type=query.node_type,
            required_features=query.required_features,
            top_k=query.max_results,
            min_score=query.min_score,
            categories=query.categories,
        )

        # Apply complexity filter if specified
        if query.exclude_complex is not None:
            matches = [
                m for m in matches if m.pattern.complexity <= query.exclude_complex
            ]

        return matches

    def _calculate_match_score(
        self,
        pattern: ModelPatternMetadata,
        node_type: EnumNodeType,
        required_features: set[str],
    ) -> float:
        """
        Calculate overall match score.

        Scoring algorithm:
        - Node type compatibility: 30% weight
        - Feature overlap: 70% weight

        Args:
            pattern: Pattern to score
            node_type: Target node type
            required_features: Required features

        Returns:
            Match score between 0.0 and 1.0

        Performance: <0.5ms per pattern
        """
        # Node type compatibility (30% weight)
        node_type_score = self._calculate_node_type_compatibility(pattern, node_type)

        # Feature overlap (70% weight)
        feature_score = self._calculate_feature_overlap(pattern, required_features)

        # Weighted combination
        total_score = (0.3 * node_type_score) + (0.7 * feature_score)

        return min(total_score, 1.0)

    def _calculate_feature_overlap(
        self,
        pattern: ModelPatternMetadata,
        required_features: set[str],
    ) -> float:
        """
        Calculate feature overlap score using Jaccard similarity.

        Jaccard similarity = |A ∩ B| / |A ∪ B|

        Args:
            pattern: Pattern to score
            required_features: Required features

        Returns:
            Overlap score between 0.0 and 1.0

        Performance: <0.3ms per pattern
        """
        # Get pattern features (tags)
        pattern_features = self._get_pattern_features(pattern)

        if not pattern_features or not required_features:
            return 0.0

        # Calculate Jaccard similarity
        intersection = len(pattern_features.intersection(required_features))
        union = len(pattern_features.union(required_features))

        if union == 0:
            return 0.0

        jaccard = intersection / union

        return jaccard

    def _calculate_node_type_compatibility(
        self,
        pattern: ModelPatternMetadata,
        node_type: EnumNodeType,
    ) -> float:
        """
        Calculate node type compatibility score.

        Args:
            pattern: Pattern to score
            node_type: Target node type

        Returns:
            1.0 if compatible, 0.0 otherwise

        Performance: <0.1ms per pattern
        """
        return 1.0 if node_type in pattern.applicable_to else 0.0

    def _get_pattern_features(self, pattern: ModelPatternMetadata) -> set[str]:
        """
        Get feature set for a pattern (with caching).

        Args:
            pattern: Pattern to extract features from

        Returns:
            Set of feature tags

        Performance: <0.1ms (cached)
        """
        # Check cache
        if pattern.pattern_id in self._feature_cache:
            return self._feature_cache[pattern.pattern_id]

        # Extract features from tags
        features = set(tag.lower() for tag in pattern.tags)

        # Cache it
        self._feature_cache[pattern.pattern_id] = features

        return features

    def _get_matched_features(
        self,
        pattern: ModelPatternMetadata,
        required_features: set[str],
    ) -> set[str]:
        """
        Get the features that matched between pattern and requirements.

        Args:
            pattern: Pattern
            required_features: Required features

        Returns:
            Set of matched features

        Performance: <0.1ms
        """
        pattern_features = self._get_pattern_features(pattern)
        return pattern_features.intersection(required_features)

    def _generate_rationale(
        self,
        pattern: ModelPatternMetadata,
        node_type: EnumNodeType,
        matched_features: set[str],
        score: float,
    ) -> str:
        """
        Generate human-readable rationale for why a pattern matched.

        Args:
            pattern: Matched pattern
            node_type: Target node type
            matched_features: Features that matched
            score: Match score

        Returns:
            Rationale string

        Performance: <0.1ms
        """
        parts = []

        # Node type match
        parts.append(f"Applicable to {node_type.value} nodes")

        # Feature matches
        if matched_features:
            features_str = ", ".join(sorted(matched_features))
            parts.append(f"Matches features: {features_str}")

        # Score
        parts.append(f"Score: {score:.2f}")

        # Pattern description (truncated)
        desc = pattern.description[:100]
        if len(pattern.description) > 100:
            desc += "..."
        parts.append(f"Pattern: {desc}")

        return ". ".join(parts)

    def clear_cache(self) -> None:
        """Clear feature cache (useful for testing)."""
        self._feature_cache.clear()
        logger.debug("PatternMatcher cache cleared")
