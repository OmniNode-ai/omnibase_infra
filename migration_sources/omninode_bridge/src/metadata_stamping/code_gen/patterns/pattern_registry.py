"""
Pattern Registry for Code Generation.

Central registry for managing and querying production patterns.
Provides pattern lookup, usage tracking, and statistics.

Performance Target: <1ms for pattern queries
"""

import logging
from collections import defaultdict
from datetime import datetime
from typing import Optional

from .models import (
    EnumNodeType,
    EnumPatternCategory,
    ModelPatternLibraryStats,
    ModelPatternMetadata,
)
from .pattern_loader import PatternLoader

logger = logging.getLogger(__name__)


class PatternRegistry:
    """
    Central registry for production patterns.

    This class provides a unified interface for accessing patterns,
    tracking usage statistics, and querying by various criteria.

    Attributes:
        loader: PatternLoader for loading patterns
        _patterns: Loaded patterns cache
        _usage_stats: Pattern usage tracking

    Performance:
        - Pattern lookup: <1ms
        - Statistics calculation: <5ms
        - Usage recording: <0.1ms
    """

    def __init__(self, loader: Optional[PatternLoader] = None):
        """
        Initialize pattern registry.

        Args:
            loader: Optional PatternLoader instance
        """
        self.loader = loader or PatternLoader()
        self._patterns: dict[str, ModelPatternMetadata] = {}
        self._usage_stats: dict[str, int] = defaultdict(int)
        self._last_loaded: Optional[datetime] = None

        logger.debug("PatternRegistry initialized")

    def load_patterns(self, force_reload: bool = False) -> None:
        """
        Load all patterns into the registry.

        Args:
            force_reload: Force reload even if already loaded

        Performance: ~20-50ms cold, <1ms warm
        """
        if self._patterns and not force_reload:
            logger.debug("Patterns already loaded, skipping")
            return

        logger.info("Loading patterns into registry...")
        self._patterns = self.loader.load_all_patterns()
        self._last_loaded = datetime.utcnow()
        logger.info(f"Loaded {len(self._patterns)} patterns into registry")

    def get_pattern(self, pattern_id: str) -> Optional[ModelPatternMetadata]:
        """
        Get a pattern by ID.

        Args:
            pattern_id: Pattern identifier (e.g., "error_handling_v1")

        Returns:
            Pattern metadata or None if not found

        Performance: <1ms
        """
        if not self._patterns:
            self.load_patterns()

        return self._patterns.get(pattern_id)

    def get_pattern_by_name(self, name: str) -> Optional[ModelPatternMetadata]:
        """
        Get a pattern by name.

        Args:
            name: Pattern name (e.g., "error_handling")

        Returns:
            Pattern metadata or None if not found

        Performance: <1ms
        """
        if not self._patterns:
            self.load_patterns()

        for pattern in self._patterns.values():
            if pattern.name == name:
                return pattern

        return None

    def get_applicable_patterns(
        self,
        node_type: EnumNodeType,
        features: set[str],
        min_score: float = 0.3,
    ) -> list[ModelPatternMetadata]:
        """
        Get patterns applicable to a node type with required features.

        This is the primary method for finding patterns during code generation.

        Args:
            node_type: Target node type
            features: Required features/tags
            min_score: Minimum match score threshold

        Returns:
            List of applicable patterns, sorted by match score

        Performance: <5ms for 21 patterns
        """
        if not self._patterns:
            self.load_patterns()

        applicable = []

        for pattern in self._patterns.values():
            # Calculate match score
            score = pattern.calculate_match_score(node_type, features)

            if score >= min_score:
                applicable.append((score, pattern))

        # Sort by score descending
        applicable.sort(key=lambda x: x[0], reverse=True)

        return [pattern for score, pattern in applicable]

    def get_patterns_by_category(
        self, category: EnumPatternCategory
    ) -> list[ModelPatternMetadata]:
        """
        Get all patterns in a category.

        Args:
            category: Pattern category

        Returns:
            List of patterns in category

        Performance: <1ms
        """
        if not self._patterns:
            self.load_patterns()

        return [p for p in self._patterns.values() if p.category == category]

    def get_patterns_by_node_type(
        self, node_type: EnumNodeType
    ) -> list[ModelPatternMetadata]:
        """
        Get all patterns applicable to a node type.

        Args:
            node_type: Node type

        Returns:
            List of applicable patterns

        Performance: <1ms
        """
        if not self._patterns:
            self.load_patterns()

        return [p for p in self._patterns.values() if node_type in p.applicable_to]

    def search_patterns(
        self,
        query: str,
        search_fields: Optional[list[str]] = None,
    ) -> list[ModelPatternMetadata]:
        """
        Search patterns by text query.

        Args:
            query: Search query
            search_fields: Fields to search (default: name, description, tags)

        Returns:
            Matching patterns

        Performance: <5ms
        """
        if not self._patterns:
            self.load_patterns()

        if search_fields is None:
            search_fields = ["name", "description", "tags"]

        query_lower = query.lower()
        matches = []

        for pattern in self._patterns.values():
            # Check each field
            for field in search_fields:
                value = getattr(pattern, field, None)

                if value is None:
                    continue

                # Handle different field types
                if isinstance(value, str):
                    if query_lower in value.lower():
                        matches.append(pattern)
                        break
                elif isinstance(value, list):
                    # For lists (like tags), check each item
                    if any(query_lower in str(item).lower() for item in value):
                        matches.append(pattern)
                        break

        return matches

    def record_usage(self, pattern_id: str) -> None:
        """
        Record that a pattern was used.

        This tracks usage statistics for analytics.

        Args:
            pattern_id: Pattern identifier

        Performance: <0.1ms
        """
        self._usage_stats[pattern_id] += 1
        logger.debug(f"Recorded usage for pattern {pattern_id}")

    def get_usage_stats(self) -> dict[str, int]:
        """
        Get usage statistics for all patterns.

        Returns:
            Dictionary mapping pattern_id to usage count

        Performance: <1ms
        """
        return dict(self._usage_stats)

    def get_most_used_patterns(self, limit: int = 10) -> list[tuple[str, int]]:
        """
        Get the most frequently used patterns.

        Args:
            limit: Maximum number of patterns to return

        Returns:
            List of (pattern_id, usage_count) tuples

        Performance: <1ms
        """
        sorted_usage = sorted(
            self._usage_stats.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_usage[:limit]

    def get_library_stats(self) -> ModelPatternLibraryStats:
        """
        Get statistics about the pattern library.

        Returns:
            Pattern library statistics

        Performance: <5ms
        """
        if not self._patterns:
            self.load_patterns()

        # Count by category
        patterns_by_category = defaultdict(int)
        for pattern in self._patterns.values():
            patterns_by_category[pattern.category.value] += 1

        # Count by node type
        patterns_by_node_type = defaultdict(int)
        for pattern in self._patterns.values():
            for node_type in pattern.applicable_to:
                patterns_by_node_type[node_type.value] += 1

        # Calculate average complexity
        if self._patterns:
            avg_complexity = sum(p.complexity for p in self._patterns.values()) / len(
                self._patterns
            )
        else:
            avg_complexity = 0.0

        return ModelPatternLibraryStats(
            total_patterns=len(self._patterns),
            patterns_by_category=dict(patterns_by_category),
            patterns_by_node_type=dict(patterns_by_node_type),
            average_complexity=round(avg_complexity, 2),
            last_updated=self._last_loaded or datetime.utcnow(),
        )

    def get_critical_patterns(self) -> list[ModelPatternMetadata]:
        """
        Get patterns marked as critical priority.

        These patterns should be applied to all generated nodes.

        Returns:
            List of critical patterns

        Performance: <1ms
        """
        if not self._patterns:
            self.load_patterns()

        # Critical patterns are typically structure patterns
        # that define the basic node skeleton
        critical_names = {
            "standard_imports",
            "class_declaration",
            "initialization_pattern",
            "error_handling",
        }

        return [p for p in self._patterns.values() if p.name in critical_names]

    def clear(self) -> None:
        """Clear registry cache (useful for testing)."""
        self._patterns.clear()
        self._usage_stats.clear()
        self._last_loaded = None
        self.loader.clear_cache()
        logger.debug("PatternRegistry cleared")
