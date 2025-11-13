#!/usr/bin/env python3
"""
Production Pattern Library for intelligent pattern matching (Phase 3, Tasks C9-C11).

Provides unified interface to discover, match, and apply production patterns
from the Phase 2 pattern generators.

Features:
- Pattern discovery and categorization
- Similarity-based pattern matching
- Pattern validation
- Usage tracking and learning

Performance Target: <10ms per pattern search
Accuracy Target: >90% pattern match relevance
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from omninode_bridge.codegen.patterns import (
    ConsulPatternGenerator,
    EventPublishingPatternGenerator,
    HealthCheckGenerator,
    LifecyclePatternGenerator,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Pattern Models
# ============================================================================


class EnumPatternCategory(str, Enum):
    """Pattern category enumeration."""

    LIFECYCLE = "lifecycle"  # Startup/shutdown patterns
    HEALTH_CHECKS = "health_checks"  # Health check patterns
    METRICS = "metrics"  # Metrics collection
    EVENT_PUBLISHING = "event_publishing"  # Kafka events
    CONSUL_INTEGRATION = "consul_integration"  # Service discovery
    ERROR_HANDLING = "error_handling"  # Error handling
    RETRY_LOGIC = "retry_logic"  # Retry mechanisms
    CIRCUIT_BREAKER = "circuit_breaker"  # Circuit breaker


@dataclass
class ModelPatternInfo:
    """
    Information about a single pattern.

    Attributes:
        name: Pattern name/identifier
        category: Pattern category
        description: Human-readable description
        applicable_node_types: Node types this pattern applies to
        required_dependencies: Required dependencies
        tags: Additional tags for matching
        generator: Generator instance (if available)
    """

    name: str
    category: EnumPatternCategory
    description: str
    applicable_node_types: list[str] = field(default_factory=list)
    required_dependencies: list[str] = field(default_factory=list)
    tags: set[str] = field(default_factory=set)
    generator: Optional[Any] = None


@dataclass
class ModelPatternMatch:
    """
    Represents a matched pattern with scoring.

    Attributes:
        pattern_info: Pattern information
        relevance_score: Relevance score (0.0-1.0)
        match_reason: Why this pattern matched
        confidence: Confidence in match (0.0-1.0)
    """

    pattern_info: ModelPatternInfo
    relevance_score: float
    match_reason: str = ""
    confidence: float = 0.9


# ============================================================================
# Production Pattern Library
# ============================================================================


class ProductionPatternLibrary:
    """
    Unified interface to production pattern generators.

    Wraps Phase 2 pattern generators and provides intelligent pattern
    discovery, matching, and application.
    """

    def __init__(self):
        """Initialize pattern library with all available generators."""
        # Initialize pattern generators
        self.lifecycle_generator = LifecyclePatternGenerator()
        self.health_check_generator = HealthCheckGenerator()
        self.event_generator = EventPublishingPatternGenerator()
        self.consul_generator = ConsulPatternGenerator()

        # Build pattern catalog
        self.patterns = self._build_pattern_catalog()

        logger.info(
            f"ProductionPatternLibrary initialized with {len(self.patterns)} patterns"
        )

    def find_matching_patterns(
        self,
        operation_type: str,
        features: set[str],
        node_type: str,
        min_relevance: float = 0.5,
    ) -> list[ModelPatternMatch]:
        """
        Find patterns matching given criteria.

        Args:
            operation_type: Type of operation (e.g., "database", "api", "kafka")
            features: Set of required features
            node_type: Node type (effect/compute/reducer/orchestrator)
            min_relevance: Minimum relevance score (default: 0.5)

        Returns:
            List of matching patterns sorted by relevance (descending)

        Example:
            >>> library = ProductionPatternLibrary()
            >>> matches = library.find_matching_patterns(
            ...     operation_type="database",
            ...     features={"health_checks", "metrics"},
            ...     node_type="effect"
            ... )
            >>> for match in matches:
            ...     print(f"{match.pattern_info.name}: {match.relevance_score:.2f}")
        """
        import time

        start_time = time.perf_counter()

        matches = []

        for pattern in self.patterns.values():
            # Check node type applicability
            if (
                pattern.applicable_node_types
                and node_type not in pattern.applicable_node_types
            ):
                continue

            # Calculate relevance score
            relevance = self._calculate_relevance(pattern, operation_type, features)

            if relevance >= min_relevance:
                match_reason = self._build_match_reason(
                    pattern, operation_type, features, relevance
                )

                matches.append(
                    ModelPatternMatch(
                        pattern_info=pattern,
                        relevance_score=relevance,
                        match_reason=match_reason,
                        confidence=0.9,
                    )
                )

        # Sort by relevance (descending)
        matches.sort(key=lambda m: m.relevance_score, reverse=True)

        search_time_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            f"Pattern search: found {len(matches)} matches "
            f"(time: {search_time_ms:.2f}ms)"
        )

        return matches

    def get_pattern_by_name(self, pattern_name: str) -> Optional[ModelPatternInfo]:
        """
        Get pattern information by name.

        Args:
            pattern_name: Pattern name

        Returns:
            Pattern info or None if not found
        """
        return self.patterns.get(pattern_name)

    def get_all_patterns(self) -> dict[str, ModelPatternInfo]:
        """
        Get all available patterns.

        Returns:
            Dictionary of pattern name → pattern info
        """
        return self.patterns

    def get_patterns_by_category(
        self, category: EnumPatternCategory
    ) -> list[ModelPatternInfo]:
        """
        Get all patterns in a category.

        Args:
            category: Pattern category

        Returns:
            List of patterns in that category
        """
        return [
            pattern
            for pattern in self.patterns.values()
            if pattern.category == category
        ]

    # ========================================================================
    # Pattern Catalog Building
    # ========================================================================

    def _build_pattern_catalog(self) -> dict[str, ModelPatternInfo]:
        """
        Build catalog of all available patterns.

        Returns:
            Dictionary of pattern name → pattern info
        """
        patterns = {}

        # Lifecycle patterns
        patterns["lifecycle"] = ModelPatternInfo(
            name="lifecycle",
            category=EnumPatternCategory.LIFECYCLE,
            description="Startup and shutdown lifecycle management",
            applicable_node_types=["effect", "compute", "reducer", "orchestrator"],
            required_dependencies=[],
            tags={"startup", "shutdown", "initialization", "cleanup"},
            generator=self.lifecycle_generator,
        )

        # Health check patterns
        patterns["health_checks"] = ModelPatternInfo(
            name="health_checks",
            category=EnumPatternCategory.HEALTH_CHECKS,
            description="Comprehensive health monitoring",
            applicable_node_types=["effect", "compute", "reducer", "orchestrator"],
            required_dependencies=[],
            tags={"health", "monitoring", "liveness", "readiness"},
            generator=self.health_check_generator,
        )

        patterns["database_health"] = ModelPatternInfo(
            name="database_health",
            category=EnumPatternCategory.HEALTH_CHECKS,
            description="Database connection health checks",
            applicable_node_types=["effect"],
            required_dependencies=["asyncpg", "psycopg2", "psycopg3", "sqlalchemy"],
            tags={"health", "database", "postgres", "mysql"},
            generator=self.health_check_generator,
        )

        patterns["kafka_health"] = ModelPatternInfo(
            name="kafka_health",
            category=EnumPatternCategory.HEALTH_CHECKS,
            description="Kafka connection health checks",
            applicable_node_types=["effect"],
            required_dependencies=["aiokafka", "confluent-kafka"],
            tags={"health", "kafka", "messaging", "events"},
            generator=self.health_check_generator,
        )

        # Metrics patterns
        patterns["metrics"] = ModelPatternInfo(
            name="metrics",
            category=EnumPatternCategory.METRICS,
            description="Comprehensive metrics collection",
            applicable_node_types=["effect", "compute", "reducer", "orchestrator"],
            required_dependencies=[],
            tags={"metrics", "observability", "prometheus", "monitoring"},
        )

        # Event publishing patterns
        patterns["event_publishing"] = ModelPatternInfo(
            name="event_publishing",
            category=EnumPatternCategory.EVENT_PUBLISHING,
            description="OnexEnvelopeV1 event publishing",
            applicable_node_types=["effect", "compute"],
            required_dependencies=["aiokafka"],
            tags={"events", "kafka", "publish", "messaging"},
            generator=self.event_generator,
        )

        # Consul integration patterns
        patterns["consul_integration"] = ModelPatternInfo(
            name="consul_integration",
            category=EnumPatternCategory.CONSUL_INTEGRATION,
            description="Service discovery and registration",
            applicable_node_types=["effect", "compute", "reducer", "orchestrator"],
            required_dependencies=["python-consul"],
            tags={"consul", "service-discovery", "registration"},
            generator=self.consul_generator,
        )

        # Error handling patterns (conceptual - no generator yet)
        patterns["error_handling"] = ModelPatternInfo(
            name="error_handling",
            category=EnumPatternCategory.ERROR_HANDLING,
            description="Structured error handling and recovery",
            applicable_node_types=["effect", "compute", "reducer", "orchestrator"],
            required_dependencies=[],
            tags={"errors", "exceptions", "recovery", "resilience"},
        )

        # Retry logic patterns (conceptual - no generator yet)
        patterns["retry_logic"] = ModelPatternInfo(
            name="retry_logic",
            category=EnumPatternCategory.RETRY_LOGIC,
            description="Exponential backoff retry mechanisms",
            applicable_node_types=["effect", "compute"],
            required_dependencies=[],
            tags={"retry", "backoff", "resilience", "fault-tolerance"},
        )

        # Circuit breaker patterns (conceptual - no generator yet)
        patterns["circuit_breaker"] = ModelPatternInfo(
            name="circuit_breaker",
            category=EnumPatternCategory.CIRCUIT_BREAKER,
            description="Circuit breaker for fault tolerance",
            applicable_node_types=["effect", "compute"],
            required_dependencies=[],
            tags={"circuit-breaker", "resilience", "fault-tolerance"},
        )

        return patterns

    # ========================================================================
    # Pattern Matching
    # ========================================================================

    def _calculate_relevance(
        self,
        pattern: ModelPatternInfo,
        operation_type: str,
        features: set[str],
    ) -> float:
        """
        Calculate relevance score for a pattern.

        Args:
            pattern: Pattern info
            operation_type: Operation type
            features: Required features

        Returns:
            Relevance score (0.0-1.0)
        """
        score = 0.0
        weights = []

        # Check if operation type in tags
        if operation_type.lower() in pattern.tags:
            score += 0.4
            weights.append(0.4)

        # Check feature overlap
        feature_overlap = len(features & pattern.tags)
        if feature_overlap > 0:
            feature_score = min(feature_overlap / len(features), 0.6)
            score += feature_score
            weights.append(feature_score)

        # Check if pattern name in features
        if pattern.name in features:
            score += 0.3
            weights.append(0.3)

        # Normalize by number of factors
        if weights:
            score = score / len(weights)

        return min(score, 1.0)

    def _build_match_reason(
        self,
        pattern: ModelPatternInfo,
        operation_type: str,
        features: set[str],
        relevance: float,
    ) -> str:
        """
        Build human-readable match reason.

        Args:
            pattern: Pattern info
            operation_type: Operation type
            features: Required features
            relevance: Relevance score

        Returns:
            Match reason string
        """
        reasons = []

        # Check operation type match
        if operation_type.lower() in pattern.tags:
            reasons.append(f"matches operation type '{operation_type}'")

        # Check feature matches
        feature_matches = features & pattern.tags
        if feature_matches:
            reasons.append(f"matches features: {', '.join(sorted(feature_matches))}")

        # Check if pattern name explicitly requested
        if pattern.name in features:
            reasons.append("explicitly requested")

        if not reasons:
            reasons.append(f"general applicability (score: {relevance:.2f})")

        return "; ".join(reasons)


__all__ = [
    "ProductionPatternLibrary",
    "ModelPatternInfo",
    "ModelPatternMatch",
    "EnumPatternCategory",
]
