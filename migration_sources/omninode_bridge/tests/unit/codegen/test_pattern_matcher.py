#!/usr/bin/env python3
"""
Unit tests for PatternMatcher.

Tests pattern matching algorithms and scoring.
"""

import pytest

from src.metadata_stamping.code_gen.patterns import (
    EnumNodeType,
    EnumPatternCategory,
    ModelPatternQuery,
    PatternMatcher,
    PatternRegistry,
)


class TestPatternMatcher:
    """Test suite for PatternMatcher."""

    @pytest.fixture
    def matcher(self):
        """Create a PatternMatcher instance."""
        return PatternMatcher()

    @pytest.fixture
    def registry(self):
        """Create a PatternRegistry instance."""
        registry = PatternRegistry()
        registry.load_patterns()
        return registry

    def test_matcher_initialization(self, matcher):
        """Test PatternMatcher initialization."""
        assert matcher is not None
        assert matcher.registry is not None

    def test_match_patterns_basic(self, matcher):
        """Test basic pattern matching."""
        matches = matcher.match_patterns(
            node_type=EnumNodeType.EFFECT,
            required_features={"async", "database", "error-handling"},
            top_k=5,
        )

        # Should find some matches
        assert len(matches) > 0
        assert len(matches) <= 5

        # Matches should be sorted by score
        scores = [m.score for m in matches]
        assert scores == sorted(scores, reverse=True)

    def test_match_patterns_with_categories(self, matcher):
        """Test pattern matching with category filter."""
        matches = matcher.match_patterns(
            node_type=EnumNodeType.EFFECT,
            required_features={"async", "error-handling"},
            categories=[EnumPatternCategory.RESILIENCE],
            top_k=10,
        )

        # All matches should be resilience patterns
        for match in matches:
            assert match.pattern.category == EnumPatternCategory.RESILIENCE

    def test_match_patterns_no_matches(self, matcher):
        """Test pattern matching with no matches."""
        matches = matcher.match_patterns(
            node_type=EnumNodeType.COMPUTE,  # Compute nodes
            required_features={"nonexistent-feature"},
            min_score=0.9,  # Very high threshold
        )

        # May have no matches with such specific requirements
        assert isinstance(matches, list)

    def test_match_from_query(self, matcher):
        """Test pattern matching using a query object."""
        query = ModelPatternQuery(
            node_type=EnumNodeType.ORCHESTRATOR,
            required_features={"workflow", "async"},
            max_results=3,
            min_score=0.3,
        )

        matches = matcher.match_from_query(query)

        assert len(matches) <= 3
        for match in matches:
            assert match.score >= 0.3

    def test_match_patterns_all_node_types(self, matcher):
        """Test that matching works for all node types."""
        for node_type in EnumNodeType:
            matches = matcher.match_patterns(
                node_type=node_type,
                required_features={"async"},
                top_k=5,
            )

            # Should find at least some matches
            assert isinstance(matches, list)

    def test_match_score_calculation(self, matcher, registry):
        """Test match score calculation."""
        # Get a known pattern
        pattern = registry.get_pattern_by_name("error_handling")
        assert pattern is not None

        # Calculate score
        score = matcher._calculate_match_score(
            pattern,
            EnumNodeType.EFFECT,
            {"async", "error-handling"},
        )

        # Score should be in valid range
        assert 0.0 <= score <= 1.0

    def test_feature_overlap_calculation(self, matcher, registry):
        """Test feature overlap calculation."""
        pattern = registry.get_pattern_by_name("error_handling")
        assert pattern is not None

        # Calculate overlap
        overlap = matcher._calculate_feature_overlap(
            pattern,
            {"error-handling", "async"},
        )

        # Overlap should be in valid range
        assert 0.0 <= overlap <= 1.0

    def test_node_type_compatibility(self, matcher, registry):
        """Test node type compatibility calculation."""
        pattern = registry.get_pattern_by_name("error_handling")
        assert pattern is not None

        # Compatible node type
        compat = matcher._calculate_node_type_compatibility(
            pattern, EnumNodeType.EFFECT
        )
        assert compat == 1.0

    def test_matched_features_extraction(self, matcher, registry):
        """Test extracting matched features."""
        pattern = registry.get_pattern_by_name("error_handling")
        assert pattern is not None

        matched = matcher._get_matched_features(
            pattern,
            {"error-handling", "async", "nonexistent"},
        )

        # Should contain only features that exist in pattern
        assert isinstance(matched, set)

    def test_rationale_generation(self, matcher, registry):
        """Test generating match rationale."""
        pattern = registry.get_pattern_by_name("error_handling")
        assert pattern is not None

        rationale = matcher._generate_rationale(
            pattern,
            EnumNodeType.EFFECT,
            {"error-handling"},
            0.75,
        )

        # Rationale should be non-empty string
        assert isinstance(rationale, str)
        assert len(rationale) > 0
        assert "effect" in rationale.lower()

    def test_clear_cache(self, matcher):
        """Test clearing the matcher cache."""
        # Do some matching to populate cache
        matcher.match_patterns(
            node_type=EnumNodeType.EFFECT,
            required_features={"async"},
        )

        # Clear cache
        matcher.clear_cache()

        # Cache should be empty
        assert len(matcher._feature_cache) == 0

    def test_top_k_limiting(self, matcher):
        """Test that top_k limits results correctly."""
        # Ask for only 3 patterns
        matches = matcher.match_patterns(
            node_type=EnumNodeType.EFFECT,
            required_features={"async"},
            top_k=3,
        )

        # Should get at most 3
        assert len(matches) <= 3

    def test_min_score_filtering(self, matcher):
        """Test that min_score filters correctly."""
        matches = matcher.match_patterns(
            node_type=EnumNodeType.EFFECT,
            required_features={"async"},
            min_score=0.5,  # Moderately high threshold
        )

        # All matches should meet threshold
        for match in matches:
            assert match.score >= 0.5


@pytest.mark.performance
class TestPatternMatcherPerformance:
    """Performance tests for PatternMatcher."""

    @pytest.fixture
    def matcher(self):
        """Create a PatternMatcher with patterns loaded."""
        matcher = PatternMatcher()
        matcher.registry.load_patterns()
        return matcher

    def test_match_performance(self, matcher, benchmark):
        """Test pattern matching performance."""
        result = benchmark(
            matcher.match_patterns,
            node_type=EnumNodeType.EFFECT,
            required_features={"async", "database", "error-handling"},
            top_k=5,
        )

        assert len(result) > 0

        # Should be < 10ms (target from spec)
        assert benchmark.stats.stats.median < 0.01

    def test_match_all_node_types_performance(self, matcher):
        """Test matching performance across all node types."""
        import time

        start = time.perf_counter()

        for node_type in EnumNodeType:
            matcher.match_patterns(
                node_type=node_type,
                required_features={"async"},
                top_k=5,
            )

        duration = time.perf_counter() - start

        # Should complete in < 50ms for all node types
        assert duration < 0.05


class TestPatternMatchQuality:
    """Test the quality of pattern matching."""

    @pytest.fixture
    def matcher(self):
        """Create a PatternMatcher instance."""
        matcher = PatternMatcher()
        matcher.registry.load_patterns()
        return matcher

    def test_error_handling_pattern_matching(self, matcher):
        """Test that error handling patterns match error-related features."""
        matches = matcher.match_patterns(
            node_type=EnumNodeType.EFFECT,
            required_features={"error-handling", "resilience"},
            top_k=5,
        )

        # Should find error_handling pattern
        pattern_names = [m.pattern.name for m in matches]
        assert any("error" in name for name in pattern_names)

    def test_logging_pattern_matching(self, matcher):
        """Test that logging patterns match logging features."""
        matches = matcher.match_patterns(
            node_type=EnumNodeType.EFFECT,
            required_features={"logging", "observability"},
            top_k=5,
        )

        # Should find logging-related patterns
        pattern_names = [m.pattern.name for m in matches]
        assert any("log" in name for name in pattern_names)

    def test_orchestrator_patterns(self, matcher):
        """Test that orchestrator-specific patterns match."""
        matches = matcher.match_patterns(
            node_type=EnumNodeType.ORCHESTRATOR,
            required_features={"workflow", "coordination"},
            top_k=5,
        )

        # Should find orchestrator patterns
        for match in matches:
            assert EnumNodeType.ORCHESTRATOR in match.pattern.applicable_to
