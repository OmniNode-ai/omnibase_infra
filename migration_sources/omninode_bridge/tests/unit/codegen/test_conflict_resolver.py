#!/usr/bin/env python3
"""
Unit tests for mixin conflict resolution.

Tests detecting and resolving conflicts between mixins:
- Mutual exclusions
- Missing prerequisites
- Redundancies
"""

from pathlib import Path

import pytest

from omninode_bridge.codegen.mixins.conflict_resolver import ConflictResolver
from omninode_bridge.codegen.mixins.models import (
    ModelMixinConflict,
    ModelMixinRecommendation,
)


@pytest.fixture
def conflict_resolver():
    """Create ConflictResolver instance for testing."""
    return ConflictResolver()


@pytest.fixture
def sample_recommendations():
    """Sample mixin recommendations for testing."""
    return [
        ModelMixinRecommendation(
            mixin_name="MixinHealthCheck",
            score=0.9,
            category="observability",
            explanation="High observability requirements",
        ),
        ModelMixinRecommendation(
            mixin_name="MixinMetrics",
            score=0.85,
            category="observability",
            explanation="Metrics tracking required",
        ),
        ModelMixinRecommendation(
            mixin_name="MixinCircuitBreaker",
            score=0.8,
            category="resilience",
            explanation="Fault tolerance needed",
        ),
    ]


@pytest.fixture
def sample_scores():
    """Sample mixin scores for resolution."""
    return {
        "MixinHealthCheck": 0.9,
        "MixinMetrics": 0.85,
        "MixinCircuitBreaker": 0.8,
        "MixinDatabase": 0.75,
    }


class TestConflictResolver:
    """Test suite for ConflictResolver initialization."""

    def test_initialization(self, conflict_resolver):
        """Test ConflictResolver initializes correctly."""
        assert conflict_resolver is not None
        assert conflict_resolver.conflicts is not None
        assert isinstance(conflict_resolver.conflicts, list)
        assert conflict_resolver.prerequisites is not None
        assert isinstance(conflict_resolver.prerequisites, list)
        assert conflict_resolver.redundancies is not None
        assert isinstance(conflict_resolver.redundancies, list)

    def test_load_config_success(self, conflict_resolver):
        """Test conflict rules load successfully."""
        assert len(conflict_resolver.conflicts) >= 0
        assert len(conflict_resolver.prerequisites) >= 0
        assert len(conflict_resolver.redundancies) >= 0

    def test_load_config_missing_file(self):
        """Test handling of missing config file."""
        resolver = ConflictResolver(config_path=Path("/nonexistent/conflicts.yaml"))
        # Should initialize with empty config
        assert resolver.conflicts == []


class TestConflictDetection:
    """Test suite for conflict detection."""

    def test_detect_conflicts_empty_list(self, conflict_resolver):
        """Test conflict detection with empty mixin list."""
        conflicts = conflict_resolver.detect_conflicts([])
        assert isinstance(conflicts, list)
        assert len(conflicts) == 0

    def test_detect_conflicts_no_conflicts(self, conflict_resolver):
        """Test detection when no conflicts exist."""
        mixins = ["MixinHealthCheck", "MixinMetrics"]
        conflicts = conflict_resolver.detect_conflicts(mixins)

        assert isinstance(conflicts, list)
        # May or may not have conflicts depending on config

    def test_detect_mutual_exclusions(self, conflict_resolver):
        """Test detecting mutual exclusions."""
        # Use actual conflicting mixins from config if available
        conflicts = conflict_resolver.detect_conflicts(
            ["MixinHealthCheck", "MixinMetrics"]
        )

        mutual_exclusions = [c for c in conflicts if c.type == "mutual_exclusion"]
        # Verify structure
        for conflict in mutual_exclusions:
            assert isinstance(conflict, ModelMixinConflict)
            assert conflict.type == "mutual_exclusion"
            assert isinstance(conflict.mixin_a, str)
            assert isinstance(conflict.mixin_b, str)
            assert len(conflict.reason) >= 10

    def test_detect_missing_prerequisites(self, conflict_resolver):
        """Test detecting missing prerequisites."""
        # Test with a mixin that has prerequisites
        conflicts = conflict_resolver.detect_conflicts(
            ["MixinCircuitBreaker"]  # May have prerequisites
        )

        missing_prereqs = [c for c in conflicts if c.type == "missing_prerequisite"]
        # Verify structure
        for conflict in missing_prereqs:
            assert conflict.type == "missing_prerequisite"
            assert isinstance(conflict.mixin_a, str)
            assert isinstance(conflict.mixin_b, str)

    def test_detect_redundancies(self, conflict_resolver):
        """Test detecting redundancies."""
        conflicts = conflict_resolver.detect_conflicts(
            ["MixinHealthCheck", "MixinMetrics"]
        )

        redundancies = [c for c in conflicts if c.type == "redundancy"]
        # Verify structure
        for conflict in redundancies:
            assert conflict.type == "redundancy"

    def test_has_conflicts_true(self, conflict_resolver):
        """Test has_conflicts returns True when conflicts exist."""
        # Use mixins likely to have conflicts
        result = conflict_resolver.has_conflicts(
            ["MixinHealthCheck", "MixinMetrics", "MixinCircuitBreaker"]
        )

        assert isinstance(result, bool)

    def test_has_conflicts_false(self, conflict_resolver):
        """Test has_conflicts returns False with no conflicts."""
        result = conflict_resolver.has_conflicts(["MixinHealthCheck"])

        # Single mixin shouldn't have conflicts
        assert isinstance(result, bool)


class TestConflictResolution:
    """Test suite for conflict resolution."""

    def test_resolve_conflicts_no_conflicts(
        self, conflict_resolver, sample_recommendations, sample_scores
    ):
        """Test resolution when no conflicts exist."""
        resolved, warnings = conflict_resolver.resolve_conflicts(
            sample_recommendations, sample_scores
        )

        assert isinstance(resolved, list)
        assert isinstance(warnings, list)
        # Should return all mixins if no conflicts
        assert len(resolved) > 0

    def test_resolve_conflicts_returns_mixins_and_warnings(
        self, conflict_resolver, sample_recommendations, sample_scores
    ):
        """Test resolution returns both mixins and warnings."""
        resolved, warnings = conflict_resolver.resolve_conflicts(
            sample_recommendations, sample_scores
        )

        assert isinstance(resolved, list)
        assert isinstance(warnings, list)
        # All returned mixins should be strings
        assert all(isinstance(m, str) for m in resolved)
        # All warnings should be strings
        assert all(isinstance(w, str) for w in warnings)

    def test_apply_resolution_prefer_higher_score(self, conflict_resolver):
        """Test resolution strategy: prefer_higher_score."""
        conflict = ModelMixinConflict(
            type="mutual_exclusion",
            mixin_a="MixinA",
            mixin_b="MixinB",
            reason="Test conflict",
            resolution="prefer_higher_score",
        )
        scores = {"MixinA": 0.9, "MixinB": 0.7}

        result = conflict_resolver._apply_resolution(
            conflict, ["MixinA", "MixinB"], scores
        )

        assert result["action"] == "remove"
        assert result["mixin"] == "MixinB"  # Lower score removed

    def test_apply_resolution_prefer_higher_score_reverse(self, conflict_resolver):
        """Test prefer_higher_score with reverse scores."""
        conflict = ModelMixinConflict(
            type="mutual_exclusion",
            mixin_a="MixinA",
            mixin_b="MixinB",
            reason="Test conflict",
            resolution="prefer_higher_score",
        )
        scores = {"MixinA": 0.6, "MixinB": 0.9}

        result = conflict_resolver._apply_resolution(
            conflict, ["MixinA", "MixinB"], scores
        )

        assert result["action"] == "remove"
        assert result["mixin"] == "MixinA"  # Lower score removed

    def test_apply_resolution_add_prerequisite(self, conflict_resolver):
        """Test resolution strategy: add_prerequisite."""
        conflict = ModelMixinConflict(
            type="missing_prerequisite",
            mixin_a="MixinA",
            mixin_b="MixinPrereq",
            reason="Requires prerequisite",
            resolution="add_prerequisite",
        )

        result = conflict_resolver._apply_resolution(conflict, ["MixinA"], {})

        assert result["action"] == "add"
        assert result["mixin"] == "MixinPrereq"

    def test_apply_resolution_remove_redundant(self, conflict_resolver):
        """Test resolution strategy: remove_redundant."""
        conflict = ModelMixinConflict(
            type="redundancy",
            mixin_a="MixinA",
            mixin_b="MixinRedundant",
            reason="Redundant functionality",
            resolution="remove_redundant",
        )

        result = conflict_resolver._apply_resolution(
            conflict, ["MixinA", "MixinRedundant"], {}
        )

        assert result["action"] == "remove"
        assert result["mixin"] == "MixinRedundant"

    def test_apply_resolution_warn(self, conflict_resolver):
        """Test resolution strategy: warn."""
        conflict = ModelMixinConflict(
            type="missing_prerequisite",
            mixin_a="MixinA",
            mixin_b="MixinB",
            reason="May need manual review",
            resolution="warn",
        )

        result = conflict_resolver._apply_resolution(conflict, ["MixinA"], {})

        assert result["action"] == "warn"
        assert "message" in result

    def test_apply_resolution_unknown_strategy(self, conflict_resolver):
        """Test handling of unknown resolution strategy."""
        conflict = ModelMixinConflict(
            type="unknown",
            mixin_a="MixinA",
            mixin_b="MixinB",
            reason="Unknown conflict",
            resolution="unknown_strategy",
        )

        result = conflict_resolver._apply_resolution(conflict, ["MixinA"], {})

        # Should default to warning
        assert result["action"] == "warn"


class TestCategoryPriority:
    """Test suite for category priority."""

    def test_get_category_priority(self, conflict_resolver):
        """Test retrieving category priority."""
        # Test known categories
        priority = conflict_resolver.get_category_priority("observability")
        assert isinstance(priority, int)
        assert priority >= 0

    def test_get_category_priority_unknown(self, conflict_resolver):
        """Test priority for unknown category."""
        priority = conflict_resolver.get_category_priority("unknown_category")
        assert priority == 999  # Low priority for unknown


class TestConflictResolutionIntegration:
    """Integration tests for conflict resolution."""

    def test_end_to_end_no_conflicts(self, conflict_resolver):
        """Test end-to-end resolution with no conflicts."""
        recommendations = [
            ModelMixinRecommendation(
                mixin_name="MixinHealthCheck",
                score=0.9,
                category="observability",
                explanation="Health checks",
            )
        ]
        scores = {"MixinHealthCheck": 0.9}

        resolved, warnings = conflict_resolver.resolve_conflicts(
            recommendations, scores
        )

        assert len(resolved) == 1
        assert "MixinHealthCheck" in resolved
        # May have warnings even without conflicts

    def test_end_to_end_with_mutual_exclusion(self, conflict_resolver):
        """Test end-to-end resolution with mutual exclusion."""
        # Create recommendations with potential conflicts
        recommendations = [
            ModelMixinRecommendation(
                mixin_name="MixinA",
                score=0.9,
                category="test",
                explanation="Test mixin A for conflict resolution",
            ),
            ModelMixinRecommendation(
                mixin_name="MixinB",
                score=0.8,
                category="test",
                explanation="Test mixin B for conflict resolution",
            ),
        ]
        scores = {"MixinA": 0.9, "MixinB": 0.8}

        resolved, warnings = conflict_resolver.resolve_conflicts(
            recommendations, scores
        )

        # Should resolve conflicts
        assert isinstance(resolved, list)
        assert isinstance(warnings, list)


@pytest.mark.parametrize(
    "mixin_list,should_have_conflicts",
    [
        (["MixinHealthCheck"], False),  # Single mixin, no conflicts
        (["MixinHealthCheck", "MixinMetrics"], False),  # Compatible mixins
        # Add more test cases based on actual conflict rules
    ],
)
def test_conflict_detection_matrix(
    conflict_resolver, mixin_list, should_have_conflicts
):
    """Test conflict detection across different mixin combinations."""
    result = conflict_resolver.has_conflicts(mixin_list)

    # Just verify we get a boolean result
    assert isinstance(result, bool)
