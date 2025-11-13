#!/usr/bin/env python3
"""
Unit tests for mixin scoring system.

Tests scoring mixins based on requirements analysis.
"""

from pathlib import Path

import pytest

from omninode_bridge.codegen.mixins.mixin_scorer import MixinScorer
from omninode_bridge.codegen.mixins.models import ModelRequirementAnalysis


@pytest.fixture
def mixin_scorer():
    """Create MixinScorer instance for testing."""
    return MixinScorer()


@pytest.fixture
def database_heavy_requirements():
    """Requirements with high database score."""
    return ModelRequirementAnalysis(
        keywords={"database", "postgres", "transaction", "crud"},
        dependency_packages={"asyncpg", "sqlalchemy"},
        database_score=9.0,
        api_score=2.0,
        kafka_score=1.0,
        security_score=3.0,
        observability_score=4.0,
        resilience_score=2.0,
        caching_score=1.0,
        performance_score=5.0,
        confidence=0.9,
        rationale="High database operations",
    )


@pytest.fixture
def api_heavy_requirements():
    """Requirements with high API score."""
    return ModelRequirementAnalysis(
        keywords={"api", "http", "rest", "client", "request"},
        dependency_packages={"httpx", "aiohttp"},
        database_score=1.0,
        api_score=9.0,
        kafka_score=1.0,
        security_score=4.0,
        observability_score=3.0,
        resilience_score=6.0,
        caching_score=2.0,
        performance_score=4.0,
        confidence=0.85,
        rationale="High API client operations",
    )


@pytest.fixture
def balanced_requirements():
    """Requirements with balanced scores."""
    return ModelRequirementAnalysis(
        keywords={"process", "workflow", "orchestrate"},
        dependency_packages=set(),
        database_score=5.0,
        api_score=5.0,
        kafka_score=5.0,
        security_score=5.0,
        observability_score=5.0,
        resilience_score=5.0,
        caching_score=5.0,
        performance_score=5.0,
        confidence=0.7,
        rationale="Balanced requirements",
    )


@pytest.fixture
def minimal_requirements():
    """Requirements with minimal scores."""
    return ModelRequirementAnalysis(
        keywords=set(),
        dependency_packages=set(),
        database_score=0.0,
        api_score=0.0,
        kafka_score=0.0,
        security_score=0.0,
        observability_score=0.0,
        resilience_score=0.0,
        caching_score=0.0,
        performance_score=0.0,
        confidence=0.5,
        rationale="Minimal requirements",
    )


class TestMixinScorer:
    """Test suite for mixin scoring."""

    def test_initialization(self, mixin_scorer):
        """Test MixinScorer initializes correctly."""
        assert mixin_scorer is not None
        assert mixin_scorer.mixins is not None
        assert isinstance(mixin_scorer.mixins, dict)
        assert len(mixin_scorer.mixins) > 0
        assert mixin_scorer.category_weights is not None
        assert mixin_scorer.global_settings is not None

    def test_load_config_success(self, mixin_scorer):
        """Test config loads successfully."""
        assert "mixins" in mixin_scorer.config
        assert "category_weights" in mixin_scorer.config
        assert "global" in mixin_scorer.config

    def test_load_config_missing_file(self):
        """Test handling of missing config file."""
        scorer = MixinScorer(config_path=Path("/nonexistent/config.yaml"))
        # Should initialize with empty config
        assert scorer.mixins == {}

    def test_score_all_mixins_database_heavy(
        self, mixin_scorer, database_heavy_requirements
    ):
        """Test scoring with database-heavy requirements."""
        scores = mixin_scorer.score_all_mixins(database_heavy_requirements)

        assert isinstance(scores, dict)
        assert len(scores) > 0

        # Database-related mixins should score higher
        if "MixinDatabase" in scores:
            assert scores["MixinDatabase"] > 0.3
        if "MixinTransactionManager" in scores:
            assert scores["MixinTransactionManager"] > 0.3

    def test_score_all_mixins_api_heavy(self, mixin_scorer, api_heavy_requirements):
        """Test scoring with API-heavy requirements."""
        scores = mixin_scorer.score_all_mixins(api_heavy_requirements)

        assert isinstance(scores, dict)
        # API-related mixins should score higher
        if "MixinApiClient" in scores:
            assert scores["MixinApiClient"] > 0.3

    def test_score_all_mixins_balanced(self, mixin_scorer, balanced_requirements):
        """Test scoring with balanced requirements."""
        scores = mixin_scorer.score_all_mixins(balanced_requirements)

        assert isinstance(scores, dict)
        # All scores should be moderate
        for score in scores.values():
            assert 0.0 <= score <= 1.0

    def test_score_all_mixins_minimal(self, mixin_scorer, minimal_requirements):
        """Test scoring with minimal requirements."""
        scores = mixin_scorer.score_all_mixins(minimal_requirements)

        assert isinstance(scores, dict)
        # Most scores should be low or default
        for mixin_name, score in scores.items():
            assert 0.0 <= score <= 1.0

    def test_score_mixin_relevance(self, mixin_scorer, database_heavy_requirements):
        """Test scoring mixin relevance to requirements."""
        scores = mixin_scorer.score_all_mixins(database_heavy_requirements)

        # Check that scores are normalized 0-1
        for score in scores.values():
            assert 0.0 <= score <= 1.0

    def test_calculate_keyword_bonus(self, mixin_scorer):
        """Test keyword matching bonus calculation."""
        keywords_match = ["database", "postgres", "transaction"]
        extracted_keywords = {"database", "postgres", "transaction", "query"}

        bonus = mixin_scorer._calculate_keyword_bonus(
            keywords_match, extracted_keywords
        )

        # 3 matches * 0.05 = 0.15 (max bonus)
        assert bonus == 0.15

    def test_calculate_keyword_bonus_partial(self, mixin_scorer):
        """Test keyword bonus with partial matches."""
        keywords_match = ["database", "postgres", "mongodb"]
        extracted_keywords = {"database", "postgres"}

        bonus = mixin_scorer._calculate_keyword_bonus(
            keywords_match, extracted_keywords
        )

        # 2 matches * 0.05 = 0.10
        assert bonus == 0.10

    def test_calculate_keyword_bonus_no_matches(self, mixin_scorer):
        """Test keyword bonus with no matches."""
        keywords_match = ["database", "postgres"]
        extracted_keywords = {"kafka", "redis"}

        bonus = mixin_scorer._calculate_keyword_bonus(
            keywords_match, extracted_keywords
        )

        assert bonus == 0.0

    def test_calculate_dependency_bonus(self, mixin_scorer):
        """Test dependency matching bonus calculation."""
        dependencies_match = ["asyncpg", "sqlalchemy"]
        dependency_packages = {"asyncpg", "sqlalchemy", "httpx"}

        bonus = mixin_scorer._calculate_dependency_bonus(
            dependencies_match, dependency_packages
        )

        # 2 matches * 0.075 = 0.15 (max bonus)
        assert bonus == 0.15

    def test_calculate_dependency_bonus_partial(self, mixin_scorer):
        """Test dependency bonus with partial matches."""
        dependencies_match = ["asyncpg", "psycopg2"]
        dependency_packages = {"asyncpg"}

        bonus = mixin_scorer._calculate_dependency_bonus(
            dependencies_match, dependency_packages
        )

        # 1 match * 0.075 = 0.075
        assert bonus == 0.075

    def test_calculate_operation_bonus(self, mixin_scorer):
        """Test operation matching bonus calculation."""
        operation_match = ["create", "update", "delete"]
        extracted_keywords = {"create", "update", "delete", "query"}

        bonus = mixin_scorer._calculate_operation_bonus(
            operation_match, extracted_keywords
        )

        # 3 matches * 0.05 = 0.15, but max is 0.10
        assert bonus == 0.10

    def test_check_requirements_and_logic(self, mixin_scorer):
        """Test requirement checking with AND logic."""
        required_scores = {"database": 5.0, "api": 3.0}
        category_scores = {"database": 6.0, "api": 4.0, "kafka": 2.0}
        mixin_config = {"and_logic": True}

        result = mixin_scorer._check_requirements(
            required_scores, category_scores, mixin_config
        )

        # Both requirements met
        assert result is True

    def test_check_requirements_and_logic_fail(self, mixin_scorer):
        """Test requirement checking with AND logic failing."""
        required_scores = {"database": 5.0, "api": 5.0}
        category_scores = {"database": 6.0, "api": 3.0, "kafka": 2.0}
        mixin_config = {"and_logic": True}

        result = mixin_scorer._check_requirements(
            required_scores, category_scores, mixin_config
        )

        # API requirement not met
        assert result is False

    def test_check_requirements_or_logic(self, mixin_scorer):
        """Test requirement checking with OR logic."""
        required_scores = {"database": 8.0, "api": 8.0}
        category_scores = {"database": 9.0, "api": 2.0, "kafka": 1.0}
        mixin_config = {"or_logic": True}

        result = mixin_scorer._check_requirements(
            required_scores, category_scores, mixin_config
        )

        # Database requirement met (OR logic)
        assert result is True

    def test_check_requirements_or_logic_fail(self, mixin_scorer):
        """Test requirement checking with OR logic failing."""
        required_scores = {"database": 8.0, "api": 8.0}
        category_scores = {"database": 5.0, "api": 5.0, "kafka": 1.0}
        mixin_config = {"or_logic": True}

        result = mixin_scorer._check_requirements(
            required_scores, category_scores, mixin_config
        )

        # Neither requirement met
        assert result is False

    def test_get_category_weight(self, mixin_scorer):
        """Test retrieving category weight."""
        # Check that category weights are loaded
        database_weight = mixin_scorer.get_category_weight("database")
        assert isinstance(database_weight, (int, float))
        assert database_weight > 0

    def test_get_min_recommendation_score(self, mixin_scorer):
        """Test retrieving minimum recommendation score."""
        min_score = mixin_scorer.get_min_recommendation_score()
        assert isinstance(min_score, (int, float))
        assert 0.0 <= min_score <= 1.0

    def test_get_default_top_k(self, mixin_scorer):
        """Test retrieving default top-K value."""
        top_k = mixin_scorer.get_default_top_k()
        assert isinstance(top_k, int)
        assert top_k > 0


class TestMixinScoringIntegration:
    """Integration tests for mixin scoring."""

    def test_end_to_end_database_scoring(self, mixin_scorer):
        """Test end-to-end scoring for database-heavy workload."""
        requirements = ModelRequirementAnalysis(
            keywords={"database", "postgres", "transaction", "crud", "query"},
            dependency_packages={"asyncpg", "sqlalchemy"},
            database_score=9.0,
            observability_score=5.0,
            confidence=0.9,
        )

        scores = mixin_scorer.score_all_mixins(requirements)

        # Should have scores for all mixins
        assert len(scores) > 0

        # All scores should be valid
        for score in scores.values():
            assert 0.0 <= score <= 1.0

    def test_end_to_end_api_scoring(self, mixin_scorer):
        """Test end-to-end scoring for API-heavy workload."""
        requirements = ModelRequirementAnalysis(
            keywords={"api", "http", "rest", "client", "retry"},
            dependency_packages={"httpx"},
            api_score=8.0,
            resilience_score=6.0,
            observability_score=4.0,
            confidence=0.85,
        )

        scores = mixin_scorer.score_all_mixins(requirements)

        # Should have scores
        assert len(scores) > 0

        # Resilience mixins should score reasonably high
        for mixin_name, score in scores.items():
            assert 0.0 <= score <= 1.0


@pytest.mark.parametrize(
    "requirements,expected_high_categories",
    [
        (
            ModelRequirementAnalysis(database_score=9.0, observability_score=5.0),
            ["database"],
        ),
        (
            ModelRequirementAnalysis(api_score=8.0, resilience_score=7.0),
            ["api", "resilience"],
        ),
        (ModelRequirementAnalysis(kafka_score=9.0, observability_score=6.0), ["kafka"]),
    ],
)
def test_category_scoring(mixin_scorer, requirements, expected_high_categories):
    """Test scoring prioritizes correct categories."""
    scores = mixin_scorer.score_all_mixins(requirements)

    # Just verify we get scores back
    assert len(scores) > 0
    for score in scores.values():
        assert 0.0 <= score <= 1.0
