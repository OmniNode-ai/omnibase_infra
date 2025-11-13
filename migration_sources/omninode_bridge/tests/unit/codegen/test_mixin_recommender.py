#!/usr/bin/env python3
"""
Unit tests for mixin recommendation system.

Tests top-K recommendation generation with explanations and usage statistics.
"""


import pytest

from omninode_bridge.codegen.mixins.mixin_recommender import MixinRecommender
from omninode_bridge.codegen.mixins.mixin_scorer import MixinScorer
from omninode_bridge.codegen.mixins.models import (
    ModelMixinRecommendation,
    ModelMixinUsageStats,
    ModelRequirementAnalysis,
)


@pytest.fixture
def mixin_recommender():
    """Create MixinRecommender instance for testing."""
    return MixinRecommender()


@pytest.fixture
def database_requirements():
    """Requirements for database-heavy operations."""
    return ModelRequirementAnalysis(
        keywords={"database", "postgres", "transaction", "crud"},
        dependency_packages={"asyncpg", "sqlalchemy"},
        database_score=9.0,
        observability_score=5.0,
        confidence=0.9,
        rationale="High database requirements",
    )


@pytest.fixture
def api_requirements():
    """Requirements for API-heavy operations."""
    return ModelRequirementAnalysis(
        keywords={"api", "http", "rest", "retry"},
        dependency_packages={"httpx"},
        api_score=8.0,
        resilience_score=6.0,
        confidence=0.85,
        rationale="High API requirements",
    )


@pytest.fixture
def minimal_requirements():
    """Minimal requirements."""
    return ModelRequirementAnalysis(
        keywords=set(),
        dependency_packages=set(),
        confidence=0.5,
        rationale="Minimal requirements",
    )


@pytest.fixture
def usage_stats():
    """Sample usage statistics for testing."""
    return {
        "MixinHealthCheck": ModelMixinUsageStats(
            mixin_name="MixinHealthCheck",
            recommended_count=100,
            accepted_count=95,
            success_count=92,
            failure_count=3,
            avg_code_quality_score=4.5,
        ),
        "MixinMetrics": ModelMixinUsageStats(
            mixin_name="MixinMetrics",
            recommended_count=80,
            accepted_count=75,
            success_count=70,
            failure_count=5,
            avg_code_quality_score=4.2,
        ),
        "MixinCircuitBreaker": ModelMixinUsageStats(
            mixin_name="MixinCircuitBreaker",
            recommended_count=50,
            accepted_count=20,
            success_count=15,
            failure_count=5,
            avg_code_quality_score=3.5,
        ),
    }


class TestMixinRecommender:
    """Test suite for mixin recommendation."""

    def test_initialization(self, mixin_recommender):
        """Test MixinRecommender initializes correctly."""
        assert mixin_recommender is not None
        assert mixin_recommender.scorer is not None
        assert isinstance(mixin_recommender.scorer, MixinScorer)
        assert isinstance(mixin_recommender.usage_stats, dict)

    def test_initialization_with_custom_scorer(self):
        """Test initialization with custom scorer."""
        custom_scorer = MixinScorer()
        recommender = MixinRecommender(mixin_scorer=custom_scorer)

        assert recommender.scorer is custom_scorer

    def test_recommend_mixins_basic(self, mixin_recommender, database_requirements):
        """Test basic mixin recommendation."""
        recommendations = mixin_recommender.recommend_mixins(database_requirements)

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        # Should return default top_k recommendations
        assert len(recommendations) <= mixin_recommender.scorer.get_default_top_k()

    def test_recommend_mixins_custom_top_k(
        self, mixin_recommender, database_requirements
    ):
        """Test recommendation with custom top_k."""
        top_k = 3
        recommendations = mixin_recommender.recommend_mixins(
            database_requirements, top_k=top_k
        )

        assert len(recommendations) <= top_k

    def test_recommend_mixins_custom_min_score(
        self, mixin_recommender, minimal_requirements
    ):
        """Test recommendation with custom min_score."""
        recommendations = mixin_recommender.recommend_mixins(
            minimal_requirements, min_score=0.8
        )

        # With high min_score and minimal requirements, may get few or no recommendations
        for rec in recommendations:
            assert rec.score >= 0.8

    def test_recommend_mixins_structure(self, mixin_recommender, database_requirements):
        """Test recommendation structure."""
        recommendations = mixin_recommender.recommend_mixins(database_requirements)

        for rec in recommendations:
            assert isinstance(rec, ModelMixinRecommendation)
            assert isinstance(rec.mixin_name, str)
            assert 0.0 <= rec.score <= 1.0
            assert isinstance(rec.category, str)
            assert isinstance(rec.explanation, str)
            assert len(rec.explanation) >= 10
            assert isinstance(rec.matched_requirements, list)
            assert isinstance(rec.prerequisites, list)
            assert isinstance(rec.conflicts_with, list)

    def test_recommend_mixins_sorted_by_score(
        self, mixin_recommender, database_requirements
    ):
        """Test recommendations are sorted by score (descending)."""
        recommendations = mixin_recommender.recommend_mixins(
            database_requirements, top_k=5
        )

        scores = [rec.score for rec in recommendations]

        # Should be sorted descending
        assert scores == sorted(scores, reverse=True)

    def test_recommend_mixins_with_usage_stats(
        self, mixin_recommender, database_requirements, usage_stats
    ):
        """Test recommendations with usage statistics."""
        mixin_recommender.load_usage_stats(usage_stats)

        recommendations = mixin_recommender.recommend_mixins(
            database_requirements, use_usage_stats=True
        )

        assert len(recommendations) > 0
        # Scores should be adjusted based on usage stats
        for rec in recommendations:
            assert 0.0 <= rec.score <= 1.0

    def test_recommend_mixins_without_usage_stats(
        self, mixin_recommender, database_requirements, usage_stats
    ):
        """Test recommendations without usage statistics."""
        mixin_recommender.load_usage_stats(usage_stats)

        recommendations = mixin_recommender.recommend_mixins(
            database_requirements, use_usage_stats=False
        )

        assert len(recommendations) > 0

    def test_adjust_scores_with_usage_stats_high_success(self, mixin_recommender):
        """Test score adjustment with high success rate."""
        base_scores = {"MixinHealthCheck": 0.6}
        stats = {
            "MixinHealthCheck": ModelMixinUsageStats(
                mixin_name="MixinHealthCheck",
                recommended_count=100,
                accepted_count=95,
                success_count=92,
                avg_code_quality_score=4.5,
            )
        }
        mixin_recommender.load_usage_stats(stats)

        adjusted = mixin_recommender._adjust_scores_with_usage_stats(base_scores)

        # Should be increased due to high success rate (>0.9), acceptance (>0.8), quality (>4.0)
        assert adjusted["MixinHealthCheck"] > base_scores["MixinHealthCheck"]

    def test_adjust_scores_with_usage_stats_low_success(self, mixin_recommender):
        """Test score adjustment with low success rate."""
        base_scores = {"MixinCircuitBreaker": 0.6}
        stats = {
            "MixinCircuitBreaker": ModelMixinUsageStats(
                mixin_name="MixinCircuitBreaker",
                recommended_count=50,
                accepted_count=20,
                success_count=15,
                avg_code_quality_score=2.5,
            )
        }
        mixin_recommender.load_usage_stats(stats)

        adjusted = mixin_recommender._adjust_scores_with_usage_stats(base_scores)

        # Should be decreased due to low success rate, acceptance, quality
        assert adjusted["MixinCircuitBreaker"] < base_scores["MixinCircuitBreaker"]

    def test_adjust_scores_with_insufficient_data(self, mixin_recommender):
        """Test score adjustment with insufficient data."""
        base_scores = {"MixinNew": 0.5}
        stats = {
            "MixinNew": ModelMixinUsageStats(
                mixin_name="MixinNew",
                recommended_count=5,  # Below threshold of 10
                accepted_count=4,
                success_count=4,
            )
        }
        mixin_recommender.load_usage_stats(stats)

        adjusted = mixin_recommender._adjust_scores_with_usage_stats(base_scores)

        # Should not be adjusted due to insufficient sample size
        assert adjusted["MixinNew"] == base_scores["MixinNew"]

    def test_create_recommendation(self, mixin_recommender, database_requirements):
        """Test creating a single recommendation."""
        mixin_name = "MixinDatabase"
        score = 0.85

        rec = mixin_recommender._create_recommendation(
            mixin_name, score, database_requirements
        )

        assert isinstance(rec, ModelMixinRecommendation)
        assert rec.mixin_name == mixin_name
        assert rec.score == score
        assert isinstance(rec.explanation, str)
        assert len(rec.explanation) >= 10

    def test_generate_explanation(self, mixin_recommender, database_requirements):
        """Test explanation generation."""
        mixin_config = {
            "primary_category": "database",
            "keywords_match": ["database", "postgres"],
            "description": "Database operations support",
        }

        explanation = mixin_recommender._generate_explanation(
            "MixinDatabase", 0.85, mixin_config, database_requirements
        )

        assert isinstance(explanation, str)
        assert len(explanation) >= 10
        assert "Confidence: 0.85" in explanation
        assert "Recommended because:" in explanation

    def test_generate_explanation_with_keyword_matches(self, mixin_recommender):
        """Test explanation includes keyword matches."""
        requirements = ModelRequirementAnalysis(
            keywords={"database", "postgres", "transaction"}, database_score=9.0
        )
        mixin_config = {
            "primary_category": "database",
            "keywords_match": ["database", "postgres", "transaction"],
        }

        explanation = mixin_recommender._generate_explanation(
            "MixinDatabase", 0.85, mixin_config, requirements
        )

        assert "keywords:" in explanation

    def test_generate_explanation_with_dependency_matches(self, mixin_recommender):
        """Test explanation includes dependency matches."""
        requirements = ModelRequirementAnalysis(
            dependency_packages={"asyncpg", "sqlalchemy"}, database_score=8.0
        )
        mixin_config = {
            "primary_category": "database",
            "dependencies_match": ["asyncpg", "sqlalchemy"],
        }

        explanation = mixin_recommender._generate_explanation(
            "MixinDatabase", 0.85, mixin_config, requirements
        )

        assert "dependencies:" in explanation

    def test_get_matched_requirements(self, mixin_recommender, database_requirements):
        """Test extracting matched requirements."""
        mixin_config = {
            "primary_category": "database",
            "keywords_match": ["database", "postgres"],
        }

        matched = mixin_recommender._get_matched_requirements(
            mixin_config, database_requirements
        )

        assert isinstance(matched, list)
        # Should have at least the primary category
        assert len(matched) > 0

    def test_load_usage_stats(self, mixin_recommender, usage_stats):
        """Test loading usage statistics."""
        mixin_recommender.load_usage_stats(usage_stats)

        assert len(mixin_recommender.usage_stats) == len(usage_stats)
        for mixin_name, stats in usage_stats.items():
            assert mixin_name in mixin_recommender.usage_stats
            assert mixin_recommender.usage_stats[mixin_name] == stats

    def test_update_usage_stat(self, mixin_recommender):
        """Test updating a single usage statistic."""
        stat = ModelMixinUsageStats(
            mixin_name="MixinTest",
            recommended_count=10,
            accepted_count=8,
            success_count=7,
        )

        mixin_recommender.update_usage_stat("MixinTest", stat)

        assert "MixinTest" in mixin_recommender.usage_stats
        assert mixin_recommender.usage_stats["MixinTest"] == stat


class TestMixinRecommendationIntegration:
    """Integration tests for mixin recommendation."""

    def test_end_to_end_database_recommendation(self, mixin_recommender):
        """Test end-to-end recommendation for database workload."""
        requirements = ModelRequirementAnalysis(
            keywords={"database", "postgres", "transaction", "crud"},
            dependency_packages={"asyncpg"},
            database_score=9.0,
            observability_score=5.0,
            confidence=0.9,
        )

        recommendations = mixin_recommender.recommend_mixins(requirements, top_k=5)

        assert len(recommendations) > 0
        assert all(isinstance(rec, ModelMixinRecommendation) for rec in recommendations)
        # Verify scores are descending
        scores = [rec.score for rec in recommendations]
        assert scores == sorted(scores, reverse=True)

    def test_end_to_end_api_recommendation(self, mixin_recommender):
        """Test end-to-end recommendation for API workload."""
        requirements = ModelRequirementAnalysis(
            keywords={"api", "http", "rest", "retry"},
            dependency_packages={"httpx"},
            api_score=8.0,
            resilience_score=6.0,
            confidence=0.85,
        )

        recommendations = mixin_recommender.recommend_mixins(requirements, top_k=5)

        assert len(recommendations) > 0
        for rec in recommendations:
            assert rec.score >= 0.0
            assert len(rec.explanation) >= 10

    def test_end_to_end_with_usage_stats(self, mixin_recommender, usage_stats):
        """Test end-to-end recommendation with usage statistics."""
        mixin_recommender.load_usage_stats(usage_stats)

        requirements = ModelRequirementAnalysis(
            keywords={"health", "metrics"}, observability_score=8.0, confidence=0.9
        )

        recommendations = mixin_recommender.recommend_mixins(
            requirements, top_k=3, use_usage_stats=True
        )

        assert len(recommendations) > 0


@pytest.mark.parametrize(
    "requirements,min_expected_recommendations",
    [
        (ModelRequirementAnalysis(database_score=9.0), 1),
        (ModelRequirementAnalysis(api_score=8.0, resilience_score=6.0), 1),
        (ModelRequirementAnalysis(kafka_score=9.0), 1),
        (ModelRequirementAnalysis(observability_score=8.0), 1),
    ],
)
def test_recommendations_for_different_workloads(
    mixin_recommender, requirements, min_expected_recommendations
):
    """Test recommendations for different workload types."""
    recommendations = mixin_recommender.recommend_mixins(requirements, top_k=10)

    assert len(recommendations) >= min_expected_recommendations
    for rec in recommendations:
        assert 0.0 <= rec.score <= 1.0
        assert len(rec.explanation) >= 10
