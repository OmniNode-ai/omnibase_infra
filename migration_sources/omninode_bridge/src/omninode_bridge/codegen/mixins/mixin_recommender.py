"""
Mixin Recommender for intelligent mixin selection.

Generates top-K mixin recommendations with explanations based on scores.
Integrates usage statistics for adaptive learning.

Performance Target: <20ms to generate recommendations
Accuracy Target: >90% relevance
"""

import logging
from pathlib import Path
from typing import Optional

from omninode_bridge.codegen.mixins.mixin_scorer import MixinScorer
from omninode_bridge.codegen.mixins.models import (
    ModelMixinRecommendation,
    ModelMixinUsageStats,
    ModelRequirementAnalysis,
)

logger = logging.getLogger(__name__)


class MixinRecommender:
    """
    Generate top-K mixin recommendations with explanations.

    Combines scoring results with usage statistics to produce ranked
    recommendations with human-readable explanations.
    """

    def __init__(
        self,
        mixin_scorer: Optional[MixinScorer] = None,
        config_path: Optional[Path] = None,
    ):
        """
        Initialize the mixin recommender.

        Args:
            mixin_scorer: MixinScorer instance (creates new if None)
            config_path: Path to scoring_config.yaml
        """
        self.scorer = mixin_scorer or MixinScorer(config_path)

        # Usage statistics (can be loaded from database)
        self.usage_stats: dict[str, ModelMixinUsageStats] = {}

    def recommend_mixins(
        self,
        requirement_analysis: ModelRequirementAnalysis,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        use_usage_stats: bool = True,
    ) -> list[ModelMixinRecommendation]:
        """
        Recommend top-K mixins with confidence scores and explanations.

        Args:
            requirement_analysis: Analysis results from RequirementsAnalyzer
            top_k: Number of recommendations (default from config)
            min_score: Minimum score threshold (default from config)
            use_usage_stats: Whether to adjust scores using usage statistics

        Returns:
            List of ModelMixinRecommendation sorted by score (descending)
        """
        # Use defaults if not specified
        if top_k is None:
            top_k = self.scorer.get_default_top_k()
        if min_score is None:
            min_score = self.scorer.get_min_recommendation_score()

        # Step 1: Score all mixins
        mixin_scores = self.scorer.score_all_mixins(requirement_analysis)

        # Step 2: Adjust scores using usage statistics (if enabled)
        if use_usage_stats and self.usage_stats:
            mixin_scores = self._adjust_scores_with_usage_stats(mixin_scores)

        # Step 3: Filter by min_score
        candidates = [
            (name, score) for name, score in mixin_scores.items() if score >= min_score
        ]

        # Step 4: Sort by score (descending)
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Step 5: Take top-K
        top_candidates = candidates[:top_k]

        # Step 6: Generate recommendations with explanations
        recommendations = []
        for mixin_name, score in top_candidates:
            recommendation = self._create_recommendation(
                mixin_name, score, requirement_analysis
            )
            recommendations.append(recommendation)

        return recommendations

    def _adjust_scores_with_usage_stats(
        self, base_scores: dict[str, float]
    ) -> dict[str, float]:
        """
        Adjust base scores using historical usage statistics.

        Adjustment factors:
        - Success rate: +0.1 if >90%, -0.1 if <50%
        - Acceptance rate: +0.05 if >80%, -0.05 if <30%
        - Code quality: +0.05 if >4.0/5.0, -0.05 if <3.0/5.0
        """
        adjusted_scores = {}

        for mixin_name, base_score in base_scores.items():
            adjusted_score = base_score

            if mixin_name in self.usage_stats:
                stats = self.usage_stats[mixin_name]

                # Require minimum sample size
                if stats.recommended_count > 10:
                    # Success rate adjustment
                    success_rate = stats.success_rate
                    if success_rate > 0.9:
                        adjusted_score += 0.1
                    elif success_rate < 0.5:
                        adjusted_score -= 0.1

                    # Acceptance rate adjustment
                    acceptance_rate = stats.acceptance_rate
                    if acceptance_rate > 0.8:
                        adjusted_score += 0.05
                    elif acceptance_rate < 0.3:
                        adjusted_score -= 0.05

                    # Code quality adjustment
                    if stats.avg_code_quality_score > 0:
                        if stats.avg_code_quality_score > 4.0:
                            adjusted_score += 0.05
                        elif stats.avg_code_quality_score < 3.0:
                            adjusted_score -= 0.05

            adjusted_scores[mixin_name] = max(0.0, min(1.0, adjusted_score))

        return adjusted_scores

    def _create_recommendation(
        self,
        mixin_name: str,
        score: float,
        requirement_analysis: ModelRequirementAnalysis,
    ) -> ModelMixinRecommendation:
        """Create a single recommendation with explanation."""
        # Get mixin configuration
        mixin_config = self.scorer.mixins.get(mixin_name, {})
        category = mixin_config.get("primary_category", "unknown")

        # Generate explanation
        explanation = self._generate_explanation(
            mixin_name, score, mixin_config, requirement_analysis
        )

        # Get matched requirements
        matched_requirements = self._get_matched_requirements(
            mixin_config, requirement_analysis
        )

        # Get prerequisites
        prerequisites = mixin_config.get("prerequisites", [])

        # Get conflicts (handled by ConflictResolver in C15)
        conflicts_with = []

        return ModelMixinRecommendation(
            mixin_name=mixin_name,
            score=score,
            category=category,
            explanation=explanation,
            matched_requirements=matched_requirements,
            prerequisites=prerequisites,
            conflicts_with=conflicts_with,
        )

    def _generate_explanation(
        self,
        mixin_name: str,
        score: float,
        mixin_config: dict,
        requirement_analysis: ModelRequirementAnalysis,
    ) -> str:
        """
        Generate human-readable explanation for recommendation.

        Template:
            "Recommended because: {reasons}. Confidence: {score}."
        """
        reasons = []

        # Check category scores
        category = mixin_config.get("primary_category", "unknown")
        category_scores = {
            "database": requirement_analysis.database_score,
            "api": requirement_analysis.api_score,
            "kafka": requirement_analysis.kafka_score,
            "security": requirement_analysis.security_score,
            "observability": requirement_analysis.observability_score,
            "resilience": requirement_analysis.resilience_score,
            "caching": requirement_analysis.caching_score,
            "performance": requirement_analysis.performance_score,
        }

        category_score = category_scores.get(category, 0.0)

        if category_score > 7.0:
            reasons.append(
                f"high {category} requirements (score: {category_score:.1f}/10)"
            )
        elif category_score > 4.0:
            reasons.append(
                f"moderate {category} requirements (score: {category_score:.1f}/10)"
            )

        # Check keyword matches
        if "keywords_match" in mixin_config:
            keyword_matches = requirement_analysis.keywords.intersection(
                set(mixin_config["keywords_match"])
            )
            if keyword_matches:
                keywords_str = ", ".join(sorted(keyword_matches)[:3])
                reasons.append(f"keywords: {keywords_str}")

        # Check dependency matches
        if "dependencies_match" in mixin_config:
            dep_matches = requirement_analysis.dependency_packages.intersection(
                set(mixin_config["dependencies_match"])
            )
            if dep_matches:
                deps_str = ", ".join(sorted(dep_matches))
                reasons.append(f"dependencies: {deps_str}")

        # Check operation matches
        if "operation_match" in mixin_config:
            op_matches = [
                op
                for op in mixin_config["operation_match"]
                if op in requirement_analysis.keywords
            ]
            if op_matches:
                ops_str = ", ".join(op_matches[:3])
                reasons.append(f"operations: {ops_str}")

        # Add description if no specific reasons
        if not reasons:
            description = mixin_config.get(
                "description", "general best practice for production nodes"
            )
            reasons.append(description)

        reason_text = "; ".join(reasons)
        return f"Recommended because: {reason_text}. Confidence: {score:.2f}."

    def _get_matched_requirements(
        self, mixin_config: dict, requirement_analysis: ModelRequirementAnalysis
    ) -> list[str]:
        """Get list of matched requirements for this mixin."""
        matched = []

        # Add primary category if high score
        category = mixin_config.get("primary_category")
        category_scores = {
            "database": requirement_analysis.database_score,
            "api": requirement_analysis.api_score,
            "kafka": requirement_analysis.kafka_score,
            "security": requirement_analysis.security_score,
            "observability": requirement_analysis.observability_score,
            "resilience": requirement_analysis.resilience_score,
            "caching": requirement_analysis.caching_score,
            "performance": requirement_analysis.performance_score,
        }

        if category_scores.get(category, 0.0) > 4.0:
            matched.append(f"{category}_operations")

        # Add matched keywords
        if "keywords_match" in mixin_config:
            keyword_matches = requirement_analysis.keywords.intersection(
                set(mixin_config["keywords_match"])
            )
            for kw in list(keyword_matches)[:3]:
                matched.append(kw)

        # Add matched dependencies
        if "dependencies_match" in mixin_config:
            dep_matches = requirement_analysis.dependency_packages.intersection(
                set(mixin_config["dependencies_match"])
            )
            matched.extend(dep_matches)

        return matched

    def load_usage_stats(self, stats: dict[str, ModelMixinUsageStats]) -> None:
        """
        Load usage statistics for adaptive scoring.

        Args:
            stats: dict mapping mixin_name → ModelMixinUsageStats
        """
        self.usage_stats = stats
        logger.info(f"Loaded usage statistics for {len(stats)} mixins")

    def update_usage_stat(self, mixin_name: str, stat: ModelMixinUsageStats) -> None:
        """
        Update usage statistics for a single mixin.

        Args:
            mixin_name: Mixin class name
            stat: Updated usage statistics
        """
        self.usage_stats[mixin_name] = stat
