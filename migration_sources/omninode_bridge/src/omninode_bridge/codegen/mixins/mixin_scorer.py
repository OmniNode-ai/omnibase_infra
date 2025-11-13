"""
Mixin Scorer for intelligent mixin selection.

Scores each mixin (0-1) based on how well it matches the analyzed requirements.
Uses scoring configuration from scoring_config.yaml.

Performance Target: <100ms to score all 21 mixins
Accuracy Target: >90% relevance
"""

import logging
from pathlib import Path
from typing import Any, Optional

import yaml

from omninode_bridge.codegen.mixins.models import ModelRequirementAnalysis

logger = logging.getLogger(__name__)


class MixinScorer:
    """
    Score mixins based on requirement analysis.

    Loads mixin configurations and applies scoring rules to determine
    which mixins best match the requirements.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the mixin scorer.

        Args:
            config_path: Path to scoring_config.yaml (defaults to same directory)
        """
        if config_path is None:
            config_path = Path(__file__).parent / "scoring_config.yaml"

        self.config = self._load_config(config_path)
        self.mixins = self.config.get("mixins", {})
        self.category_weights = self.config.get("category_weights", {})
        self.global_settings = self.config.get("global", {})

    def _load_config(self, config_path: Path) -> dict[str, Any]:
        """Load scoring configuration from YAML file."""
        try:
            with open(config_path) as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load scoring config: {e}")
            return {}

    def score_all_mixins(
        self, requirement_analysis: ModelRequirementAnalysis
    ) -> dict[str, float]:
        """
        Score all mixins in single pass.

        Args:
            requirement_analysis: Analysis results from RequirementsAnalyzer

        Returns:
            dict mapping mixin_name → score (0-1)
        """
        scores = {}

        for mixin_name, mixin_config in self.mixins.items():
            scores[mixin_name] = self._calculate_mixin_score(
                mixin_name, mixin_config, requirement_analysis
            )

        return scores

    def _calculate_mixin_score(
        self,
        mixin_name: str,
        mixin_config: dict[str, Any],
        requirement_analysis: ModelRequirementAnalysis,
    ) -> float:
        """
        Calculate score for single mixin.

        Scoring steps:
        1. Check required_scores thresholds
        2. Apply keyword matching bonus
        3. Apply dependency matching bonus
        4. Apply operation matching bonus
        5. Apply boost factors
        6. Apply category weight
        7. Normalize to 0-1

        Returns:
            Score (0-1) where >0.5 means "recommended"
        """
        # Get category scores
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

        # Get mixin configuration
        primary_category = mixin_config.get("primary_category", "observability")
        required_scores = mixin_config.get("required_scores", {})

        # Step 1: Check required_scores threshold
        meets_requirements = self._check_requirements(
            required_scores, category_scores, mixin_config
        )

        if not meets_requirements:
            # Return default score if requirements not met
            return mixin_config.get("default_score", 0.0)

        # Base score from primary category (0-0.5)
        score = (category_scores.get(primary_category, 0.0) / 10.0) * 0.5

        # If default_score is specified and base score is 0, use default_score as starting point
        # This handles cases like MixinHealthCheck/MixinMetrics that should always be recommended
        if "default_score" in mixin_config and score == 0.0:
            score = mixin_config["default_score"]

        # Step 2: Keyword matching bonus (0-0.15)
        if "keywords_match" in mixin_config:
            keyword_bonus = self._calculate_keyword_bonus(
                mixin_config["keywords_match"], requirement_analysis.keywords
            )
            score += keyword_bonus

        # Step 3: Dependency matching bonus (0-0.15)
        if "dependencies_match" in mixin_config:
            dependency_bonus = self._calculate_dependency_bonus(
                mixin_config["dependencies_match"],
                requirement_analysis.dependency_packages,
            )
            score += dependency_bonus

        # Step 4: Operation matching bonus (0-0.1)
        if "operation_match" in mixin_config:
            operation_bonus = self._calculate_operation_bonus(
                mixin_config["operation_match"], requirement_analysis.keywords
            )
            score += operation_bonus

        # Step 5: Apply boost factors (0-0.1)
        if "boost_factors" in mixin_config:
            boost_bonus = self._calculate_boost_bonus(
                mixin_config["boost_factors"], category_scores
            )
            score += boost_bonus

        # Step 6: Apply category weight
        category_weight = self.category_weights.get(primary_category, 1.0)
        score *= category_weight

        # Step 7: Apply mixin-specific weight
        mixin_weight = mixin_config.get("weight", 1.0)
        score *= mixin_weight

        return min(1.0, score)

    def _check_requirements(
        self,
        required_scores: dict[str, float],
        category_scores: dict[str, float],
        mixin_config: dict[str, Any],
    ) -> bool:
        """Check if category scores meet mixin requirements."""
        if not required_scores:
            return True

        # Check logic type
        or_logic = mixin_config.get("or_logic", False)
        and_logic = mixin_config.get("and_logic", False)

        requirement_checks = [
            category_scores.get(cat, 0.0) >= threshold
            for cat, threshold in required_scores.items()
        ]

        if or_logic:
            # Any requirement met = pass
            return any(requirement_checks)
        elif and_logic:
            # All requirements met = pass
            return all(requirement_checks)
        else:
            # Default: primary category must meet threshold
            primary_category = mixin_config.get("primary_category")
            if primary_category in required_scores:
                return (
                    category_scores.get(primary_category, 0.0)
                    >= required_scores[primary_category]
                )
            return True

    def _calculate_keyword_bonus(
        self, keywords_match: list[str], extracted_keywords: set[str]
    ) -> float:
        """Calculate bonus from keyword matching (0-0.15)."""
        matches = len(set(keywords_match).intersection(extracted_keywords))
        return min(0.15, matches * 0.05)

    def _calculate_dependency_bonus(
        self, dependencies_match: list[str], dependency_packages: set[str]
    ) -> float:
        """Calculate bonus from dependency matching (0-0.15)."""
        matches = len(set(dependencies_match).intersection(dependency_packages))
        return min(0.15, matches * 0.075)

    def _calculate_operation_bonus(
        self, operation_match: list[str], extracted_keywords: set[str]
    ) -> float:
        """Calculate bonus from operation matching (0-0.1)."""
        matches = sum(1 for op_word in operation_match if op_word in extracted_keywords)
        return min(0.1, matches * 0.05)

    def _calculate_boost_bonus(
        self, boost_factors: dict[str, float], category_scores: dict[str, float]
    ) -> float:
        """Calculate bonus from boost factors (0-0.1)."""
        total_boost = 0.0
        for boost_category, boost_weight in boost_factors.items():
            if category_scores.get(boost_category, 0.0) > 5.0:
                total_boost += boost_weight * (category_scores[boost_category] / 10.0)
        return min(0.1, total_boost)

    def get_category_weight(self, category: str) -> float:
        """Get weight for a specific category."""
        return self.category_weights.get(category, 1.0)

    def get_min_recommendation_score(self) -> float:
        """Get minimum score threshold for recommendations."""
        return self.global_settings.get("min_recommendation_score", 0.5)

    def get_default_top_k(self) -> int:
        """Get default number of recommendations."""
        return self.global_settings.get("default_top_k", 5)
