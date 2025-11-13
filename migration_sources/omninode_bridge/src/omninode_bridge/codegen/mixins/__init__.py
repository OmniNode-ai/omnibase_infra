"""
Intelligent Mixin Selection - Phase 3 Implementation.

This module provides intelligent mixin recommendation based on requirements analysis.

Components:
- RequirementsAnalyzer: Extract and categorize requirements from PRD
- MixinScorer: Score mixins based on requirement match
- MixinRecommender: Generate top-K recommendations with explanations
- ConflictResolver: Detect and resolve mixin conflicts

Performance: <200ms total pipeline
Accuracy: >90% recommendation relevance
"""

from omninode_bridge.codegen.mixins.conflict_resolver import ConflictResolver
from omninode_bridge.codegen.mixins.mixin_recommender import MixinRecommender
from omninode_bridge.codegen.mixins.mixin_scorer import MixinScorer
from omninode_bridge.codegen.mixins.models import (
    ModelMixinConflict,
    ModelMixinRecommendation,
    ModelMixinUsageStats,
    ModelRequirementAnalysis,
)
from omninode_bridge.codegen.mixins.requirements_analyzer import RequirementsAnalyzer

__all__ = [
    "ModelMixinRecommendation",
    "ModelRequirementAnalysis",
    "ModelMixinConflict",
    "ModelMixinUsageStats",
    "RequirementsAnalyzer",
    "MixinScorer",
    "MixinRecommender",
    "ConflictResolver",
]

__version__ = "1.0.0"
