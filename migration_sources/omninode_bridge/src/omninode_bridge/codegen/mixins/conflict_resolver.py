"""
Conflict Resolver for intelligent mixin selection.

Detects and resolves conflicts between recommended mixins:
- Mutual exclusions
- Missing prerequisites
- Redundancies

Performance Target: <20ms for conflict detection and resolution
Accuracy Target: 100% conflict detection
"""

import logging
from pathlib import Path
from typing import Any, Optional

import yaml

from omninode_bridge.codegen.mixins.models import (
    ModelMixinConflict,
    ModelMixinRecommendation,
)

logger = logging.getLogger(__name__)


class ConflictResolver:
    """
    Detect and resolve conflicts between mixins.

    Ensures recommended mixin combinations are valid and consistent.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the conflict resolver.

        Args:
            config_path: Path to conflict_rules.yaml (defaults to same directory)
        """
        if config_path is None:
            config_path = Path(__file__).parent / "conflict_rules.yaml"

        self.config = self._load_config(config_path)
        self.conflicts = self.config.get("conflicts", [])
        self.prerequisites = self.config.get("prerequisites", [])
        self.redundancies = self.config.get("redundancies", [])
        self.priorities = self.config.get("priorities", [])

    def _load_config(self, config_path: Path) -> dict[str, Any]:
        """Load conflict rules from YAML file."""
        try:
            with open(config_path) as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load conflict rules: {e}")
            return {}

    def resolve_conflicts(
        self,
        recommendations: list[ModelMixinRecommendation],
        scores: dict[str, float],
    ) -> tuple[list[str], list[str]]:
        """
        Resolve conflicts and return final mixin list.

        Args:
            recommendations: List of recommended mixins
            scores: Mixin scores for conflict resolution

        Returns:
            Tuple of (resolved_mixins, warnings)
                - resolved_mixins: Final list of mixin names
                - warnings: List of warning messages for manual review
        """
        mixin_names = [rec.mixin_name for rec in recommendations]
        warnings = []

        # Step 1: Detect all conflicts
        detected_conflicts = self.detect_conflicts(mixin_names)

        # Step 2: Resolve conflicts
        for conflict in detected_conflicts:
            resolution_result = self._apply_resolution(conflict, mixin_names, scores)

            if resolution_result["action"] == "remove":
                if resolution_result["mixin"] in mixin_names:
                    mixin_names.remove(resolution_result["mixin"])
                    warnings.append(
                        f"Removed {resolution_result['mixin']}: {conflict.reason}"
                    )
            elif resolution_result["action"] == "add":
                if resolution_result["mixin"] not in mixin_names:
                    mixin_names.append(resolution_result["mixin"])
                    warnings.append(
                        f"Added {resolution_result['mixin']}: {conflict.reason}"
                    )
            elif resolution_result["action"] == "warn":
                warnings.append(resolution_result["message"])

        return mixin_names, warnings

    def detect_conflicts(self, mixins: list[str]) -> list[ModelMixinConflict]:
        """
        Detect all conflicts in mixin list.

        Args:
            mixins: List of mixin names to check

        Returns:
            List of detected conflicts
        """
        conflicts_found = []

        # Check mutual exclusions
        for rule in self.conflicts:
            if rule["mixin_a"] in mixins and rule["mixin_b"] in mixins:
                conflicts_found.append(
                    ModelMixinConflict(
                        type="mutual_exclusion",
                        mixin_a=rule["mixin_a"],
                        mixin_b=rule["mixin_b"],
                        reason=rule["reason"],
                        resolution=rule["resolution"],
                    )
                )

        # Check missing prerequisites
        for rule in self.prerequisites:
            if rule["mixin"] in mixins:
                for required in rule["requires"]:
                    if required not in mixins:
                        conflicts_found.append(
                            ModelMixinConflict(
                                type="missing_prerequisite",
                                mixin_a=rule["mixin"],
                                mixin_b=required,
                                reason=rule["reason"],
                                resolution=(
                                    "add_prerequisite"
                                    if rule.get("auto_add", True)
                                    else "warn"
                                ),
                            )
                        )

        # Check redundancies
        for rule in self.redundancies:
            if rule["mixin"] in mixins:
                for included in rule["includes"]:
                    if included in mixins:
                        conflicts_found.append(
                            ModelMixinConflict(
                                type="redundancy",
                                mixin_a=rule["mixin"],
                                mixin_b=included,
                                reason=rule["reason"],
                                resolution="remove_redundant",
                            )
                        )

        return conflicts_found

    def _apply_resolution(
        self,
        conflict: ModelMixinConflict,
        current_mixins: list[str],
        scores: dict[str, float],
    ) -> dict[str, Any]:
        """
        Apply resolution strategy to conflict.

        Returns:
            dict with 'action' and 'mixin' or 'message'
        """
        resolution = conflict.resolution

        if resolution == "prefer_higher_score":
            # Keep mixin with higher score
            score_a = scores.get(conflict.mixin_a, 0.0)
            score_b = scores.get(conflict.mixin_b, 0.0)

            if score_a >= score_b:
                return {"action": "remove", "mixin": conflict.mixin_b}
            else:
                return {"action": "remove", "mixin": conflict.mixin_a}

        elif resolution == "prefer_event_driven":
            # Prefer event-driven paradigm
            return {"action": "remove", "mixin": conflict.mixin_b}

        elif resolution == "prefer_wrapper":
            # Prefer service wrapper over base class
            return {"action": "remove", "mixin": conflict.mixin_b}

        elif resolution == "add_prerequisite":
            # Add missing prerequisite
            return {"action": "add", "mixin": conflict.mixin_b}

        elif resolution == "remove_redundant":
            # Remove redundant mixin
            return {"action": "remove", "mixin": conflict.mixin_b}

        elif resolution == "warn":
            # Just warn, don't auto-resolve
            return {
                "action": "warn",
                "message": f"Warning: {conflict.mixin_a} may require {conflict.mixin_b}. {conflict.reason}",
            }

        else:
            # Unknown resolution strategy
            return {
                "action": "warn",
                "message": f"Unknown resolution strategy: {resolution}",
            }

    def get_category_priority(self, category: str) -> int:
        """Get priority for a category (lower number = higher priority)."""
        for priority_rule in self.priorities:
            if priority_rule["category"] == category:
                return priority_rule["priority"]
        return 999  # Low priority for unknown categories

    def has_conflicts(self, mixins: list[str]) -> bool:
        """Check if mixin list has any conflicts."""
        return len(self.detect_conflicts(mixins)) > 0
