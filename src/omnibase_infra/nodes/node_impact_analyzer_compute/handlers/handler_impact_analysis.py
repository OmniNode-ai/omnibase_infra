# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Handler for computing artifact impact from change triggers.

Pure deterministic compute: no I/O, no external dependencies.
Implements the explicit scoring table defined in OMN-3925.
"""

from __future__ import annotations

import fnmatch
import logging
from typing import Literal

from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
)
from omnibase_infra.nodes.node_artifact_change_detector_effect.models.model_update_trigger import (
    ModelUpdateTrigger,
)
from omnibase_infra.nodes.node_impact_analyzer_compute.handlers.constants import (
    ACTION_THRESHOLD_REGENERATE,
    ACTION_THRESHOLD_REVIEW,
    MERGE_POLICY_ORDER,
    POLICY_FLOORS,
    SCOPE_MULTIPLIER_PR,
    SCOPE_MULTIPLIER_STRUCTURAL,
    STRUCTURAL_TRIGGER_TYPES,
)
from omnibase_infra.nodes.node_impact_analyzer_compute.models.model_impact_analysis_result import (
    ModelImpactAnalysisResult,
)
from omnibase_infra.nodes.node_impact_analyzer_compute.models.model_impacted_artifact import (
    ModelImpactedArtifact,
)
from omnibase_infra.registry.models.model_artifact_registry import ModelArtifactRegistry
from omnibase_infra.registry.models.model_artifact_registry_entry import (
    ModelArtifactRegistryEntry,
)

logger = logging.getLogger(__name__)


class HandlerImpactAnalysis:
    """Compute impact of a change trigger against the artifact registry.

    Pure function: deterministic, no external I/O.

    Implements the scoring table from OMN-3925:
    - Match changed files against source_triggers[].pattern using fnmatch
    - base = matched_triggers / total_triggers
    - scope_multiplier: 1.0 for structural, 0.7 for PR triggers
    - policy_floor enforced per update_policy
    - impact_strength = max(base * scope_multiplier, policy_floor), capped at 1.0
    - required_action: >=0.8 → regenerate, >=0.5 → review, >0.0 + require/strict → review
    - highest_merge_policy = max(update_policy) over all impacted artifacts

    Special case: manual_plan_request with empty changed_files matches ALL
    triggers for artifacts in the source_repo.
    """

    @property
    def handler_id(self) -> str:
        """Unique handler identifier."""
        return "handler-impact-analysis"

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role: infrastructure handler."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Behavioral classification: pure deterministic compute."""
        return EnumHandlerTypeCategory.COMPUTE

    def analyze(
        self,
        trigger: ModelUpdateTrigger,
        registry: ModelArtifactRegistry,
    ) -> ModelImpactAnalysisResult:
        """Analyze the impact of a change trigger against the registry.

        Args:
            trigger: The change trigger with file list and trigger type.
            registry: The artifact registry to match against.

        Returns:
            ModelImpactAnalysisResult containing impacted artifacts and
            the highest merge policy across all impacted artifacts.
        """
        is_manual_empty = (
            trigger.trigger_type == "manual_plan_request"
            and len(trigger.changed_files) == 0
        )

        impacted: list[ModelImpactedArtifact] = []
        highest_policy_level = 0

        for entry in registry.artifacts:
            artifact = self._score_artifact(
                entry=entry,
                trigger=trigger,
                is_manual_empty=is_manual_empty,
            )
            if artifact is not None:
                impacted.append(artifact)
                policy_level = MERGE_POLICY_ORDER.get(entry.update_policy, 0)
                highest_policy_level = max(highest_policy_level, policy_level)

        # Determine highest merge policy string from level
        highest_merge_policy = _level_to_policy(highest_policy_level)

        result = ModelImpactAnalysisResult(
            source_trigger_id=trigger.trigger_id,
            impacted_artifacts=impacted,
            highest_merge_policy=highest_merge_policy,
        )

        logger.info(
            "Impact analysis complete: trigger=%s type=%s impacted=%d highest_policy=%s",
            trigger.trigger_id,
            trigger.trigger_type,
            len(impacted),
            highest_merge_policy,
        )

        return result

    def _score_artifact(
        self,
        entry: ModelArtifactRegistryEntry,
        trigger: ModelUpdateTrigger,
        is_manual_empty: bool,
    ) -> ModelImpactedArtifact | None:
        """Score a single artifact against the trigger.

        Returns None if the artifact has no impact (matched_triggers == 0).
        """
        if not entry.source_triggers:
            return None

        total_triggers = len(entry.source_triggers)

        if is_manual_empty and entry.repo == trigger.source_repo:
            # manual_plan_request with empty changed_files: match all triggers
            matched_triggers = total_triggers
            reason_codes = ["manual_reconciliation"]
        else:
            matched_triggers, reason_codes = self._count_matches(
                entry=entry,
                changed_files=trigger.changed_files,
            )

        if matched_triggers == 0:
            return None

        # Compute impact strength
        base = matched_triggers / total_triggers

        if trigger.trigger_type in STRUCTURAL_TRIGGER_TYPES:
            scope_multiplier = SCOPE_MULTIPLIER_STRUCTURAL
        else:
            scope_multiplier = SCOPE_MULTIPLIER_PR

        policy_floor = POLICY_FLOORS.get(entry.update_policy, 0.0)
        impact_strength = max(base * scope_multiplier, policy_floor)
        impact_strength = min(impact_strength, 1.0)

        required_action = self._assign_action(impact_strength, entry.update_policy)

        if required_action == "none":
            return None

        return ModelImpactedArtifact(
            artifact_id=entry.artifact_id,
            artifact_type=entry.artifact_type,
            path=entry.path,
            impact_strength=impact_strength,
            reason_codes=reason_codes,
            required_action=required_action,
        )

    @staticmethod
    def _count_matches(
        entry: ModelArtifactRegistryEntry,
        changed_files: list[str],
    ) -> tuple[int, list[str]]:
        """Count how many triggers match at least one changed file.

        Each trigger counts at most once regardless of how many files match.

        Returns:
            Tuple of (matched_trigger_count, reason_codes).
        """
        matched = 0
        reason_codes: list[str] = []

        for trigger in entry.source_triggers:
            trigger_matched = any(
                fnmatch.fnmatch(changed_file, trigger.pattern)
                for changed_file in changed_files
            )
            if trigger_matched:
                matched += 1
                # Derive reason code from trigger pattern
                reason_codes.append(_pattern_to_reason_code(trigger.pattern))

        return matched, list(dict.fromkeys(reason_codes))  # deduplicate, preserve order

    @staticmethod
    def _assign_action(
        impact_strength: float,
        update_policy: str,
    ) -> Literal["none", "review", "regenerate"]:
        """Assign required_action based on impact strength and policy.

        Implements the deterministic mapping table from OMN-3925:
        - >= ACTION_THRESHOLD_REGENERATE → regenerate
        - >= ACTION_THRESHOLD_REVIEW → review
        - > 0.0 with require/strict policy → review
        - else → none
        """
        if impact_strength >= ACTION_THRESHOLD_REGENERATE:
            return "regenerate"
        if impact_strength >= ACTION_THRESHOLD_REVIEW:
            return "review"
        if impact_strength > 0.0 and update_policy in ("require", "strict"):
            return "review"
        return "none"


def _level_to_policy(level: int) -> Literal["none", "warn", "require", "strict"]:
    """Convert a numeric policy level back to the policy string."""
    for policy, lvl in MERGE_POLICY_ORDER.items():
        if lvl == level:
            return policy  # type: ignore[return-value]
    return "none"


def _pattern_to_reason_code(pattern: str) -> str:
    """Derive a reason code from a file pattern.

    Maps well-known patterns to canonical reason codes. Falls back to
    'config_changed' for unrecognized patterns.
    """
    if "contract.yaml" in pattern:
        return "contract_yaml_changed"
    if "handler" in pattern:
        return "handler_routing_changed"
    if "schema" in pattern:
        return "schema_changed"
    if "script" in pattern or pattern.endswith((".sh", ".py")):
        return "script_changed"
    if "event_bus" in pattern or "topics" in pattern:
        return "event_bus_topics_changed"
    return "config_changed"


__all__: list[str] = ["HandlerImpactAnalysis"]
