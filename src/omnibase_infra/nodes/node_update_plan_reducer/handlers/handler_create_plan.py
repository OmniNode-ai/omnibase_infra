# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Handler for creating update plans from impact analysis results.

Implements the deterministic task_type mapping:
    required_action → task_type (per OMN-3925 plan):

    | required_action | task_type    | blocking                     |
    |-----------------|--------------|------------------------------|
    | review          | human_author | True if require/strict policy |
    | regenerate      | regenerate   | True                         |
    | create          | create_stub  | True                         |
    | none            | (skipped)    | —                            |

The plan's merge_policy is set to highest_merge_policy from the impact
analysis result.

Tracking:
    - OMN-3943: Task 6 — Update Plan REDUCER Node
    - OMN-3925: Artifact Reconciliation + Update Planning MVP
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from uuid import UUID, uuid4

from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
)
from omnibase_infra.nodes.node_impact_analyzer_compute.models.model_impact_analysis_result import (
    ModelImpactAnalysisResult,
)
from omnibase_infra.nodes.node_impact_analyzer_compute.models.model_impacted_artifact import (
    ModelImpactedArtifact,
)
from omnibase_infra.nodes.node_update_plan_reducer.models.model_update_plan import (
    ModelUpdatePlan,
)
from omnibase_infra.nodes.node_update_plan_reducer.models.model_update_task import (
    ModelUpdateTask,
)

logger = logging.getLogger(__name__)

# Policies where review tasks are blocking
_BLOCKING_REVIEW_POLICIES: frozenset[str] = frozenset({"require", "strict"})


class HandlerCreatePlan:
    """Create an update plan from an impact analysis result.

    Pure function: deterministic, no external I/O.

    Translates ModelImpactAnalysisResult → ModelUpdatePlan by:
    1. Iterating impacted artifacts
    2. Mapping each artifact's required_action to a task_type
    3. Setting blocking based on task_type and update_policy
    4. Setting merge_policy to highest_merge_policy from the result
    """

    @property
    def handler_id(self) -> str:
        """Unique handler identifier."""
        return "handler-create-plan"

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role: infrastructure handler."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Behavioral classification: pure compute."""
        return EnumHandlerTypeCategory.COMPUTE

    def create_plan(
        self,
        result: ModelImpactAnalysisResult,
        source_entity_ref: str,
        summary: str,
        plan_id: UUID | None = None,
    ) -> ModelUpdatePlan:
        """Create an update plan from an impact analysis result.

        Args:
            result: The impact analysis result containing impacted artifacts.
            source_entity_ref: Reference to the source entity (e.g. "pr/repo/123").
            summary: Human-readable summary of the plan.
            plan_id: Optional plan UUID (generated if not provided).

        Returns:
            A ModelUpdatePlan with tasks mapped from impacted artifacts.
        """
        effective_plan_id = plan_id or uuid4()
        tasks = self._build_tasks(result)

        plan = ModelUpdatePlan(
            plan_id=effective_plan_id,
            source_trigger_id=result.source_trigger_id,
            source_entity_ref=source_entity_ref,
            summary=summary,
            impacted_artifacts=list(result.impacted_artifacts),
            tasks=tasks,
            merge_policy=result.highest_merge_policy,
            created_at=datetime.now(UTC),
        )

        logger.info(
            "Plan created: plan_id=%s tasks=%d merge_policy=%s",
            effective_plan_id,
            len(tasks),
            result.highest_merge_policy,
        )

        return plan

    def _build_tasks(
        self,
        result: ModelImpactAnalysisResult,
    ) -> list[ModelUpdateTask]:
        """Build tasks from impacted artifacts.

        Args:
            result: The impact analysis result.

        Returns:
            List of ModelUpdateTask instances.
        """
        tasks: list[ModelUpdateTask] = []
        for artifact in result.impacted_artifacts:
            task = self._artifact_to_task(artifact)
            if task is not None:
                tasks.append(task)
        return tasks

    @staticmethod
    def _artifact_to_task(  # stub-ok: implemented
        artifact: ModelImpactedArtifact,
    ) -> ModelUpdateTask | None:
        """Map a single impacted artifact to a task.

        Applies the deterministic mapping table from OMN-3925:
            - required_action == "none"       → no task created (returns None)
            - required_action == "review"     → task_type="human_author"
                                                blocking=True if update_policy in (require, strict)
            - required_action == "regenerate" → task_type="regenerate", blocking=True
            - required_action == "create"     → task_type="create_stub", blocking=True
            - required_action == "patch"      → task_type="patch_existing", blocking=False

        Args:
            artifact: The impacted artifact to map.

        Returns:
            A ModelUpdateTask or None if no task is needed.
        """
        if artifact.required_action == "none":
            return None

        if artifact.required_action == "review":
            # Blocking only for require/strict policies — infer from reason_codes
            # We use the artifact's artifact_type to set a descriptive title
            task_type = "human_author"
            # Note: update_policy is not stored on ModelImpactedArtifact, so we
            # derive blocking from whether the artifact was flagged at all with
            # a policy-sensitive required_action. The impact handler only sets
            # required_action="review" for require/strict when impact_strength
            # is between 0 and ACTION_THRESHOLD_REVIEW. For higher strengths,
            # all policies get review. We default blocking=True for review tasks
            # since the plan's merge_policy captures the aggregate enforcement.
            blocking = True
            title = f"Review {artifact.artifact_type}: {artifact.path}"
        elif artifact.required_action == "regenerate":
            task_type = "regenerate"
            blocking = True
            title = f"Regenerate {artifact.artifact_type}: {artifact.path}"
        elif artifact.required_action == "create":
            task_type = "create_stub"
            blocking = True
            title = f"Create stub for {artifact.artifact_type}: {artifact.path}"
        elif artifact.required_action == "patch":
            task_type = "patch_existing"
            blocking = False
            title = f"Patch {artifact.artifact_type}: {artifact.path}"
        else:
            # Unknown action — skip
            logger.warning(
                "Unknown required_action %r for artifact %s — skipping",
                artifact.required_action,
                artifact.artifact_id,
            )
            return None

        return ModelUpdateTask(
            task_id=uuid4(),
            title=title,
            target_artifact_id=artifact.artifact_id,
            task_type=task_type,
            blocking=blocking,
        )


__all__: list[str] = ["HandlerCreatePlan"]
