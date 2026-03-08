# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for artifact reconciliation domain event models."""

from datetime import UTC, datetime

import pytest


@pytest.mark.unit
class TestUpdateTrigger:
    def test_pr_opened_trigger(self) -> None:
        from omnibase_infra.nodes.node_artifact_change_detector_effect.models.model_update_trigger import (
            ModelUpdateTrigger,
        )

        trigger = ModelUpdateTrigger(
            trigger_id="t-001",
            trigger_type="pr_opened",
            source_repo="omnibase_infra",
            source_ref="refs/pull/123/head",
            changed_files=["src/omnibase_infra/nodes/node_foo/contract.yaml"],
            timestamp=datetime.now(UTC),
        )
        assert trigger.trigger_type == "pr_opened"
        assert len(trigger.changed_files) == 1

    def test_contract_changed_trigger(self) -> None:
        from omnibase_infra.nodes.node_artifact_change_detector_effect.models.model_update_trigger import (
            ModelUpdateTrigger,
        )

        trigger = ModelUpdateTrigger(
            trigger_id="t-002",
            trigger_type="contract_changed",
            source_repo="omnibase_infra",
            changed_files=["src/omnibase_infra/nodes/node_bar/contract.yaml"],
            timestamp=datetime.now(UTC),
        )
        assert trigger.trigger_type == "contract_changed"

    def test_manual_trigger_with_empty_changed_files(self) -> None:
        from omnibase_infra.nodes.node_artifact_change_detector_effect.models.model_update_trigger import (
            ModelUpdateTrigger,
        )

        trigger = ModelUpdateTrigger(
            trigger_id="t-003",
            trigger_type="manual_plan_request",
            source_repo="omnibase_infra",
            changed_files=[],
            reason="Manual check after migration",
            timestamp=datetime.now(UTC),
        )
        assert trigger.changed_files == []
        assert trigger.reason == "Manual check after migration"

    def test_trigger_rejects_invalid_type(self) -> None:
        from omnibase_infra.nodes.node_artifact_change_detector_effect.models.model_update_trigger import (
            ModelUpdateTrigger,
        )

        with pytest.raises(ValueError):
            ModelUpdateTrigger(
                trigger_id="t-bad",
                trigger_type="invalid_type",
                source_repo="omnibase_infra",
                changed_files=[],
                timestamp=datetime.now(UTC),
            )

    def test_trigger_rejects_extra_fields(self) -> None:
        from omnibase_infra.nodes.node_artifact_change_detector_effect.models.model_update_trigger import (
            ModelUpdateTrigger,
        )

        with pytest.raises(ValueError):
            ModelUpdateTrigger(
                trigger_id="t-bad",
                trigger_type="pr_opened",
                source_repo="omnibase_infra",
                changed_files=[],
                timestamp=datetime.now(UTC),
                bogus_field="nope",  # type: ignore[call-arg]
            )

    def test_trigger_is_frozen(self) -> None:
        from omnibase_infra.nodes.node_artifact_change_detector_effect.models.model_update_trigger import (
            ModelUpdateTrigger,
        )

        trigger = ModelUpdateTrigger(
            trigger_id="t-freeze",
            trigger_type="pr_opened",
            source_repo="omnibase_infra",
            changed_files=[],
            timestamp=datetime.now(UTC),
        )
        with pytest.raises(ValueError):
            trigger.trigger_id = "modified"  # type: ignore[misc]


@pytest.mark.unit
class TestImpactedArtifact:
    def test_impacted_artifact(self) -> None:
        from omnibase_infra.nodes.node_impact_analyzer_compute.models.model_impacted_artifact import (
            ModelImpactedArtifact,
        )

        artifact = ModelImpactedArtifact(
            artifact_id="doc-handler-lifecycle",
            artifact_type="doc",
            path="docs/architecture/handler_lifecycle.md",
            impact_strength=0.85,
            reason_codes=["contract_yaml_changed"],
            required_action="review",
        )
        assert artifact.impact_strength == 0.85

    def test_impact_strength_bounds_upper(self) -> None:
        from omnibase_infra.nodes.node_impact_analyzer_compute.models.model_impacted_artifact import (
            ModelImpactedArtifact,
        )

        with pytest.raises(ValueError):
            ModelImpactedArtifact(
                artifact_id="x",
                artifact_type="doc",
                path="x",
                impact_strength=1.5,
                reason_codes=[],
                required_action="none",
            )

    def test_impact_strength_bounds_lower(self) -> None:
        from omnibase_infra.nodes.node_impact_analyzer_compute.models.model_impacted_artifact import (
            ModelImpactedArtifact,
        )

        with pytest.raises(ValueError):
            ModelImpactedArtifact(
                artifact_id="x",
                artifact_type="doc",
                path="x",
                impact_strength=-0.1,
                reason_codes=[],
                required_action="none",
            )

    def test_impact_strength_at_boundaries(self) -> None:
        from omnibase_infra.nodes.node_impact_analyzer_compute.models.model_impacted_artifact import (
            ModelImpactedArtifact,
        )

        # Exactly 0.0 and 1.0 should be valid
        a0 = ModelImpactedArtifact(
            artifact_id="a",
            artifact_type="doc",
            path="p",
            impact_strength=0.0,
            reason_codes=[],
            required_action="none",
        )
        a1 = ModelImpactedArtifact(
            artifact_id="b",
            artifact_type="doc",
            path="p",
            impact_strength=1.0,
            reason_codes=[],
            required_action="none",
        )
        assert a0.impact_strength == 0.0
        assert a1.impact_strength == 1.0

    def test_rejects_invalid_artifact_type(self) -> None:
        from omnibase_infra.nodes.node_impact_analyzer_compute.models.model_impacted_artifact import (
            ModelImpactedArtifact,
        )

        with pytest.raises(ValueError):
            ModelImpactedArtifact(
                artifact_id="x",
                artifact_type="invalid",
                path="x",
                impact_strength=0.5,
                reason_codes=[],
                required_action="none",
            )


@pytest.mark.unit
class TestImpactAnalysisResult:
    def test_result_with_artifacts(self) -> None:
        from omnibase_infra.nodes.node_impact_analyzer_compute.models.model_impact_analysis_result import (
            ModelImpactAnalysisResult,
        )
        from omnibase_infra.nodes.node_impact_analyzer_compute.models.model_impacted_artifact import (
            ModelImpactedArtifact,
        )

        artifact = ModelImpactedArtifact(
            artifact_id="doc-test",
            artifact_type="doc",
            path="docs/test.md",
            impact_strength=0.7,
            reason_codes=["contract_yaml_changed"],
            required_action="review",
        )
        result = ModelImpactAnalysisResult(
            source_trigger_id="t-001",
            impacted_artifacts=[artifact],
            highest_merge_policy="require",
        )
        assert len(result.impacted_artifacts) == 1
        assert result.highest_merge_policy == "require"

    def test_empty_result(self) -> None:
        from omnibase_infra.nodes.node_impact_analyzer_compute.models.model_impact_analysis_result import (
            ModelImpactAnalysisResult,
        )

        result = ModelImpactAnalysisResult(
            source_trigger_id="t-empty",
            impacted_artifacts=[],
            highest_merge_policy="none",
        )
        assert len(result.impacted_artifacts) == 0


@pytest.mark.unit
class TestUpdateTask:
    def test_update_task_defaults(self) -> None:
        from omnibase_infra.nodes.node_update_plan_reducer.models.model_update_task import (
            ModelUpdateTask,
        )

        task = ModelUpdateTask(
            task_id="ut-001",
            title="Review handler lifecycle doc",
            target_artifact_id="doc-handler-lifecycle",
            task_type="human_author",
            blocking=True,
        )
        assert task.status == "planned"
        assert task.depends_on == []
        assert task.owner_hint is None

    def test_task_rejects_invalid_type(self) -> None:
        from omnibase_infra.nodes.node_update_plan_reducer.models.model_update_task import (
            ModelUpdateTask,
        )

        with pytest.raises(ValueError):
            ModelUpdateTask(
                task_id="ut-bad",
                title="Bad task",
                target_artifact_id="x",
                task_type="invalid",
            )

    def test_task_rejects_invalid_status(self) -> None:
        from omnibase_infra.nodes.node_update_plan_reducer.models.model_update_task import (
            ModelUpdateTask,
        )

        with pytest.raises(ValueError):
            ModelUpdateTask(
                task_id="ut-bad",
                title="Bad task",
                target_artifact_id="x",
                task_type="human_author",
                status="invalid",
            )


@pytest.mark.unit
class TestUpdatePlan:
    def test_update_plan_with_tasks(self) -> None:
        from omnibase_infra.nodes.node_update_plan_reducer.models.model_update_plan import (
            ModelUpdatePlan,
        )
        from omnibase_infra.nodes.node_update_plan_reducer.models.model_update_task import (
            ModelUpdateTask,
        )

        task = ModelUpdateTask(
            task_id="ut-001",
            title="Review handler lifecycle doc",
            target_artifact_id="doc-handler-lifecycle",
            task_type="human_author",
            blocking=True,
        )
        plan = ModelUpdatePlan(
            plan_id="plan-001",
            source_trigger_id="t-001",
            source_entity_ref="pr/omnibase_infra/123",
            summary="PR #123 modified contract.yaml",
            impacted_artifacts=[],
            tasks=[task],
            merge_policy="require",
            created_at=datetime.now(UTC),
        )
        assert plan.merge_policy == "require"
        assert len(plan.tasks) == 1
        assert plan.tasks[0].status == "planned"

    def test_plan_rejects_invalid_merge_policy(self) -> None:
        from omnibase_infra.nodes.node_update_plan_reducer.models.model_update_plan import (
            ModelUpdatePlan,
        )

        with pytest.raises(ValueError):
            ModelUpdatePlan(
                plan_id="plan-bad",
                source_trigger_id="t-001",
                source_entity_ref="pr/test/1",
                summary="Bad plan",
                impacted_artifacts=[],
                tasks=[],
                merge_policy="invalid",
                created_at=datetime.now(UTC),
            )

    def test_plan_rejects_extra_fields(self) -> None:
        from omnibase_infra.nodes.node_update_plan_reducer.models.model_update_plan import (
            ModelUpdatePlan,
        )

        with pytest.raises(ValueError):
            ModelUpdatePlan(
                plan_id="plan-bad",
                source_trigger_id="t-001",
                source_entity_ref="pr/test/1",
                summary="Bad plan",
                impacted_artifacts=[],
                tasks=[],
                merge_policy="none",
                created_at=datetime.now(UTC),
                bogus="nope",  # type: ignore[call-arg]
            )
