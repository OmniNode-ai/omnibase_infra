# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Integration tests for the full artifact reconciliation pipeline.

Wires all 4 nodes with in-memory event bus (direct handler calls, no Kafka):

    ModelUpdateTrigger
         |
         v
    HandlerImpactAnalysis (COMPUTE) -> ModelImpactAnalysisResult
         |
         v
    HandlerCreatePlan (REDUCER) -> ModelUpdatePlan
         |
         +-- HandlerPlanToPRComment (ORCHESTRATOR) -> ModelPRCommentResult
         |
         +-- HandlerPlanToYaml (ORCHESTRATOR) -> ModelYamlEmitResult

Tests:
    TestArtifactReconciliationPipeline:
        - test_trigger_produces_impact_analysis_result
        - test_impact_result_produces_update_plan
        - test_pr_comment_handler_posts_for_pr_trigger
        - test_pr_comment_handler_skips_non_pr_trigger
        - test_pr_comment_handler_idempotent_anchor
        - test_yaml_handler_produces_correct_payload
        - test_full_pipeline_end_to_end

Tracking:
    - OMN-3944: Task 7 - Reconciliation ORCHESTRATOR Node
    - OMN-3925: Artifact Reconciliation + Update Planning MVP
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
import yaml

from omnibase_infra.nodes.node_artifact_change_detector_effect.models.model_update_trigger import (
    ModelUpdateTrigger,
)
from omnibase_infra.nodes.node_artifact_reconciliation_orchestrator.handlers.handler_plan_to_pr_comment import (
    HandlerPlanToPRComment,
    _anchor_for_plan,
    _build_markdown_table,
)
from omnibase_infra.nodes.node_artifact_reconciliation_orchestrator.handlers.handler_plan_to_yaml import (
    HandlerPlanToYaml,
)
from omnibase_infra.nodes.node_artifact_reconciliation_orchestrator.models.model_pr_comment_result import (
    ModelPRCommentResult,
)
from omnibase_infra.nodes.node_artifact_reconciliation_orchestrator.models.model_yaml_emit_result import (
    ModelYamlEmitResult,
)
from omnibase_infra.nodes.node_impact_analyzer_compute.handlers.handler_impact_analysis import (
    HandlerImpactAnalysis,
)
from omnibase_infra.nodes.node_impact_analyzer_compute.models.model_impact_analysis_result import (
    ModelImpactAnalysisResult,
)
from omnibase_infra.nodes.node_update_plan_reducer.handlers.handler_create_plan import (
    HandlerCreatePlan,
)
from omnibase_infra.nodes.node_update_plan_reducer.models.model_update_plan import (
    ModelUpdatePlan,
)
from omnibase_infra.registry.models.model_artifact_registry import ModelArtifactRegistry
from omnibase_infra.registry.models.model_artifact_registry_entry import (
    ModelArtifactRegistryEntry,
)
from omnibase_infra.registry.models.model_source_trigger import ModelSourceTrigger

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def artifact_id_1() -> UUID:
    """UUID for the first test artifact."""
    return UUID("a1b2c3d4-0001-0001-0001-000000000001")


@pytest.fixture
def artifact_id_2() -> UUID:
    """UUID for the second test artifact."""
    return UUID("a1b2c3d4-0001-0001-0001-000000000002")


@pytest.fixture
def test_registry(artifact_id_1: UUID, artifact_id_2: UUID) -> ModelArtifactRegistry:
    """Minimal test artifact registry with two entries."""
    return ModelArtifactRegistry(
        version="1.0.0",
        description="Test registry",
        artifacts=[
            ModelArtifactRegistryEntry(
                artifact_id=artifact_id_1,
                artifact_type="doc",
                title="Handler Protocol Architecture",
                path="docs/architecture/handler_protocol.md",
                repo="omnibase_infra",
                owner_hint="platform-core",
                update_policy="require",
                source_triggers=[
                    ModelSourceTrigger(
                        pattern="src/omnibase_infra/nodes/*/contract.yaml",
                        change_scope="structural",
                    ),
                ],
            ),
            ModelArtifactRegistryEntry(
                artifact_id=artifact_id_2,
                artifact_type="reference",
                title="Topic Catalog",
                path="docs/architecture/TOPIC_CATALOG.md",
                repo="omnibase_infra",
                owner_hint="platform-core",
                update_policy="strict",
                source_triggers=[
                    ModelSourceTrigger(
                        pattern="src/omnibase_infra/nodes/*/contract.yaml",
                        change_scope="structural",
                    ),
                ],
            ),
        ],
    )


@pytest.fixture
def pr_trigger() -> ModelUpdateTrigger:
    """A PR-opened trigger that modifies contract.yaml files."""
    return ModelUpdateTrigger(
        trigger_id=uuid4(),
        trigger_type="pr_opened",
        source_repo="omnibase_infra",
        source_ref="refs/pull/123/head",
        changed_files=["src/omnibase_infra/nodes/node_foo/contract.yaml"],
        timestamp=datetime.now(UTC),
    )


@pytest.fixture
def manual_trigger() -> ModelUpdateTrigger:
    """A manual plan request trigger."""
    return ModelUpdateTrigger(
        trigger_id=uuid4(),
        trigger_type="manual_plan_request",
        source_repo="omnibase_infra",
        changed_files=[],
        reason="Manual reconciliation check",
        timestamp=datetime.now(UTC),
    )


@pytest.fixture
def impact_handler() -> HandlerImpactAnalysis:
    """The impact analysis COMPUTE handler."""
    return HandlerImpactAnalysis()


@pytest.fixture
def plan_handler() -> HandlerCreatePlan:
    """The update plan REDUCER handler."""
    return HandlerCreatePlan()


@pytest.fixture
def yaml_handler() -> HandlerPlanToYaml:
    """The plan-to-YAML ORCHESTRATOR handler."""
    return HandlerPlanToYaml()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestArtifactReconciliationPipeline:
    """Integration tests for the full 4-node artifact reconciliation pipeline."""

    def test_trigger_produces_impact_analysis_result(
        self,
        pr_trigger: ModelUpdateTrigger,
        test_registry: ModelArtifactRegistry,
        impact_handler: HandlerImpactAnalysis,
    ) -> None:
        """Fire ModelUpdateTrigger and assert ModelImpactAnalysisResult is produced."""
        result = impact_handler.analyze(pr_trigger, test_registry)

        assert isinstance(result, ModelImpactAnalysisResult)
        assert result.source_trigger_id == pr_trigger.trigger_id
        # Both artifacts should be impacted (contract.yaml trigger matches)
        assert len(result.impacted_artifacts) == 2
        # highest policy should be "strict" (max of "require" and "strict")
        assert result.highest_merge_policy == "strict"

    def test_impact_result_produces_update_plan(
        self,
        pr_trigger: ModelUpdateTrigger,
        test_registry: ModelArtifactRegistry,
        impact_handler: HandlerImpactAnalysis,
        plan_handler: HandlerCreatePlan,
    ) -> None:
        """Assert ModelUpdatePlan is created from ModelImpactAnalysisResult."""
        result = impact_handler.analyze(pr_trigger, test_registry)
        plan = plan_handler.create_plan(
            result=result,
            source_entity_ref="pr/OmniNode-ai/omnibase_infra/123",
            summary="PR #123 modified contract.yaml",
        )

        assert isinstance(plan, ModelUpdatePlan)
        assert plan.source_trigger_id == pr_trigger.trigger_id
        assert plan.merge_policy == "strict"
        assert len(plan.tasks) > 0
        # All tasks reference valid artifact IDs
        artifact_ids = {a.artifact_id for a in plan.impacted_artifacts}
        for task in plan.tasks:
            assert task.target_artifact_id in artifact_ids

    @pytest.mark.asyncio
    async def test_pr_comment_handler_posts_for_pr_trigger(
        self,
        pr_trigger: ModelUpdateTrigger,
        test_registry: ModelArtifactRegistry,
        impact_handler: HandlerImpactAnalysis,
        plan_handler: HandlerCreatePlan,
    ) -> None:
        """Assert PR comment event is emitted with correct markdown for PR trigger."""
        result = impact_handler.analyze(pr_trigger, test_registry)
        plan = plan_handler.create_plan(
            result=result,
            source_entity_ref="pr/OmniNode-ai/omnibase_infra/123",
            summary="PR #123 modified contract.yaml",
        )

        # Build expected markdown to verify structure
        expected_markdown = _build_markdown_table(plan)
        assert "Artifact | Type | Strength | Action | Owner" in expected_markdown
        assert plan.summary in expected_markdown
        assert plan.merge_policy in expected_markdown

        # Mock httpx to avoid real network calls
        mock_response_list = MagicMock()
        mock_response_list.raise_for_status = MagicMock()
        mock_response_list.json = MagicMock(return_value=[])  # no existing comments

        mock_response_post = MagicMock()
        mock_response_post.raise_for_status = MagicMock()
        mock_response_post.json = MagicMock(return_value={"id": 42})

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response_list)
        mock_client.post = AsyncMock(return_value=mock_response_post)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        handler = HandlerPlanToPRComment()
        with patch("httpx.AsyncClient", return_value=mock_client):
            comment_result = await handler.post_plan_comment(
                plan=plan,
                trigger_type=pr_trigger.trigger_type,
            )

        assert isinstance(comment_result, ModelPRCommentResult)
        assert comment_result.posted is True
        assert comment_result.comment_id == 42
        assert comment_result.error is None
        assert comment_result.skipped is False

        # Verify the comment body contains our anchor
        post_call_args = mock_client.post.call_args
        posted_body: str = post_call_args.kwargs["json"]["body"]
        anchor = _anchor_for_plan(plan.plan_id)
        assert anchor in posted_body
        # Verify markdown table is in the body
        assert "| Artifact |" in posted_body

    @pytest.mark.asyncio
    async def test_pr_comment_handler_skips_non_pr_trigger(
        self,
        manual_trigger: ModelUpdateTrigger,
        test_registry: ModelArtifactRegistry,
        impact_handler: HandlerImpactAnalysis,
        plan_handler: HandlerCreatePlan,
    ) -> None:
        """Assert PR comment handler skips non-PR trigger types."""
        result = impact_handler.analyze(manual_trigger, test_registry)
        plan = plan_handler.create_plan(
            result=result,
            source_entity_ref="manual/omnibase_infra",
            summary="Manual reconciliation",
        )

        handler = HandlerPlanToPRComment()
        comment_result = await handler.post_plan_comment(
            plan=plan,
            trigger_type=manual_trigger.trigger_type,
        )

        assert isinstance(comment_result, ModelPRCommentResult)
        assert comment_result.posted is False
        assert comment_result.skipped is True
        assert comment_result.comment_id is None

    @pytest.mark.asyncio
    async def test_pr_comment_handler_idempotent_anchor(
        self,
        pr_trigger: ModelUpdateTrigger,
        test_registry: ModelArtifactRegistry,
        impact_handler: HandlerImpactAnalysis,
        plan_handler: HandlerCreatePlan,
    ) -> None:
        """Assert handler updates existing comment when anchor is found (idempotent)."""
        result = impact_handler.analyze(pr_trigger, test_registry)
        plan = plan_handler.create_plan(
            result=result,
            source_entity_ref="pr/OmniNode-ai/omnibase_infra/123",
            summary="PR #123 modified contract.yaml",
        )

        anchor = _anchor_for_plan(plan.plan_id)
        existing_comment_id = 99

        # Simulate existing comment with our anchor
        mock_response_list = MagicMock()
        mock_response_list.raise_for_status = MagicMock()
        mock_response_list.json = MagicMock(
            return_value=[
                {"id": existing_comment_id, "body": f"{anchor}\n\nOld content"}
            ]
        )

        mock_response_patch = MagicMock()
        mock_response_patch.raise_for_status = MagicMock()
        mock_response_patch.json = MagicMock(return_value={"id": existing_comment_id})

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response_list)
        mock_client.patch = AsyncMock(return_value=mock_response_patch)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        handler = HandlerPlanToPRComment()
        with patch("httpx.AsyncClient", return_value=mock_client):
            comment_result = await handler.post_plan_comment(
                plan=plan,
                trigger_type=pr_trigger.trigger_type,
            )

        assert isinstance(comment_result, ModelPRCommentResult)
        assert comment_result.posted is True
        assert comment_result.comment_id == existing_comment_id
        # PATCH called (not POST)
        mock_client.patch.assert_called_once()
        mock_client.post.assert_not_called()

    def test_yaml_handler_produces_correct_payload(
        self,
        pr_trigger: ModelUpdateTrigger,
        test_registry: ModelArtifactRegistry,
        impact_handler: HandlerImpactAnalysis,
        plan_handler: HandlerCreatePlan,
        yaml_handler: HandlerPlanToYaml,
    ) -> None:
        """Assert YAML event is emitted with correct payload and topic."""
        result = impact_handler.analyze(pr_trigger, test_registry)
        plan = plan_handler.create_plan(
            result=result,
            source_entity_ref="pr/OmniNode-ai/omnibase_infra/123",
            summary="PR #123 modified contract.yaml",
        )

        yaml_result = yaml_handler.serialize_plan(plan=plan)

        assert isinstance(yaml_result, ModelYamlEmitResult)
        assert yaml_result.plan_id == plan.plan_id
        assert yaml_result.topic == "onex.evt.artifact.update-plan-emitted.v1"

        # Verify YAML is valid and contains plan data
        parsed = yaml.safe_load(yaml_result.yaml_payload)
        assert isinstance(parsed, dict)
        assert str(parsed["plan_id"]) == str(plan.plan_id)
        assert parsed["merge_policy"] == "strict"
        assert "impacted_artifacts" in parsed
        assert "tasks" in parsed

    @pytest.mark.asyncio
    async def test_full_pipeline_end_to_end(
        self,
        pr_trigger: ModelUpdateTrigger,
        test_registry: ModelArtifactRegistry,
        impact_handler: HandlerImpactAnalysis,
        plan_handler: HandlerCreatePlan,
        yaml_handler: HandlerPlanToYaml,
        artifact_id_1: UUID,
        artifact_id_2: UUID,
    ) -> None:
        """Full end-to-end pipeline: trigger -> impact -> plan -> PR comment + YAML.

        Wires all 4 nodes with in-memory event bus (no Kafka):
        1. ModelUpdateTrigger -> HandlerImpactAnalysis -> ModelImpactAnalysisResult
        2. ModelImpactAnalysisResult -> HandlerCreatePlan -> ModelUpdatePlan
        3. ModelUpdatePlan -> HandlerPlanToPRComment -> ModelPRCommentResult
        4. ModelUpdatePlan -> HandlerPlanToYaml -> ModelYamlEmitResult
        """
        # Stage 1: Impact analysis
        impact_result = impact_handler.analyze(pr_trigger, test_registry)
        assert isinstance(impact_result, ModelImpactAnalysisResult)
        assert impact_result.source_trigger_id == pr_trigger.trigger_id
        assert len(impact_result.impacted_artifacts) == 2

        # Stage 2: Plan creation
        plan = plan_handler.create_plan(
            result=impact_result,
            source_entity_ref="pr/OmniNode-ai/omnibase_infra/123",
            summary="PR #123 modified contract.yaml",
        )
        assert isinstance(plan, ModelUpdatePlan)
        assert plan.merge_policy == "strict"
        assert len(plan.tasks) >= 1

        # Stage 3: PR comment (mocked HTTP)
        mock_response_list = MagicMock()
        mock_response_list.raise_for_status = MagicMock()
        mock_response_list.json = MagicMock(return_value=[])

        mock_response_post = MagicMock()
        mock_response_post.raise_for_status = MagicMock()
        mock_response_post.json = MagicMock(return_value={"id": 777})

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response_list)
        mock_client.post = AsyncMock(return_value=mock_response_post)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        pr_handler = HandlerPlanToPRComment()
        with patch("httpx.AsyncClient", return_value=mock_client):
            pr_result = await pr_handler.post_plan_comment(
                plan=plan,
                trigger_type="pr_opened",
            )

        assert isinstance(pr_result, ModelPRCommentResult)
        assert pr_result.posted is True
        assert pr_result.comment_id == 777
        assert pr_result.skipped is False

        # Verify comment body contains correct markdown table
        post_call_args = mock_client.post.call_args
        comment_body: str = post_call_args.kwargs["json"]["body"]
        assert "| Artifact | Type | Strength | Action | Owner |" in comment_body
        assert "docs/architecture/handler_protocol.md" in comment_body
        assert "docs/architecture/TOPIC_CATALOG.md" in comment_body
        assert f"<!-- onex-artifact-plan:{plan.plan_id} -->" in comment_body

        # Stage 4: YAML emission
        yaml_result = yaml_handler.serialize_plan(plan=plan)
        assert isinstance(yaml_result, ModelYamlEmitResult)
        assert yaml_result.topic == "onex.evt.artifact.update-plan-emitted.v1"
        assert yaml_result.plan_id == plan.plan_id

        parsed_yaml = yaml.safe_load(yaml_result.yaml_payload)
        assert str(parsed_yaml["plan_id"]) == str(plan.plan_id)
        assert parsed_yaml["merge_policy"] == "strict"
        assert len(parsed_yaml["impacted_artifacts"]) == 2
        assert len(parsed_yaml["tasks"]) >= 1
