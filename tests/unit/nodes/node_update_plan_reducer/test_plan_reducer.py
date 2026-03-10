# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Tests for update plan reducer FSM and handler logic.

Covers:
- FSM state transitions (valid and invalid)
- Guard predicates (can_create_plan, can_post_comment, etc.)
- HandlerCreatePlan.create_plan() task_type mapping
- Idempotency (duplicate event detection)
- Immutability enforcement
- EnumUpdatePlanState enum values

Tracking:
    - OMN-3943: Task 6 — Update Plan REDUCER Node
    - OMN-3925: Artifact Reconciliation + Update Planning MVP
"""

from __future__ import annotations

from datetime import UTC, datetime, timezone
from uuid import UUID, uuid4

import pytest

pytestmark = pytest.mark.unit

from omnibase_infra.enums import EnumUpdatePlanState
from omnibase_infra.errors import RuntimeHostError
from omnibase_infra.nodes.node_impact_analyzer_compute.models.model_impact_analysis_result import (
    ModelImpactAnalysisResult,
)
from omnibase_infra.nodes.node_impact_analyzer_compute.models.model_impacted_artifact import (
    ModelImpactedArtifact,
)
from omnibase_infra.nodes.node_update_plan_reducer.handlers.handler_create_plan import (
    HandlerCreatePlan,
)
from omnibase_infra.nodes.node_update_plan_reducer.models.model_update_plan_state import (
    ModelUpdatePlanState,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_artifact(
    required_action: str = "review",
    artifact_type: str = "doc",
    impact_strength: float = 0.7,
) -> ModelImpactedArtifact:
    """Build a minimal ModelImpactedArtifact for testing."""
    return ModelImpactedArtifact(
        artifact_id=uuid4(),
        artifact_type=artifact_type,  # type: ignore[arg-type]
        path=f"docs/{artifact_type}/test.md",
        impact_strength=impact_strength,
        reason_codes=["contract_yaml_changed"],
        required_action=required_action,  # type: ignore[arg-type]
    )


def _make_result(
    artifacts: list[ModelImpactedArtifact] | None = None,
    highest_merge_policy: str = "require",
) -> ModelImpactAnalysisResult:
    """Build a minimal ModelImpactAnalysisResult for testing."""
    return ModelImpactAnalysisResult(
        source_trigger_id=uuid4(),
        impacted_artifacts=artifacts or [_make_artifact()],
        highest_merge_policy=highest_merge_policy,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# EnumUpdatePlanState
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEnumUpdatePlanState:
    """Tests for EnumUpdatePlanState enum values."""

    def test_all_states_present(self) -> None:
        """All MVP FSM states are present."""
        values = {e.value for e in EnumUpdatePlanState}
        assert values == {
            "idle",
            "created",
            "comment_posted",
            "yaml_emitted",
            "closed",
            "waived",
        }

    def test_str_enum(self) -> None:
        """EnumUpdatePlanState is a str enum for JSON serialization."""
        assert EnumUpdatePlanState.IDLE == "idle"
        assert EnumUpdatePlanState.CREATED == "created"
        assert EnumUpdatePlanState.COMMENT_POSTED == "comment_posted"
        assert EnumUpdatePlanState.YAML_EMITTED == "yaml_emitted"
        assert EnumUpdatePlanState.CLOSED == "closed"
        assert EnumUpdatePlanState.WAIVED == "waived"


# ---------------------------------------------------------------------------
# ModelUpdatePlanState — default state
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestModelUpdatePlanStateDefaults:
    """Tests for default ModelUpdatePlanState values."""

    def test_default_state_is_idle(self) -> None:
        """Default status is IDLE with no plan_id."""
        state = ModelUpdatePlanState()
        assert state.status == EnumUpdatePlanState.IDLE
        assert state.plan_id is None
        assert state.last_processed_event_id is None

    def test_immutability(self) -> None:
        """State model is frozen — assignment raises."""
        state = ModelUpdatePlanState()
        with pytest.raises(Exception):
            state.status = EnumUpdatePlanState.CREATED  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        """Extra fields are rejected."""
        with pytest.raises(ValueError):
            ModelUpdatePlanState(unknown="oops")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# FSM: valid transitions (happy path)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestModelUpdatePlanStateValidTransitions:
    """Tests for valid FSM transitions."""

    def test_idle_to_created(self) -> None:
        """idle -> created via with_plan_created."""
        state = ModelUpdatePlanState()
        plan_id = uuid4()
        event_id = uuid4()
        new_state = state.with_plan_created(plan_id, event_id)

        assert new_state.status == EnumUpdatePlanState.CREATED
        assert new_state.plan_id == plan_id
        assert new_state.last_processed_event_id == event_id

        # Original is unchanged (immutable)
        assert state.status == EnumUpdatePlanState.IDLE

    def test_created_to_comment_posted(self) -> None:
        """created -> comment_posted via with_comment_posted."""
        plan_id = uuid4()
        state = ModelUpdatePlanState(
            status=EnumUpdatePlanState.CREATED,
            plan_id=plan_id,
        )
        event_id = uuid4()
        new_state = state.with_comment_posted(event_id)

        assert new_state.status == EnumUpdatePlanState.COMMENT_POSTED
        assert new_state.plan_id == plan_id
        assert new_state.last_processed_event_id == event_id

    def test_comment_posted_to_yaml_emitted(self) -> None:
        """comment_posted -> yaml_emitted via with_yaml_emitted."""
        plan_id = uuid4()
        state = ModelUpdatePlanState(
            status=EnumUpdatePlanState.COMMENT_POSTED,
            plan_id=plan_id,
        )
        event_id = uuid4()
        new_state = state.with_yaml_emitted(event_id)

        assert new_state.status == EnumUpdatePlanState.YAML_EMITTED
        assert new_state.plan_id == plan_id

    def test_yaml_emitted_to_closed(self) -> None:
        """yaml_emitted -> closed via with_closed."""
        plan_id = uuid4()
        state = ModelUpdatePlanState(
            status=EnumUpdatePlanState.YAML_EMITTED,
            plan_id=plan_id,
        )
        event_id = uuid4()
        new_state = state.with_closed(event_id)

        assert new_state.status == EnumUpdatePlanState.CLOSED
        assert new_state.plan_id == plan_id

    def test_created_to_waived(self) -> None:
        """created -> waived via with_waived (immediate waiver)."""
        plan_id = uuid4()
        state = ModelUpdatePlanState(
            status=EnumUpdatePlanState.CREATED,
            plan_id=plan_id,
        )
        event_id = uuid4()
        new_state = state.with_waived(event_id)

        assert new_state.status == EnumUpdatePlanState.WAIVED

    def test_comment_posted_to_waived(self) -> None:
        """comment_posted -> waived via with_waived."""
        plan_id = uuid4()
        state = ModelUpdatePlanState(
            status=EnumUpdatePlanState.COMMENT_POSTED,
            plan_id=plan_id,
        )
        event_id = uuid4()
        new_state = state.with_waived(event_id)

        assert new_state.status == EnumUpdatePlanState.WAIVED

    def test_full_lifecycle_happy_path(self) -> None:
        """Complete lifecycle: idle -> created -> comment_posted -> yaml_emitted -> closed."""
        state = ModelUpdatePlanState()
        plan_id = uuid4()

        state = state.with_plan_created(plan_id, uuid4())
        assert state.status == EnumUpdatePlanState.CREATED

        state = state.with_comment_posted(uuid4())
        assert state.status == EnumUpdatePlanState.COMMENT_POSTED

        state = state.with_yaml_emitted(uuid4())
        assert state.status == EnumUpdatePlanState.YAML_EMITTED

        state = state.with_closed(uuid4())
        assert state.status == EnumUpdatePlanState.CLOSED
        assert state.plan_id == plan_id

    def test_full_lifecycle_waived_path(self) -> None:
        """Waiver path: idle -> created -> waived."""
        state = ModelUpdatePlanState()

        state = state.with_plan_created(uuid4(), uuid4())
        assert state.status == EnumUpdatePlanState.CREATED

        state = state.with_waived(uuid4())
        assert state.status == EnumUpdatePlanState.WAIVED


# ---------------------------------------------------------------------------
# FSM: invalid transitions
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestModelUpdatePlanStateInvalidTransitions:
    """Tests for invalid FSM transitions — all should raise RuntimeHostError."""

    @pytest.mark.parametrize(
        ("source_status", "transition_method", "expected_match"),
        [
            # with_plan_created: only valid from IDLE
            (
                EnumUpdatePlanState.CREATED,
                "with_plan_created",
                "requires IDLE",
            ),
            (
                EnumUpdatePlanState.COMMENT_POSTED,
                "with_plan_created",
                "requires IDLE",
            ),
            (
                EnumUpdatePlanState.YAML_EMITTED,
                "with_plan_created",
                "requires IDLE",
            ),
            (
                EnumUpdatePlanState.CLOSED,
                "with_plan_created",
                "requires IDLE",
            ),
            (
                EnumUpdatePlanState.WAIVED,
                "with_plan_created",
                "requires IDLE",
            ),
            # with_comment_posted: only valid from CREATED
            (
                EnumUpdatePlanState.IDLE,
                "with_comment_posted",
                "requires CREATED",
            ),
            (
                EnumUpdatePlanState.COMMENT_POSTED,
                "with_comment_posted",
                "requires CREATED",
            ),
            (
                EnumUpdatePlanState.YAML_EMITTED,
                "with_comment_posted",
                "requires CREATED",
            ),
            (
                EnumUpdatePlanState.CLOSED,
                "with_comment_posted",
                "requires CREATED",
            ),
            (
                EnumUpdatePlanState.WAIVED,
                "with_comment_posted",
                "requires CREATED",
            ),
            # with_yaml_emitted: only valid from COMMENT_POSTED
            (
                EnumUpdatePlanState.IDLE,
                "with_yaml_emitted",
                "requires COMMENT_POSTED",
            ),
            (
                EnumUpdatePlanState.CREATED,
                "with_yaml_emitted",
                "requires COMMENT_POSTED",
            ),
            (
                EnumUpdatePlanState.YAML_EMITTED,
                "with_yaml_emitted",
                "requires COMMENT_POSTED",
            ),
            (
                EnumUpdatePlanState.CLOSED,
                "with_yaml_emitted",
                "requires COMMENT_POSTED",
            ),
            (
                EnumUpdatePlanState.WAIVED,
                "with_yaml_emitted",
                "requires COMMENT_POSTED",
            ),
            # with_closed: only valid from YAML_EMITTED
            (
                EnumUpdatePlanState.IDLE,
                "with_closed",
                "requires YAML_EMITTED",
            ),
            (
                EnumUpdatePlanState.CREATED,
                "with_closed",
                "requires YAML_EMITTED",
            ),
            (
                EnumUpdatePlanState.COMMENT_POSTED,
                "with_closed",
                "requires YAML_EMITTED",
            ),
            (
                EnumUpdatePlanState.CLOSED,
                "with_closed",
                "requires YAML_EMITTED",
            ),
            (
                EnumUpdatePlanState.WAIVED,
                "with_closed",
                "requires YAML_EMITTED",
            ),
            # with_waived: only valid from CREATED or COMMENT_POSTED
            (
                EnumUpdatePlanState.IDLE,
                "with_waived",
                "requires CREATED or COMMENT_POSTED",
            ),
            (
                EnumUpdatePlanState.YAML_EMITTED,
                "with_waived",
                "requires CREATED or COMMENT_POSTED",
            ),
            (
                EnumUpdatePlanState.CLOSED,
                "with_waived",
                "requires CREATED or COMMENT_POSTED",
            ),
            (
                EnumUpdatePlanState.WAIVED,
                "with_waived",
                "requires CREATED or COMMENT_POSTED",
            ),
        ],
        ids=[
            "create-from-CREATED",
            "create-from-COMMENT_POSTED",
            "create-from-YAML_EMITTED",
            "create-from-CLOSED",
            "create-from-WAIVED",
            "comment-from-IDLE",
            "comment-from-COMMENT_POSTED",
            "comment-from-YAML_EMITTED",
            "comment-from-CLOSED",
            "comment-from-WAIVED",
            "yaml-from-IDLE",
            "yaml-from-CREATED",
            "yaml-from-YAML_EMITTED",
            "yaml-from-CLOSED",
            "yaml-from-WAIVED",
            "close-from-IDLE",
            "close-from-CREATED",
            "close-from-COMMENT_POSTED",
            "close-from-CLOSED",
            "close-from-WAIVED",
            "waive-from-IDLE",
            "waive-from-YAML_EMITTED",
            "waive-from-CLOSED",
            "waive-from-WAIVED",
        ],
    )
    def test_invalid_transition_raises(
        self,
        source_status: EnumUpdatePlanState,
        transition_method: str,
        expected_match: str,
    ) -> None:
        """Every invalid (source_state, transition) pair raises RuntimeHostError."""
        # States that need a plan_id to construct validly
        needs_plan_id = {
            EnumUpdatePlanState.CREATED,
            EnumUpdatePlanState.COMMENT_POSTED,
            EnumUpdatePlanState.YAML_EMITTED,
            EnumUpdatePlanState.CLOSED,
            EnumUpdatePlanState.WAIVED,
        }
        kwargs: dict[str, EnumUpdatePlanState | UUID | None] = {"status": source_status}
        if source_status in needs_plan_id:
            kwargs["plan_id"] = uuid4()
        state = ModelUpdatePlanState(**kwargs)  # type: ignore[arg-type]

        method = getattr(state, transition_method)
        with pytest.raises(RuntimeHostError, match=expected_match):
            if transition_method == "with_plan_created":
                method(uuid4(), uuid4())
            else:
                method(uuid4())


# ---------------------------------------------------------------------------
# Guard predicates
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestModelUpdatePlanStateGuards:
    """Tests for can_* guard predicates."""

    def test_can_create_plan_only_in_idle(self) -> None:
        """can_create_plan is True only in IDLE state."""
        assert ModelUpdatePlanState().can_create_plan()

        for status in [
            EnumUpdatePlanState.CREATED,
            EnumUpdatePlanState.COMMENT_POSTED,
            EnumUpdatePlanState.YAML_EMITTED,
            EnumUpdatePlanState.CLOSED,
            EnumUpdatePlanState.WAIVED,
        ]:
            state = ModelUpdatePlanState(status=status, plan_id=uuid4())
            assert not state.can_create_plan(), f"Expected False for status={status}"

    def test_can_waive_from_created_or_comment_posted(self) -> None:
        """can_waive is True from CREATED or COMMENT_POSTED."""
        plan_id = uuid4()

        assert ModelUpdatePlanState(
            status=EnumUpdatePlanState.CREATED,
            plan_id=plan_id,
        ).can_waive()

        assert ModelUpdatePlanState(
            status=EnumUpdatePlanState.COMMENT_POSTED,
            plan_id=plan_id,
        ).can_waive()

        # Not from other states
        assert not ModelUpdatePlanState().can_waive()
        assert not ModelUpdatePlanState(
            status=EnumUpdatePlanState.YAML_EMITTED,
            plan_id=plan_id,
        ).can_waive()
        assert not ModelUpdatePlanState(
            status=EnumUpdatePlanState.CLOSED,
            plan_id=plan_id,
        ).can_waive()

    def test_is_terminal_for_closed_and_waived(self) -> None:
        """is_terminal is True for CLOSED and WAIVED only."""
        assert ModelUpdatePlanState(
            status=EnumUpdatePlanState.CLOSED,
            plan_id=uuid4(),
        ).is_terminal()

        assert ModelUpdatePlanState(
            status=EnumUpdatePlanState.WAIVED,
            plan_id=uuid4(),
        ).is_terminal()

        for status in [
            EnumUpdatePlanState.IDLE,
            EnumUpdatePlanState.CREATED,
            EnumUpdatePlanState.COMMENT_POSTED,
            EnumUpdatePlanState.YAML_EMITTED,
        ]:
            state = ModelUpdatePlanState(status=status, plan_id=uuid4())
            assert not state.is_terminal(), f"Expected non-terminal for status={status}"

    def test_is_duplicate_event_detects_replay(self) -> None:
        """is_duplicate_event detects consecutive event replay."""
        state = ModelUpdatePlanState()
        event_id = uuid4()
        new_state = state.with_plan_created(uuid4(), event_id)

        assert new_state.is_duplicate_event(event_id)
        assert not new_state.is_duplicate_event(uuid4())


# ---------------------------------------------------------------------------
# HandlerCreatePlan — task_type mapping
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHandlerCreatePlan:
    """Tests for HandlerCreatePlan.create_plan() task_type mapping."""

    def test_review_action_maps_to_human_author(self) -> None:
        """required_action=review → task_type=human_author."""
        handler = HandlerCreatePlan()
        artifact = _make_artifact(required_action="review")
        result = _make_result(artifacts=[artifact], highest_merge_policy="require")

        plan = handler.create_plan(
            result=result,
            source_entity_ref="pr/omnibase_infra/123",
            summary="Test plan",
        )

        assert len(plan.tasks) == 1
        assert plan.tasks[0].task_type == "human_author"
        assert plan.tasks[0].blocking is True
        assert plan.tasks[0].target_artifact_id == artifact.artifact_id

    def test_regenerate_action_maps_to_regenerate(self) -> None:
        """required_action=regenerate → task_type=regenerate, blocking=True."""
        handler = HandlerCreatePlan()
        artifact = _make_artifact(required_action="regenerate", impact_strength=0.9)
        result = _make_result(artifacts=[artifact], highest_merge_policy="strict")

        plan = handler.create_plan(
            result=result,
            source_entity_ref="pr/omnibase_infra/124",
            summary="Regenerate plan",
        )

        assert len(plan.tasks) == 1
        assert plan.tasks[0].task_type == "regenerate"
        assert plan.tasks[0].blocking is True

    def test_create_action_maps_to_create_stub(self) -> None:
        """required_action=create → task_type=create_stub, blocking=True."""
        handler = HandlerCreatePlan()
        artifact = _make_artifact(required_action="create")
        result = _make_result(artifacts=[artifact], highest_merge_policy="require")

        plan = handler.create_plan(
            result=result,
            source_entity_ref="pr/omnibase_infra/125",
            summary="Create stub plan",
        )

        assert len(plan.tasks) == 1
        assert plan.tasks[0].task_type == "create_stub"
        assert plan.tasks[0].blocking is True

    def test_none_action_produces_no_task(self) -> None:
        """required_action=none → no task created."""
        handler = HandlerCreatePlan()
        artifact = _make_artifact(required_action="none")
        result = ModelImpactAnalysisResult(
            source_trigger_id=uuid4(),
            impacted_artifacts=[artifact],
            highest_merge_policy="none",
        )

        plan = handler.create_plan(
            result=result,
            source_entity_ref="pr/omnibase_infra/126",
            summary="No-op plan",
        )

        assert len(plan.tasks) == 0

    def test_patch_action_maps_to_patch_existing(self) -> None:
        """required_action=patch → task_type=patch_existing, blocking=False."""
        handler = HandlerCreatePlan()
        artifact = _make_artifact(required_action="patch")
        result = _make_result(artifacts=[artifact], highest_merge_policy="warn")

        plan = handler.create_plan(
            result=result,
            source_entity_ref="pr/omnibase_infra/127",
            summary="Patch plan",
        )

        assert len(plan.tasks) == 1
        assert plan.tasks[0].task_type == "patch_existing"
        assert plan.tasks[0].blocking is False

    def test_merge_policy_set_from_result(self) -> None:
        """Plan merge_policy is taken from result.highest_merge_policy."""
        handler = HandlerCreatePlan()
        result = _make_result(highest_merge_policy="strict")

        plan = handler.create_plan(
            result=result,
            source_entity_ref="pr/omnibase_infra/128",
            summary="Strict policy plan",
        )

        assert plan.merge_policy == "strict"

    def test_plan_id_generated_if_not_provided(self) -> None:
        """plan_id is generated automatically if not passed."""
        handler = HandlerCreatePlan()
        result = _make_result()

        plan = handler.create_plan(
            result=result,
            source_entity_ref="pr/omnibase_infra/129",
            summary="Auto id plan",
        )

        assert isinstance(plan.plan_id, UUID)

    def test_plan_id_preserved_if_provided(self) -> None:
        """plan_id is preserved if explicitly provided."""
        handler = HandlerCreatePlan()
        result = _make_result()
        explicit_id = uuid4()

        plan = handler.create_plan(
            result=result,
            source_entity_ref="pr/omnibase_infra/130",
            summary="Explicit id plan",
            plan_id=explicit_id,
        )

        assert plan.plan_id == explicit_id

    def test_multiple_artifacts_produce_multiple_tasks(self) -> None:
        """Multiple impacted artifacts produce one task each (if action != none)."""
        handler = HandlerCreatePlan()
        artifacts = [
            _make_artifact(required_action="review"),
            _make_artifact(required_action="regenerate", impact_strength=0.9),
            _make_artifact(required_action="none"),  # Should be skipped
        ]
        result = _make_result(artifacts=artifacts, highest_merge_policy="strict")

        plan = handler.create_plan(
            result=result,
            source_entity_ref="pr/omnibase_infra/131",
            summary="Multi-artifact plan",
        )

        assert len(plan.tasks) == 2
        task_types = {t.task_type for t in plan.tasks}
        assert task_types == {"human_author", "regenerate"}

    def test_plan_source_trigger_id_matches_result(self) -> None:
        """Plan source_trigger_id matches the result's source_trigger_id."""
        handler = HandlerCreatePlan()
        result = _make_result()

        plan = handler.create_plan(
            result=result,
            source_entity_ref="pr/omnibase_infra/132",
            summary="Trigger id test",
        )

        assert plan.source_trigger_id == result.source_trigger_id

    def test_plan_created_at_is_recent(self) -> None:
        """Plan created_at is a timezone-aware datetime."""
        handler = HandlerCreatePlan()
        result = _make_result()

        before = datetime.now(UTC)
        plan = handler.create_plan(
            result=result,
            source_entity_ref="pr/omnibase_infra/133",
            summary="Timestamp test",
        )
        after = datetime.now(UTC)

        assert plan.created_at.tzinfo is not None
        assert before <= plan.created_at <= after

    def test_empty_artifacts_produces_empty_plan(self) -> None:
        """Empty impacted_artifacts list produces a plan with no tasks."""
        handler = HandlerCreatePlan()
        result = ModelImpactAnalysisResult(
            source_trigger_id=uuid4(),
            impacted_artifacts=[],
            highest_merge_policy="none",
        )

        plan = handler.create_plan(
            result=result,
            source_entity_ref="pr/omnibase_infra/134",
            summary="Empty plan",
        )

        assert len(plan.tasks) == 0
        assert len(plan.impacted_artifacts) == 0
        assert plan.merge_policy == "none"
