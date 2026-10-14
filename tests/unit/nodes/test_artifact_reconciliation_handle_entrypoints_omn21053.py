# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""OMN-21053 -- the seams between the four artifact reconciliation nodes.

Each test pins one hop the dev lane (2026-10-12T12:15:31Z) or the bus-chain test
showed to be unreachable through the runtime's dispatch rule: a handler is invoked
through ``handle()`` / ``handle_async()`` with its wire payload validated into the
single typed request it declares.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from omnibase_infra.enums import EnumUpdateTriggerType
from omnibase_infra.nodes.node_artifact_change_detector_effect.handlers.handler_manual_trigger import (
    HandlerManualTrigger,
)
from omnibase_infra.nodes.node_artifact_change_detector_effect.handlers.handler_pr_webhook_ingestion import (
    HandlerPRWebhookIngestion,
)
from omnibase_infra.nodes.node_artifact_change_detector_effect.models.model_update_trigger import (
    ModelUpdateTrigger,
)
from omnibase_infra.nodes.node_artifact_reconciliation_orchestrator.handlers.handler_plan_to_pr_comment import (
    HandlerPlanToPRComment,
)
from omnibase_infra.nodes.node_artifact_reconciliation_orchestrator.handlers.handler_plan_to_yaml import (
    HandlerPlanToYaml,
)
from omnibase_infra.nodes.node_impact_analyzer_compute.handlers.handler_impact_analysis import (
    HandlerImpactAnalysis,
)
from omnibase_infra.nodes.node_impact_analyzer_compute.models.model_impact_analysis_request import (
    ModelImpactAnalysisRequest,
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
from omnibase_infra.registry.loader import load_packaged_artifact_registry

pytestmark = [pytest.mark.unit]

_TS = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.mark.parametrize(
    "handler_class",
    [
        HandlerManualTrigger,
        HandlerPRWebhookIngestion,
        HandlerImpactAnalysis,
        HandlerCreatePlan,
        HandlerPlanToPRComment,
        HandlerPlanToYaml,
    ],
)
def test_every_chain_handler_exposes_a_dispatch_entrypoint(
    handler_class: type,
) -> None:
    """The runtime binds ``handle`` / ``handle_async`` and nothing else."""
    assert callable(getattr(handler_class, "handle", None)), handler_class.__name__


def test_request_accepts_the_bare_trigger_the_detector_publishes() -> None:
    trigger = ModelUpdateTrigger(
        trigger_id=uuid4(),
        trigger_type="manual_plan_request",
        source_repo="omnibase_infra",
        changed_files=[],
        timestamp=_TS,
    )

    request = ModelImpactAnalysisRequest.model_validate(trigger.model_dump(mode="json"))

    assert request.trigger == trigger
    assert request.registry is load_packaged_artifact_registry()
    # Positive control: the packaged registry is non-empty, so "scores against it" is real.
    assert len(request.registry.artifacts) > 0


def test_request_still_accepts_an_explicit_trigger_and_registry() -> None:
    trigger = ModelUpdateTrigger(
        trigger_id=uuid4(),
        trigger_type="pr_opened",
        source_repo="omnibase_infra",
        changed_files=["a.py"],
        timestamp=_TS,
    )
    registry = load_packaged_artifact_registry().model_copy(update={"artifacts": []})

    request = ModelImpactAnalysisRequest(trigger=trigger, registry=registry)

    assert request.registry.artifacts == []


def test_impact_result_carries_the_trigger_context() -> None:
    trigger = ModelUpdateTrigger(
        trigger_id=uuid4(),
        trigger_type="pr_opened",
        source_repo="OmniNode-ai/omnibase_infra",
        source_ref="refs/pull/7/head",
        changed_files=["x.py"],
        timestamp=_TS,
    )

    result = HandlerImpactAnalysis().handle(
        ModelImpactAnalysisRequest.model_validate(trigger.model_dump(mode="json"))
    )

    assert result.source_trigger_id == trigger.trigger_id
    assert result.trigger_type == "pr_opened"
    assert result.source_repo == "OmniNode-ai/omnibase_infra"
    assert result.source_ref == "refs/pull/7/head"


def _result(
    trigger_type: str, source_repo: str, source_ref: str | None
) -> ModelImpactAnalysisResult:
    return ModelImpactAnalysisResult(
        source_trigger_id=uuid4(),
        trigger_type=EnumUpdateTriggerType(trigger_type),
        source_repo=source_repo,
        source_ref=source_ref,
        impacted_artifacts=[],
        highest_merge_policy="none",
    )


@pytest.mark.parametrize(
    ("trigger_type", "source_repo", "source_ref", "expected_ref"),
    [
        (
            "pr_opened",
            "OmniNode-ai/omnibase_infra",
            "refs/pull/7/head",
            "pr/OmniNode-ai/omnibase_infra/7",
        ),
        (
            "pr_merged",
            "OmniNode-ai/omnibase_infra",
            "refs/pull/9/head",
            "pr/OmniNode-ai/omnibase_infra/9",
        ),
        ("manual_plan_request", "omnibase_infra", None, "manual/omnibase_infra"),
        ("contract_changed", "omnibase_infra", None, "change/omnibase_infra"),
        # A PR trigger whose ref is not a pull ref is NOT named as a PR: no comment target.
        (
            "pr_opened",
            "OmniNode-ai/omnibase_infra",
            "refs/heads/x",
            "change/OmniNode-ai/omnibase_infra",
        ),
        (
            "pr_opened",
            "OmniNode-ai/omnibase_infra",
            None,
            "change/OmniNode-ai/omnibase_infra",
        ),
    ],
)
def test_reducer_handle_names_the_entity_and_keeps_the_trigger_type(
    trigger_type: str, source_repo: str, source_ref: str | None, expected_ref: str
) -> None:
    result = _result(trigger_type, source_repo, source_ref)

    plan = HandlerCreatePlan().handle(result)

    assert plan.source_entity_ref == expected_ref
    assert plan.trigger_type == trigger_type
    assert plan.source_trigger_id == result.source_trigger_id


def _plan(trigger_type: str, source_entity_ref: str) -> ModelUpdatePlan:
    return ModelUpdatePlan(
        plan_id=UUID(int=1),
        source_trigger_id=UUID(int=2),
        trigger_type=EnumUpdateTriggerType(trigger_type),
        source_entity_ref=source_entity_ref,
        summary="s",
        impacted_artifacts=[],
        tasks=[],
        merge_policy="none",
        created_at=_TS,
    )


@pytest.mark.asyncio
async def test_pr_comment_handle_returns_nothing_for_a_non_pr_plan() -> None:
    """A skip is not an outcome: nothing is returned, so nothing is published."""
    handler = HandlerPlanToPRComment(github_token="")

    assert (
        await handler.handle(_plan("manual_plan_request", "manual/omnibase_infra"))
        is None
    )


@pytest.mark.asyncio
async def test_pr_comment_handle_reports_an_unparseable_pr_reference() -> None:
    """A PR plan that cannot be posted is still an outcome, with its error."""
    handler = HandlerPlanToPRComment(github_token="")

    result = await handler.handle(_plan("pr_opened", "not-a-pr-reference"))

    assert result is not None
    assert result.posted is False
    assert result.error is not None


def test_yaml_handle_serializes_the_plan_to_the_emitted_topic() -> None:
    plan = _plan("manual_plan_request", "manual/omnibase_infra")

    emitted = HandlerPlanToYaml().handle(plan)

    assert emitted.plan_id == plan.plan_id
    assert emitted.topic.endswith("update-plan-emitted.v1")
    assert "manual/omnibase_infra" in emitted.yaml_payload
