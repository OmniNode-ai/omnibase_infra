# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""OMN-21053 -- parity proof for the canonical def-B flip of three reconciliation nodes.

``node_artifact_change_detector_effect``, ``node_artifact_reconciliation_orchestrator``
and ``node_update_plan_reducer`` sat in ``canonical_handler_shape_baseline.py`` because
their handlers exposed only op-methods (``ingest_pr_webhook_event``,
``post_plan_comment``, ``create_plan``) and no ``handle``. Each test below asserts the
``handle`` entrypoint FIRST, so it fails on its own assertion against the pre-flip
handler (never on an import error), and then pins the behavior the flip must keep.

``CORPUS_*`` is the deterministic input corpus bound (by ``input_hash``) into the
adequacy receipts and hand-flip proofs under ``scripts/ci/adequacy_receipts/``.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import UUID

import pytest

from omnibase_infra.nodes.node_artifact_change_detector_effect.handlers.handler_pr_webhook_ingestion import (
    HandlerPRWebhookIngestion,
)
from omnibase_infra.nodes.node_artifact_change_detector_effect.models.model_pr_webhook_event import (
    ModelPRWebhookEvent,
)
from omnibase_infra.nodes.node_artifact_reconciliation_orchestrator.handlers.handler_plan_to_pr_comment import (
    HandlerPlanToPRComment,
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

pytestmark = [pytest.mark.unit]

_TS = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)


def _event(action: str, *, merged: bool = False) -> ModelPRWebhookEvent:
    return ModelPRWebhookEvent.model_validate(
        {
            "action": action,
            "repo": "OmniNode-ai/omnibase_infra",
            "pr_number": 7,
            "head_ref": "feature",
            "head_sha": "abc123",
            "changed_files": ["src/a.py"],
            "merged": merged,
        }
    )


CORPUS_WEBHOOK: tuple[ModelPRWebhookEvent, ...] = (
    _event("opened"),
    _event("synchronize"),
    _event("closed", merged=True),
    _event("closed", merged=False),
)


def _result(trigger_type: str, source_ref: str | None) -> ModelImpactAnalysisResult:
    return ModelImpactAnalysisResult.model_validate(
        {
            "source_trigger_id": UUID(int=1),
            "trigger_type": trigger_type,
            "source_repo": "OmniNode-ai/omnibase_infra",
            "source_ref": source_ref,
            "impacted_artifacts": [],
            "highest_merge_policy": "none",
        }
    )


def corpus_impact() -> tuple[ModelImpactAnalysisResult, ...]:
    """Built on call: the trigger context fields only exist after the flip."""
    return (
        _result("pr_opened", "refs/pull/7/head"),
        _result("manual_plan_request", None),
        _result("pr_opened", "refs/heads/x"),
    )


def _plan(trigger_type: str, source_entity_ref: str) -> ModelUpdatePlan:
    return ModelUpdatePlan.model_validate(
        {
            "plan_id": UUID(int=1),
            "source_trigger_id": UUID(int=2),
            "trigger_type": trigger_type,
            "source_entity_ref": source_entity_ref,
            "summary": "s",
            "impacted_artifacts": [],
            "tasks": [],
            "merge_policy": "none",
            "created_at": _TS,
        }
    )


def corpus_plan() -> tuple[ModelUpdatePlan, ...]:
    """Built on call: the trigger type field only exists after the flip."""
    return (
        _plan("manual_plan_request", "manual/omnibase_infra"),
        _plan("pr_opened", "not-a-pr-reference"),
    )


def test_webhook_handler_exposes_handle_entrypoint() -> None:
    assert callable(getattr(HandlerPRWebhookIngestion, "handle", None))
    assert not hasattr(HandlerPRWebhookIngestion, "ingest_pr_webhook_event")


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        (CORPUS_WEBHOOK[0], "pr_opened"),
        (CORPUS_WEBHOOK[1], "pr_updated"),
        (CORPUS_WEBHOOK[2], "pr_merged"),
        (CORPUS_WEBHOOK[3], "pr_updated"),
    ],
)
def test_webhook_handle_keeps_the_action_mapping(
    event: ModelPRWebhookEvent, expected: str
) -> None:
    assert callable(getattr(HandlerPRWebhookIngestion, "handle", None))

    trigger = HandlerPRWebhookIngestion().handle(event)

    assert trigger.trigger_type == expected
    assert trigger.changed_files == ["src/a.py"]


def test_create_plan_handler_exposes_handle_entrypoint() -> None:
    assert callable(getattr(HandlerCreatePlan, "handle", None))


def test_create_plan_handle_returns_a_plan_for_the_result() -> None:
    assert callable(getattr(HandlerCreatePlan, "handle", None))

    result = corpus_impact()[0]

    plan = HandlerCreatePlan().handle(result)

    assert plan.source_trigger_id == result.source_trigger_id
    assert plan.tasks == []


def test_pr_comment_handler_exposes_handle_entrypoint() -> None:
    assert callable(getattr(HandlerPlanToPRComment, "handle", None))


def test_pr_comment_handle_returns_nothing_for_a_non_pr_plan() -> None:
    assert callable(getattr(HandlerPlanToPRComment, "handle", None))

    outcome = asyncio.run(
        HandlerPlanToPRComment(github_token="").handle(corpus_plan()[0])
    )

    assert outcome is None
