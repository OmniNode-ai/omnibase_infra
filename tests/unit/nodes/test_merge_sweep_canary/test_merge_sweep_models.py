# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for merge-sweep models — frozen, extra=forbid, serialization."""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.nodes.node_merge_sweep_auto_merge_effect.models import (
    ModelAutoMergeOutcome,
    ModelAutoMergeRequest,
    ModelAutoMergeResult,
)
from omnibase_infra.nodes.node_merge_sweep_classify_compute.models import (
    ModelClassifyInput,
    ModelClassifyResult,
    ModelPRClassification,
)
from omnibase_infra.nodes.node_merge_sweep_pr_list_effect.models import (
    ModelPRInfo,
    ModelPRListRequest,
    ModelPRListResult,
)
from omnibase_infra.nodes.node_merge_sweep_workflow_orchestrator.models import (
    ModelMergeSweepCommand,
    ModelMergeSweepResult,
)


@pytest.mark.unit
class TestPRInfoModel:
    """Tests for ModelPRInfo."""

    def test_create_minimal(self):
        pr = ModelPRInfo(number=1, repo="OmniNode-ai/test")
        assert pr.number == 1
        assert pr.repo == "OmniNode-ai/test"
        assert pr.is_draft is False
        assert pr.ci_status == "UNKNOWN"

    def test_frozen(self):
        pr = ModelPRInfo(number=1, repo="OmniNode-ai/test")
        with pytest.raises(ValidationError):
            pr.number = 2  # type: ignore[misc]

    def test_extra_forbid(self):
        with pytest.raises(ValidationError):
            ModelPRInfo(number=1, repo="test", bogus="field")  # type: ignore[call-arg]

    def test_full_fields(self):
        pr = ModelPRInfo(
            number=42,
            repo="OmniNode-ai/omniclaude",
            title="feat: new feature",
            head_ref="feature-branch",
            base_ref="main",
            author="jonahgabriel",
            is_draft=False,
            mergeable="MERGEABLE",
            review_decision="APPROVED",
            ci_status="SUCCESS",
            has_auto_merge=False,
            labels=("enhancement",),
            updated_at="2026-04-01T12:00:00Z",
        )
        assert pr.author == "jonahgabriel"
        assert pr.labels == ("enhancement",)


@pytest.mark.unit
class TestPRListModels:
    """Tests for PR list request/result models."""

    def test_request_creation(self):
        cid = uuid4()
        req = ModelPRListRequest(
            correlation_id=cid,
            repos=("OmniNode-ai/omniclaude", "OmniNode-ai/omnibase_core"),
        )
        assert len(req.repos) == 2
        assert req.since == ""

    def test_result_with_prs(self):
        cid = uuid4()
        pr = ModelPRInfo(number=1, repo="OmniNode-ai/test")
        result = ModelPRListResult(
            correlation_id=cid,
            prs=(pr,),
            repos_scanned=("OmniNode-ai/test",),
        )
        assert len(result.prs) == 1
        assert result.success is True


@pytest.mark.unit
class TestClassificationModels:
    """Tests for classification models."""

    def test_classification_track_a(self):
        pr = ModelPRInfo(number=1, repo="OmniNode-ai/test")
        c = ModelPRClassification(pr=pr, track="A", reason="CI green")
        assert c.track == "A"

    def test_classification_invalid_track(self):
        pr = ModelPRInfo(number=1, repo="OmniNode-ai/test")
        with pytest.raises(ValidationError):
            ModelPRClassification(pr=pr, track="C")  # type: ignore[arg-type]

    def test_classify_input(self):
        cid = uuid4()
        pr = ModelPRInfo(number=1, repo="OmniNode-ai/test")
        inp = ModelClassifyInput(correlation_id=cid, prs=(pr,))
        assert inp.require_approval is True

    def test_classify_result(self):
        cid = uuid4()
        result = ModelClassifyResult(correlation_id=cid, total_classified=5)
        assert result.total_classified == 5
        assert len(result.track_a) == 0


@pytest.mark.unit
class TestAutoMergeModels:
    """Tests for auto-merge models."""

    def test_request(self):
        cid = uuid4()
        req = ModelAutoMergeRequest(
            correlation_id=cid,
            prs=(("OmniNode-ai/test", 42),),
        )
        assert req.merge_method == "squash"
        assert req.dry_run is False

    def test_outcome_success(self):
        o = ModelAutoMergeOutcome(repo="OmniNode-ai/test", pr_number=42)
        assert o.success is True

    def test_outcome_failure(self):
        o = ModelAutoMergeOutcome(
            repo="OmniNode-ai/test",
            pr_number=42,
            success=False,
            error_message="Branch protection",
        )
        assert not o.success

    def test_result(self):
        cid = uuid4()
        result = ModelAutoMergeResult(
            correlation_id=cid, total_enabled=3, total_failed=1, success=False
        )
        assert result.total_enabled == 3


@pytest.mark.unit
class TestMergeSweepCommandAndResult:
    """Tests for orchestrator command and result models."""

    def test_command_defaults(self):
        cid = uuid4()
        cmd = ModelMergeSweepCommand(
            correlation_id=cid,
            repos=("OmniNode-ai/omniclaude",),
        )
        assert cmd.merge_method == "squash"
        assert cmd.require_approval is True
        assert cmd.dry_run is False

    def test_result_nothing_to_merge(self):
        cid = uuid4()
        result = ModelMergeSweepResult(
            correlation_id=cid,
            status="nothing_to_merge",
        )
        assert result.total_prs_scanned == 0

    def test_result_complete(self):
        cid = uuid4()
        result = ModelMergeSweepResult(
            correlation_id=cid,
            status="complete",
            total_prs_scanned=10,
            track_a_count=6,
            track_b_count=3,
            skipped_count=1,
            auto_merge_enabled=6,
            repos_scanned=("OmniNode-ai/test",),
        )
        assert result.auto_merge_enabled == 6
