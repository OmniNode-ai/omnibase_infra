# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for HandlerClassifyPRs — pure compute, no I/O."""

from __future__ import annotations

from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_merge_sweep_classify_compute.handlers.handler_classify_prs import (
    HandlerClassifyPRs,
)
from omnibase_infra.nodes.node_merge_sweep_pr_list_effect.models.model_pr_info import (
    ModelPRInfo,
)


def _make_pr(**overrides) -> ModelPRInfo:
    """Helper to create a PR with sensible defaults."""
    defaults = {
        "number": 1,
        "repo": "OmniNode-ai/test",
        "title": "test PR",
        "mergeable": "MERGEABLE",
        "review_decision": "APPROVED",
        "ci_status": "SUCCESS",
        "is_draft": False,
        "has_auto_merge": False,
    }
    defaults.update(overrides)
    return ModelPRInfo(**defaults)


@pytest.mark.unit
class TestHandlerClassifyPRs:
    """Tests for the PR classification compute handler."""

    @pytest.fixture
    def handler(self) -> HandlerClassifyPRs:
        return HandlerClassifyPRs()

    @pytest.mark.asyncio
    async def test_track_a_merge_ready(self, handler: HandlerClassifyPRs):
        """PR with green CI, approved, mergeable goes to Track A."""
        pr = _make_pr()
        result = await handler.handle(prs=(pr,), correlation_id=uuid4())

        assert len(result.track_a) == 1
        assert result.track_a[0].track == "A"
        assert len(result.track_b) == 0

    @pytest.mark.asyncio
    async def test_track_b_ci_failure(self, handler: HandlerClassifyPRs):
        """PR with CI failure goes to Track B."""
        pr = _make_pr(ci_status="FAILURE")
        result = await handler.handle(prs=(pr,), correlation_id=uuid4())

        assert len(result.track_a) == 0
        assert len(result.track_b) == 1
        assert "CI FAILURE" in result.track_b[0].reason

    @pytest.mark.asyncio
    async def test_track_b_changes_requested(self, handler: HandlerClassifyPRs):
        """PR with changes requested goes to Track B."""
        pr = _make_pr(review_decision="CHANGES_REQUESTED")
        result = await handler.handle(prs=(pr,), correlation_id=uuid4())

        assert len(result.track_b) == 1
        assert "CHANGES_REQUESTED" in result.track_b[0].reason

    @pytest.mark.asyncio
    async def test_track_b_conflicting(self, handler: HandlerClassifyPRs):
        """PR with merge conflicts goes to Track B."""
        pr = _make_pr(mergeable="CONFLICTING")
        result = await handler.handle(prs=(pr,), correlation_id=uuid4())

        assert len(result.track_b) == 1

    @pytest.mark.asyncio
    async def test_skip_draft(self, handler: HandlerClassifyPRs):
        """Draft PR is skipped."""
        pr = _make_pr(is_draft=True)
        result = await handler.handle(prs=(pr,), correlation_id=uuid4())

        assert len(result.skipped) == 1
        assert result.skipped[0].track == "SKIP"
        assert "Draft" in result.skipped[0].reason

    @pytest.mark.asyncio
    async def test_skip_auto_merge_enabled(self, handler: HandlerClassifyPRs):
        """PR with auto-merge already enabled is skipped."""
        pr = _make_pr(has_auto_merge=True)
        result = await handler.handle(prs=(pr,), correlation_id=uuid4())

        assert len(result.skipped) == 1
        assert "Auto-merge already" in result.skipped[0].reason

    @pytest.mark.asyncio
    async def test_no_approval_required(self, handler: HandlerClassifyPRs):
        """Track A with require_approval=False ignores review status."""
        pr = _make_pr(review_decision="")
        result = await handler.handle(
            prs=(pr,), correlation_id=uuid4(), require_approval=False
        )

        assert len(result.track_a) == 1

    @pytest.mark.asyncio
    async def test_mixed_batch(self, handler: HandlerClassifyPRs):
        """Mixed batch of PRs classifies correctly."""
        prs = (
            _make_pr(number=1),  # Track A
            _make_pr(number=2, ci_status="FAILURE"),  # Track B
            _make_pr(number=3, is_draft=True),  # SKIP
            _make_pr(number=4, mergeable="CONFLICTING"),  # Track B
            _make_pr(number=5, has_auto_merge=True),  # SKIP
        )
        result = await handler.handle(prs=prs, correlation_id=uuid4())

        assert result.total_classified == 5
        assert len(result.track_a) == 1
        assert len(result.track_b) == 2
        assert len(result.skipped) == 2

    @pytest.mark.asyncio
    async def test_empty_input(self, handler: HandlerClassifyPRs):
        """Empty PR list produces empty result."""
        result = await handler.handle(prs=(), correlation_id=uuid4())

        assert result.total_classified == 0
        assert len(result.track_a) == 0

    @pytest.mark.asyncio
    async def test_correlation_id_preserved(self, handler: HandlerClassifyPRs):
        """Correlation ID is preserved in result."""
        cid = uuid4()
        result = await handler.handle(prs=(), correlation_id=cid)
        assert result.correlation_id == cid
