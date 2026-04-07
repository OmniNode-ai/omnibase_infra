# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for merge_sweep_runner — classification logic and handler wiring."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_closeout_effect.handlers.handler_closeout import (
    HandlerCloseout,
)
from omnibase_infra.nodes.node_closeout_effect.handlers.merge_sweep_runner import (
    PRInfo,
    _classify_pr,
    run_merge_sweep,
)


# ---------------------------------------------------------------------------
# Classification unit tests (pure logic, no I/O)
# ---------------------------------------------------------------------------
class TestClassifyPR:
    def test_draft_pr_skipped(self) -> None:
        pr = PRInfo(
            number=1,
            title="WIP",
            repo="OmniNode-ai/test",
            mergeable="MERGEABLE",
            merge_state_status="CLEAN",
            is_draft=True,
        )
        track, reason = _classify_pr(pr)
        assert track == "skip"
        assert "Draft" in reason

    def test_merge_ready_track_a(self) -> None:
        pr = PRInfo(
            number=2,
            title="Ready PR",
            repo="OmniNode-ai/test",
            mergeable="MERGEABLE",
            merge_state_status="CLEAN",
            review_decision="APPROVED",
            required_checks_pass=True,
        )
        track, _ = _classify_pr(pr)
        assert track == "A"

    def test_stale_branch_track_a_update(self) -> None:
        pr = PRInfo(
            number=3,
            title="Stale",
            repo="OmniNode-ai/test",
            mergeable="MERGEABLE",
            merge_state_status="BEHIND",
        )
        track, _ = _classify_pr(pr)
        assert track == "A-update"

    def test_blocked_threads_track_a_resolve(self) -> None:
        pr = PRInfo(
            number=4,
            title="Blocked",
            repo="OmniNode-ai/test",
            mergeable="MERGEABLE",
            merge_state_status="BLOCKED",
            required_checks_pass=True,
        )
        track, _ = _classify_pr(pr)
        assert track == "A-resolve"

    def test_conflicting_track_b(self) -> None:
        pr = PRInfo(
            number=5,
            title="Conflicts",
            repo="OmniNode-ai/test",
            mergeable="CONFLICTING",
            merge_state_status="DIRTY",
        )
        track, _ = _classify_pr(pr)
        assert track == "B"

    def test_ci_failing_track_b(self) -> None:
        pr = PRInfo(
            number=6,
            title="CI Red",
            repo="OmniNode-ai/test",
            mergeable="MERGEABLE",
            merge_state_status="CLEAN",
            required_checks_pass=False,
        )
        track, _ = _classify_pr(pr)
        assert track == "B"

    def test_unknown_mergeable_track_a_update(self) -> None:
        pr = PRInfo(
            number=7,
            title="Unknown",
            repo="OmniNode-ai/test",
            mergeable="UNKNOWN",
            merge_state_status="UNKNOWN",
        )
        track, _ = _classify_pr(pr)
        assert track == "A-update"

    def test_no_review_still_merge_ready(self) -> None:
        pr = PRInfo(
            number=8,
            title="No review",
            repo="OmniNode-ai/test",
            mergeable="MERGEABLE",
            merge_state_status="CLEAN",
            review_decision=None,
            required_checks_pass=True,
        )
        track, _ = _classify_pr(pr)
        assert track == "A"


# ---------------------------------------------------------------------------
# Integration: run_merge_sweep with mocked gh CLI
# ---------------------------------------------------------------------------
def _make_gh_search_output(prs: list[dict]) -> str:
    """Build fake gh search output JSON."""
    return json.dumps(prs)


@pytest.mark.asyncio
async def test_run_merge_sweep_dry_run_no_actions() -> None:
    """Dry run should classify but not enable auto-merge."""
    fake_prs = [
        {
            "number": 10,
            "title": "Ready PR",
            "isDraft": False,
            "mergeable": "MERGEABLE",
            "mergeStateStatus": "CLEAN",
            "reviewDecision": "APPROVED",
            "labels": [],
            "statusCheckRollup": [],
            "headRefName": "feature-x",
            "repository": {"nameWithOwner": "OmniNode-ai/repo"},
        },
    ]

    with patch(
        "omnibase_infra.nodes.node_closeout_effect.handlers.merge_sweep_runner._run_gh",
        new_callable=AsyncMock,
        return_value=json.dumps(fake_prs),
    ) as mock_gh:
        result = await run_merge_sweep(dry_run=True)

    assert len(result.classified) == 1
    assert result.classified[0].track == "A"
    assert result.auto_merge_enabled == 0
    # Should only have called gh search, not gh pr merge
    assert mock_gh.call_count == 1


@pytest.mark.asyncio
async def test_run_merge_sweep_enables_auto_merge() -> None:
    """Non-dry-run should enable auto-merge on Track A PRs."""
    fake_prs = [
        {
            "number": 20,
            "title": "Merge me",
            "isDraft": False,
            "mergeable": "MERGEABLE",
            "mergeStateStatus": "CLEAN",
            "reviewDecision": "APPROVED",
            "labels": [],
            "statusCheckRollup": [],
            "headRefName": "feature-y",
            "repository": {"nameWithOwner": "OmniNode-ai/repo"},
        },
    ]

    with patch(
        "omnibase_infra.nodes.node_closeout_effect.handlers.merge_sweep_runner._run_gh",
        new_callable=AsyncMock,
        return_value=json.dumps(fake_prs),
    ) as mock_gh:
        result = await run_merge_sweep(dry_run=False)

    assert result.auto_merge_enabled == 1
    # Called once for search, once for auto-merge
    assert mock_gh.call_count == 2


@pytest.mark.asyncio
async def test_run_merge_sweep_skips_dependabot() -> None:
    """PRs with 'dependencies' label should be skipped."""
    fake_prs = [
        {
            "number": 30,
            "title": "Bump something",
            "isDraft": False,
            "mergeable": "MERGEABLE",
            "mergeStateStatus": "CLEAN",
            "reviewDecision": "APPROVED",
            "labels": [{"name": "dependencies"}],
            "statusCheckRollup": [],
            "headRefName": "dependabot/bump",
            "repository": {"nameWithOwner": "OmniNode-ai/repo"},
        },
    ]

    with patch(
        "omnibase_infra.nodes.node_closeout_effect.handlers.merge_sweep_runner._run_gh",
        new_callable=AsyncMock,
        return_value=json.dumps(fake_prs),
    ):
        result = await run_merge_sweep(dry_run=True)

    assert len(result.classified) == 0


# ---------------------------------------------------------------------------
# HandlerCloseout integration
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_handler_closeout_invokes_merge_sweep() -> None:
    """HandlerCloseout should call run_merge_sweep and report results."""
    fake_prs = [
        {
            "number": 40,
            "title": "Ready",
            "isDraft": False,
            "mergeable": "MERGEABLE",
            "mergeStateStatus": "CLEAN",
            "reviewDecision": "APPROVED",
            "labels": [],
            "statusCheckRollup": [],
            "headRefName": "feat",
            "repository": {"nameWithOwner": "OmniNode-ai/repo"},
        },
    ]

    with patch(
        "omnibase_infra.nodes.node_closeout_effect.handlers.merge_sweep_runner._run_gh",
        new_callable=AsyncMock,
        return_value=json.dumps(fake_prs),
    ):
        handler = HandlerCloseout()
        result = await handler.handle(correlation_id=uuid4(), dry_run=False)

    assert result.merge_sweep_completed is True
    assert result.prs_merged == 1


@pytest.mark.asyncio
async def test_handler_closeout_dry_run_no_side_effects() -> None:
    """Dry run should return synthetic success without calling gh."""
    handler = HandlerCloseout()
    result = await handler.handle(correlation_id=uuid4(), dry_run=True)

    assert result.merge_sweep_completed is True
    assert result.prs_merged == 0
    assert "dry_run" in result.warnings[0]
