# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration test: full merge-sweep workflow with in-memory orchestration."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_merge_sweep_auto_merge_effect.handlers.handler_auto_merge import (
    HandlerAutoMerge,
)
from omnibase_infra.nodes.node_merge_sweep_classify_compute.handlers.handler_classify_prs import (
    HandlerClassifyPRs,
)
from omnibase_infra.nodes.node_merge_sweep_pr_list_effect.handlers.handler_pr_list import (
    HandlerPRList,
)
from omnibase_infra.nodes.node_merge_sweep_workflow_orchestrator.handlers.handler_auto_merge_complete import (
    HandlerAutoMergeComplete,
)
from omnibase_infra.nodes.node_merge_sweep_workflow_orchestrator.handlers.handler_classify_complete import (
    HandlerClassifyComplete,
)
from omnibase_infra.nodes.node_merge_sweep_workflow_orchestrator.handlers.handler_merge_sweep_initiate import (
    HandlerMergeSweepInitiate,
)
from omnibase_infra.nodes.node_merge_sweep_workflow_orchestrator.handlers.handler_pr_list_complete import (
    HandlerPRListComplete,
)


@pytest.mark.unit
class TestMergeSweepWorkflowIntegration:
    """End-to-end test of the merge-sweep workflow using handlers directly.

    This simulates the orchestrator dispatching handlers in sequence
    without requiring the full event bus or runtime.
    """

    @pytest.mark.asyncio
    async def test_full_workflow_with_track_a_prs(self):
        """Run complete workflow: scan -> classify -> auto-merge."""
        cid = uuid4()
        repos = ("OmniNode-ai/test-repo",)

        # Step 1: Initiate (ORCHESTRATOR handler)
        initiate_handler = HandlerMergeSweepInitiate()
        pr_list_request = await initiate_handler.handle(
            correlation_id=cid,
            repos=repos,
        )
        assert pr_list_request.correlation_id == cid
        assert pr_list_request.repos == repos

        # Step 2: PR List (EFFECT) — mock gh CLI
        mock_prs = [
            {
                "number": 1,
                "title": "feat: ready PR",
                "author": {"login": "user1"},
                "mergeable": "MERGEABLE",
                "reviewDecision": "APPROVED",
                "statusCheckRollup": [{"conclusion": "SUCCESS"}],
                "isDraft": False,
                "autoMergeRequest": None,
                "labels": [],
            },
            {
                "number": 2,
                "title": "fix: failing CI",
                "author": {"login": "user2"},
                "mergeable": "MERGEABLE",
                "reviewDecision": "APPROVED",
                "statusCheckRollup": [{"conclusion": "FAILURE"}],
                "isDraft": False,
                "autoMergeRequest": None,
                "labels": [],
            },
            {
                "number": 3,
                "title": "wip: draft",
                "author": {"login": "user3"},
                "isDraft": True,
                "autoMergeRequest": None,
                "labels": [],
            },
        ]
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (
            json.dumps(mock_prs).encode(),
            b"",
        )
        mock_proc.returncode = 0

        pr_list_handler = HandlerPRList()
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            pr_list_result = await pr_list_handler.handle(
                repos=pr_list_request.repos,
                correlation_id=pr_list_request.correlation_id,
            )
        assert len(pr_list_result.prs) == 3

        # Step 3: PR List Complete (ORCHESTRATOR handler)
        pr_list_complete = HandlerPRListComplete()
        classify_input = await pr_list_complete.handle(
            correlation_id=pr_list_result.correlation_id,
            prs=pr_list_result.prs,
        )
        assert len(classify_input.prs) == 3

        # Step 4: Classify (COMPUTE)
        classify_handler = HandlerClassifyPRs()
        classify_result = await classify_handler.handle(
            prs=classify_input.prs,
            correlation_id=classify_input.correlation_id,
            require_approval=classify_input.require_approval,
        )
        assert len(classify_result.track_a) == 1  # PR #1
        assert len(classify_result.track_b) == 1  # PR #2 (CI failure)
        assert len(classify_result.skipped) == 1  # PR #3 (draft)

        # Step 5: Classify Complete (ORCHESTRATOR handler)
        classify_complete = HandlerClassifyComplete()
        auto_merge_request = await classify_complete.handle(
            correlation_id=classify_result.correlation_id,
            track_a=classify_result.track_a,
            dry_run=True,
        )
        assert len(auto_merge_request.prs) == 1
        assert auto_merge_request.dry_run is True

        # Step 6: Auto-merge (EFFECT) — dry run
        auto_merge_handler = HandlerAutoMerge()
        auto_merge_result = await auto_merge_handler.handle(
            prs=auto_merge_request.prs,
            correlation_id=auto_merge_request.correlation_id,
            merge_method=auto_merge_request.merge_method,
            dry_run=auto_merge_request.dry_run,
        )
        assert len(auto_merge_result.outcomes) == 1
        assert auto_merge_result.outcomes[0].dry_run is True

        # Step 7: Auto-merge Complete (ORCHESTRATOR handler)
        auto_merge_complete = HandlerAutoMergeComplete()
        final_result = await auto_merge_complete.handle(
            correlation_id=auto_merge_result.correlation_id,
            outcomes=auto_merge_result.outcomes,
            total_prs_scanned=len(pr_list_result.prs),
            track_a_count=len(classify_result.track_a),
            track_b_count=len(classify_result.track_b),
            skipped_count=len(classify_result.skipped),
            repos_scanned=pr_list_result.repos_scanned,
            dry_run=True,
        )

        # Final assertions
        assert final_result.correlation_id == cid
        assert (
            final_result.status == "complete"
        )  # dry run: 0 enabled, 0 failed, 3 scanned
        assert final_result.total_prs_scanned == 3
        assert final_result.track_a_count == 1
        assert final_result.track_b_count == 1
        assert final_result.skipped_count == 1
        assert final_result.dry_run is True

    @pytest.mark.asyncio
    async def test_workflow_no_prs(self):
        """Workflow with no PRs found produces nothing_to_merge."""
        cid = uuid4()

        # Scan returns empty
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"[]", b"")
        mock_proc.returncode = 0

        pr_list_handler = HandlerPRList()
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            pr_list_result = await pr_list_handler.handle(
                repos=("OmniNode-ai/test",),
                correlation_id=cid,
            )
        assert len(pr_list_result.prs) == 0

        # Classify empty
        classify_handler = HandlerClassifyPRs()
        classify_result = await classify_handler.handle(
            prs=pr_list_result.prs,
            correlation_id=cid,
        )
        assert classify_result.total_classified == 0

        # Final result
        auto_merge_complete = HandlerAutoMergeComplete()
        final_result = await auto_merge_complete.handle(
            correlation_id=cid,
            outcomes=(),
            total_prs_scanned=0,
        )
        assert final_result.status == "nothing_to_merge"
