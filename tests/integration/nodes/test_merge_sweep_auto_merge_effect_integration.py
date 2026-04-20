# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration test for node_merge_sweep_auto_merge_effect (OMN-9276).

Verifies the end-to-end contract of HandlerAutoMerge: given a CLEAN PR on a
merge-queue repo, the handler arms auto-merge AND enqueues the PR into the
merge queue. The regression this guards against is the original bug where
auto-merge was armed but the PR never entered the queue, so CLEAN PRs sat
idle forever while the merge-sweep loop reported "success".

Subprocess calls are mocked because hitting the real GitHub API from CI
would need credentials and would mutate real PRs — but the test asserts the
exact sequence and argument shape of the three required gh invocations,
which is the actual contract between this handler and GitHub.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_merge_sweep_auto_merge_effect.handlers.handler_auto_merge import (
    HandlerAutoMerge,
)
from omnibase_infra.nodes.node_merge_sweep_auto_merge_effect.models.model_auto_merge_result import (
    ModelAutoMergeResult,
)


def _proc(returncode: int, stdout: bytes = b"", stderr: bytes = b"") -> AsyncMock:
    proc = AsyncMock()
    proc.communicate.return_value = (stdout, stderr)
    proc.returncode = returncode
    return proc


@pytest.mark.integration
class TestAutoMergeEffectContract:
    """End-to-end contract for the auto-merge effect node."""

    @pytest.mark.asyncio
    async def test_merge_queue_repo_full_lifecycle(self) -> None:
        """CLEAN PR on merge-queue repo: arm, fetch id, enqueue — all three calls."""
        handler = HandlerAutoMerge()
        call_log: list[tuple[str, ...]] = []

        def _side_effect(*args, **_kwargs):
            call_log.append(tuple(args))
            # Determine which call by inspecting argv
            if "merge" in args and "--auto" in args:
                return _proc(0)
            if "view" in args and "--json" in args:
                return _proc(0, stdout=b'{"id":"PR_kwDOABCDEF12345"}')
            if "graphql" in args:
                return _proc(
                    0,
                    stdout=(
                        b'{"data":{"enqueuePullRequest":'
                        b'{"mergeQueueEntry":{"position":7}}}}'
                    ),
                )
            return _proc(1, stderr=b"unexpected call in test")

        cid = uuid4()
        with patch("asyncio.create_subprocess_exec", side_effect=_side_effect):
            result: ModelAutoMergeResult = await handler.handle(
                prs=(("OmniNode-ai/omnimarket", 999),),
                correlation_id=cid,
            )

        assert isinstance(result, ModelAutoMergeResult)
        assert result.correlation_id == cid
        assert result.total_enabled == 1
        assert result.total_enqueued == 1
        assert result.total_failed == 0
        assert result.success is True

        outcome = result.outcomes[0]
        assert outcome.repo == "OmniNode-ai/omnimarket"
        assert outcome.pr_number == 999
        assert outcome.success is True
        assert outcome.enqueued is True
        assert outcome.queue_position == 7
        assert outcome.enqueue_error == ""
        assert outcome.enqueue_skipped is False

        assert len(call_log) == 3, f"expected 3 gh calls, got {len(call_log)}"
        merge_args, view_args, enqueue_args = call_log

        assert merge_args[:4] == ("gh", "pr", "merge", "999")
        assert "--auto" in merge_args
        assert "--repo" in merge_args
        assert "OmniNode-ai/omnimarket" in merge_args

        assert view_args[:4] == ("gh", "pr", "view", "999")
        assert "--json" in view_args
        assert "id" in view_args

        assert enqueue_args[:3] == ("gh", "api", "graphql")
        joined = " ".join(enqueue_args)
        assert "enqueuePullRequest" in joined
        assert "id=PR_kwDOABCDEF12345" in joined

    @pytest.mark.asyncio
    async def test_legacy_repo_without_merge_queue_still_succeeds(self) -> None:
        """Repos without a merge queue are handled gracefully (enqueue_skipped)."""
        handler = HandlerAutoMerge()
        responses = iter(
            [
                _proc(0),
                _proc(0, stdout=b'{"id":"PR_kwDOABCDEF99999"}'),
                _proc(
                    1,
                    stderr=(
                        b"GraphQL: Base branch does not have a merge queue "
                        b"enabled (enqueuePullRequest)"
                    ),
                ),
            ]
        )
        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=lambda *_a, **_kw: next(responses),
        ):
            result = await handler.handle(
                prs=(("OmniNode-ai/legacy-repo", 42),),
                correlation_id=uuid4(),
            )

        assert result.total_enabled == 1
        assert result.total_enqueued == 0
        assert result.total_failed == 0
        assert result.success is True
        assert result.outcomes[0].enqueue_skipped is True
        assert result.outcomes[0].enqueued is False
        assert result.outcomes[0].enqueue_error == ""
