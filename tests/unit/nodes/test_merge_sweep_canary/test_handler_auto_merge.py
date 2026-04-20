# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for HandlerAutoMerge — mocked subprocess for gh CLI.

Post-OMN-9276: handler now arms auto-merge, fetches PR node_id, and enqueues
into the merge queue via GraphQL. Tests cover the three subprocess calls per PR
and the fall-back paths (no merge queue, enqueue failure, node_id lookup fail).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_merge_sweep_auto_merge_effect.handlers.handler_auto_merge import (
    HandlerAutoMerge,
)

_PR_VIEW_ID_JSON = b'{"id":"PR_kwDOR6jjtc7TyYqU"}'
_ENQUEUE_SUCCESS_JSON = (
    b'{"data":{"enqueuePullRequest":{"mergeQueueEntry":{"position":1}}}}'
)
_ENQUEUE_NO_QUEUE_STDERR = (
    b"GraphQL: Base branch does not have a merge queue enabled (enqueuePullRequest)"
)


def _mk_proc(returncode: int, stdout: bytes = b"", stderr: bytes = b"") -> AsyncMock:
    proc = AsyncMock()
    proc.communicate.return_value = (stdout, stderr)
    proc.returncode = returncode
    return proc


def _subprocess_side_effect(responses: list[AsyncMock]):
    """Return a side_effect that yields the next mock proc per subprocess call."""
    iterator = iter(responses)

    def _side_effect(*_args, **_kwargs):
        return next(iterator)

    return _side_effect


@pytest.mark.unit
class TestHandlerAutoMerge:
    """Tests for the auto-merge-and-enqueue effect handler."""

    @pytest.fixture
    def handler(self) -> HandlerAutoMerge:
        return HandlerAutoMerge()

    @pytest.mark.asyncio
    async def test_dry_run(self, handler: HandlerAutoMerge) -> None:
        """Dry run does not call gh CLI and marks enqueue_skipped."""
        cid = uuid4()
        result = await handler.handle(
            prs=(("OmniNode-ai/test", 42),),
            correlation_id=cid,
            dry_run=True,
        )

        assert len(result.outcomes) == 1
        assert result.outcomes[0].success is True
        assert result.outcomes[0].dry_run is True
        assert result.outcomes[0].enqueue_skipped is True
        assert result.outcomes[0].enqueued is False
        assert result.total_enabled == 0
        assert result.total_enqueued == 0

    @pytest.mark.asyncio
    async def test_successful_merge_and_enqueue(
        self, handler: HandlerAutoMerge
    ) -> None:
        """Happy path: auto-merge armed, node_id fetched, PR enqueued."""
        responses = [
            _mk_proc(0),  # gh pr merge --auto
            _mk_proc(0, stdout=_PR_VIEW_ID_JSON),  # gh pr view --json id
            _mk_proc(0, stdout=_ENQUEUE_SUCCESS_JSON),  # gh api graphql
        ]
        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=_subprocess_side_effect(responses),
        ):
            result = await handler.handle(
                prs=(("OmniNode-ai/test", 42),),
                correlation_id=uuid4(),
            )

        assert result.total_enabled == 1
        assert result.total_enqueued == 1
        assert result.total_failed == 0
        assert result.success is True
        outcome = result.outcomes[0]
        assert outcome.enqueued is True
        assert outcome.queue_position == 1
        assert outcome.enqueue_error == ""
        assert outcome.enqueue_skipped is False

    @pytest.mark.asyncio
    async def test_repo_without_merge_queue(self, handler: HandlerAutoMerge) -> None:
        """When repo has no merge queue, enqueue is skipped, not failed."""
        responses = [
            _mk_proc(0),
            _mk_proc(0, stdout=_PR_VIEW_ID_JSON),
            _mk_proc(1, stderr=_ENQUEUE_NO_QUEUE_STDERR),
        ]
        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=_subprocess_side_effect(responses),
        ):
            result = await handler.handle(
                prs=(("OmniNode-ai/legacy", 7),),
                correlation_id=uuid4(),
            )

        assert result.total_enabled == 1
        assert result.total_enqueued == 0
        assert result.total_failed == 0
        assert result.success is True
        outcome = result.outcomes[0]
        assert outcome.enqueued is False
        assert outcome.enqueue_skipped is True
        assert outcome.enqueue_error == ""

    @pytest.mark.asyncio
    async def test_enqueue_failure_does_not_fail_overall(
        self, handler: HandlerAutoMerge
    ) -> None:
        """If auto-merge is armed but enqueue fails, PR is not counted as failed."""
        responses = [
            _mk_proc(0),
            _mk_proc(0, stdout=_PR_VIEW_ID_JSON),
            _mk_proc(1, stderr=b"GraphQL: rate limited"),
        ]
        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=_subprocess_side_effect(responses),
        ):
            result = await handler.handle(
                prs=(("OmniNode-ai/test", 42),),
                correlation_id=uuid4(),
            )

        assert result.total_enabled == 1
        assert result.total_enqueued == 0
        assert result.total_failed == 0
        assert result.success is True
        outcome = result.outcomes[0]
        assert outcome.enqueued is False
        assert "rate limited" in outcome.enqueue_error
        assert outcome.enqueue_skipped is False

    @pytest.mark.asyncio
    async def test_failed_merge_skips_enqueue(self, handler: HandlerAutoMerge) -> None:
        """If gh pr merge --auto fails, enqueue is not attempted."""
        responses = [_mk_proc(1, stderr=b"branch protection error")]
        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=_subprocess_side_effect(responses),
        ):
            result = await handler.handle(
                prs=(("OmniNode-ai/test", 42),),
                correlation_id=uuid4(),
            )

        assert result.total_enabled == 0
        assert result.total_enqueued == 0
        assert result.total_failed == 1
        assert result.success is False
        outcome = result.outcomes[0]
        assert outcome.enqueue_skipped is True

    @pytest.mark.asyncio
    async def test_node_id_fetch_failure(self, handler: HandlerAutoMerge) -> None:
        """If node_id fetch fails, auto-merge stays armed but enqueue error recorded."""
        responses = [
            _mk_proc(0),
            _mk_proc(1, stderr=b"PR not found"),
        ]
        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=_subprocess_side_effect(responses),
        ):
            result = await handler.handle(
                prs=(("OmniNode-ai/test", 42),),
                correlation_id=uuid4(),
            )

        assert result.total_enabled == 1
        assert result.total_enqueued == 0
        assert result.total_failed == 0
        outcome = result.outcomes[0]
        assert "PR not found" in outcome.enqueue_error

    @pytest.mark.asyncio
    async def test_multiple_prs(self, handler: HandlerAutoMerge) -> None:
        """Multiple PRs processed in parallel — all enqueued."""
        responses = [
            # PR 1: merge, view, enqueue
            _mk_proc(0),
            _mk_proc(0, stdout=_PR_VIEW_ID_JSON),
            _mk_proc(0, stdout=_ENQUEUE_SUCCESS_JSON),
            # PR 2
            _mk_proc(0),
            _mk_proc(0, stdout=_PR_VIEW_ID_JSON),
            _mk_proc(
                0,
                stdout=(
                    b'{"data":{"enqueuePullRequest":'
                    b'{"mergeQueueEntry":{"position":2}}}}'
                ),
            ),
            # PR 3
            _mk_proc(0),
            _mk_proc(0, stdout=_PR_VIEW_ID_JSON),
            _mk_proc(
                0,
                stdout=(
                    b'{"data":{"enqueuePullRequest":'
                    b'{"mergeQueueEntry":{"position":3}}}}'
                ),
            ),
        ]
        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=_subprocess_side_effect(responses),
        ):
            result = await handler.handle(
                prs=(
                    ("OmniNode-ai/test", 1),
                    ("OmniNode-ai/test", 2),
                    ("OmniNode-ai/other", 3),
                ),
                correlation_id=uuid4(),
            )

        assert len(result.outcomes) == 3
        assert result.total_enabled == 3
        assert result.total_enqueued == 3

    @pytest.mark.asyncio
    async def test_merge_method_rebase(self, handler: HandlerAutoMerge) -> None:
        """Rebase merge method is passed to gh pr merge."""
        responses = [
            _mk_proc(0),
            _mk_proc(0, stdout=_PR_VIEW_ID_JSON),
            _mk_proc(0, stdout=_ENQUEUE_SUCCESS_JSON),
        ]
        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=_subprocess_side_effect(responses),
        ) as mock_exec:
            await handler.handle(
                prs=(("OmniNode-ai/test", 42),),
                correlation_id=uuid4(),
                merge_method="rebase",
            )

        first_call_args = mock_exec.call_args_list[0][0]
        assert "--rebase" in first_call_args

    @pytest.mark.asyncio
    async def test_correlation_id_preserved(self, handler: HandlerAutoMerge) -> None:
        """Correlation ID preserved through empty-PR dry run."""
        cid = uuid4()
        result = await handler.handle(prs=(), correlation_id=cid, dry_run=True)
        assert result.correlation_id == cid

    @pytest.mark.asyncio
    async def test_enqueue_mutation_passed_correctly(
        self, handler: HandlerAutoMerge
    ) -> None:
        """The enqueuePullRequest GraphQL mutation is invoked with the PR node_id."""
        responses = [
            _mk_proc(0),
            _mk_proc(0, stdout=_PR_VIEW_ID_JSON),
            _mk_proc(0, stdout=_ENQUEUE_SUCCESS_JSON),
        ]
        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=_subprocess_side_effect(responses),
        ) as mock_exec:
            await handler.handle(
                prs=(("OmniNode-ai/test", 42),),
                correlation_id=uuid4(),
            )

        enqueue_call_args = mock_exec.call_args_list[2][0]
        assert "graphql" in enqueue_call_args
        assert any("enqueuePullRequest" in a for a in enqueue_call_args)
        assert any("id=PR_kwDOR6jjtc7TyYqU" in a for a in enqueue_call_args)
