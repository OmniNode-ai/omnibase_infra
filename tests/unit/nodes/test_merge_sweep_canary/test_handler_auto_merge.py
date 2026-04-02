# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for HandlerAutoMerge — mocked subprocess for gh CLI."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_merge_sweep_auto_merge_effect.handlers.handler_auto_merge import (
    HandlerAutoMerge,
)


@pytest.mark.unit
class TestHandlerAutoMerge:
    """Tests for the auto-merge effect handler with mocked subprocess."""

    @pytest.fixture
    def handler(self) -> HandlerAutoMerge:
        return HandlerAutoMerge()

    @pytest.mark.asyncio
    async def test_dry_run(self, handler: HandlerAutoMerge):
        """Dry run does not call gh CLI."""
        cid = uuid4()
        result = await handler.handle(
            prs=(("OmniNode-ai/test", 42),),
            correlation_id=cid,
            dry_run=True,
        )

        assert len(result.outcomes) == 1
        assert result.outcomes[0].success is True
        assert result.outcomes[0].dry_run is True
        assert result.total_enabled == 0  # dry run doesn't count as enabled

    @pytest.mark.asyncio
    async def test_successful_merge(self, handler: HandlerAutoMerge):
        """Successful auto-merge enable."""
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"")
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await handler.handle(
                prs=(("OmniNode-ai/test", 42),),
                correlation_id=uuid4(),
            )

        assert result.total_enabled == 1
        assert result.total_failed == 0
        assert result.success is True

    @pytest.mark.asyncio
    async def test_failed_merge(self, handler: HandlerAutoMerge):
        """Failed auto-merge reports error."""
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"branch protection error")
        mock_proc.returncode = 1

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await handler.handle(
                prs=(("OmniNode-ai/test", 42),),
                correlation_id=uuid4(),
            )

        assert result.total_enabled == 0
        assert result.total_failed == 1
        assert result.success is False

    @pytest.mark.asyncio
    async def test_multiple_prs(self, handler: HandlerAutoMerge):
        """Multiple PRs processed in parallel."""
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"")
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
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

    @pytest.mark.asyncio
    async def test_merge_method_rebase(self, handler: HandlerAutoMerge):
        """Rebase merge method is passed to gh CLI."""
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"")
        mock_proc.returncode = 0

        with patch(
            "asyncio.create_subprocess_exec", return_value=mock_proc
        ) as mock_exec:
            await handler.handle(
                prs=(("OmniNode-ai/test", 42),),
                correlation_id=uuid4(),
                merge_method="rebase",
            )

        # Verify --rebase was passed
        call_args = mock_exec.call_args[0]
        assert "--rebase" in call_args

    @pytest.mark.asyncio
    async def test_correlation_id_preserved(self, handler: HandlerAutoMerge):
        """Correlation ID preserved."""
        cid = uuid4()
        result = await handler.handle(prs=(), correlation_id=cid, dry_run=True)
        assert result.correlation_id == cid
