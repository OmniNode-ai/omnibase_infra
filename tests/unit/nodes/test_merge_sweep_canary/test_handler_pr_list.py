# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for HandlerPRList — mocked subprocess for GitHub API calls."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_merge_sweep_pr_list_effect.handlers.handler_pr_list import (
    HandlerPRList,
    _extract_ci_status,
    _pr_json_to_model,
)


@pytest.mark.unit
class TestCIStatusExtraction:
    """Tests for _extract_ci_status helper."""

    def test_empty_rollup(self):
        assert _extract_ci_status({}) == "UNKNOWN"

    def test_all_success(self):
        pr = {"statusCheckRollup": [{"conclusion": "SUCCESS"}]}
        assert _extract_ci_status(pr) == "SUCCESS"

    def test_failure_present(self):
        pr = {
            "statusCheckRollup": [
                {"conclusion": "SUCCESS"},
                {"conclusion": "FAILURE"},
            ]
        }
        assert _extract_ci_status(pr) == "FAILURE"

    def test_pending_present(self):
        pr = {
            "statusCheckRollup": [
                {"conclusion": "SUCCESS"},
                {"state": "PENDING"},
            ]
        }
        assert _extract_ci_status(pr) == "PENDING"


@pytest.mark.unit
class TestPRJsonToModel:
    """Tests for _pr_json_to_model helper."""

    def test_minimal_json(self):
        raw = {"number": 42}
        pr = _pr_json_to_model(raw, "OmniNode-ai/test")
        assert pr.number == 42
        assert pr.repo == "OmniNode-ai/test"
        assert pr.author == ""

    def test_full_json(self):
        raw = {
            "number": 42,
            "title": "feat: something",
            "headRefName": "feat-branch",
            "baseRefName": "main",
            "author": {"login": "jonahgabriel"},
            "isDraft": False,
            "mergeable": "MERGEABLE",
            "reviewDecision": "APPROVED",
            "statusCheckRollup": [{"conclusion": "SUCCESS"}],
            "autoMergeRequest": {"enabledAt": "2026-04-01"},
            "labels": [{"name": "enhancement"}],
            "updatedAt": "2026-04-01T12:00:00Z",
        }
        pr = _pr_json_to_model(raw, "OmniNode-ai/test")
        assert pr.title == "feat: something"
        assert pr.author == "jonahgabriel"
        assert pr.has_auto_merge is True
        assert pr.labels == ("enhancement",)
        assert pr.ci_status == "SUCCESS"


@pytest.mark.unit
class TestHandlerPRList:
    """Tests for the PR list effect handler with mocked subprocess."""

    @pytest.fixture
    def handler(self) -> HandlerPRList:
        return HandlerPRList()

    @pytest.mark.asyncio
    async def test_successful_scan(self, handler: HandlerPRList):
        """Successful scan returns PRs."""
        mock_prs = [
            {"number": 1, "title": "PR 1", "author": {"login": "user1"}},
            {"number": 2, "title": "PR 2", "author": {"login": "user2"}},
        ]
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (
            json.dumps(mock_prs).encode(),
            b"",
        )
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await handler.handle(
                repos=("OmniNode-ai/test",), correlation_id=uuid4()
            )

        assert result.success is True
        assert len(result.prs) == 2
        assert result.repos_scanned == ("OmniNode-ai/test",)

    @pytest.mark.asyncio
    async def test_scan_failure(self, handler: HandlerPRList):
        """Failed scan marks repo as failed."""
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"error: not found")
        mock_proc.returncode = 1

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await handler.handle(
                repos=("OmniNode-ai/nonexistent",), correlation_id=uuid4()
            )

        assert result.success is False
        assert len(result.repos_failed) == 1
        assert len(result.prs) == 0

    @pytest.mark.asyncio
    async def test_author_filter(self, handler: HandlerPRList):
        """Author filter limits results."""
        mock_prs = [
            {"number": 1, "author": {"login": "user1"}},
            {"number": 2, "author": {"login": "user2"}},
        ]
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (
            json.dumps(mock_prs).encode(),
            b"",
        )
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await handler.handle(
                repos=("OmniNode-ai/test",),
                correlation_id=uuid4(),
                authors=("user1",),
            )

        assert len(result.prs) == 1
        assert result.prs[0].author == "user1"

    @pytest.mark.asyncio
    async def test_label_filter(self, handler: HandlerPRList):
        """Label filter limits results."""
        mock_prs = [
            {"number": 1, "labels": [{"name": "bug"}]},
            {"number": 2, "labels": [{"name": "enhancement"}]},
        ]
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (
            json.dumps(mock_prs).encode(),
            b"",
        )
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await handler.handle(
                repos=("OmniNode-ai/test",),
                correlation_id=uuid4(),
                labels=("bug",),
            )

        assert len(result.prs) == 1
        assert result.prs[0].number == 1

    @pytest.mark.asyncio
    async def test_correlation_id_preserved(self, handler: HandlerPRList):
        """Correlation ID preserved in result."""
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"[]", b"")
        mock_proc.returncode = 0
        cid = uuid4()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await handler.handle(
                repos=("OmniNode-ai/test",), correlation_id=cid
            )

        assert result.correlation_id == cid
