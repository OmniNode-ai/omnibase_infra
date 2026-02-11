# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for the RRH emit effect node handlers.

Tests that run against real system resources (git repo, filesystem).
The ``integration`` marker is auto-applied by conftest for this directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.models.rrh import ModelRRHRepoState
from omnibase_infra.nodes.node_rrh_emit_effect.handlers.handler_repo_state_collect import (
    HandlerRepoStateCollect,
)


class TestHandlerRepoStateCollectIntegration:
    @pytest.fixture
    def handler(self) -> HandlerRepoStateCollect:
        return HandlerRepoStateCollect()

    @pytest.mark.anyio
    async def test_collects_from_real_repo(
        self, handler: HandlerRepoStateCollect
    ) -> None:
        """Collect state from the actual repo hosting this test suite."""
        repo_path = str(Path(__file__).resolve().parents[3])
        result = await handler.handle(repo_path)
        assert isinstance(result, ModelRRHRepoState)
        assert result.branch  # Should have a branch
        assert result.head_sha  # Should have a SHA
        assert result.repo_root  # Should have a root path
