# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for HandlerScopeExtract."""

from __future__ import annotations

from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_scope_extract_compute.handlers.handler_scope_extract import (
    HandlerScopeExtract,
)


@pytest.mark.unit
class TestHandlerScopeExtract:
    """Tests for the scope extraction compute handler."""

    @pytest.fixture
    def handler(self) -> HandlerScopeExtract:
        return HandlerScopeExtract()

    @pytest.mark.asyncio
    async def test_extract_files_from_backticks(self, handler: HandlerScopeExtract):
        """Extract file paths from backtick-quoted strings."""
        content = """
# Plan

Modify `src/api.py` and `tests/test_api.py` to add validation.
Also update `config/settings.yaml`.
"""
        cid = uuid4()
        result = await handler.handle(
            content=content, plan_file_path="plan.md", correlation_id=cid
        )

        assert "src/api.py" in result.files
        assert "tests/test_api.py" in result.files
        assert "config/settings.yaml" in result.files

    @pytest.mark.asyncio
    async def test_extract_directories(self, handler: HandlerScopeExtract):
        """Extract directory paths (ending in /)."""
        content = """
Changes in `plugins/onex/hooks/` and `src/omniclaude/`.
"""
        cid = uuid4()
        result = await handler.handle(
            content=content, plan_file_path="plan.md", correlation_id=cid
        )

        assert "plugins/onex/hooks/" in result.directories
        assert "src/omniclaude/" in result.directories

    @pytest.mark.asyncio
    async def test_extract_repos(self, handler: HandlerScopeExtract):
        """Extract known repo names."""
        content = """
This affects omniclaude and omnibase_core repositories.
"""
        cid = uuid4()
        result = await handler.handle(
            content=content, plan_file_path="plan.md", correlation_id=cid
        )

        assert "omniclaude" in result.repos
        assert "omnibase_core" in result.repos

    @pytest.mark.asyncio
    async def test_extract_systems(self, handler: HandlerScopeExtract):
        """Extract system keywords."""
        content = """
Modify the hooks system and update CLAUDE.md.
"""
        cid = uuid4()
        result = await handler.handle(
            content=content, plan_file_path="plan.md", correlation_id=cid
        )

        assert "hooks" in result.systems
        assert "CLAUDE.md" in result.systems

    @pytest.mark.asyncio
    async def test_extract_files_affected_section(self, handler: HandlerScopeExtract):
        """Extract from 'Files affected' sections."""
        content = """
# Plan

Files affected:
- `src/handler.py`
- `tests/test_handler.py`
- `plugins/onex/skills/`
"""
        cid = uuid4()
        result = await handler.handle(
            content=content, plan_file_path="plan.md", correlation_id=cid
        )

        assert "src/handler.py" in result.files
        assert "tests/test_handler.py" in result.files
        assert "plugins/onex/skills/" in result.directories

    @pytest.mark.asyncio
    async def test_empty_content(self, handler: HandlerScopeExtract):
        """Empty content returns empty scope."""
        cid = uuid4()
        result = await handler.handle(
            content="", plan_file_path="empty.md", correlation_id=cid
        )

        assert result.files == ()
        assert result.directories == ()
        assert result.repos == ()
        assert result.systems == ()

    @pytest.mark.asyncio
    async def test_deduplication(self, handler: HandlerScopeExtract):
        """Duplicate entries are deduplicated."""
        content = """
Edit `src/api.py` then update `src/api.py` again.
"""
        cid = uuid4()
        result = await handler.handle(
            content=content, plan_file_path="plan.md", correlation_id=cid
        )

        assert result.files.count("src/api.py") == 1
