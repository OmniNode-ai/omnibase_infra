# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for HandlerScopeFileRead."""

from __future__ import annotations

from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_scope_file_read_effect.handlers.handler_scope_file_read import (
    HandlerScopeFileRead,
)


@pytest.mark.unit
class TestHandlerScopeFileRead:
    """Tests for the file read effect handler."""

    @pytest.fixture
    def handler(self) -> HandlerScopeFileRead:
        return HandlerScopeFileRead()

    @pytest.mark.asyncio
    async def test_read_existing_file(self, handler: HandlerScopeFileRead, tmp_path):
        """Read an existing file returns content."""
        plan_file = tmp_path / "plan.md"
        plan_file.write_text("# My Plan\n\nSome content.", encoding="utf-8")
        cid = uuid4()

        result = await handler.handle(file_path=str(plan_file), correlation_id=cid)

        assert result.success is True
        assert result.content == "# My Plan\n\nSome content."
        assert result.correlation_id == cid
        assert result.error_message == ""

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self, handler: HandlerScopeFileRead, tmp_path):
        """Reading a nonexistent file returns error."""
        cid = uuid4()

        result = await handler.handle(
            file_path=str(tmp_path / "missing.md"), correlation_id=cid
        )

        assert result.success is False
        assert "not found" in result.error_message.lower()
        assert result.content == ""

    @pytest.mark.asyncio
    async def test_correlation_id_preserved(
        self, handler: HandlerScopeFileRead, tmp_path
    ):
        """Correlation ID is preserved in result."""
        plan_file = tmp_path / "test.md"
        plan_file.write_text("content", encoding="utf-8")
        cid = uuid4()

        result = await handler.handle(file_path=str(plan_file), correlation_id=cid)

        assert result.correlation_id == cid
