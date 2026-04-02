# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for HandlerScopeManifestWrite."""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_scope_manifest_write_effect.handlers.handler_scope_manifest_write import (
    HandlerScopeManifestWrite,
)


@pytest.mark.unit
class TestHandlerScopeManifestWrite:
    """Tests for the manifest write effect handler."""

    @pytest.fixture
    def handler(self) -> HandlerScopeManifestWrite:
        return HandlerScopeManifestWrite()

    @pytest.mark.asyncio
    async def test_write_manifest(self, handler: HandlerScopeManifestWrite, tmp_path):
        """Write a manifest file and verify contents."""
        output = tmp_path / "scope-manifest.json"
        cid = uuid4()

        result = await handler.handle(
            output_path=str(output),
            plan_file_path="docs/plan.md",
            files=("src/api.py", "tests/test_api.py"),
            directories=("src/",),
            repos=("omniclaude",),
            systems=("hooks",),
            adjacent_files=(),
            correlation_id=cid,
        )

        assert result.success is True
        assert result.correlation_id == cid
        assert output.exists()

        manifest = json.loads(output.read_text(encoding="utf-8"))
        assert manifest["version"] == "1.0.0"
        assert manifest["plan_file"] == "docs/plan.md"
        assert manifest["repos"] == ["omniclaude"]
        assert manifest["files"] == ["src/api.py", "tests/test_api.py"]
        assert manifest["directories"] == ["src/"]
        assert manifest["systems"] == ["hooks"]

    @pytest.mark.asyncio
    async def test_write_creates_parent_dirs(
        self, handler: HandlerScopeManifestWrite, tmp_path
    ):
        """Writing to a nested path creates parent directories."""
        output = tmp_path / "nested" / "deep" / "manifest.json"
        cid = uuid4()

        result = await handler.handle(
            output_path=str(output),
            plan_file_path="plan.md",
            files=(),
            directories=(),
            repos=(),
            systems=(),
            adjacent_files=(),
            correlation_id=cid,
        )

        assert result.success is True
        assert output.exists()

    @pytest.mark.asyncio
    async def test_write_to_readonly_path(
        self, handler: HandlerScopeManifestWrite, tmp_path
    ):
        """Writing to an unwritable path returns error."""
        # Use /proc (Linux) or a clearly invalid path
        cid = uuid4()

        result = await handler.handle(
            output_path="/dev/null/impossible/manifest.json",
            plan_file_path="plan.md",
            files=(),
            directories=(),
            repos=(),
            systems=(),
            adjacent_files=(),
            correlation_id=cid,
        )

        assert result.success is False
        assert result.error_message != ""
