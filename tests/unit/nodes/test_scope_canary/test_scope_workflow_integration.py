# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration test: full scope-check workflow with in-memory orchestration."""

from __future__ import annotations

from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_scope_extract_compute.handlers.handler_scope_extract import (
    HandlerScopeExtract,
)
from omnibase_infra.nodes.node_scope_file_read_effect.handlers.handler_scope_file_read import (
    HandlerScopeFileRead,
)
from omnibase_infra.nodes.node_scope_manifest_write_effect.handlers.handler_scope_manifest_write import (
    HandlerScopeManifestWrite,
)


@pytest.mark.unit
class TestScopeWorkflowIntegration:
    """End-to-end test of the scope-check workflow using handlers directly.

    This simulates the orchestrator dispatching handlers in sequence
    without requiring the full event bus or runtime.
    """

    @pytest.mark.asyncio
    async def test_full_workflow(self, tmp_path):
        """Run the complete scope-check workflow: read -> extract -> write."""
        # Setup: create a plan file
        plan_file = tmp_path / "test-plan.md"
        plan_file.write_text(
            """# Test Plan

Modify `src/omniclaude/hooks/handler.py` and update `plugins/onex/skills/`.

Files affected:
- `src/handler.py`
- `tests/test_handler.py`

This impacts the omniclaude and omnibase_core repositories.
The hooks system needs updating.
""",
            encoding="utf-8",
        )
        manifest_path = tmp_path / "scope-manifest.json"
        cid = uuid4()

        # Step 1: File read (EFFECT)
        file_reader = HandlerScopeFileRead()
        read_result = await file_reader.handle(
            file_path=str(plan_file), correlation_id=cid
        )
        assert read_result.success is True

        # Step 2: Scope extraction (COMPUTE)
        extractor = HandlerScopeExtract()
        extracted = await extractor.handle(
            content=read_result.content,
            plan_file_path=str(plan_file),
            correlation_id=cid,
        )
        assert len(extracted.files) > 0
        assert "omniclaude" in extracted.repos
        assert "hooks" in extracted.systems

        # Step 3: Manifest write (EFFECT)
        writer = HandlerScopeManifestWrite()
        written = await writer.handle(
            output_path=str(manifest_path),
            plan_file_path=str(plan_file),
            files=extracted.files,
            directories=extracted.directories,
            repos=extracted.repos,
            systems=extracted.systems,
            adjacent_files=extracted.adjacent_files,
            correlation_id=cid,
        )
        assert written.success is True
        assert manifest_path.exists()

        # Verify manifest content
        import json

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["version"] == "1.0.0"
        assert "omniclaude" in manifest["repos"]
        assert len(manifest["files"]) > 0
