# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Read back the scope manifest written to the command's explicit destination."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_scope_extract_compute.handlers.handler_scope_extract import (
    HandlerScopeExtract,
)
from omnibase_infra.nodes.node_scope_extract_compute.models.model_scope_extract_input import (
    ModelScopeExtractInput,
)
from omnibase_infra.nodes.node_scope_file_read_effect.handlers.handler_scope_file_read import (
    HandlerScopeFileRead,
)
from omnibase_infra.nodes.node_scope_manifest_write_effect.handlers.handler_scope_manifest_write import (
    HandlerScopeManifestWrite,
)
from omnibase_infra.nodes.node_scope_workflow_orchestrator.handlers.handler_scope_check_initiate import (
    HandlerScopeCheckInitiate,
)
from omnibase_infra.nodes.node_scope_workflow_orchestrator.models.model_scope_check_command import (
    ModelScopeCheckCommand,
)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest.mark.parametrize("destination", ["selected.json", "nested/custom-scope.json"])
async def test_deploy_scope_manifest_readback(tmp_path: Path, destination: str) -> None:
    """The real handlers preserve the destination from command to disk write."""
    plan = tmp_path / "plan.md"
    plan.write_text("Update `src/handler.py` in omnibase_infra.\n", encoding="utf-8")
    manifest_path = tmp_path / destination
    correlation_id = uuid4()
    command = ModelScopeCheckCommand(
        correlation_id=correlation_id,
        plan_file_path=str(plan),
        output_path=str(manifest_path),
    )

    request = await HandlerScopeCheckInitiate().handle(command)
    assert request.output_path == command.output_path
    read = await HandlerScopeFileRead().handle(
        file_path=request.file_path,
        correlation_id=request.correlation_id,
        output_path=request.output_path,
    )
    assert read.success, read.error_message
    assert read.output_path == command.output_path
    extracted = await HandlerScopeExtract().handle(
        ModelScopeExtractInput(
            correlation_id=read.correlation_id,
            plan_file_path=read.file_path,
            output_path=read.output_path,
            content=read.content,
        )
    )
    assert extracted.output_path == command.output_path
    written = await HandlerScopeManifestWrite().handle(
        output_path=extracted.output_path,
        plan_file_path=extracted.plan_file_path,
        files=extracted.files,
        directories=extracted.directories,
        repos=extracted.repos,
        systems=extracted.systems,
        adjacent_files=extracted.adjacent_files,
        correlation_id=extracted.correlation_id,
    )

    assert written.success, written.error_message
    assert written.correlation_id == correlation_id
    assert written.manifest_path == str(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["plan_file"] == str(plan)
    assert "src/handler.py" in manifest["files"]
    assert "omnibase_infra" in manifest["repos"]
    assert sorted(tmp_path.rglob("*.json")) == [manifest_path]
