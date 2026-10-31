# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed request contract for the scope manifest writer (OMN-21536)."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.nodes.node_scope_manifest_write_effect.handlers.handler_scope_manifest_write import (
    HandlerScopeManifestWrite,
)
from omnibase_infra.nodes.node_scope_manifest_write_effect.models import (
    ModelScopeManifestWriteRequest,
)


async def test_handle_typed_request(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "scope-manifest.json"
    correlation_id = uuid4()
    request = ModelScopeManifestWriteRequest(
        output_path=str(output),
        plan_file_path="docs/plan.md",
        files=("src/api.py", "tests/test_api.py"),
        directories=("src/",),
        repos=("omniclaude",),
        systems=("hooks",),
        adjacent_files=(),
        correlation_id=correlation_id,
    )
    handler = HandlerScopeManifestWrite()

    result = await handler.handle(request)

    assert result.success is True
    assert result.correlation_id == correlation_id
    assert result.manifest_path == str(output.resolve())
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["version"] == "1.0.0"
    assert manifest["plan_file"] == "docs/plan.md"
    assert manifest["files"] == ["src/api.py", "tests/test_api.py"]
    assert manifest["directories"] == ["src/"]
    assert manifest["repos"] == ["omniclaude"]
    assert manifest["systems"] == ["hooks"]
    assert manifest["adjacent_files"] == []
    assert list(inspect.signature(handler.handle).parameters) == ["request"]
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ModelScopeManifestWriteRequest.model_validate(
            {**request.model_dump(), "unknown_field": True}
        )
