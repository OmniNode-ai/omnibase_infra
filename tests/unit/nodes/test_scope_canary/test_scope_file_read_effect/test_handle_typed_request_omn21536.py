# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed request contract for the scope file read handler (OMN-21536)."""

from __future__ import annotations

import inspect
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.nodes.node_scope_file_read_effect.handlers.handler_scope_file_read import (
    HandlerScopeFileRead,
)
from omnibase_infra.nodes.node_scope_file_read_effect.models import (
    ModelScopeFileReadRequest,
)


async def test_handle_typed_request(tmp_path: Path) -> None:
    plan_file = tmp_path / "plan.md"
    plan_file.write_text("# My Plan\n\nSome content.", encoding="utf-8")
    correlation_id = uuid4()
    request = ModelScopeFileReadRequest(
        file_path=str(plan_file), correlation_id=correlation_id
    )
    handler = HandlerScopeFileRead()

    result = await handler.handle(request)

    assert result.success is True
    assert result.content == "# My Plan\n\nSome content."
    assert result.correlation_id == correlation_id
    assert result.output_path == "~/.claude/scope-manifest.json"
    assert result.error_message == ""
    assert list(inspect.signature(handler.handle).parameters) == ["request"]
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ModelScopeFileReadRequest.model_validate(
            {**request.model_dump(), "unknown_field": True}
        )
