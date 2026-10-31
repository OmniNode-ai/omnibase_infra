# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Typed repository request preserves collected git state and errors."""

import inspect
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from omnibase_infra.nodes.node_rrh_emit_effect.handlers.handler_repo_state_collect import (
    HandlerRepoStateCollect,
)
from omnibase_infra.nodes.node_rrh_emit_effect.models import (
    ModelRepoStateCollectRequest,
)


async def test_handle_request_collects_repo_state() -> None:
    handler = HandlerRepoStateCollect()
    request = ModelRepoStateCollectRequest(repo_path="/fake/repo")
    with patch.object(
        handler,
        "_git",
        new=AsyncMock(side_effect=["main\n", "abc123\n", "", "/fake/repo\n", ""]),
    ) as git:
        result = await handler.handle(request)

    assert result.branch == "main"
    assert result.head_sha == "abc123"
    assert result.repo_root == "/fake/repo"
    assert result.remote_url == ""
    assert result.is_dirty is False
    git.assert_any_await(request.repo_path, "rev-parse", "HEAD")


async def test_handle_request_rejects_relative_path() -> None:
    request = ModelRepoStateCollectRequest(repo_path="relative/path")
    with pytest.raises(ValueError, match="non-empty absolute path"):
        await HandlerRepoStateCollect().handle(request)


def test_request_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ModelRepoStateCollectRequest.model_validate(
            {"repo_path": "/fake/repo", "unknown_field": True}
        )


def test_handle_has_one_request_parameter() -> None:
    assert list(inspect.signature(HandlerRepoStateCollect().handle).parameters) == [
        "request"
    ]
