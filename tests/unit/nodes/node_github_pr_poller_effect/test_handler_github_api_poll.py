# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for HandlerGitHubApiPoll adapter wiring."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from omnibase_infra.nodes.node_github_pr_poller_effect.handlers.handler_github_api_poll import (
    HandlerGitHubApiPoll,
)
from omnibase_infra.nodes.node_github_pr_poller_effect.models.model_github_poller_config import (
    ModelGitHubPollerConfig,
)


class _FakeGitHubClient:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.repos: list[str] = []

    def fetch_open_prs_for_triage(self, repo: str) -> list[dict[str, object]]:
        self.repos.append(repo)
        return [
            {
                "number": 42,
                "title": "Ready PR",
                "draft": False,
                "labels": [],
                "updated_at": datetime.now(tz=UTC).isoformat(),
                "combined_status": "success",
                "review_states": ["APPROVED"],
            }
        ]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_handle_uses_github_http_client_adapter() -> None:
    created_clients: list[_FakeGitHubClient] = []

    def factory(**kwargs: Any) -> _FakeGitHubClient:
        client = _FakeGitHubClient(**kwargs)
        created_clients.append(client)
        return client

    handler = HandlerGitHubApiPoll(
        api_base="https://github.example.test",
        http_timeout=5.0,
        github_token="fake-token",
        github_client_factory=factory,
    )
    config = ModelGitHubPollerConfig(
        repos=["OmniNode-ai/omnimarket"],
        poll_interval_seconds=10,
        github_token_env_var="GITHUB_TOKEN",
    )

    result = await handler.handle(config)

    assert result.errors == []
    assert result.repos_polled == ["OmniNode-ai/omnimarket"]
    assert result.prs_polled == 1
    assert result.pending_events == [
        {
            "event_type": "onex.evt.github.pr-status.v1",
            "repo": "OmniNode-ai/omnimarket",
            "pr_number": 42,
            "triage_state": "ready_to_merge",
            "title": "Ready PR",
            "is_draft": False,
            "partition_key": "OmniNode-ai/omnimarket:42",
        }
    ]
    assert created_clients[0].kwargs == {
        "token": "fake-token",
        "rest_base": "https://github.example.test",
        "timeout": 5.0,
    }
    assert created_clients[0].repos == ["OmniNode-ai/omnimarket"]


class _FakeDraftGitHubClient(_FakeGitHubClient):
    """PR fixture with draft=True (OMN-14394 seam gap regression coverage)."""

    def fetch_open_prs_for_triage(self, repo: str) -> list[dict[str, object]]:
        self.repos.append(repo)
        return [
            {
                "number": 7,
                "title": "WIP PR",
                "draft": True,
                "labels": [],
                "updated_at": datetime.now(tz=UTC).isoformat(),
                "combined_status": "pending",
                "review_states": [],
            }
        ]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_handle_publishes_is_draft_true_for_draft_pr() -> None:
    """OMN-14394: is_draft must round-trip pr['draft'] onto the published event
    -- this is the field the pr_state projection (OMN-14375) and the
    OMN-14374 read skill's ModelOpenPrSummary.is_draft depend on."""

    def factory(**kwargs: Any) -> _FakeDraftGitHubClient:
        return _FakeDraftGitHubClient(**kwargs)

    handler = HandlerGitHubApiPoll(github_client_factory=factory)
    config = ModelGitHubPollerConfig(
        repos=["OmniNode-ai/omnimarket"],
        poll_interval_seconds=10,
    )

    result = await handler.handle(config)

    assert result.pending_events == [
        {
            "event_type": "onex.evt.github.pr-status.v1",
            "repo": "OmniNode-ai/omnimarket",
            "pr_number": 7,
            "triage_state": "draft",
            "title": "WIP PR",
            "is_draft": True,
            "partition_key": "OmniNode-ai/omnimarket:7",
        }
    ]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_handle_surfaces_github_client_initialization_errors() -> None:
    def factory(**_kwargs: Any) -> _FakeGitHubClient:
        raise RuntimeError("missing token")

    handler = HandlerGitHubApiPoll(github_client_factory=factory)
    config = ModelGitHubPollerConfig(
        repos=["OmniNode-ai/omnimarket"],
        poll_interval_seconds=10,
    )

    result = await handler.handle(config)

    assert result.repos_polled == []
    assert result.prs_polled == 0
    assert result.pending_events == []
    assert len(result.errors) == 1
    assert "Error initializing GitHub client: RuntimeError" in result.errors[0]
