# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for `resolve_project_tracker()` — central tracker DI authority.

Required branches (OMN-20595, Operating Rule 8: fail fast, no silent fallback):
    1. Linear selected (the default), LINEAR_API_KEY / LINEAR_TOKEN present
       → AdapterLinearGraphQLProjectTracker
    2. Linear selected and no key → raises InfraAuthenticationError; never the stub
    3. LocalStubProjectTracker only when the caller selects it explicitly

Plus an OMN-10048 regression: when a token is set, the resolved adapter
MUST be functional — calls to ``tracker.get_issue()`` must NOT raise
``NotImplementedError`` outside ``node_session_compose``. The previous
MCP-callable adapter required external callable injection that no
production callsite ever wired, so every domain method raised
``NotImplementedError``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from omnibase_infra.adapters.project_tracker.linear_graphql_project_tracker_adapter import (
    AdapterLinearGraphQLProjectTracker,
)
from omnibase_infra.adapters.project_tracker.local_stub_project_tracker import (
    LocalStubProjectTracker,
)
from omnibase_infra.enums.enum_project_tracker_backend import (
    EnumProjectTrackerBackend,
)
from omnibase_infra.errors import InfraAuthenticationError
from omnibase_infra.services.project_tracker.resolver import resolve_project_tracker

pytestmark = pytest.mark.unit


_ISSUE_FIXTURE: dict[str, object] = {
    "id": "uuid-1",
    "identifier": "OMN-1",
    "title": "Test issue",
    "description": "body",
    "priority": 2,
    "url": "https://linear.app/omninode/issue/OMN-1",
    "createdAt": "2026-04-27T00:00:00.000Z",
    "updatedAt": "2026-04-27T01:00:00.000Z",
    "state": {"name": "Backlog", "type": "backlog"},
    "assignee": {"id": "u1", "name": "Tester"},
    "labels": {"nodes": []},
    "team": {"id": "t1", "name": "Omninode"},
    "project": None,
}


class TestResolveProjectTracker:
    def test_fails_loud_when_linear_selected_and_no_key(self, tmp_path: Path) -> None:
        """No key on a caller that asked for Linear raises; it never returns the stub."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(InfraAuthenticationError):
                resolve_project_tracker(state_root=tmp_path)

    def test_fails_loud_when_linear_selected_explicitly_and_no_key(
        self, tmp_path: Path
    ) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(InfraAuthenticationError):
                resolve_project_tracker(
                    state_root=tmp_path, backend=EnumProjectTrackerBackend.LINEAR
                )

    def test_fails_loud_on_blank_key(self, tmp_path: Path) -> None:
        with patch.dict("os.environ", {"LINEAR_API_KEY": "   "}, clear=True):
            with pytest.raises(InfraAuthenticationError):
                resolve_project_tracker(state_root=tmp_path)

    def test_returns_local_stub_only_when_selected(self, tmp_path: Path) -> None:
        with patch.dict("os.environ", {"LINEAR_API_KEY": "fake"}, clear=True):
            tracker = resolve_project_tracker(
                state_root=tmp_path, backend=EnumProjectTrackerBackend.LOCAL_STUB
            )
            assert isinstance(tracker, LocalStubProjectTracker)

    def test_returns_linear_graphql_adapter_when_token_present(
        self, tmp_path: Path
    ) -> None:
        with patch.dict("os.environ", {"LINEAR_TOKEN": "fake"}, clear=True):
            tracker = resolve_project_tracker(state_root=tmp_path)
            assert isinstance(tracker, AdapterLinearGraphQLProjectTracker)

    def test_api_key_env_var_also_selects_linear(self, tmp_path: Path) -> None:
        """LINEAR_API_KEY must be honored in addition to LINEAR_TOKEN."""
        with patch.dict("os.environ", {"LINEAR_API_KEY": "fake"}, clear=True):
            tracker = resolve_project_tracker(state_root=tmp_path)
            assert isinstance(tracker, AdapterLinearGraphQLProjectTracker)


class TestOmn10048Regression:
    """Resolver must return a working adapter from any context.

    The previous MCP-callable adapter raised ``NotImplementedError`` on every
    domain method outside ``node_session_compose`` because the MCP-to-Python
    bridge it required was never wired. The GraphQL adapter has no such
    dependency — ``tracker.get_issue()`` succeeds against a mocked transport.
    """

    @pytest.mark.asyncio
    async def test_get_issue_does_not_raise_notimplemented(
        self, tmp_path: Path
    ) -> None:
        with patch.dict("os.environ", {"LINEAR_API_KEY": "fake-token"}, clear=True):
            tracker = resolve_project_tracker(state_root=tmp_path)

        assert isinstance(tracker, AdapterLinearGraphQLProjectTracker)

        # Swap in a mock httpx transport so we can prove get_issue executes
        # end-to-end without contacting the real Linear API.
        await tracker._client.aclose()
        tracker._client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    200, json={"data": {"issue": _ISSUE_FIXTURE}}
                )
            ),
            headers={"Authorization": "fake-token"},
        )
        tracker._owns_client = True

        try:
            issue = await tracker.get_issue("OMN-1")
        except NotImplementedError as exc:  # pragma: no cover — regression guard
            pytest.fail(
                f"OMN-10048 regression: tracker.get_issue raised "
                f"NotImplementedError: {exc}"
            )
        else:
            assert issue.identifier == "OMN-1"
        await tracker.close()
