# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""MCP-backed implementation of ProtocolProjectTracker for Linear.

Wraps ``mcp__linear-server__*`` MCP tool calls behind ProtocolProjectTracker.
Uses callable injection so tests can pass fake callables without requiring
a live MCP server.

Selected by ``resolve_project_tracker()`` when ``LINEAR_TOKEN`` or
``LINEAR_API_KEY`` is present in the environment; otherwise
``LocalStubProjectTracker`` is used.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from omnibase_infra.adapters.project_tracker.model_stub_comment import ModelStubComment
from omnibase_infra.adapters.project_tracker.model_stub_issue import ModelStubIssue
from omnibase_infra.adapters.project_tracker.model_stub_project import ModelStubProject


class LinearHealthStatus:
    """Minimal health status for the Linear adapter.

    Satisfies the structural shape of ProtocolServiceHealthStatus without
    importing it to keep this adapter free of SPI runtime imports.
    """

    def __init__(self, status: str = "healthy") -> None:
        self.service_id = "linear-project-tracker"
        self.status = status
        self.diagnostics: dict[str, str] = {}


def _parse_dt(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        # Linear returns RFC3339 with trailing 'Z'
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    return datetime.now(UTC)


def _issue_from_mcp(d: dict[str, object]) -> ModelStubIssue:  # stub-ok
    """Translate a Linear MCP issue dict into ModelStubIssue wire shape."""
    state_raw = d.get("state")
    if isinstance(state_raw, dict):
        state = str(state_raw.get("name") or state_raw.get("type") or "unknown")
    else:
        state = str(state_raw) if state_raw is not None else "unknown"

    priority_raw = d.get("priority")
    if isinstance(priority_raw, dict):
        priority: str | None = (
            str(priority_raw.get("name")) if priority_raw.get("name") else None
        )
    elif priority_raw is None:
        priority = None
    else:
        priority = str(priority_raw)

    assignee_raw = d.get("assignee")
    if isinstance(assignee_raw, dict):
        assignee: str | None = (
            str(assignee_raw.get("name") or assignee_raw.get("id"))
            if assignee_raw
            else None
        )
    elif assignee_raw is None:
        assignee = None
    else:
        assignee = str(assignee_raw)

    labels_raw = d.get("labels", [])
    labels: list[str] = []
    if isinstance(labels_raw, list):
        for item in labels_raw:
            if isinstance(item, dict):
                label_name = item.get("name") or item.get("id")
                if label_name is not None:
                    labels.append(str(label_name))
            elif item is not None:
                labels.append(str(item))

    team_raw = d.get("team")
    if isinstance(team_raw, dict):
        team: str | None = str(team_raw.get("name") or team_raw.get("id")) or None
    elif team_raw is None:
        team = None
    else:
        team = str(team_raw)

    return ModelStubIssue(
        id=str(d.get("id", "")),
        identifier=str(d.get("identifier", "")),
        title=str(d.get("title", "")),
        description=(
            str(d["description"]) if d.get("description") is not None else None
        ),
        state=state,
        priority=priority,
        assignee=assignee,
        labels=labels,
        team=team,
        project_id=(str(d["project_id"]) if d.get("project_id") is not None else None),
        url=str(d["url"]) if d.get("url") is not None else None,
        created_at=_parse_dt(d.get("createdAt") or d.get("created_at")),
        updated_at=_parse_dt(d.get("updatedAt") or d.get("updated_at")),
    )


def _comment_from_mcp(d: dict[str, object]) -> ModelStubComment:
    user_raw = d.get("user")
    if isinstance(user_raw, dict):
        author = str(user_raw.get("name") or user_raw.get("id") or "linear-user")
    elif user_raw is None:
        author = "linear-user"
    else:
        author = str(user_raw)
    return ModelStubComment(
        id=str(d.get("id", "")),
        body=str(d.get("body", "")),
        author=author,
        created_at=_parse_dt(d.get("createdAt") or d.get("created_at")),
    )


def _project_from_mcp(d: dict[str, object]) -> ModelStubProject:
    progress_raw = d.get("progress", 0.0)
    progress = float(progress_raw) if isinstance(progress_raw, (int, float)) else 0.0
    state_raw = d.get("state")
    if isinstance(state_raw, dict):
        state: str | None = (
            str(state_raw.get("name")) if state_raw.get("name") else None
        )
    elif state_raw is None:
        state = None
    else:
        state = str(state_raw)
    return ModelStubProject(
        id=str(d.get("id", "")),
        name=str(d.get("name", "")),
        description=(
            str(d["description"]) if d.get("description") is not None else None
        ),
        state=state,
        progress=progress,
        url=str(d["url"]) if d.get("url") is not None else None,
    )


class AdapterLinearProjectTracker:
    """MCP-backed ProtocolProjectTracker implementation for Linear.

    Constructor accepts optional callables for each MCP operation. When a
    callable is ``None``, the adapter raises NotImplementedError if the
    corresponding method is invoked. This keeps the adapter pure-Python
    (no global MCP import) and fully testable via callable injection.

    The callable contract mirrors the Linear MCP tool signatures — each
    callable accepts keyword arguments and returns a dict (for single-
    entity operations) or a list of dicts (for list/search operations).
    """

    def __init__(
        self,
        *,
        mcp_get_issue: Callable[..., dict[str, object]] | None = None,
        mcp_list_issues: Callable[..., list[dict[str, object]]] | None = None,
        mcp_create_issue: Callable[..., dict[str, object]] | None = None,
        mcp_update_issue: Callable[..., dict[str, object]] | None = None,
        mcp_search_issues: Callable[..., list[dict[str, object]]] | None = None,
        mcp_add_comment: Callable[..., dict[str, object]] | None = None,
        mcp_get_project: Callable[..., dict[str, object]] | None = None,
        mcp_list_projects: Callable[..., list[dict[str, object]]] | None = None,
    ) -> None:
        self._mcp_get_issue = mcp_get_issue
        self._mcp_list_issues = mcp_list_issues
        self._mcp_create_issue = mcp_create_issue
        self._mcp_update_issue = mcp_update_issue
        self._mcp_search_issues = mcp_search_issues
        self._mcp_add_comment = mcp_add_comment
        self._mcp_get_project = mcp_get_project
        self._mcp_list_projects = mcp_list_projects
        self._connected = False

    # -- lifecycle --

    async def connect(self) -> bool:
        self._connected = True
        return True

    async def health_check(self) -> LinearHealthStatus:
        return LinearHealthStatus(
            status="healthy" if self._connected else "not_connected"
        )

    async def get_capabilities(self) -> list[str]:
        return ["read", "write"]

    async def close(self, timeout_seconds: float = 30.0) -> None:
        self._connected = False

    # -- internal helpers --

    @staticmethod
    def _require(
        callable_: Callable[..., object] | None, name: str
    ) -> Callable[..., object]:
        if callable_ is None:
            raise NotImplementedError(
                f"AdapterLinearProjectTracker: '{name}' callable not injected"
            )
        return callable_

    # -- domain operations --

    async def get_issue(self, issue_id: str) -> ModelStubIssue:
        fn = self._require(self._mcp_get_issue, "mcp_get_issue")
        result = fn(id=issue_id)
        if not isinstance(result, dict) or not result:
            raise KeyError(f"Issue not found: {issue_id}")
        return _issue_from_mcp(result)

    async def list_issues(
        self, filters: dict[str, str] | None = None, limit: int = 50
    ) -> list[ModelStubIssue]:
        fn = self._require(self._mcp_list_issues, "mcp_list_issues")
        kwargs: dict[str, object] = {"limit": limit}
        if filters:
            kwargs.update(filters)
        result = fn(**kwargs)
        if not isinstance(result, list):
            return []
        return [_issue_from_mcp(d) for d in result if isinstance(d, dict)]

    async def create_issue(
        self,
        title: str,
        description: str,
        labels: list[str] | None = None,
        assignee: str | None = None,
        priority: str | None = None,
        team: str | None = None,
    ) -> ModelStubIssue:
        fn = self._require(self._mcp_create_issue, "mcp_create_issue")
        kwargs: dict[str, object] = {"title": title, "description": description}
        if labels is not None:
            kwargs["labels"] = labels
        if assignee is not None:
            kwargs["assignee"] = assignee
        if priority is not None:
            kwargs["priority"] = priority
        if team is not None:
            kwargs["team"] = team
        result = fn(**kwargs)
        if not isinstance(result, dict):
            raise RuntimeError("create_issue: MCP returned non-dict result")
        return _issue_from_mcp(result)

    async def update_issue(
        self, issue_id: str, updates: dict[str, str]
    ) -> ModelStubIssue:
        fn = self._require(self._mcp_update_issue, "mcp_update_issue")
        result = fn(id=issue_id, **updates)
        if not isinstance(result, dict) or not result:
            raise KeyError(f"Issue not found: {issue_id}")
        return _issue_from_mcp(result)

    async def search_issues(self, query: str, limit: int = 50) -> list[ModelStubIssue]:
        fn = self._require(self._mcp_search_issues, "mcp_search_issues")
        result = fn(query=query, limit=limit)
        if not isinstance(result, list):
            return []
        return [_issue_from_mcp(d) for d in result if isinstance(d, dict)]

    async def add_comment(self, issue_id: str, body: str) -> ModelStubComment:
        fn = self._require(self._mcp_add_comment, "mcp_add_comment")
        result = fn(issueId=issue_id, body=body)
        if not isinstance(result, dict) or not result:
            raise KeyError(f"Issue not found: {issue_id}")
        return _comment_from_mcp(result)

    async def get_project(self, project_id: str) -> ModelStubProject:
        fn = self._require(self._mcp_get_project, "mcp_get_project")
        result = fn(id=project_id)
        if not isinstance(result, dict) or not result:
            raise KeyError(f"Project not found: {project_id}")
        return _project_from_mcp(result)

    async def list_projects(self, limit: int = 50) -> list[ModelStubProject]:
        fn = self._require(self._mcp_list_projects, "mcp_list_projects")
        result = fn(limit=limit)
        if not isinstance(result, list):
            return []
        return [_project_from_mcp(d) for d in result if isinstance(d, dict)]
