# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Linear GraphQL HTTP implementation of ProtocolProjectTracker.

Calls the Linear GraphQL API directly via ``httpx.AsyncClient`` using the
``LINEAR_API_KEY`` (or ``LINEAR_TOKEN``) credential from the environment.
Selected by ``resolve_project_tracker()`` whenever a Linear credential is
available — works from any Python context (no MCP-runtime dependency).

Replaces the previous MCP-callable ``AdapterLinearProjectTracker`` (deleted
in OMN-10048) which required external callable injection that no production
callsite ever wired.

Strategy reference: OMN-7709 ("Primary: Linear GraphQL API").
"""

from __future__ import annotations

import logging
import os
from types import TracebackType
from typing import Final, cast

import httpx

from omnibase_infra.adapters.project_tracker.model_stub_comment import ModelStubComment
from omnibase_infra.adapters.project_tracker.model_stub_issue import ModelStubIssue
from omnibase_infra.adapters.project_tracker.model_stub_project import ModelStubProject
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraAuthenticationError,
    InfraConnectionError,
    InfraRateLimitedError,
    InfraTimeoutError,
    ModelInfraErrorContext,
    ModelTimeoutErrorContext,
)
from omnibase_infra.mixins import MixinAsyncCircuitBreaker

logger = logging.getLogger(__name__)

# Default Linear GraphQL endpoint; overridable via constructor.
DEFAULT_LINEAR_GRAPHQL_ENDPOINT: Final[str] = "https://api.linear.app/graphql"
# Default request timeout for Linear API calls.
DEFAULT_TIMEOUT_SECONDS: Final[float] = 30.0
# Service identifier for circuit breaker / error context.
SERVICE_NAME: Final[str] = "linear-graphql"

# Common GraphQL field projections — kept tight to match the wire models
# (ModelStubIssue, ModelStubComment, ModelStubProject).
_ISSUE_FIELDS: Final[str] = """
id
identifier
title
description
priority
url
createdAt
updatedAt
state { name type }
assignee { id name }
labels { nodes { id name } }
team { id name }
project { id }
"""

_COMMENT_FIELDS: Final[str] = """
id
body
createdAt
user { id name }
"""

_PROJECT_FIELDS: Final[str] = """
id
name
description
state
progress
url
"""

# Module-level GraphQL queries (frozen). One per ProtocolProjectTracker method.

_QUERY_VIEWER: Final[str] = """
query Viewer {
    viewer { id name }
}
"""

_QUERY_GET_ISSUE: Final[str] = (
    """
query GetIssue($id: String!) {
    issue(id: $id) { """
    + _ISSUE_FIELDS
    + """ }
}
"""
)

_QUERY_LIST_ISSUES: Final[str] = (
    """
query ListIssues($first: Int!, $filter: IssueFilter) {
    issues(first: $first, filter: $filter) {
        nodes { """
    + _ISSUE_FIELDS
    + """ }
    }
}
"""
)

_QUERY_SEARCH_ISSUES: Final[str] = (
    """
query SearchIssues($term: String!, $first: Int!) {
    searchIssues(term: $term, first: $first) {
        nodes { """
    + _ISSUE_FIELDS
    + """ }
    }
}
"""
)

_MUTATION_CREATE_ISSUE: Final[str] = (
    """
mutation CreateIssue($input: IssueCreateInput!) {
    issueCreate(input: $input) {
        success
        issue { """
    + _ISSUE_FIELDS
    + """ }
    }
}
"""
)

_MUTATION_UPDATE_ISSUE: Final[str] = (
    """
mutation UpdateIssue($id: String!, $input: IssueUpdateInput!) {
    issueUpdate(id: $id, input: $input) {
        success
        issue { """
    + _ISSUE_FIELDS
    + """ }
    }
}
"""
)

_MUTATION_ADD_COMMENT: Final[str] = (
    """
mutation AddComment($input: CommentCreateInput!) {
    commentCreate(input: $input) {
        success
        comment { """
    + _COMMENT_FIELDS
    + """ }
    }
}
"""
)

_QUERY_GET_PROJECT: Final[str] = (
    """
query GetProject($id: String!) {
    project(id: $id) { """
    + _PROJECT_FIELDS
    + """ }
}
"""
)

_QUERY_LIST_PROJECTS: Final[str] = (
    """
query ListProjects($first: Int!) {
    projects(first: $first) {
        nodes { """
    + _PROJECT_FIELDS
    + """ }
    }
}
"""
)


class LinearGraphQLHealthStatus:
    """Minimal health status — satisfies ProtocolServiceHealthStatus shape."""

    def __init__(self, status: str = "healthy") -> None:
        self.service_id: str = SERVICE_NAME
        self.status: str = status
        self.diagnostics: dict[str, str] = {}


def _coerce_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _issue_from_graphql(d: dict[str, object]) -> ModelStubIssue:
    """Translate a Linear GraphQL Issue node into the wire-model issue."""
    state_obj = d.get("state") or {}
    state = (
        str(state_obj.get("name") or state_obj.get("type") or "unknown")
        if isinstance(state_obj, dict)
        else "unknown"
    )

    assignee_obj = d.get("assignee") or {}
    assignee = (
        str(assignee_obj.get("name") or assignee_obj.get("id"))
        if isinstance(assignee_obj, dict) and assignee_obj
        else None
    )

    team_obj = d.get("team") or {}
    team = (
        str(team_obj.get("name") or team_obj.get("id"))
        if isinstance(team_obj, dict) and team_obj
        else None
    )

    project_obj = d.get("project") or {}
    project_id = (
        str(project_obj.get("id"))
        if isinstance(project_obj, dict) and project_obj.get("id")
        else None
    )

    labels_container = d.get("labels") or {}
    labels: list[str] = []
    if isinstance(labels_container, dict):
        nodes = labels_container.get("nodes")
        if isinstance(nodes, list):
            for item in nodes:
                if isinstance(item, dict):
                    name = item.get("name") or item.get("id")
                    if name is not None:
                        labels.append(str(name))

    priority_raw = d.get("priority")
    priority = str(priority_raw) if priority_raw is not None else None

    from datetime import UTC, datetime

    def _parse_dt(value: object) -> datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        return datetime.now(UTC)

    return ModelStubIssue(
        id=str(d.get("id", "")),
        identifier=str(d.get("identifier", "")),
        title=str(d.get("title", "")),
        description=_coerce_str(d.get("description")),
        state=state,
        priority=priority,
        assignee=assignee,
        labels=labels,
        team=team,
        project_id=project_id,
        url=_coerce_str(d.get("url")),
        created_at=_parse_dt(d.get("createdAt")),
        updated_at=_parse_dt(d.get("updatedAt")),
    )


def _comment_from_graphql(d: dict[str, object]) -> ModelStubComment:
    from datetime import UTC, datetime

    user_obj = d.get("user") or {}
    author = (
        str(user_obj.get("name") or user_obj.get("id") or "linear-user")
        if isinstance(user_obj, dict) and user_obj
        else "linear-user"
    )

    created_at_raw = d.get("createdAt")
    if isinstance(created_at_raw, str):
        created_at = datetime.fromisoformat(created_at_raw.replace("Z", "+00:00"))
    else:
        created_at = datetime.now(UTC)

    return ModelStubComment(
        id=str(d.get("id", "")),
        body=str(d.get("body", "")),
        author=author,
        created_at=created_at,
    )


def _project_from_graphql(d: dict[str, object]) -> ModelStubProject:
    progress_raw = d.get("progress", 0.0)
    progress = float(progress_raw) if isinstance(progress_raw, (int, float)) else 0.0
    state_raw = d.get("state")
    state = str(state_raw) if state_raw is not None else None
    return ModelStubProject(
        id=str(d.get("id", "")),
        name=str(d.get("name", "")),
        description=_coerce_str(d.get("description")),
        state=state,
        progress=progress,
        url=_coerce_str(d.get("url")),
    )


class AdapterLinearGraphQLProjectTracker(MixinAsyncCircuitBreaker):
    """Linear GraphQL implementation of ProtocolProjectTracker.

    Calls Linear's public GraphQL endpoint via ``httpx.AsyncClient``. Auth
    via ``LINEAR_API_KEY`` (or ``LINEAR_TOKEN``).

    Designed for any Python caller — no MCP runtime dependency. Selected by
    ``resolve_project_tracker()`` when a Linear credential is present.

    Resilience: wraps ``MixinAsyncCircuitBreaker`` with HTTP transport,
    threshold=5, reset_timeout=60s. Connection / timeout / auth errors are
    mapped to the typed ``Infra*Error`` hierarchy with sanitized context
    (the API key is never included in error messages).
    """

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str = DEFAULT_LINEAR_GRAPHQL_ENDPOINT,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Construct the Linear GraphQL adapter.

        Args:
            api_key: Linear API key. Defaults to ``LINEAR_API_KEY`` env var,
                falling back to ``LINEAR_TOKEN``. Raises if neither is set.
            endpoint: GraphQL endpoint URL.
            timeout_seconds: Request timeout in seconds.
            client: Optional pre-built ``httpx.AsyncClient`` (used by tests
                to inject a mock transport). When provided, the adapter does
                NOT take ownership — the caller is responsible for closing
                it. Otherwise the adapter constructs and owns its client.

        Raises:
            InfraAuthenticationError: If no Linear credential is available.
        """
        resolved_key = (
            api_key
            or os.environ.get("LINEAR_API_KEY")
            or os.environ.get("LINEAR_TOKEN")
        )
        if not resolved_key:
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.HTTP,
                operation="construct",
                target_name=SERVICE_NAME,
            )
            raise InfraAuthenticationError(
                "AdapterLinearGraphQLProjectTracker requires LINEAR_API_KEY "
                "or LINEAR_TOKEN in the environment",
                context=context,
                auth_method="api_key",
            )

        self._api_key: str = resolved_key
        self._endpoint: str = endpoint
        self._timeout_seconds: float = timeout_seconds
        self._owns_client: bool = client is None
        self._client: httpx.AsyncClient = client or httpx.AsyncClient(
            timeout=timeout_seconds,
            headers={
                "Authorization": resolved_key,
                "Content-Type": "application/json",
            },
        )
        self._connected: bool = False

        self._init_circuit_breaker(
            threshold=5,
            reset_timeout=60.0,
            service_name=SERVICE_NAME,
            transport_type=EnumInfraTransportType.HTTP,
            half_open_successes=1,
        )

    # -- lifecycle (ProtocolExternalService) --

    async def connect(self) -> bool:
        """Verify the Linear API key by issuing a viewer query.

        Returns:
            True if Linear responded with a viewer object.

        Raises:
            InfraAuthenticationError: If Linear rejected the credential.
            InfraConnectionError: If Linear is unreachable.
        """
        data = await self._execute(_QUERY_VIEWER, operation="connect")
        viewer = data.get("viewer") if isinstance(data, dict) else None
        self._connected = bool(viewer)
        return self._connected

    async def health_check(self) -> LinearGraphQLHealthStatus:
        return LinearGraphQLHealthStatus(
            status="healthy" if self._connected else "not_connected",
        )

    async def get_capabilities(self) -> list[str]:
        return ["read", "write"]

    async def close(self, timeout_seconds: float = 30.0) -> None:
        self._connected = False
        # Cancel circuit breaker active recovery task to avoid leaking it.
        await self.cancel_active_recovery()
        if self._owns_client:
            await self._client.aclose()

    # -- async context manager (ergonomic resource management) --

    async def __aenter__(self) -> AdapterLinearGraphQLProjectTracker:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()

    # -- domain operations (ProtocolProjectTracker) --

    async def get_issue(self, issue_id: str) -> ModelStubIssue:
        data = await self._execute(
            _QUERY_GET_ISSUE,
            operation="get_issue",
            variables={"id": issue_id},
        )
        issue = data.get("issue") if isinstance(data, dict) else None
        if not isinstance(issue, dict) or not issue:
            raise KeyError(f"Issue not found: {issue_id}")
        return _issue_from_graphql(issue)

    async def list_issues(
        self, filters: dict[str, str] | None = None, limit: int = 50
    ) -> list[ModelStubIssue]:
        graphql_filter: dict[str, object] | None = None
        if filters:
            # Linear's IssueFilter uses object-shaped sub-filters. For the
            # narrow string-filter shape ProtocolProjectTracker exposes,
            # translate {"state": "Done"} → {"state": {"name": {"eq": "Done"}}}
            # and pass through other simple eq-shaped filters where possible.
            graphql_filter = {}
            for key, val in filters.items():
                if key in ("state", "team", "assignee"):
                    graphql_filter[key] = {"name": {"eq": val}}
                else:
                    graphql_filter[key] = {"eq": val}

        data = await self._execute(
            _QUERY_LIST_ISSUES,
            operation="list_issues",
            variables={"first": limit, "filter": graphql_filter},
        )
        nodes = self._extract_nodes(data, "issues")
        return [_issue_from_graphql(n) for n in nodes]

    async def create_issue(
        self,
        title: str,
        description: str,
        labels: list[str] | None = None,
        assignee: str | None = None,
        priority: str | None = None,
        team: str | None = None,
    ) -> ModelStubIssue:
        if team is None:
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.HTTP,
                operation="create_issue",
                target_name=SERVICE_NAME,
            )
            raise InfraConnectionError(
                "Linear create_issue requires a team identifier",
                context=context,
            )

        graphql_input: dict[str, object] = {
            "title": title,
            "description": description,
            "teamId": team,
        }
        if labels is not None:
            graphql_input["labelIds"] = labels
        if assignee is not None:
            graphql_input["assigneeId"] = assignee
        if priority is not None:
            try:
                graphql_input["priority"] = int(priority)
            except (TypeError, ValueError):
                # Linear priority is an int (0..4); leave caller responsible
                # for valid values. Fall through with the raw value so the
                # API surfaces the validation error rather than swallowing it.
                graphql_input["priority"] = priority

        data = await self._execute(
            _MUTATION_CREATE_ISSUE,
            operation="create_issue",
            variables={"input": graphql_input},
        )
        wrapper = data.get("issueCreate") if isinstance(data, dict) else None
        if not isinstance(wrapper, dict) or not wrapper.get("success"):
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.HTTP,
                operation="create_issue",
                target_name=SERVICE_NAME,
            )
            raise InfraConnectionError(
                "Linear issueCreate mutation returned success=false",
                context=context,
            )
        issue = wrapper.get("issue")
        if not isinstance(issue, dict):
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.HTTP,
                operation="create_issue",
                target_name=SERVICE_NAME,
            )
            raise InfraConnectionError(
                "Linear issueCreate mutation returned no issue object",
                context=context,
            )
        return _issue_from_graphql(issue)

    async def update_issue(
        self, issue_id: str, updates: dict[str, str]
    ) -> ModelStubIssue:
        # Translate caller-friendly keys to Linear input fields. The
        # protocol's ``updates: dict[str, str]`` shape is intentionally
        # narrow; pass through unknown keys verbatim so the API surfaces
        # any schema validation errors.
        graphql_input: dict[str, object] = dict(updates)

        data = await self._execute(
            _MUTATION_UPDATE_ISSUE,
            operation="update_issue",
            variables={"id": issue_id, "input": graphql_input},
        )
        wrapper = data.get("issueUpdate") if isinstance(data, dict) else None
        if not isinstance(wrapper, dict) or not wrapper.get("success"):
            raise KeyError(f"Issue not found or update failed: {issue_id}")
        issue = wrapper.get("issue")
        if not isinstance(issue, dict):
            raise KeyError(f"Issue not found: {issue_id}")
        return _issue_from_graphql(issue)

    async def search_issues(self, query: str, limit: int = 50) -> list[ModelStubIssue]:
        data = await self._execute(
            _QUERY_SEARCH_ISSUES,
            operation="search_issues",
            variables={"term": query, "first": limit},
        )
        nodes = self._extract_nodes(data, "searchIssues")
        return [_issue_from_graphql(n) for n in nodes]

    async def add_comment(self, issue_id: str, body: str) -> ModelStubComment:
        data = await self._execute(
            _MUTATION_ADD_COMMENT,
            operation="add_comment",
            variables={"input": {"issueId": issue_id, "body": body}},
        )
        wrapper = data.get("commentCreate") if isinstance(data, dict) else None
        if not isinstance(wrapper, dict) or not wrapper.get("success"):
            raise KeyError(f"Issue not found: {issue_id}")
        comment = wrapper.get("comment")
        if not isinstance(comment, dict):
            raise KeyError(f"Issue not found: {issue_id}")
        return _comment_from_graphql(comment)

    async def get_project(self, project_id: str) -> ModelStubProject:
        data = await self._execute(
            _QUERY_GET_PROJECT,
            operation="get_project",
            variables={"id": project_id},
        )
        project = data.get("project") if isinstance(data, dict) else None
        if not isinstance(project, dict) or not project:
            raise KeyError(f"Project not found: {project_id}")
        return _project_from_graphql(project)

    async def list_projects(self, limit: int = 50) -> list[ModelStubProject]:
        data = await self._execute(
            _QUERY_LIST_PROJECTS,
            operation="list_projects",
            variables={"first": limit},
        )
        nodes = self._extract_nodes(data, "projects")
        return [_project_from_graphql(n) for n in nodes]

    # -- internal --

    @staticmethod
    def _extract_nodes(data: object, root_key: str) -> list[dict[str, object]]:
        if not isinstance(data, dict):
            return []
        container = data.get(root_key)
        if not isinstance(container, dict):
            return []
        nodes = container.get("nodes")
        if not isinstance(nodes, list):
            return []
        return [n for n in nodes if isinstance(n, dict)]

    async def _execute(
        self,
        query: str,
        operation: str,
        variables: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Execute a GraphQL operation through the circuit breaker.

        Maps transport-level failures to the typed Infra* error hierarchy
        with sanitized context (no API key in any error message).
        """
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(operation=operation)

        payload: dict[str, object] = {"query": query}
        if variables is not None:
            payload["variables"] = variables

        try:
            response = await self._client.post(self._endpoint, json=payload)
        except httpx.TimeoutException as exc:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(operation=operation)
            timeout_ctx = ModelTimeoutErrorContext(
                transport_type=EnumInfraTransportType.HTTP,
                operation=operation,
                target_name=SERVICE_NAME,
                timeout_seconds=self._timeout_seconds,
            )
            raise InfraTimeoutError(
                f"Linear GraphQL request timed out: {operation}",
                context=timeout_ctx,
            ) from exc
        except httpx.HTTPError as exc:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(operation=operation)
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.HTTP,
                operation=operation,
                target_name=SERVICE_NAME,
            )
            raise InfraConnectionError(
                f"Linear GraphQL transport error: {operation}",
                context=context,
            ) from exc

        # Handle HTTP-level failures.
        status = response.status_code
        if status in (401, 403):
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(operation=operation)
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.HTTP,
                operation=operation,
                target_name=SERVICE_NAME,
            )
            raise InfraAuthenticationError(
                f"Linear API rejected credential (HTTP {status})",
                context=context,
                auth_method="api_key",
            )
        if status == 429:
            async with self._circuit_breaker_lock:
                # Rate-limit responses do NOT count toward the circuit-breaker
                # failure budget — they are an expected backpressure signal.
                pass
            retry_after_value = response.headers.get("Retry-After")
            retry_after_seconds: float | None
            try:
                retry_after_seconds = (
                    float(retry_after_value) if retry_after_value else None
                )
            except (TypeError, ValueError):
                retry_after_seconds = None
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.HTTP,
                operation=operation,
                target_name=SERVICE_NAME,
            )
            raise InfraRateLimitedError(
                "Linear API rate limit exceeded",
                context=context,
                retry_after_seconds=retry_after_seconds,
            )
        if status >= 500 or status >= 400:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(operation=operation)
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.HTTP,
                operation=operation,
                target_name=SERVICE_NAME,
            )
            raise InfraConnectionError(
                f"Linear GraphQL HTTP {status}",
                context=context,
            )

        # Decode body and surface GraphQL-level errors (200 OK + errors[]).
        try:
            body = response.json()
        except ValueError as exc:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(operation=operation)
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.HTTP,
                operation=operation,
                target_name=SERVICE_NAME,
            )
            raise InfraConnectionError(
                "Linear GraphQL response was not valid JSON",
                context=context,
            ) from exc

        if not isinstance(body, dict):
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(operation=operation)
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.HTTP,
                operation=operation,
                target_name=SERVICE_NAME,
            )
            raise InfraConnectionError(
                "Linear GraphQL response was not a JSON object",
                context=context,
            )

        errors = body.get("errors")
        if errors:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(operation=operation)
            # Sanitize: surface only the first message and its extensions code.
            first_msg = "unknown"
            if isinstance(errors, list) and errors and isinstance(errors[0], dict):
                first_msg = str(errors[0].get("message", "unknown"))
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.HTTP,
                operation=operation,
                target_name=SERVICE_NAME,
            )
            raise InfraConnectionError(
                f"Linear GraphQL error: {first_msg}",
                context=context,
            )

        async with self._circuit_breaker_lock:
            await self._reset_circuit_breaker()

        data = body.get("data")
        if not isinstance(data, dict):
            return {}
        return cast("dict[str, object]", data)


__all__ = [
    "AdapterLinearGraphQLProjectTracker",
    "LinearGraphQLHealthStatus",
]
