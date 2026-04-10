# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Linear API adapter implementing ProtocolTicketService.

Provides read operations (get_ticket, list_tickets, get_ticket_status,
search via list_tickets filters) against the Linear GraphQL API.
Write operations (create_ticket, update_ticket_status, add_comment)
raise NotImplementedError — they will be wired in a future ticket.

Constructor Injection:
    ``linear_api_key`` is required and must be provided by the caller
    (e.g. from ``os.environ["LINEAR_API_KEY"]``). The adapter does NOT
    read ``os.environ`` directly.

Auth:
    ``Authorization: {linear_api_key}`` header — no "Bearer" prefix,
    matching the established pattern in handler_linear_db_error_reporter.py.

Related Tickets:
    - OMN-7587: Create Linear API adapter implementing ProtocolTicketService
    - OMN-7575: Build loop migration epic
"""

from __future__ import annotations

from uuid import uuid4

import httpx

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraConnectionError,
    InfraUnavailableError,
)
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)

# Metadata dict type for unimplemented create_ticket stub.
# ONEX_EXCLUDE: dict_str_any - unimplemented stub; no domain type exists yet
_TicketMetadata = dict[str, object]

# Linear GraphQL endpoint
_LINEAR_API_URL: str = "https://api.linear.app/graphql"

# Default timeout for Linear API calls (seconds)
_DEFAULT_TIMEOUT_SECONDS: float = 15.0

# ---------------------------------------------------------------------------
# GraphQL queries
# ---------------------------------------------------------------------------

_ISSUE_QUERY: str = """
query GetIssue($id: String!) {
  issue(id: $id) {
    id
    identifier
    title
    description
    url
    priority
    state { name }
    assignee { name email }
    labels { nodes { name } }
    createdAt
    updatedAt
  }
}
"""

_ISSUES_LIST_QUERY: str = """
query ListIssues($teamId: String, $first: Int!, $filter: IssueFilter) {
  issues(first: $first, filter: $filter) {
    nodes {
      id
      identifier
      title
      description
      url
      priority
      state { name }
      assignee { name email }
      labels { nodes { name } }
      createdAt
      updatedAt
    }
  }
}
"""

_ISSUE_BY_IDENTIFIER_QUERY: str = """
query GetIssueByIdentifier($identifier: String!) {
  issueSearch(query: $identifier, first: 1) {
    nodes {
      id
      identifier
      title
      description
      url
      priority
      state { name }
      assignee { name email }
      labels { nodes { name } }
      createdAt
      updatedAt
    }
  }
}
"""


def _normalise_issue(raw: dict[str, object]) -> dict[str, object]:
    """Flatten a Linear GraphQL issue node into a ContextValue-compatible dict.

    Converts nested objects (state, assignee, labels) to flat string values
    so callers get a uniform dict[str, str | list[str] | None].
    """
    state_raw = raw.get("state")
    assignee_raw = raw.get("assignee")
    labels_raw = raw.get("labels", {})

    return {
        "id": raw.get("id", ""),
        "identifier": raw.get("identifier", ""),
        "title": raw.get("title", ""),
        "description": raw.get("description", ""),
        "url": raw.get("url", ""),
        "priority": raw.get("priority"),
        "status": state_raw.get("name", "") if isinstance(state_raw, dict) else "",
        "assignee_name": (
            assignee_raw.get("name", "") if isinstance(assignee_raw, dict) else None
        ),
        "assignee_email": (
            assignee_raw.get("email", "") if isinstance(assignee_raw, dict) else None
        ),
        "labels": [
            n.get("name", "")
            for n in (
                labels_raw.get("nodes", []) if isinstance(labels_raw, dict) else []
            )
        ],
        "created_at": raw.get("createdAt", ""),
        "updated_at": raw.get("updatedAt", ""),
    }


def _build_issue_filter(filters: dict[str, object]) -> dict[str, object]:
    """Translate simple key-value filters to Linear IssueFilter object.

    Supported filter keys:
        - status: str  -> state.name.eq
        - assignee: str -> assignee.name.eq
        - label: str -> labels.name.eq
        - team: str -> team.key.eq
    """
    gql_filter: dict[str, object] = {}

    if "status" in filters:
        gql_filter["state"] = {"name": {"eq": filters["status"]}}
    if "assignee" in filters:
        gql_filter["assignee"] = {"name": {"eq": filters["assignee"]}}
    if "label" in filters:
        gql_filter["labels"] = {"name": {"eq": filters["label"]}}
    if "team" in filters:
        gql_filter["team"] = {"key": {"eq": filters["team"]}}

    return gql_filter


class AdapterTicketLinear:
    """Linear API adapter implementing ProtocolTicketService.

    Read methods (get_ticket, list_tickets, get_ticket_status, health_check)
    call the Linear GraphQL API. Write methods (create_ticket,
    update_ticket_status, add_comment) raise NotImplementedError.

    Args:
        linear_api_key: Linear API key (required, non-empty).
        timeout: HTTP timeout for Linear API calls in seconds.
    """

    def __init__(
        self,
        linear_api_key: str,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        if not linear_api_key:
            raise ValueError("AdapterTicketLinear requires a non-empty linear_api_key")
        self._api_key = linear_api_key
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        """Build HTTP headers for Linear API calls."""
        return {
            "Authorization": self._api_key,
            "Content-Type": "application/json",
        }

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazily create or return the shared httpx client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def _execute_graphql(
        self,
        query: str,
        variables: dict[str, object],
        operation: str,
    ) -> dict[str, object]:
        """Execute a GraphQL query against the Linear API.

        Args:
            query: GraphQL query string.
            variables: Query variables.
            operation: Operation name for error context.

        Returns:
            The ``data`` portion of the GraphQL response.

        Raises:
            InfraConnectionError: On network / HTTP errors.
            InfraUnavailableError: On GraphQL-level errors from Linear.
        """
        correlation_id = uuid4()
        client = await self._get_client()

        try:
            response = await client.post(
                _LINEAR_API_URL,
                json={"query": query, "variables": variables},
                headers=self._headers(),
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.HTTP,
                operation=operation,
                target_name="linear_api",
            )
            raise InfraConnectionError(
                f"Linear API returned HTTP {exc.response.status_code} "
                f"for operation={operation}",
                context=context,
            ) from exc
        except httpx.HTTPError as exc:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.HTTP,
                operation=operation,
                target_name="linear_api",
            )
            raise InfraConnectionError(
                f"Linear API connection failed for operation={operation}: {exc}",
                context=context,
            ) from exc

        data = response.json()
        errors = data.get("errors")
        if errors:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.HTTP,
                operation=operation,
                target_name="linear_api",
            )
            raise InfraUnavailableError(
                f"Linear GraphQL errors for operation={operation}: {errors}",
                context=context,
            )

        result: dict[str, object] = data.get("data", {})
        return result

    # ------------------------------------------------------------------
    # ProtocolTicketService — read methods
    # ------------------------------------------------------------------

    async def get_ticket(self, ticket_id: str) -> dict[str, object]:
        """Retrieve ticket details by identifier.

        Accepts both Linear UUIDs and human-readable identifiers
        (e.g. "OMN-1234"). For human-readable identifiers, uses
        issueSearch; for UUIDs, uses the issue query directly.

        Args:
            ticket_id: Linear issue UUID or human-readable identifier.

        Returns:
            Normalised ticket dict.

        Raises:
            KeyError: If the ticket is not found.
            InfraConnectionError: On network errors.
        """
        # Heuristic: Linear UUIDs are 36-char strings with dashes
        is_uuid = len(ticket_id) == 36 and "-" in ticket_id

        if is_uuid:
            data = await self._execute_graphql(
                _ISSUE_QUERY,
                {"id": ticket_id},
                "get_ticket",
            )
            issue = data.get("issue")
            if not issue or not isinstance(issue, dict):
                raise KeyError(f"Ticket not found: {ticket_id}")
            return _normalise_issue(issue)

        # Human-readable identifier — use search
        data = await self._execute_graphql(
            _ISSUE_BY_IDENTIFIER_QUERY,
            {"identifier": ticket_id},
            "get_ticket_by_identifier",
        )
        search_result = data.get("issueSearch")
        if not isinstance(search_result, dict):
            raise KeyError(f"Ticket not found: {ticket_id}")
        nodes_raw = search_result.get("nodes", [])
        if not isinstance(nodes_raw, list) or not nodes_raw:
            raise KeyError(f"Ticket not found: {ticket_id}")
        first = nodes_raw[0]
        if not isinstance(first, dict):
            raise KeyError(f"Ticket not found: {ticket_id}")
        return _normalise_issue(first)

    async def list_tickets(
        self,
        filters: dict[str, object] | None = None,
        limit: int = 50,
    ) -> list[dict[str, object]]:
        """List tickets matching optional filters.

        Args:
            filters: Optional filter criteria. Supported keys:
                status, assignee, label, team.
            limit: Maximum number of tickets to return.

        Returns:
            List of normalised ticket dicts.
        """
        variables: dict[str, object] = {"first": min(limit, 250)}
        if filters:
            gql_filter = _build_issue_filter(filters)
            if gql_filter:
                variables["filter"] = gql_filter

        data = await self._execute_graphql(
            _ISSUES_LIST_QUERY,
            variables,
            "list_tickets",
        )
        issues_result = data.get("issues")
        if not isinstance(issues_result, dict):
            return []
        nodes_raw = issues_result.get("nodes", [])
        if not isinstance(nodes_raw, list):
            return []
        return [_normalise_issue(n) for n in nodes_raw if isinstance(n, dict)]

    async def get_ticket_status(self, ticket_id: str) -> str:
        """Get the current status of a ticket.

        Args:
            ticket_id: Linear issue UUID or human-readable identifier.

        Returns:
            Status name string (e.g. "In Progress", "Done").

        Raises:
            KeyError: If ticket not found.
        """
        ticket = await self.get_ticket(ticket_id)
        status = ticket.get("status", "")
        if not status:
            raise KeyError(f"Ticket {ticket_id} has no status")
        return str(status)

    async def health_check(self) -> bool:
        """Check if the Linear API is reachable.

        Returns:
            True if reachable, False otherwise.
        """
        try:
            client = await self._get_client()
            response = await client.post(
                _LINEAR_API_URL,
                json={
                    "query": "query { viewer { id } }",
                    "variables": {},
                },
                headers=self._headers(),
            )
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    # ------------------------------------------------------------------
    # ProtocolTicketService — write methods (not yet implemented)
    # ------------------------------------------------------------------

    async def create_ticket(
        self,
        title: str,
        description: str,
        labels: list[str] | None = None,
        assignee: str | None = None,
        metadata: _TicketMetadata | None = None,
    ) -> str:
        """Create a new ticket. Not yet implemented.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(  # stub-ok: write methods deferred to OMN-7587
            "AdapterTicketLinear.create_ticket is not yet implemented "
            "(OMN-7587: write methods deferred)"
        )

    async def update_ticket_status(self, ticket_id: str, status: str) -> bool:
        """Update ticket status. Not yet implemented.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(  # stub-ok: write methods deferred to OMN-7587
            "AdapterTicketLinear.update_ticket_status is not yet implemented "
            "(OMN-7587: write methods deferred)"
        )

    async def add_comment(self, ticket_id: str, body: str) -> str:
        """Add a comment to a ticket. Not yet implemented.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(  # stub-ok: write methods deferred to OMN-7587
            "AdapterTicketLinear.add_comment is not yet implemented "
            "(OMN-7587: write methods deferred)"
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self, timeout_seconds: float = 30.0) -> None:
        """Release the httpx client.

        Args:
            timeout_seconds: Maximum time to wait for cleanup.
        """
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


__all__ = ["AdapterTicketLinear"]
