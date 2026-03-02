# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Linear DB Error Reporter Handler - PostgreSQL error -> Linear ticket.

This handler processes ``ModelDbErrorEvent`` payloads received from the
``onex.evt.omnibase-infra.db-error.v1`` Kafka topic and creates Linear
tickets for unique PostgreSQL errors.

Architecture:
    1. Dedup check — SELECT from ``db_error_tickets`` WHERE fingerprint = $1
    2. If found   — UPDATE occurrence_count, return skipped=True
    3. If new     — POST Linear GraphQL issueCreate mutation via httpx
    4. INSERT into ``db_error_tickets``
    5. Return     — ModelDbErrorTicketResult(created=True, ...)

Deduplication:
    Each PostgreSQL error is identified by a 32-char SHA-256 fingerprint
    computed from the normalised error fields (see ModelDbErrorEvent).
    A fingerprint that already exists in ``db_error_tickets`` does NOT
    generate a new Linear ticket — it only increments ``occurrence_count``
    and updates ``last_seen_at``.

Linear API:
    Uses the Linear GraphQL ``issueCreate`` mutation via httpx.
    Auth header: ``Authorization: {linear_api_key}``  (no "Bearer" prefix —
    matches the pattern in docs/tools/generate_ticket_plan.py).

Constructor Injection:
    - ``linear_api_key: str`` — from env ``LINEAR_API_KEY``
    - ``linear_team_id: str`` — from env ``LINEAR_TEAM_ID``
    - ``db_pool`` — asyncpg.Pool (injected by caller / plugin)

Coroutine Safety:
    This handler is stateless across calls (no mutable instance state is
    mutated after __init__).  Concurrent calls with different payloads are
    safe.

Related Tickets:
    - OMN-3408: Kafka Consumer -> Linear Ticket Reporter (ONEX Node)
    - OMN-3407: PostgreSQL Error Emitter (hard prerequisite)
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import httpx

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.handlers.models.model_db_error_event import ModelDbErrorEvent
from omnibase_infra.handlers.models.model_db_error_ticket_result import (
    ModelDbErrorTicketResult,
)
from omnibase_infra.utils.util_error_sanitization import sanitize_error_message

if TYPE_CHECKING:
    import asyncpg

logger = logging.getLogger(__name__)

# Linear GraphQL endpoint
_LINEAR_API_URL: str = "https://api.linear.app/graphql"

# Default timeout for Linear API calls (seconds)
_DEFAULT_TIMEOUT_SECONDS: float = 15.0

# Max length for error_message in the Linear ticket title
_MAX_TITLE_MESSAGE_LENGTH: int = 80

# GraphQL mutation used for issue creation
_ISSUE_CREATE_MUTATION: str = """
mutation CreateIssue($teamId: String!, $title: String!, $description: String!, $priority: Int!) {
  issueCreate(input: {
    teamId: $teamId
    title: $title
    description: $description
    priority: $priority
  }) {
    issue {
      id
      identifier
      url
    }
  }
}
"""


def _build_ticket_title(event: ModelDbErrorEvent) -> str:
    """Build the Linear ticket title from a db error event.

    Format: ``[DB ERROR] {error_code or 'UNKNOWN'}: {short_message} ({table_name or 'unknown table'})``
    """
    code = event.error_code or "UNKNOWN"
    table = event.table_name or "unknown table"
    short_message = event.error_message
    if len(short_message) > _MAX_TITLE_MESSAGE_LENGTH:
        short_message = short_message[:_MAX_TITLE_MESSAGE_LENGTH] + "..."
    return f"[DB ERROR] {code}: {short_message} ({table})"


def _build_ticket_description(event: ModelDbErrorEvent) -> str:
    """Build the Linear ticket description from a db error event."""
    sql_block = event.sql_statement or "(not captured)"
    first_seen = event.first_seen_at.isoformat()
    return (
        "## PostgreSQL Error\n\n"
        f"**Error**: {event.error_message}\n"
        f"**Hint**: {event.hint or 'none'}\n"
        f"**SQL**:\n```sql\n{sql_block}\n```\n\n"
        f"**Table**: {event.table_name or 'unknown'}\n"
        f"**Service**: {event.service}\n"
        f"**First seen**: {first_seen}\n"
        f"**Fingerprint**: {event.fingerprint}\n"
    )


class HandlerLinearDbErrorReporter:
    """Handler that creates Linear tickets for unique PostgreSQL errors.

    Implements the ``report_error`` operation declared in
    ``node_db_error_linear_effect/contract.yaml``.

    Lifecycle:
        1. ``handle(event)`` called for each Kafka message
        2. SELECT from ``db_error_tickets`` for dedup
        3. If found: UPDATE occurrence_count + last_seen_at, return skipped
        4. If new: call Linear API, INSERT into table, return created

    Args:
        linear_api_key: Linear API key (``LINEAR_API_KEY`` env var).
        linear_team_id: Linear team UUID (``LINEAR_TEAM_ID`` env var).
        db_pool: asyncpg connection pool for ``db_error_tickets``.
        timeout: HTTP timeout for Linear API calls (seconds).
    """

    def __init__(
        self,
        linear_api_key: str | None = None,
        linear_team_id: str | None = None,
        db_pool: asyncpg.Pool | None = None,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._linear_api_key: str = linear_api_key or os.environ.get(
            "LINEAR_API_KEY", ""
        )
        self._linear_team_id: str = linear_team_id or os.environ.get(
            "LINEAR_TEAM_ID", ""
        )
        self._db_pool = db_pool
        self._timeout = timeout

    @property
    def handler_type(self) -> EnumHandlerType:
        """Infrastructure handler — performs external I/O (Linear API + Postgres)."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Effect category — side-effecting I/O operations."""
        return EnumHandlerTypeCategory.EFFECT

    async def handle(self, event: ModelDbErrorEvent) -> ModelDbErrorTicketResult:
        """Process a db error event: dedup check -> Linear create -> DB insert.

        Args:
            event: Structured PostgreSQL error event from Kafka.

        Returns:
            ModelDbErrorTicketResult with created/skipped flag, issue info,
            and current occurrence_count.
        """
        if not self._linear_api_key:
            logger.warning(
                "LINEAR_API_KEY not set — skipping Linear ticket creation "
                "(fingerprint=%s)",
                event.fingerprint,
            )
            return ModelDbErrorTicketResult(
                error="LINEAR_API_KEY not configured",
            )

        if not self._linear_team_id:
            logger.warning(
                "LINEAR_TEAM_ID not set — skipping Linear ticket creation "
                "(fingerprint=%s)",
                event.fingerprint,
            )
            return ModelDbErrorTicketResult(
                error="LINEAR_TEAM_ID not configured",
            )

        # 1. Dedup check
        existing = await self._lookup_fingerprint(event.fingerprint)
        if existing is not None:
            issue_id, issue_url, current_count = existing
            new_count = await self._increment_occurrence(event.fingerprint)
            logger.info(
                "DB error fingerprint already tracked — incrementing occurrence_count "
                "(fingerprint=%s, issue_id=%s, occurrence_count=%d)",
                event.fingerprint,
                issue_id,
                new_count,
            )
            return ModelDbErrorTicketResult(
                skipped=True,
                issue_id=issue_id,
                issue_url=issue_url,
                occurrence_count=new_count,
            )

        # 2. Create Linear ticket
        try:
            linear_issue_id, linear_issue_url = await self._create_linear_issue(event)
        except Exception as exc:
            sanitized = sanitize_error_message(exc)
            logger.error(
                "Failed to create Linear ticket (fingerprint=%s): %s",
                event.fingerprint,
                sanitized,
            )
            return ModelDbErrorTicketResult(error=sanitized)

        # 3. Insert into db_error_tickets
        await self._insert_db_error_ticket(event, linear_issue_id, linear_issue_url)

        logger.info(
            "Created Linear ticket for db error (fingerprint=%s, issue_id=%s, url=%s)",
            event.fingerprint,
            linear_issue_id,
            linear_issue_url,
        )
        return ModelDbErrorTicketResult(
            created=True,
            issue_id=linear_issue_id,
            issue_url=linear_issue_url,
            occurrence_count=1,
        )

    async def _lookup_fingerprint(
        self,
        fingerprint: str,
    ) -> tuple[str, str | None, int] | None:
        """SELECT from db_error_tickets for the given fingerprint.

        Returns:
            Tuple of (linear_issue_id, linear_issue_url, occurrence_count)
            if found, else None.
        """
        if self._db_pool is None:
            logger.debug(
                "No db_pool — skipping dedup lookup (fingerprint=%s)", fingerprint
            )
            return None

        async with self._db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT linear_issue_id, linear_issue_url, occurrence_count "
                "FROM db_error_tickets WHERE fingerprint = $1",
                fingerprint,
            )
        if row is None:
            return None
        return (
            str(row["linear_issue_id"]),
            str(row["linear_issue_url"]) if row["linear_issue_url"] else None,
            int(row["occurrence_count"]),
        )

    async def _increment_occurrence(self, fingerprint: str) -> int:
        """UPDATE occurrence_count and last_seen_at for an existing fingerprint.

        Returns:
            New occurrence_count after increment.
        """
        if self._db_pool is None:
            return 1

        async with self._db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "UPDATE db_error_tickets "
                "SET last_seen_at = NOW(), occurrence_count = occurrence_count + 1 "
                "WHERE fingerprint = $1 "
                "RETURNING occurrence_count",
                fingerprint,
            )
        return int(row["occurrence_count"]) if row else 1

    async def _create_linear_issue(self, event: ModelDbErrorEvent) -> tuple[str, str]:
        """Call the Linear GraphQL API to create a new issue.

        Returns:
            Tuple of (issue_id, issue_url).

        Raises:
            httpx.HTTPStatusError: On 4xx/5xx responses from Linear API.
            ValueError: When the GraphQL response is missing expected fields.
        """
        title = _build_ticket_title(event)
        description = _build_ticket_description(event)

        payload = {
            "query": _ISSUE_CREATE_MUTATION,
            "variables": {
                "teamId": self._linear_team_id,
                "title": title,
                "description": description,
                "priority": 3,  # Normal priority
            },
        }

        # Auth header: no "Bearer" prefix — matches generate_ticket_plan.py pattern
        headers = {
            "Authorization": self._linear_api_key,
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(_LINEAR_API_URL, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        errors = data.get("errors")
        if errors:
            raise ValueError(f"Linear API returned GraphQL errors: {errors}")

        issue_data = data.get("data", {}).get("issueCreate", {}).get("issue", {})
        issue_id: str = issue_data.get("id", "")
        issue_url: str = issue_data.get("url", "")

        if not issue_id:
            raise ValueError(f"Linear issueCreate returned no issue.id (data={data!r})")

        return issue_id, issue_url

    async def _insert_db_error_ticket(
        self,
        event: ModelDbErrorEvent,
        linear_issue_id: str,
        linear_issue_url: str,
    ) -> None:
        """INSERT a new row into db_error_tickets.

        No-ops gracefully if db_pool is None (logs a warning).
        """
        if self._db_pool is None:
            logger.warning(
                "No db_pool — cannot persist db_error_ticket record "
                "(fingerprint=%s, issue_id=%s)",
                event.fingerprint,
                linear_issue_id,
            )
            return

        async with self._db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO db_error_tickets
                    (fingerprint, error_code, error_message, table_name, service,
                     linear_issue_id, linear_issue_url, occurrence_count,
                     first_seen_at, last_seen_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, 1, $8, NOW())
                ON CONFLICT (fingerprint) DO NOTHING
                """,
                event.fingerprint,
                event.error_code,
                event.error_message,
                event.table_name,
                event.service,
                linear_issue_id,
                linear_issue_url,
                event.first_seen_at,
            )


__all__ = ["HandlerLinearDbErrorReporter"]
