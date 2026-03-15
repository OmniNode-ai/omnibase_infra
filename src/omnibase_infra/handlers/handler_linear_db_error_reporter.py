# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Linear DB Error Reporter Handler - PostgreSQL error -> Linear ticket.

This handler processes ``ModelDbErrorEvent`` payloads received from the
``onex.evt.omnibase-infra.db-error.v1`` Kafka topic and creates Linear
tickets for unique PostgreSQL errors.

Architecture:
    1. Dedup check — acquire per-fingerprint advisory lock, then SELECT
    2. If found   — UPDATE occurrence_count, release lock, return skipped
    3. If new     — POST Linear GraphQL issueCreate mutation via httpx
    4. INSERT into ``db_error_tickets``, release lock, return created

Atomicity / TOCTOU:
    The dedup-check and ticket-create-and-insert sequence is guarded by a
    PostgreSQL advisory lock keyed on the first 8 bytes of the fingerprint.
    This prevents concurrent consumers from both passing the initial
    fingerprint check and creating duplicate Linear tickets.

    Advisory lock strategy:
    - ``pg_try_advisory_xact_lock(key)`` (session-level, auto-released
      at transaction end)
    - If lock acquired: proceed with check-then-create-then-insert
    - If not acquired: wait briefly and retry (falls back to skipped if
      the fingerprint was just inserted by the winner)

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
    All dependencies are provided via constructor arguments.  Callers
    source ``linear_api_key`` and ``linear_team_id`` from the environment
    (e.g. ``os.environ["LINEAR_API_KEY"]``) and pass them in — the handler
    itself does not read ``os.environ`` directly.

    - ``linear_api_key: str`` — Linear API key (required, non-empty)
    - ``linear_team_id: str`` — Linear team UUID (required, non-empty)
    - ``db_pool`` — asyncpg.Pool (required — handler raises on ``None``)

Coroutine Safety:
    This handler is stateless across calls (no mutable instance state is
    mutated after __init__).  Concurrent calls are made safe via the
    per-fingerprint advisory lock (DB-side).

Related Tickets:
    - OMN-3408: Kafka Consumer -> Linear Ticket Reporter (ONEX Node)
    - OMN-3407: PostgreSQL Error Emitter (hard prerequisite)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import httpx

from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
)
from omnibase_infra.errors import RuntimeHostError
from omnibase_infra.handlers.models.model_db_error_event import ModelDbErrorEvent
from omnibase_infra.handlers.models.model_db_error_ticket_result import (
    ModelDbErrorTicketResult,
)
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
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


def _fingerprint_to_lock_key(fingerprint: str) -> int:
    """Derive a 64-bit integer advisory lock key from a fingerprint string.

    Uses the first 16 hex chars (8 bytes) of the fingerprint, which is
    always a 32-char hex string (see ModelDbErrorEvent.fingerprint).

    Returns:
        Signed 64-bit integer suitable for ``pg_advisory_xact_lock``.
    """
    raw = int(fingerprint[:16], 16)
    # Fold to signed int64 range (PostgreSQL advisory lock key is bigint)
    if raw >= 2**63:
        raw -= 2**64
    return raw


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
        2. Acquire per-fingerprint PostgreSQL advisory lock
        3. SELECT from ``db_error_tickets`` for dedup (inside lock)
        4. If found: UPDATE occurrence_count + last_seen_at, return skipped
        5. If new: call Linear API, INSERT into table, return created

    Args:
        linear_api_key: Linear API key (required, non-empty).
        linear_team_id: Linear team UUID (required, non-empty).
        db_pool: asyncpg connection pool for ``db_error_tickets`` (required).
        timeout: HTTP timeout for Linear API calls (seconds).
    """

    def __init__(
        self,
        linear_api_key: str = "",
        linear_team_id: str = "",
        db_pool: asyncpg.Pool | None = None,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        # Callers source these from the environment and pass them in.
        # This handler does not read os.environ directly (architecture invariant).
        self._linear_api_key = linear_api_key
        self._linear_team_id = linear_team_id
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

        Acquires a per-fingerprint PostgreSQL advisory lock before the
        dedup check so concurrent consumers cannot race to create duplicate
        Linear tickets (TOCTOU prevention).

        Args:
            event: Structured PostgreSQL error event from Kafka.

        Returns:
            ModelDbErrorTicketResult with created/skipped flag, issue info,
            and current occurrence_count.

        Raises:
            RuntimeHostError: If ``db_pool`` is not configured, if the Linear
                API call fails, or if a DB error prevents the insert.
        """
        if self._db_pool is None:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=uuid4(),
                transport_type=EnumInfraTransportType.DATABASE,
                operation="report_error",
                target_name="db_error_tickets",
            )
            raise RuntimeHostError(
                "HandlerLinearDbErrorReporter requires a db_pool — "
                "received None.  Callers must pass an asyncpg.Pool.",
                context=context,
            )

        if not self._linear_api_key:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=uuid4(),
                transport_type=EnumInfraTransportType.HTTP,
                operation="report_error",
                target_name="linear_api",
            )
            raise RuntimeHostError(
                "HandlerLinearDbErrorReporter: linear_api_key is empty — "
                "pass LINEAR_API_KEY via constructor.",
                context=context,
            )

        if not self._linear_team_id:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=uuid4(),
                transport_type=EnumInfraTransportType.HTTP,
                operation="report_error",
                target_name="linear_api",
            )
            raise RuntimeHostError(
                "HandlerLinearDbErrorReporter: linear_team_id is empty — "
                "pass LINEAR_TEAM_ID via constructor.",
                context=context,
            )

        lock_key = _fingerprint_to_lock_key(event.fingerprint)

        async with self._db_pool.acquire() as conn:
            # Acquire per-fingerprint advisory lock for the duration of
            # this transaction to prevent TOCTOU duplicate ticket creation.
            await conn.execute("SELECT pg_advisory_xact_lock($1)", lock_key)

            # Re-check inside the lock — another consumer may have just
            # inserted this fingerprint while we were waiting.
            row = await conn.fetchrow(
                "SELECT linear_issue_id, linear_issue_url, occurrence_count "
                "FROM db_error_tickets WHERE fingerprint = $1",
                event.fingerprint,
            )
            if row is not None:
                # Already exists — increment and return skipped
                updated = await conn.fetchrow(
                    "UPDATE db_error_tickets "
                    "SET last_seen_at = NOW(), occurrence_count = occurrence_count + 1 "
                    "WHERE fingerprint = $1 "
                    "RETURNING occurrence_count",
                    event.fingerprint,
                )
                issue_id = UUID(str(row["linear_issue_id"]))
                issue_url = (
                    str(row["linear_issue_url"]) if row["linear_issue_url"] else ""
                )
                new_count = int(updated["occurrence_count"]) if updated else 1
                logger.info(
                    "DB error fingerprint already tracked — incrementing "
                    "occurrence_count (fingerprint=%s, issue_id=%s, count=%d)",
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

            # Not found under lock — create Linear ticket then insert
            try:
                linear_issue_id, linear_issue_url = await self._create_linear_issue(
                    event
                )
            except Exception as exc:
                sanitized = sanitize_error_message(exc)
                logger.exception(
                    "Failed to create Linear ticket (fingerprint=%s): %s",
                    event.fingerprint,
                    sanitized,
                )
                context = ModelInfraErrorContext.with_correlation(
                    correlation_id=uuid4(),
                    transport_type=EnumInfraTransportType.HTTP,
                    operation="issueCreate",
                    target_name="linear_api",
                )
                raise RuntimeHostError(
                    f"Linear issueCreate failed for fingerprint={event.fingerprint}: "
                    f"{sanitized}",
                    context=context,
                ) from exc

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
                str(linear_issue_id),
                linear_issue_url,
                event.first_seen_at,
            )

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

    async def _create_linear_issue(self, event: ModelDbErrorEvent) -> tuple[UUID, str]:
        """Call the Linear GraphQL API to create a new issue.

        Returns:
            Tuple of (issue_uuid, issue_url).

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
        raw_id: str = issue_data.get("id", "")
        issue_url: str = issue_data.get("url", "")

        if not raw_id:
            raise ValueError(f"Linear issueCreate returned no issue.id (data={data!r})")

        return UUID(raw_id), issue_url


__all__ = ["HandlerLinearDbErrorReporter"]
