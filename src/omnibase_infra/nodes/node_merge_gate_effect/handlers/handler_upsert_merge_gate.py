# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler for upserting merge gate decisions and opening Linear quarantine tickets.

Behavior:
    1. Upsert the merge gate decision into ``merge_gate_decisions``
       (ON CONFLICT (pr_ref, head_sha) DO UPDATE).
    2. If decision == "QUARANTINE": open a Linear ticket with violation details.
    3. Return ModelBackendResult with success/failure.

SQL:
    INSERT INTO merge_gate_decisions (...)
    ON CONFLICT (pr_ref, head_sha) DO UPDATE SET
        gate_id = EXCLUDED.gate_id,
        base_sha = EXCLUDED.base_sha,
        decision = EXCLUDED.decision,
        tier = EXCLUDED.tier,
        violations = EXCLUDED.violations,
        run_id = EXCLUDED.run_id,
        correlation_id = EXCLUDED.correlation_id,
        run_fingerprint = EXCLUDED.run_fingerprint,
        decided_at = EXCLUDED.decided_at;

Linear Quarantine:
    When decision == QUARANTINE, a Linear issue is created via GraphQL
    ``issueMutation`` with quarantine details. Requires LINEAR_API_KEY
    and LINEAR_TEAM_ID environment variables. If Linear is not configured,
    the quarantine ticket is skipped with a warning log (does not fail
    the overall upsert).

Related Tickets:
    - OMN-3140: NodeMergeGateEffect + migration
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING
from uuid import UUID

import httpx

from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumPostgresErrorCode,
)
from omnibase_infra.mixins.mixin_postgres_op_executor import MixinPostgresOpExecutor
from omnibase_infra.models.model_backend_result import ModelBackendResult

if TYPE_CHECKING:
    import asyncpg

    from omnibase_infra.nodes.node_merge_gate_effect.models.model_merge_gate_result import (
        ModelMergeGateResult,
    )

logger = logging.getLogger(__name__)

# =============================================================================
# SQL Statements
# =============================================================================

SQL_UPSERT_MERGE_GATE = """
INSERT INTO merge_gate_decisions (
    gate_id, pr_ref, head_sha, base_sha,
    decision, tier, violations,
    run_id, correlation_id, run_fingerprint,
    decided_at
) VALUES (
    $1, $2, $3, $4,
    $5, $6, $7::jsonb,
    $8, $9, $10,
    $11
)
ON CONFLICT (pr_ref, head_sha) DO UPDATE SET
    gate_id         = EXCLUDED.gate_id,
    base_sha        = EXCLUDED.base_sha,
    decision        = EXCLUDED.decision,
    tier            = EXCLUDED.tier,
    violations      = EXCLUDED.violations,
    run_id          = EXCLUDED.run_id,
    correlation_id  = EXCLUDED.correlation_id,
    run_fingerprint = EXCLUDED.run_fingerprint,
    decided_at      = EXCLUDED.decided_at
RETURNING (xmax = 0) AS was_insert;
"""

# =============================================================================
# Linear GraphQL
# =============================================================================

LINEAR_GRAPHQL_URL = "https://api.linear.app/graphql"

LINEAR_CREATE_ISSUE_MUTATION = """
mutation CreateQuarantineIssue($title: String!, $teamId: String!, $description: String!, $priority: Int!) {
    issueCreate(input: {
        title: $title,
        teamId: $teamId,
        description: $description,
        priority: $priority
    }) {
        success
        issue {
            id
            identifier
            url
        }
    }
}
"""


# =============================================================================
# Handler
# =============================================================================


class HandlerUpsertMergeGate(MixinPostgresOpExecutor):
    """Upsert merge gate decisions and open Linear quarantine tickets.

    The handler upserts a merge gate decision into the ``merge_gate_decisions``
    table using ON CONFLICT (pr_ref, head_sha) DO UPDATE for idempotency.
    When the decision is QUARANTINE, a Linear ticket is opened with the
    violation details.

    Attributes:
        _pool: asyncpg connection pool for database operations.

    Example:
        >>> pool = await asyncpg.create_pool(dsn="...")
        >>> handler = HandlerUpsertMergeGate(pool)
        >>> result = await handler.handle(payload, correlation_id)
        >>> result.success
        True
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        """Initialise handler with asyncpg connection pool.

        Args:
            pool: asyncpg connection pool. Should be pre-configured and ready.
        """
        self._pool = pool

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role of this handler."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Behavioral classification of this handler."""
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self,
        payload: ModelMergeGateResult,
        correlation_id: UUID,
    ) -> ModelBackendResult:
        """Execute merge gate decision upsert and optional quarantine ticket.

        Args:
            payload: Merge gate decision payload from Kafka event.
            correlation_id: Request correlation ID for distributed tracing.

        Returns:
            ModelBackendResult with success=True when upsert committed.
            Linear ticket failures are logged but do not cause success=False.
        """
        # Resolve a single effective correlation ID used end-to-end
        effective_cid = payload.correlation_id or correlation_id
        return await self._execute_postgres_op(
            op_error_code=EnumPostgresErrorCode.UPSERT_ERROR,
            correlation_id=effective_cid,
            log_context={
                "pr_ref": payload.pr_ref,
                "head_sha": payload.head_sha,
                "decision": payload.decision,
            },
            fn=lambda: self._upsert_and_quarantine(payload, effective_cid),
        )

    async def _upsert_and_quarantine(
        self,
        payload: ModelMergeGateResult,
        correlation_id: UUID,
    ) -> None:
        """Upsert the decision and open quarantine ticket if needed.

        Args:
            payload: Merge gate decision payload.
            correlation_id: Correlation ID for tracing.
        """
        # Serialize violations to JSON
        violations_json = json.dumps(
            [v.model_dump(mode="json") for v in payload.violations]
        )

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                SQL_UPSERT_MERGE_GATE,
                payload.gate_id,
                payload.pr_ref,
                payload.head_sha,
                payload.base_sha,
                payload.decision,
                payload.tier,
                violations_json,
                payload.run_id,
                correlation_id,
                payload.run_fingerprint,
                payload.decided_at,
            )

        was_insert = row["was_insert"] if row else True
        logger.info(
            "Merge gate decision upserted",
            extra={
                "pr_ref": payload.pr_ref,
                "head_sha": payload.head_sha,
                "decision": payload.decision,
                "tier": payload.tier,
                "operation": "insert" if was_insert else "update",
                "correlation_id": str(correlation_id),
            },
        )

        # Open Linear quarantine ticket only on first insert (idempotent).
        # Re-evaluations for the same (pr_ref, head_sha) are updates and
        # must not create duplicate tickets.
        if payload.decision == "QUARANTINE" and was_insert:
            await self._open_quarantine_ticket(payload, correlation_id)

    async def _open_quarantine_ticket(
        self,
        payload: ModelMergeGateResult,
        correlation_id: UUID,
    ) -> None:
        """Open a Linear ticket for a QUARANTINE decision.

        If LINEAR_API_KEY or LINEAR_TEAM_ID are not configured, the ticket
        creation is skipped with a warning. Linear API failures are caught
        and logged but do not cause the overall handler to fail.

        Args:
            payload: Merge gate decision payload.
            correlation_id: Correlation ID for tracing.
        """
        api_key = os.environ.get(
            "LINEAR_API_KEY"
        )  # ONEX_EXCLUDE: Linear API key has no config injection path
        team_id = os.environ.get(
            "LINEAR_TEAM_ID"
        )  # ONEX_EXCLUDE: Linear team ID has no config injection path

        if not api_key or not team_id:
            logger.warning(
                "QUARANTINE decision but LINEAR_API_KEY or LINEAR_TEAM_ID not set; "
                "skipping quarantine ticket creation",
                extra={
                    "pr_ref": payload.pr_ref,
                    "correlation_id": str(correlation_id),
                },
            )
            return

        # Build ticket content
        violation_lines = "\n".join(
            f"- **{v.rule_code}** ({v.severity}): {v.message}"
            for v in payload.violations
        )
        title = f"QUARANTINE: {payload.pr_ref} ({payload.tier})"
        description = (
            f"## Merge Gate Quarantine\n\n"
            f"**PR**: {payload.pr_ref}\n"
            f"**Head SHA**: `{payload.head_sha}`\n"
            f"**Base SHA**: `{payload.base_sha}`\n"
            f"**Tier**: {payload.tier}\n"
            f"**Gate ID**: `{payload.gate_id}`\n"
            f"**Decided at**: {payload.decided_at.isoformat()}\n\n"
            f"## Violations\n\n"
            f"{violation_lines or '_No violations recorded._'}\n\n"
            f"## Context\n\n"
            f"- Correlation ID: `{correlation_id}`\n"
            f"- Run ID: `{payload.run_id or 'N/A'}`\n"
            f"- Run Fingerprint: `{payload.run_fingerprint or 'N/A'}`\n"
        )

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    LINEAR_GRAPHQL_URL,
                    headers={
                        "Authorization": api_key,
                        "Content-Type": "application/json",
                    },
                    json={
                        "query": LINEAR_CREATE_ISSUE_MUTATION,
                        "variables": {
                            "title": title,
                            "teamId": team_id,
                            "description": description,
                            "priority": 1,  # Urgent
                        },
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            issue_data = data.get("data", {}).get("issueCreate", {})
            if issue_data.get("success"):
                issue = issue_data.get("issue", {})
                logger.info(
                    "Quarantine Linear ticket created",
                    extra={
                        "pr_ref": payload.pr_ref,
                        "linear_issue_id": issue.get("identifier"),
                        "linear_issue_url": issue.get("url"),
                        "correlation_id": str(correlation_id),
                    },
                )
            else:
                # Log only the top-level success flag and errors array (if any)
                # to avoid leaking raw Linear response payloads.
                errors = data.get("errors", [])
                logger.warning(
                    "Linear issueCreate returned success=false",
                    extra={
                        "pr_ref": payload.pr_ref,
                        "error_count": len(errors),
                        "error_messages": [e.get("message", "") for e in errors][:3],
                        "correlation_id": str(correlation_id),
                    },
                )
        except (
            Exception  # noqa: BLE001 — boundary: logs warning and degrades
        ) as exc:  # ONEX: broad catch — ticket failure must not fail upsert
            logger.warning(
                "Failed to create quarantine Linear ticket (upsert still succeeded)",
                extra={
                    "pr_ref": payload.pr_ref,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "correlation_id": str(correlation_id),
                },
            )


__all__: list[str] = ["HandlerUpsertMergeGate"]
