# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Two-stage decision write handler for NodeDecisionStoreEffect.

Stage 1 (committed independently):
    1. Normalize scope_services — sort, lowercase, validate against ALLOWED_DOMAINS
       service slugs (no strict repo-slug allowlist; domain is validated)
    2. Normalize scope_domain — lowercase, validate against ALLOWED_DOMAINS
    3. Validate created_at — reject if > CREATED_AT_FUTURE_LIMIT_SECONDS in future
       vs DB now()
    4. Enforce supersession: if superseded_by is set → force status=SUPERSEDED
    5. INSERT INTO decision_store ON CONFLICT (decision_id) DO UPDATE
    6. Return Stage1Result with status ("recorded" or "updated") and old_status

Stage 2 (runs after Stage 1 commit — failure NEVER rolls back Stage 1):
    1. Query ACTIVE decisions in same scope_domain + scope_layer
    2. Run structural_confidence() pure function against each pair
    3. For pairs >= 0.3: write to decision_conflicts via INSERT ON CONFLICT DO NOTHING
    4. Enforce ACTIVE invariant: structural_confidence >= 0.9 + both ACTIVE without
       DISMISSED conflict or supersession link → set new decision to PROPOSED, emit HIGH
    5. Return list of conflict pairs written

Structural confidence function (pure, no DB):
    if a.scope_domain != b.scope_domain: return 0.0
    if a.scope_layer != b.scope_layer: return 0.4
    a_svc, b_svc = set(a.scope_services), set(b.scope_services)
    if not a_svc and not b_svc: return 0.9
    if not a_svc or not b_svc: return 0.8
    if a_svc == b_svc: return 1.0
    if a_svc & b_svc: return 0.7
    return 0.3

ALLOWED_DOMAINS = ['transport', 'data-model', 'auth', 'api', 'infra', 'testing',
                   'code-structure', 'security', 'observability', 'custom']

Related Tickets:
    - OMN-2765: NodeDecisionStoreEffect implementation
    - OMN-2764: DB migrations (decision_store, decision_conflicts tables)
    - OMN-2763: omnibase_core ModelDecisionStoreEntry / ModelDecisionConflict
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumPostgresErrorCode,
)
from omnibase_infra.errors import ModelInfraErrorContext, RepositoryExecutionError
from omnibase_infra.mixins.mixin_postgres_op_executor import MixinPostgresOpExecutor
from omnibase_infra.models.model_backend_result import ModelBackendResult

if TYPE_CHECKING:
    import asyncpg

    from omnibase_infra.nodes.node_decision_store_effect.models.model_payload_write_decision import (
        ModelPayloadWriteDecision,
    )

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

ALLOWED_DOMAINS: frozenset[str] = frozenset(
    [
        "transport",
        "data-model",
        "auth",
        "api",
        "infra",
        "testing",
        "code-structure",
        "security",
        "observability",
        "custom",
    ]
)

# Reject created_at timestamps more than this many seconds in the future vs DB now().
CREATED_AT_FUTURE_LIMIT_SECONDS: int = 300  # 5 minutes

# Structural confidence threshold for writing a conflict record.
CONFLICT_WRITE_THRESHOLD: float = 0.3

# Structural confidence threshold for enforcing the ACTIVE invariant.
ACTIVE_INVARIANT_THRESHOLD: float = 0.9

# =============================================================================
# SQL Statements
# =============================================================================

# Stage 1: Upsert into decision_store
SQL_UPSERT_DECISION = """
INSERT INTO decision_store (
    decision_id, correlation_id, title, decision_type, status,
    scope_domain, scope_services, scope_layer, rationale, alternatives,
    tags, source, epic_id, supersedes, superseded_by,
    created_at, db_written_at, created_by
) VALUES (
    $1, $2, $3, $4, $5,
    $6, $7::jsonb, $8, $9, $10::jsonb,
    $11::jsonb, $12, $13, $14::jsonb, $15,
    $16, NOW(), $17
)
ON CONFLICT (decision_id) DO UPDATE SET
    correlation_id  = EXCLUDED.correlation_id,
    title           = EXCLUDED.title,
    decision_type   = EXCLUDED.decision_type,
    status          = EXCLUDED.status,
    scope_domain    = EXCLUDED.scope_domain,
    scope_services  = EXCLUDED.scope_services,
    scope_layer     = EXCLUDED.scope_layer,
    rationale       = EXCLUDED.rationale,
    alternatives    = EXCLUDED.alternatives,
    tags            = EXCLUDED.tags,
    source          = EXCLUDED.source,
    epic_id         = EXCLUDED.epic_id,
    supersedes      = EXCLUDED.supersedes,
    superseded_by   = EXCLUDED.superseded_by,
    created_at      = EXCLUDED.created_at,
    created_by      = EXCLUDED.created_by
RETURNING
    (xmax = 0) AS was_insert,
    status AS new_status;
"""

# Stage 1: Fetch old status before upsert (for change detection)
SQL_FETCH_OLD_STATUS = """
SELECT status FROM decision_store WHERE decision_id = $1;
"""

# Stage 1: DB server time (for created_at validation)
SQL_NOW = "SELECT NOW() AT TIME ZONE 'UTC';"

# Stage 2: Query active decisions in same domain + layer (excluding the just-written one)
SQL_ACTIVE_SAME_SCOPE = """
SELECT
    decision_id,
    scope_domain,
    scope_services,
    scope_layer,
    status,
    superseded_by
FROM decision_store
WHERE
    scope_domain = $1
    AND scope_layer = $2
    AND status = 'ACTIVE'
    AND decision_id != $3;
"""

# Stage 2: Check if a DISMISSED conflict already exists for a pair
SQL_CHECK_DISMISSED_CONFLICT = """
SELECT 1 FROM decision_conflicts
WHERE decision_min_id = $1
  AND decision_max_id = $2
  AND status = 'DISMISSED'
LIMIT 1;
"""

# Stage 2: Idempotent insert into decision_conflicts
SQL_INSERT_CONFLICT = """
INSERT INTO decision_conflicts (
    conflict_id, decision_min_id, decision_max_id,
    structural_confidence, final_severity, status, detected_at
) VALUES (
    gen_random_uuid(), $1, $2, $3, $4, 'OPEN', NOW()
)
ON CONFLICT (decision_min_id, decision_max_id) DO NOTHING
RETURNING conflict_id;
"""

# Stage 2: Demote new decision to PROPOSED when ACTIVE invariant is violated
SQL_DEMOTE_TO_PROPOSED = """
UPDATE decision_store SET status = 'PROPOSED' WHERE decision_id = $1;
"""


# =============================================================================
# Internal data containers
# =============================================================================


@dataclass
class ActiveDecisionRow:
    """Lightweight row container for decisions fetched in Stage 2."""

    decision_id: UUID
    scope_domain: str
    scope_services: list[str]
    scope_layer: str
    status: str
    superseded_by: UUID | None


@dataclass
class Stage1Result:
    """Result of the Stage 1 upsert."""

    was_insert: bool
    new_status: str
    old_status: str | None = None


@dataclass
class ConflictWritten:
    """A single conflict pair written in Stage 2."""

    decision_min_id: UUID
    decision_max_id: UUID
    confidence: float
    final_severity: str
    was_new: bool  # False when ON CONFLICT DO NOTHING skipped the row


@dataclass
class Stage2Result:
    """Aggregate result of Stage 2 conflict detection."""

    conflicts_written: list[ConflictWritten] = field(default_factory=list)
    new_decision_demoted: bool = False  # True if ACTIVE invariant forced PROPOSED


@dataclass
class DecisionScopeKey:
    """Scope fields used by structural_confidence().

    Bundles the three scope attributes for a decision so that
    structural_confidence() takes two model arguments rather than six primitives.
    """

    scope_domain: str
    scope_layer: str
    scope_services: list[str]


# =============================================================================
# Pure function: structural_confidence
# =============================================================================


def structural_confidence(a: DecisionScopeKey, b: DecisionScopeKey) -> float:
    """Compute structural conflict confidence between two decisions.

    Pure function — no I/O.

    Args:
        a: Scope key for the first decision.
        b: Scope key for the second decision.

    Returns:
        0.0  — different domains (no conflict possible)
        0.4  — same domain, different layer (weak signal)
        0.9  — same domain + layer, both empty service lists
        0.8  — same domain + layer, one side is empty
        1.0  — same domain + layer + identical non-empty service sets
        0.7  — same domain + layer, overlapping non-empty service sets
        0.3  — same domain + layer, disjoint non-empty service sets
    """
    if a.scope_domain != b.scope_domain:
        return 0.0
    if a.scope_layer != b.scope_layer:
        return 0.4
    a_svc: set[str] = set(a.scope_services)
    b_svc: set[str] = set(b.scope_services)
    if not a_svc and not b_svc:
        return 0.9
    if not a_svc or not b_svc:
        return 0.8
    if a_svc == b_svc:
        return 1.0
    if a_svc & b_svc:
        return 0.7
    return 0.3


def _severity_for_confidence(confidence: float) -> str:
    """Map structural confidence score to a final_severity string."""
    if confidence >= ACTIVE_INVARIANT_THRESHOLD:
        return "HIGH"
    if confidence >= 0.6:
        return "MEDIUM"
    return "LOW"


def _ordered_pair(a: UUID, b: UUID) -> tuple[UUID, UUID]:
    """Return (min, max) UUID pair preserving DB constraint chk_conflict_pair_order."""
    return (a, b) if a < b else (b, a)


# =============================================================================
# Handler
# =============================================================================


class HandlerWriteDecision(MixinPostgresOpExecutor):
    """Two-stage decision write handler.

    Stage 1 commits independently; Stage 2 failure NEVER rolls back Stage 1.

    The handler uses two separate asyncpg connections (one per stage) to ensure
    Stage 1 is committed before Stage 2 begins. Any exception in Stage 2 is
    logged and surfaced in the result metadata but does NOT cause the overall
    handle() call to return success=False (Stage 1 committed = success).

    Attributes:
        _pool: asyncpg connection pool for database operations.

    Example:
        >>> pool = await asyncpg.create_pool(dsn="...")
        >>> handler = HandlerWriteDecision(pool)
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
        payload: ModelPayloadWriteDecision,
        correlation_id: UUID,
    ) -> ModelBackendResult:
        """Execute the two-stage decision write.

        Uses _execute_postgres_op to wrap Stage 1. Stage 2 runs after Stage 1
        commits and its exceptions are caught internally, preserving Stage 1
        success semantics.

        Args:
            payload: Decision write payload.
            correlation_id: Request correlation ID for distributed tracing.

        Returns:
            ModelBackendResult with success=True when Stage 1 committed.
            Stage 2 errors are logged but do not cause success=False.
        """
        return await self._execute_postgres_op(
            op_error_code=EnumPostgresErrorCode.UPSERT_ERROR,
            correlation_id=correlation_id,
            log_context={
                "decision_id": str(payload.decision_id),
                "scope_domain": payload.scope_domain,
            },
            fn=lambda: self._run_two_stage(payload, correlation_id),
        )

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------

    def _normalize_scope_services(
        self,
        raw: list[str],
    ) -> list[str]:
        """Return sorted list of lowercase service slugs."""
        return sorted(s.lower() for s in raw)

    def _normalize_scope_domain(self, raw: str) -> str:
        """Return lowercase domain, raising ValueError if not in ALLOWED_DOMAINS."""
        normalized = raw.lower()
        if normalized not in ALLOWED_DOMAINS:
            raise ValueError(
                f"scope_domain {raw!r} is not in ALLOWED_DOMAINS. "
                f"Allowed: {sorted(ALLOWED_DOMAINS)}"
            )
        return normalized

    # ------------------------------------------------------------------
    # Stage 1: validate + upsert
    # ------------------------------------------------------------------

    async def _run_stage1(
        self,
        payload: ModelPayloadWriteDecision,
        correlation_id: UUID,
        conn: asyncpg.Connection,
    ) -> Stage1Result:
        """Validate and upsert the decision record in a single connection.

        Raises:
            ValueError: For domain or timestamp validation failures.
            RepositoryExecutionError: If the DB returns no row on upsert.
        """
        # 1. Normalise scope_services and scope_domain
        scope_services = self._normalize_scope_services(payload.scope_services)
        scope_domain = self._normalize_scope_domain(payload.scope_domain)

        # 2. Validate created_at vs DB now()
        db_now_row = await conn.fetchval(SQL_NOW)
        db_now: datetime = (
            db_now_row.replace(tzinfo=UTC) if db_now_row.tzinfo is None else db_now_row
        )
        created_at = payload.created_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        delta = (created_at - db_now).total_seconds()
        if delta > CREATED_AT_FUTURE_LIMIT_SECONDS:
            raise ValueError(
                f"created_at is {delta:.0f}s in the future (limit: "
                f"{CREATED_AT_FUTURE_LIMIT_SECONDS}s). Rejected."
            )

        # 3. Fetch old status for change-event discrimination
        old_status_row = await conn.fetchrow(SQL_FETCH_OLD_STATUS, payload.decision_id)
        old_status: str | None = old_status_row["status"] if old_status_row else None

        # 4. Enforce supersession: superseded_by set → force SUPERSEDED
        effective_status: str = payload.status
        if payload.superseded_by is not None:
            effective_status = "SUPERSEDED"

        # 5. Upsert
        row = await conn.fetchrow(
            SQL_UPSERT_DECISION,
            payload.decision_id,
            payload.correlation_id,
            payload.title,
            payload.decision_type,
            effective_status,
            scope_domain,
            json.dumps(scope_services),
            payload.scope_layer,
            payload.rationale,
            json.dumps(payload.alternatives),
            json.dumps(payload.tags),
            payload.source,
            payload.epic_id,
            json.dumps([str(u) for u in payload.supersedes]),
            payload.superseded_by,
            payload.created_at,
            payload.created_by,
        )

        if row is None:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type="db",
                operation="decision_upsert",
            )
            raise RepositoryExecutionError(
                "decision_store upsert returned no row",
                context=context,
            )

        was_insert: bool = row["was_insert"]
        new_status: str = row["new_status"]

        logger.info(
            "Stage 1 complete: decision upserted",
            extra={
                "decision_id": str(payload.decision_id),
                "operation": "insert" if was_insert else "update",
                "new_status": new_status,
                "old_status": old_status,
                "correlation_id": str(correlation_id),
            },
        )

        return Stage1Result(
            was_insert=was_insert,
            new_status=new_status,
            old_status=old_status,
        )

    # ------------------------------------------------------------------
    # Stage 2: conflict detection
    # ------------------------------------------------------------------

    async def _run_stage2(
        self,
        payload: ModelPayloadWriteDecision,
        stage1: Stage1Result,
        correlation_id: UUID,
        conn: asyncpg.Connection,
    ) -> Stage2Result:
        """Detect and record structural conflicts for the newly-written decision.

        This runs in a SEPARATE connection after Stage 1 commits. Any exception
        here is caught by the caller (_run_two_stage) and does not roll back
        Stage 1.

        Args:
            payload: The original write-decision payload.
            stage1: Result from Stage 1 (status, was_insert).
            correlation_id: Correlation ID for tracing.
            conn: A fresh asyncpg connection (Stage 1 already committed).

        Returns:
            Stage2Result with lists of conflicts written and demotion flag.
        """
        # Only run conflict detection for ACTIVE decisions.
        effective_status = stage1.new_status
        if effective_status != "ACTIVE":
            logger.debug(
                "Stage 2 skipped: decision status is not ACTIVE",
                extra={
                    "decision_id": str(payload.decision_id),
                    "status": effective_status,
                    "correlation_id": str(correlation_id),
                },
            )
            return Stage2Result()

        scope_domain_lower = payload.scope_domain.lower()
        scope_services_normalized = self._normalize_scope_services(
            payload.scope_services
        )
        new_scope = DecisionScopeKey(
            scope_domain=scope_domain_lower,
            scope_layer=payload.scope_layer,
            scope_services=scope_services_normalized,
        )

        # 1. Query ACTIVE decisions in same domain + layer (excluding this one)
        rows = await conn.fetch(
            SQL_ACTIVE_SAME_SCOPE,
            scope_domain_lower,
            payload.scope_layer,
            payload.decision_id,
        )

        result = Stage2Result()

        for row in rows:
            other_id: UUID = row["decision_id"]
            other_domain: str = row["scope_domain"]
            other_layer: str = row["scope_layer"]
            # scope_services stored as JSONB array of strings
            raw_svc = row["scope_services"]
            other_services: list[str] = (
                json.loads(raw_svc) if isinstance(raw_svc, str) else list(raw_svc)
            )

            other_scope = DecisionScopeKey(
                scope_domain=other_domain,
                scope_layer=other_layer,
                scope_services=other_services,
            )
            score = structural_confidence(new_scope, other_scope)

            if score < CONFLICT_WRITE_THRESHOLD:
                continue

            severity = _severity_for_confidence(score)
            min_id, max_id = _ordered_pair(payload.decision_id, other_id)

            # Check ACTIVE invariant: score >= 0.9 and both ACTIVE
            if score >= ACTIVE_INVARIANT_THRESHOLD:
                # Check for existing DISMISSED conflict (invariant exception)
                dismissed = await conn.fetchval(
                    SQL_CHECK_DISMISSED_CONFLICT, min_id, max_id
                )
                # Check supersession link
                this_superseded = payload.superseded_by is not None
                other_superseded = row["superseded_by"] is not None

                if not dismissed and not this_superseded and not other_superseded:
                    # Invariant violated: demote new decision to PROPOSED
                    await conn.execute(SQL_DEMOTE_TO_PROPOSED, payload.decision_id)
                    result.new_decision_demoted = True
                    logger.warning(
                        "Stage 2: ACTIVE invariant violated — demoted to PROPOSED",
                        extra={
                            "decision_id": str(payload.decision_id),
                            "conflicting_id": str(other_id),
                            "confidence": score,
                            "correlation_id": str(correlation_id),
                        },
                    )

            # Write conflict record (idempotent)
            conflict_row = await conn.fetchrow(
                SQL_INSERT_CONFLICT, min_id, max_id, score, severity
            )
            was_new = conflict_row is not None

            result.conflicts_written.append(
                ConflictWritten(
                    decision_min_id=min_id,
                    decision_max_id=max_id,
                    confidence=score,
                    final_severity=severity,
                    was_new=was_new,
                )
            )

            if was_new:
                logger.info(
                    "Stage 2: new conflict recorded",
                    extra={
                        "conflict_min_id": str(min_id),
                        "conflict_max_id": str(max_id),
                        "confidence": score,
                        "severity": severity,
                        "correlation_id": str(correlation_id),
                    },
                )

        return result

    # ------------------------------------------------------------------
    # Orchestration: two-stage execution
    # ------------------------------------------------------------------

    async def _run_two_stage(
        self,
        payload: ModelPayloadWriteDecision,
        correlation_id: UUID,
    ) -> None:
        """Orchestrate Stage 1 (committed) then Stage 2 (best-effort).

        Stage 1 runs inside a transaction that commits before Stage 2 begins.
        Stage 2 runs in a separate connection so its failure cannot roll back
        Stage 1. Stage 2 exceptions are caught, logged, and suppressed.
        """
        # ---- Stage 1 ----
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                stage1 = await self._run_stage1(payload, correlation_id, conn)
            # Transaction committed here (context manager __aexit__)

        # ---- Stage 2 (separate connection — failure never rolls back Stage 1) ----
        try:
            async with self._pool.acquire() as conn2:
                await self._run_stage2(payload, stage1, correlation_id, conn2)
        except Exception as exc:  # ONEX: intentional broad catch for stage isolation
            logger.warning(
                "Stage 2 failed (Stage 1 already committed — no rollback)",
                extra={
                    "decision_id": str(payload.decision_id),
                    "correlation_id": str(correlation_id),
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )


__all__: list[str] = [
    "ActiveDecisionRow",
    "ConflictWritten",
    "DecisionScopeKey",
    "HandlerWriteDecision",
    "Stage1Result",
    "Stage2Result",
    "structural_confidence",
    "ALLOWED_DOMAINS",
    "CONFLICT_WRITE_THRESHOLD",
    "ACTIVE_INVARIANT_THRESHOLD",
]
