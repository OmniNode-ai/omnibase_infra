# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""EFFECT handler that correlates raw savings signals and computes estimates.

Replaces the legacy ``ServiceSavingsEstimator`` (services/observability/
savings_estimation/consumer.py, deleted alongside this handler landing) which
held per-session correlation state in an in-memory ``OrderedDict`` buffer.
Per the OMN-16293 architecture decision, correlation state now lives entirely
in Postgres (the "projection surface") — every raw signal is INSERTed the
instant it is ingested, and the periodic batch step queries fresh state each
tick instead of reading Python instance memory. This mirrors
``HandlerBaselinesBatchCompute`` (node_baselines_batch_compute): an EFFECT
handler with an injected asyncpg pool and publisher, invoked directly from a
periodic ``asyncio`` loop in ``service_kernel.py`` rather than through the
generic operation-match auto-wiring.

Two responsibilities:
    1. Ingest: one row per raw signal event, written immediately (no
       buffering) into ``savings_injection_signals`` /
       ``savings_validator_catch_signals``.
       ``llm-call-completed`` and ``session-outcome`` signals are NOT
       ingested here — they are already projected by omnimarket's
       ``node_projection_llm_cost`` / ``node_projection_session_outcome``
       into ``llm_call_metrics`` / ``session_outcomes``, read directly by
       the correlation batch below (same cross-repo read-only pattern
       ``HandlerBaselinesBatchCompute`` uses for ``agent_routing_decisions``
       / ``agent_actions``).
    2. Correlate: a periodic batch step finds sessions with unfinalized
       signals that are "ready" (a ``session_outcomes`` row past the grace
       window, or signals older than the timeout), builds a
       ``ModelSavingsEstimationInput`` per session, calls the existing pure
       ``HandlerSavingsEstimation`` (unchanged), applies the same
       counterfactual-resolution and validator-catch heuristic-savings
       post-processing the legacy consumer applied, and publishes the result
       to ``onex.evt.omnibase-infra.savings-estimated.v1``. Persistence into
       ``savings_estimates`` is downstream and cross-repo (omnimarket's
       ``node_projection_savings``, already live and idle) — this handler
       never writes that table, only reads it for idempotency.

The dispatch-outcome-evaluated branch of the legacy correlator (task-level
savings from delegate-skill dispatch evaluation) is intentionally OUT of
scope here — it is covered by the active OMN-15800 savings.v1 dashboard
workstream.

The one cross-domain read, and why it is guarded (OMN-16770)
------------------------------------------------------------
Four of the five relations ``_find_ready_sessions`` joins are INTERNAL-domain
and granted to this pool's principal. The fifth, ``savings_estimates``, is
not: it is TENANT-domain, under ``FORCE ROW LEVEL SECURITY`` with a policy
predicated on the ``app.tenant_id`` GUC, and this pool connects as
``omninode_runtime`` — NOSUPERUSER, NOBYPASSRLS, non-owner by design
(OMN-16843). Because the handler takes a RAW injected ``asyncpg`` pool rather
than a contract-declared table operation, that read never passes through
``ProjectionTableOperation._assert_read_declared``, the seam that fails closed
on an undeclared relation. Nothing could refuse it at declaration time, so it
was only ever going to surface at runtime.

``_assert_idempotency_read_is_scoped`` is that missing refusal, moved to the
last place it can still be made: the connection itself, immediately before the
statement that depends on it. See
:class:`SavingsCorrelationUnscopedReadError` for why a ``GRANT`` is not the
fix.

Ticket: OMN-16293, OMN-16770
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from uuid import UUID

import asyncpg

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.errors.error_infra import RuntimeHostError
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.nodes.node_savings_estimation_compute.handlers.handler_savings_estimation import (
    HandlerSavingsEstimation,
)
from omnibase_infra.nodes.node_savings_estimation_compute.models.enum_catch_severity import (
    EnumCatchSeverity,
)
from omnibase_infra.nodes.node_savings_estimation_compute.models.enum_model_tier import (
    EnumModelTier,
)
from omnibase_infra.nodes.node_savings_estimation_compute.models.enum_savings_category import (
    EnumSavingsCategory,
)
from omnibase_infra.nodes.node_savings_estimation_compute.models.model_effectiveness_entry import (
    ModelEffectivenessEntry,
)
from omnibase_infra.nodes.node_savings_estimation_compute.models.model_savings_category import (
    ModelSavingsCategory,
)
from omnibase_infra.nodes.node_savings_estimation_compute.models.model_savings_correlation_batch_command import (
    ModelSavingsCorrelationBatchCommand,
)
from omnibase_infra.nodes.node_savings_estimation_compute.models.model_savings_correlation_batch_output import (
    ModelSavingsCorrelationBatchOutput,
)
from omnibase_infra.nodes.node_savings_estimation_compute.models.model_savings_estimation_input import (
    ModelSavingsEstimationInput,
)
from omnibase_infra.topics import SUFFIX_SAVINGS_ESTIMATED
from omnibase_infra.utils.util_db_transaction import set_statement_timeout
from omnibase_infra.utils.util_error_sanitization import sanitize_error_message

logger = logging.getLogger(__name__)

DEFAULT_GRACE_WINDOW_SECONDS: float = 30.0
DEFAULT_SESSION_TIMEOUT_SECONDS: float = 3600.0
DEFAULT_LOOKBACK_HOURS: int = 48
DEFAULT_BATCH_SIZE: int = 50
DEFAULT_QUERY_TIMEOUT: float = 10.0


# ---------------------------------------------------------------------------
# OMN-16770: the tenant-scope seam on the one cross-domain read this handler
# makes. Everything below exists because a GRANT is NOT the fix here.
# ---------------------------------------------------------------------------

#: The transaction-local setting every canonical tenant policy on this
#: platform compares ``tenant_id`` against.
TENANT_GUC: str = "app.tenant_id"

#: The one relation ``_find_ready_sessions`` reads that this node neither owns
#: nor writes. It is TENANT-domain, written cross-repo by omnibase_infra's
#: vendored ``node_projection_savings`` over the tenant binding, and put under
#: ``ENABLE`` + ``FORCE ROW LEVEL SECURITY`` by
#: ``081_savings_estimates_rls_tenant_isolation.sql`` with the policy
#: ``tenant_id = current_setting('app.tenant_id', true)``.
IDEMPOTENCY_RELATION: str = "savings_estimates"

#: Both premises in ONE round trip, so they are answered at the same instant,
#: on the same pooled connection that is about to run the anti-join. Reading
#: them separately, or on a different connection, would prove nothing about
#: the statement that actually matters.
_TENANT_SCOPE_PROBE_SQL = """
    SELECT
        row_security_active($1::regclass) AS rls_enforced,
        nullif(current_setting($2, true), '') AS tenant_scope
"""


class SavingsCorrelationUnscopedReadError(RuntimeHostError):
    """The idempotency read cannot be trusted, so the batch refuses to run.

    OMN-16770. ``_find_ready_sessions`` closes its candidate query with

        ``AND NOT EXISTS (SELECT 1 FROM savings_estimates se WHERE ...)``

    an anti-join over a FORCE-RLS relation whose policy is predicated on the
    ``app.tenant_id`` GUC. The pool that runs it is built from
    ``OMNINODE_INTERNAL_DB_URL``, whose principal ``omninode_runtime`` is
    pinned NOSUPERUSER / NOBYPASSRLS / non-owner (OMN-16843,
    ``docker/docker-compose.infra.yml``).

    Postgres row-level security fails OPEN from the caller's point of view.
    With the GUC unset, ``tenant_id = current_setting('app.tenant_id', true)``
    evaluates to NULL for every row, so the subquery matches nothing and
    raises nothing. ``NOT EXISTS`` becomes universally true: every candidate
    session reads as "never finalized" and the batch re-publishes an estimate
    for every session on every 60s tick, forever.

    That is why the obvious remediation for the live
    ``InsufficientPrivilegeError`` — ``GRANT SELECT ON savings_estimates TO
    omninode_runtime`` — is the WRONG fix. It does not make the read correct;
    it makes the read silently, unboundedly wrong, and converts a visible
    error into no error at all. This class is what a grant would run into
    instead.

    Refusing is the honest outcome, not a degradation: this handler carries no
    tenant attribution of its own (its ``savings_injection_signals`` /
    ``savings_validator_catch_signals`` tables have no ``tenant_id`` column),
    so there is no scope for it to bind that would not be invented — and
    inventing one is exactly what the OMN-16831 ruling forbids. The durable
    close is to stop reading a TENANT relation for INTERNAL idempotency at
    all; until that lands, this refusal is what keeps the anti-join from
    inverting.
    """


async def _assert_idempotency_read_is_scoped(conn: asyncpg.Connection) -> None:
    """Refuse the candidate query unless its anti-join can answer truthfully.

    Two outcomes are legitimate and are allowed through:

    * ``row_security_active`` is FALSE — the connection is the table owner
      without FORCE, a BYPASSRLS role, or a superuser. It sees every tenant's
      rows, so the anti-join is answered truthfully. This is the compose-lane
      case ``081``'s own header calls out.
    * RLS applies AND a non-blank ``app.tenant_id`` is bound — the read is
      scoped, and the anti-join is answered truthfully for that scope.

    Everything else refuses, including an unreadable premise: a probe that
    answers nothing must not be read as "no RLS here", or a permission or
    driver change that silences it would re-open the inversion with no signal.
    """
    row = await conn.fetchrow(_TENANT_SCOPE_PROBE_SQL, IDEMPOTENCY_RELATION, TENANT_GUC)
    if row is None:
        raise SavingsCorrelationUnscopedReadError(
            f"Savings correlation refused: could not read whether row-level "
            f"security applies to {IDEMPOTENCY_RELATION} on this connection. "
            f"The {IDEMPOTENCY_RELATION} anti-join is only sound when the "
            f"answer is known, so an unanswerable probe fails closed "
            f"(OMN-16770)."
        )

    rls_enforced = row["rls_enforced"]
    if not isinstance(rls_enforced, bool):
        raise SavingsCorrelationUnscopedReadError(
            f"Savings correlation refused: row_security_active("
            f"{IDEMPOTENCY_RELATION}) returned "
            f"{type(rls_enforced).__name__}, not a boolean. The anti-join's "
            f"visibility premise is unresolved, so the batch fails closed "
            f"(OMN-16770)."
        )

    if not rls_enforced:
        return

    tenant_scope = row["tenant_scope"]
    if isinstance(tenant_scope, str) and tenant_scope.strip():
        return

    raise SavingsCorrelationUnscopedReadError(
        f"Savings correlation refused: row-level security applies to "
        f"{IDEMPOTENCY_RELATION} on this connection and no {TENANT_GUC} is "
        f"bound, so its policy would filter every row and the "
        f"`NOT EXISTS ({IDEMPOTENCY_RELATION})` anti-join would report every "
        f"session as un-finalized — re-publishing an estimate for every "
        f"session on every tick. Granting SELECT does not fix this; it is "
        f"what produces it. Either read as a role row-level security does not "
        f"apply to, or bind {TENANT_GUC} for the scope being read "
        f"(OMN-16770)."
    )


# ---------------------------------------------------------------------------
# Severity classification for validator catches (ported unchanged from the
# legacy ServiceSavingsEstimator — the heuristic formulas are not part of the
# architecture change, only their state source is).
# ---------------------------------------------------------------------------

_SEVERITY_SAVINGS_USD: dict[EnumCatchSeverity, float] = {
    EnumCatchSeverity.CRITICAL: 0.50,
    EnumCatchSeverity.MAJOR: 0.20,
    EnumCatchSeverity.MINOR: 0.05,
}

_SEVERITY_TOKENS_SAVED: dict[EnumCatchSeverity, int] = {
    EnumCatchSeverity.CRITICAL: 2000,
    EnumCatchSeverity.MAJOR: 800,
    EnumCatchSeverity.MINOR: 200,
}

_SEVERITY_CONFIDENCE: dict[EnumCatchSeverity, float] = {
    EnumCatchSeverity.CRITICAL: 0.7,
    EnumCatchSeverity.MAJOR: 0.6,
    EnumCatchSeverity.MINOR: 0.4,
}

# Counterfactual model: the highest-cost configured routing candidate.
_COUNTERFACTUAL_MODEL_MAP: dict[str, str] = {
    "claude-sonnet-4": "claude-opus-4-6",
    "claude-3-5-sonnet": "claude-opus-4-6",
    "claude-3.5-sonnet": "claude-opus-4-6",
    "claude-opus-4-6": "claude-opus-4-6",
    "claude-3-opus": "claude-opus-4-6",
}


def _resolve_counterfactual(actual_model_id: str) -> str:
    """Resolve the counterfactual model for a given actual model.

    Never returns None — downstream (omnimarket's node_projection_savings)
    treats an absent/empty ``model_cloud_baseline`` as a malformed event and
    routes it to the DLQ (OMN-14533), so this must always resolve to
    something. Falls back to ``actual_model_id`` itself when no tier match is
    found.
    """
    lower = actual_model_id.lower()
    for key, value in _COUNTERFACTUAL_MODEL_MAP.items():
        if key in lower:
            return value
    return actual_model_id


def _classify_severity(raw: str) -> EnumCatchSeverity:
    lower = raw.lower().strip()
    if lower in ("critical", "error", "fatal"):
        return EnumCatchSeverity.CRITICAL
    if lower in ("major", "warning", "warn"):
        return EnumCatchSeverity.MAJOR
    return EnumCatchSeverity.MINOR


def _model_tier_from_id(model_id: str) -> EnumModelTier:
    lower = model_id.lower()
    if "sonnet" in lower:
        return EnumModelTier.SONNET
    return EnumModelTier.OPUS


def _compute_validator_catch_savings(
    severities: list[EnumCatchSeverity],
) -> tuple[float, int, float]:
    """Compute heuristic avoided-rework savings from validator catches.

    Applies diminishing returns so a session with many MINOR catches cannot
    claim unbounded savings. Ported unchanged from the legacy consumer.

    Returns:
        (total_savings_usd, total_tokens_saved, avg_confidence).
    """
    if not severities:
        return 0.0, 0, 0.0

    total_usd = 0.0
    total_tokens = 0
    confidence_sum = 0.0

    sorted_severities = sorted(severities)

    for idx, severity in enumerate(sorted_severities):
        diminishing_factor = 1.0 / (1.0 + 0.3 * idx)
        base_usd = _SEVERITY_SAVINGS_USD.get(severity, 0.05)
        base_tokens = _SEVERITY_TOKENS_SAVED.get(severity, 200)
        confidence = _SEVERITY_CONFIDENCE.get(severity, 0.4)

        total_usd += base_usd * diminishing_factor
        total_tokens += int(base_tokens * diminishing_factor)
        confidence_sum += confidence

    avg_confidence = confidence_sum / len(severities)
    return round(total_usd, 10), total_tokens, round(avg_confidence, 4)


# ---------------------------------------------------------------------------
# Signal row shapes (read from Postgres, replacing the legacy in-memory
# SessionBuffer/InjectionSignal/ValidatorCatchSignal/LlmCallSignal dataclasses)
# ---------------------------------------------------------------------------


@dataclass  # internal-dataclass-ok: query-result row shape, not a wire model
class InjectionRow:
    tokens_injected: int
    patterns_count: int


@dataclass  # internal-dataclass-ok: query-result row shape, not a wire model
class ValidatorCatchRow:
    severity: EnumCatchSeverity


@dataclass  # internal-dataclass-ok: query-result row shape, not a wire model
class LlmCallRow:
    model_id: str
    prompt_tokens: int
    completion_tokens: int


def _build_effectiveness_entries(
    injection_rows: list[InjectionRow],
    llm_rows: list[LlmCallRow],
    validator_rows: list[ValidatorCatchRow],
    *,
    has_session_outcome: bool,
) -> tuple[ModelEffectivenessEntry, ...]:
    """Convert queried signal rows into effectiveness entries.

    Mirrors the legacy ``_build_effectiveness_entries`` exactly, adapted to
    read from freshly-queried Postgres rows instead of a SessionBuffer.
    """
    tier = EnumModelTier.OPUS
    if llm_rows:
        tier = _model_tier_from_id(llm_rows[0].model_id)

    entries: list[ModelEffectivenessEntry] = []

    for row in injection_rows:
        if row.tokens_injected > 0:
            utilization = (
                min(row.patterns_count / 10.0, 1.0) if row.patterns_count > 0 else 0.5
            )
            entries.append(
                ModelEffectivenessEntry(
                    utilization_score=round(utilization, 4),
                    patterns_count=row.patterns_count,
                    tokens_saved=row.tokens_injected,
                    model_tier=tier,
                    is_output_tokens=False,
                )
            )

    if not entries and llm_rows:
        total_tokens = sum(r.prompt_tokens + r.completion_tokens for r in llm_rows)
        if total_tokens > 0:
            entries.append(
                ModelEffectivenessEntry(
                    utilization_score=0.0,
                    patterns_count=0,
                    tokens_saved=0,
                    model_tier=tier,
                    is_output_tokens=False,
                )
            )

    if not entries and validator_rows and has_session_outcome:
        entries.append(
            ModelEffectivenessEntry(
                utilization_score=0.0,
                patterns_count=0,
                tokens_saved=0,
                model_tier=tier,
                is_output_tokens=False,
            )
        )

    return tuple(entries)


@runtime_checkable
class ProtocolPublisher(Protocol):
    """Protocol matching PublisherTopicScoped.publish signature."""

    async def __call__(
        self,
        event_type: str,
        payload: object,
        topic: str | None,
        correlation_id: object,
        **kwargs: object,
    ) -> bool: ...


class HandlerSavingsCorrelation:
    """EFFECT handler: ingests raw savings signals and correlates savings.

    Attributes:
        _pool: Injected asyncpg connection pool.
        _publisher: Optional async callable for publishing to Kafka.
        _estimation_handler: The existing pure COMPUTE handler.
    """

    def __init__(
        self,
        pool: asyncpg.Pool,
        publisher: Callable[..., Awaitable[bool]] | None = None,
        estimation_handler: HandlerSavingsEstimation | None = None,
        grace_window_seconds: float = DEFAULT_GRACE_WINDOW_SECONDS,
        session_timeout_seconds: float = DEFAULT_SESSION_TIMEOUT_SECONDS,
        lookback_hours: int = DEFAULT_LOOKBACK_HOURS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        query_timeout: float = DEFAULT_QUERY_TIMEOUT,
    ) -> None:
        self._pool = pool
        self._publisher = publisher
        self._estimation_handler = estimation_handler or HandlerSavingsEstimation()
        self._grace_window_seconds = grace_window_seconds
        self._session_timeout_seconds = session_timeout_seconds
        self._lookback_hours = lookback_hours
        self._batch_size = batch_size
        self._query_timeout = query_timeout

    @property
    def handler_id(self) -> str:
        return "handler-savings-correlation"

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    # ------------------------------------------------------------------
    # Ingest: one INSERT per raw signal event, no buffering.
    # ------------------------------------------------------------------

    async def ingest_injection_event(self, payload: dict[str, object]) -> None:
        """Persist one onex.evt.omniclaude.context-injected.v1 event."""
        session_id = str(payload.get("session_id", "")).strip()
        tokens_injected = _coerce_int(payload.get("tokens_injected"))
        patterns_count = _coerce_int(payload.get("patterns_count"))
        if not session_id or tokens_injected <= 0:
            return
        async with self._pool.acquire() as conn:
            await set_statement_timeout(conn, self._query_timeout * 1000)
            await conn.execute(
                """
                INSERT INTO omninode_internal.savings_injection_signals
                    (session_id, tokens_injected, patterns_count)
                VALUES ($1, $2, $3)
                """,
                session_id,
                tokens_injected,
                patterns_count,
            )

    async def ingest_validator_catch_event(
        self, topic: str, payload: dict[str, object]
    ) -> None:
        """Persist one validator-catch or pattern-enforcement event."""
        session_id = str(payload.get("session_id", "")).strip()
        if not session_id:
            return
        severity = _classify_severity(str(payload.get("severity", "minor")))
        validator_type = str(payload.get("validator_type", ""))
        source_event_type = (
            "pattern-enforcement"
            if "pattern-enforcement" in topic
            else "validator-catch"
        )
        async with self._pool.acquire() as conn:
            await set_statement_timeout(conn, self._query_timeout * 1000)
            await conn.execute(
                """
                INSERT INTO omninode_internal.savings_validator_catch_signals
                    (session_id, severity, validator_type, source_event_type)
                VALUES ($1, $2, $3, $4)
                """,
                session_id,
                severity.value,
                validator_type,
                source_event_type,
            )

    # ------------------------------------------------------------------
    # Correlate: periodic batch step.
    # ------------------------------------------------------------------

    async def handle(
        self, command: ModelSavingsCorrelationBatchCommand
    ) -> ModelSavingsCorrelationBatchOutput:
        return await self.run_correlation_batch(command)

    async def run_correlation_batch(
        self, command: ModelSavingsCorrelationBatchCommand
    ) -> ModelSavingsCorrelationBatchOutput:
        """Find ready sessions, compute savings, and publish estimates."""
        correlation_id = command.correlation_id
        ready_session_ids = await self._find_ready_sessions()

        finalized = 0
        errors: list[str] = []

        for session_id in ready_session_ids:
            try:
                published = await self._finalize_session(session_id, correlation_id)
                if published:
                    finalized += 1
            except Exception as exc:  # noqa: BLE001 — one bad session must not kill the tick
                safe_msg = sanitize_error_message(exc)
                msg = f"session {session_id} failed to finalize: {safe_msg}"
                logger.warning(
                    "Savings correlation: %s", msg, extra={"session_id": session_id}
                )
                errors.append(msg)

        return ModelSavingsCorrelationBatchOutput(
            sessions_finalized=finalized,
            sessions_skipped_incomplete=0,
            errors=tuple(errors),
        )

    async def _find_ready_sessions(self) -> list[str]:
        sql = """
            WITH candidate_sessions AS (
                SELECT session_id, MIN(created_at) AS earliest_signal_at
                FROM (
                    SELECT session_id, created_at FROM omninode_internal.savings_injection_signals
                    UNION ALL
                    SELECT session_id, created_at FROM omninode_internal.savings_validator_catch_signals
                    UNION ALL
                    SELECT session_id, created_at FROM llm_call_metrics
                        WHERE session_id IS NOT NULL
                ) all_signals
                WHERE created_at > NOW() - make_interval(hours => $1::int)
                GROUP BY session_id
            )
            SELECT cs.session_id
            FROM candidate_sessions cs
            LEFT JOIN session_outcomes so ON so.session_id = cs.session_id
            WHERE (
                (
                    so.session_id IS NOT NULL
                    AND so.emitted_at <= NOW() - make_interval(secs => $2::double precision)
                )
                OR (
                    so.session_id IS NULL
                    AND cs.earliest_signal_at
                        <= NOW() - make_interval(secs => $3::double precision)
                )
            )
            AND NOT EXISTS (
                SELECT 1 FROM savings_estimates se WHERE se.session_id = cs.session_id
            )
            ORDER BY cs.session_id
            LIMIT $4
        """
        async with self._pool.acquire() as conn:
            await set_statement_timeout(conn, self._query_timeout * 1000)
            # OMN-16770: the anti-join above reads savings_estimates, a
            # FORCE-RLS tenant relation this node neither owns nor writes. The
            # probe runs FIRST, on this same connection: a refusal raised
            # after the query would already have executed the inverted
            # anti-join, and its all-sessions answer is the whole harm.
            await _assert_idempotency_read_is_scoped(conn)
            rows = await conn.fetch(
                sql,
                self._lookback_hours,
                self._grace_window_seconds,
                self._session_timeout_seconds,
                self._batch_size,
            )
        return [str(row["session_id"]) for row in rows]

    async def _finalize_session(self, session_id: str, correlation_id: UUID) -> bool:
        async with self._pool.acquire() as conn:
            await set_statement_timeout(conn, self._query_timeout * 1000)
            injection_rows = await conn.fetch(
                "SELECT tokens_injected, patterns_count "
                "FROM omninode_internal.savings_injection_signals WHERE session_id = $1 "
                "ORDER BY created_at",
                session_id,
            )
            validator_rows = await conn.fetch(
                "SELECT severity FROM omninode_internal.savings_validator_catch_signals "
                "WHERE session_id = $1 ORDER BY created_at",
                session_id,
            )
            llm_rows = await conn.fetch(
                "SELECT model_id, prompt_tokens, completion_tokens "
                "FROM llm_call_metrics WHERE session_id = $1 ORDER BY created_at",
                session_id,
            )
            outcome_row = await conn.fetchrow(
                "SELECT outcome FROM session_outcomes WHERE session_id = $1",
                session_id,
            )

        injection = [
            InjectionRow(
                tokens_injected=int(r["tokens_injected"]),
                patterns_count=int(r["patterns_count"]),
            )
            for r in injection_rows
        ]
        validator = [
            ValidatorCatchRow(severity=EnumCatchSeverity(r["severity"]))
            for r in validator_rows
        ]
        llm = [
            LlmCallRow(
                model_id=str(r["model_id"] or ""),
                prompt_tokens=int(r["prompt_tokens"] or 0),
                completion_tokens=int(r["completion_tokens"] or 0),
            )
            for r in llm_rows
        ]

        entries = _build_effectiveness_entries(
            injection, llm, validator, has_session_outcome=outcome_row is not None
        )
        if not entries:
            # Nothing to estimate for this session — not an error, just
            # nothing to publish (e.g. only a bare session-outcome with no
            # measurable signal ever arrived).
            return False

        actual_total_tokens = sum(r.prompt_tokens + r.completion_tokens for r in llm)
        actual_model_id = (
            llm[0].model_id if llm and llm[0].model_id else "claude-opus-4-6"
        )

        estimation_input = ModelSavingsEstimationInput(
            session_id=session_id,
            effectiveness_entries=entries,
            actual_total_tokens=actual_total_tokens,
            actual_model_id=actual_model_id,
        )

        estimate = await self._estimation_handler.handle(estimation_input)

        counterfactual = _resolve_counterfactual(actual_model_id)
        heuristic_usd, heuristic_tokens, heuristic_confidence = (
            _compute_validator_catch_savings([row.severity for row in validator])
        )

        categories = list(estimate.categories)
        if heuristic_usd > 0:
            categories.append(
                ModelSavingsCategory(
                    category=EnumSavingsCategory.VALIDATOR_CATCH,
                    savings_usd=heuristic_usd,
                    tokens_saved=heuristic_tokens,
                    confidence=heuristic_confidence,
                )
            )

        estimated_total_savings = round(estimate.direct_savings_usd + heuristic_usd, 10)
        estimated_total_tokens = estimate.direct_tokens_saved + heuristic_tokens
        heuristic_confidence_avg = (
            round(
                (estimate.heuristic_confidence_avg + heuristic_confidence) / 2.0,
                4,
            )
            if heuristic_usd > 0 and estimate.heuristic_confidence_avg > 0
            else (
                heuristic_confidence
                if heuristic_usd > 0
                else estimate.heuristic_confidence_avg
            )
        )

        payload = estimate.model_dump(mode="json")
        payload["counterfactual_model_id"] = counterfactual
        payload["heuristic_savings_usd"] = heuristic_usd
        payload["categories"] = [c.model_dump(mode="json") for c in categories]
        payload["estimated_total_savings_usd"] = estimated_total_savings
        payload["estimated_total_tokens_saved"] = estimated_total_tokens
        payload["heuristic_confidence_avg"] = heuristic_confidence_avg
        payload["model_local"] = actual_model_id
        payload["model_cloud_baseline"] = counterfactual
        payload["local_cost_usd"] = estimate.actual_cost_usd
        payload["cloud_cost_usd"] = round(
            estimate.actual_cost_usd + estimated_total_savings, 10
        )
        payload["savings_usd"] = estimated_total_savings
        # event_timestamp is what omnimarket's node_projection_savings actually
        # requires (ModelSavingsEstimatedEvent); reuse the same instant the
        # pure COMPUTE handler already stamped as timestamp_iso rather than
        # taking a second, slightly-later datetime.now() reading.
        payload["event_timestamp"] = payload["timestamp_iso"]

        if self._publisher is None:
            logger.warning(
                "Savings correlation: no publisher configured, dropping "
                "computed estimate for session %s",
                session_id,
            )
            return False

        await self._publisher(
            event_type="savings.estimated",
            payload=payload,
            topic=SUFFIX_SAVINGS_ESTIMATED,
            correlation_id=correlation_id,
        )
        logger.info(
            "Savings correlation: published estimate for session=%s "
            "savings=$%.6f (cid=%s)",
            session_id,
            estimated_total_savings,
            correlation_id,
        )
        return True


def _coerce_int(value: object, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def decode_event_message(message: ModelEventMessage) -> tuple[str, dict[str, object]]:
    """Decode a typed event-bus message into a (topic, payload) pair.

    Both the Kafka and in-memory event buses deliver a
    :class:`ModelEventMessage` to a consumer's ``on_message`` callback — never
    a raw ``dict`` or ``str``. The message body is the JSON payload carried
    in the typed ``value`` field (bytes). This decodes that field directly
    off the typed model. It does NOT call ``.get()`` on the message, which
    has no such method (OMN-13149).

    Ported from the legacy ``services/observability/savings_estimation/
    consumer.py`` (deleted alongside this handler landing) — unchanged.

    Args:
        message: The typed event-bus message delivered by the consumer
            callback. ``message.topic`` is the correlation topic and
            ``message.value`` is the JSON-encoded payload.

    Returns:
        A ``(topic, payload)`` pair ready for
        :meth:`HandlerSavingsCorrelation.ingest_injection_event` /
        :meth:`HandlerSavingsCorrelation.ingest_validator_catch_event`.

    Raises:
        TypeError: If the decoded payload is not a JSON object.
    """
    payload = json.loads(message.value)
    if not isinstance(payload, dict):
        raise TypeError(
            "savings correlation payload must be a JSON object, "
            f"got {type(payload).__name__} on topic {message.topic!r}"
        )
    return message.topic, payload


__all__: list[str] = [
    "IDEMPOTENCY_RELATION",
    "TENANT_GUC",
    "EnumCatchSeverity",
    "HandlerSavingsCorrelation",
    "ProtocolPublisher",
    "SavingsCorrelationUnscopedReadError",
    "decode_event_message",
]
