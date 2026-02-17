# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""PostgreSQL Writer for LLM cost aggregation.

This module provides a PostgreSQL writer for persisting LLM cost aggregation
data consumed from Kafka. It handles upsert semantics for the
``llm_cost_aggregates`` table and insert-only semantics for ``llm_call_metrics``.

Design Decisions:
    - Pool injection: asyncpg.Pool is injected, not created/managed
    - Batch inserts for llm_call_metrics (append-only)
    - Upsert for llm_cost_aggregates (UNIQUE on aggregation_key + window)
    - Event deduplication via event_id tracking (in-memory set, bounded)
    - Circuit breaker: MixinAsyncCircuitBreaker for resilience

Idempotency Contract:
    | Table               | Unique Key                       | Conflict Action     |
    |---------------------|----------------------------------|---------------------|
    | llm_call_metrics    | id (UUID PK)                     | DO NOTHING          |
    | llm_cost_aggregates | (aggregation_key, window)        | DO UPDATE (additive)|

Related Tickets:
    - OMN-2240: E1-T4 LLM cost aggregation service
    - OMN-2236: llm_call_metrics + llm_cost_aggregates migration 031
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
from collections import OrderedDict
from decimal import Decimal
from uuid import UUID, uuid4

import asyncpg

from omnibase_core.types import JsonType
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError
from omnibase_infra.mixins import MixinAsyncCircuitBreaker

logger = logging.getLogger(__name__)

# Maximum number of event IDs to track for deduplication.
# Uses an LRU-style OrderedDict to bound memory usage.
_MAX_DEDUP_CACHE_SIZE: int = 50_000

# Aggregation windows matching the cost_aggregation_window enum in PostgreSQL.
AGGREGATION_WINDOWS: tuple[str, ...] = ("24h", "7d", "30d")

# Aggregation key prefixes for the composite key format.
_KEY_PREFIX_SESSION: str = "session"
_KEY_PREFIX_MODEL: str = "model"
_KEY_PREFIX_REPO: str = "repo"
_KEY_PREFIX_PATTERN: str = "pattern"


class WriterLlmCostAggregationPostgres(MixinAsyncCircuitBreaker):
    """PostgreSQL writer for LLM cost aggregation.

    Provides batch write methods for llm_call_metrics (raw events) and
    llm_cost_aggregates (rolling window aggregations) with idempotency
    guarantees and circuit breaker resilience.

    The writer tracks event IDs in an in-memory bounded cache to prevent
    double-counting on Kafka consumer replay. Events whose ID has already
    been processed are silently skipped.

    Aggregation keys use the format ``<prefix>:<value>`` where prefix is one
    of: session, model, repo, pattern. Each event produces multiple
    aggregation rows (one per dimension per window).

    Attributes:
        _pool: Injected asyncpg connection pool.
        _dedup_cache: Bounded OrderedDict for event ID deduplication.
        DEFAULT_QUERY_TIMEOUT_SECONDS: Default timeout for database queries (30s).
    """

    DEFAULT_QUERY_TIMEOUT_SECONDS: float = 30.0

    def __init__(
        self,
        pool: asyncpg.Pool,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_reset_timeout: float = 60.0,
        circuit_breaker_half_open_successes: int = 1,
        query_timeout: float = DEFAULT_QUERY_TIMEOUT_SECONDS,
    ) -> None:
        """Initialize the writer with connection pool.

        Args:
            pool: asyncpg connection pool (lifecycle managed externally).
            circuit_breaker_threshold: Failures before circuit opens.
            circuit_breaker_reset_timeout: Seconds before circuit half-opens.
            circuit_breaker_half_open_successes: Successes to close from half-open.
            query_timeout: Statement timeout for database queries in seconds.
        """
        # Validate query_timeout before storing
        if not math.isfinite(query_timeout) or query_timeout <= 0:
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="__init__",
            )
            raise ProtocolConfigurationError(
                f"query_timeout must be a finite positive number, got {query_timeout!r}",
                context=context,
                parameter="query_timeout",
                value=str(query_timeout),
            )

        self._pool = pool
        self._query_timeout = query_timeout
        self._dedup_cache: OrderedDict[str, bool] = OrderedDict()
        self._dedup_lock = asyncio.Lock()

        # Initialize circuit breaker
        self._init_circuit_breaker(
            threshold=circuit_breaker_threshold,
            reset_timeout=circuit_breaker_reset_timeout,
            service_name="llm-cost-aggregation-writer",
            transport_type=EnumInfraTransportType.DATABASE,
            half_open_successes=circuit_breaker_half_open_successes,
        )

    def _statement_timeout_ms(self) -> int:
        """Compute the statement timeout in milliseconds as a bounded integer.

        PostgreSQL's ``SET LOCAL statement_timeout`` does not support
        parameterized queries (``$1`` placeholders), so the value must be
        interpolated into the SQL string. This method converts
        ``_query_timeout`` (validated as a finite positive float in
        ``__init__``) to a bounded integer to guarantee the interpolated
        value is safe.

        Returns:
            Timeout in milliseconds, clamped to [1, 600_000] (10 minutes max).
        """
        timeout_ms = int(self._query_timeout * 1000)
        # Defensive bounds: even though __init__ validates query_timeout > 0,
        # clamp to a safe range in case the attribute is mutated post-init.
        timeout_ms = max(timeout_ms, 1)
        timeout_ms = min(timeout_ms, 600_000)
        return timeout_ms

    def _is_duplicate(self, event_id: str) -> bool:
        """Check if an event ID has already been processed.

        Uses an LRU-style bounded cache. If the cache exceeds
        ``_MAX_DEDUP_CACHE_SIZE``, the oldest entries are evicted.

        This method only **checks** the cache; it does NOT add the event ID.
        Callers must explicitly call ``_mark_seen`` after successful
        persistence to avoid marking events as "seen" before they are
        actually written to the database.

        Precondition:
            **Caller must hold** ``_dedup_lock``.  This method reads
            ``_dedup_cache`` without acquiring the lock internally.
            This is a deliberate design choice: callers
            (``write_call_metrics`` and ``write_cost_aggregates``) batch
            multiple ``_is_duplicate`` calls under a single lock acquisition
            to avoid per-event lock overhead.  Making this method acquire
            the lock itself would cause deadlocks with the existing callers
            that already hold it.

        Args:
            event_id: Unique event identifier to check.

        Returns:
            True if the event was already seen, False otherwise.
        """
        if event_id in self._dedup_cache:
            # Move to end (most recently seen)
            self._dedup_cache.move_to_end(event_id)
            return True

        return False

    def _mark_seen(self, event_id: str) -> None:
        """Record an event ID as successfully persisted.

        Adds the event ID to the bounded LRU dedup cache. This must be
        called only AFTER the corresponding database write has committed,
        so that a failed write does not prevent retries.

        Precondition:
            **Caller must hold** ``_dedup_lock``.  See ``_is_duplicate``
            docstring for rationale.

        Args:
            event_id: Unique event identifier to mark as seen.
        """
        self._dedup_cache[event_id] = True

        # Evict oldest if over capacity
        while len(self._dedup_cache) > _MAX_DEDUP_CACHE_SIZE:
            self._dedup_cache.popitem(last=False)

    async def write_call_metrics(
        self,
        events: list[dict[str, object]],
        correlation_id: UUID | None = None,
    ) -> int:
        """Write raw LLM call metrics to the llm_call_metrics table.

        Each event is inserted as a new row. Duplicate event IDs (based on
        the in-memory dedup cache) are silently skipped. Database-level
        conflicts on the UUID primary key use DO NOTHING for safety.

        Args:
            events: List of event dictionaries from ContractLlmCallMetrics.
            correlation_id: Correlation ID for tracing.

        Returns:
            Number of rows successfully written.

        Raises:
            InfraUnavailableError: If the circuit breaker is open.
        """
        if not events:
            return 0

        if correlation_id is None:
            correlation_id = uuid4()

        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("write_call_metrics", correlation_id)

        # Filter duplicates using stable dedup keys. Both write_call_metrics
        # and write_cost_aggregates call _is_duplicate() independently with
        # different key prefixes ("" vs "agg:") so that each write path tracks
        # its own dedup state. This intentionally doubles the cache entries per
        # event but keeps the two write paths decoupled -- a failure in one
        # does not affect dedup tracking in the other.
        #
        # NOTE: _is_duplicate() only checks the cache; it does NOT add entries.
        # Events are marked as seen via _mark_seen() only AFTER successful
        # database persistence to prevent data loss on write failures.
        unique_events: list[tuple[str, dict[str, object]]] = []
        seen_in_batch: set[str] = set()
        empty_dedup_fields_count = 0
        async with self._dedup_lock:
            for event in events:
                event_id = _derive_stable_dedup_key(event)
                if _has_empty_dedup_fields(event):
                    empty_dedup_fields_count += 1
                if not self._is_duplicate(event_id) and event_id not in seen_in_batch:
                    unique_events.append((event_id, event))
                    seen_in_batch.add(event_id)

        if empty_dedup_fields_count > 0:
            logger.warning(
                "Batch contained %d event(s) with all dedup fallback fields empty; "
                "deduplication may not work correctly for these events",
                empty_dedup_fields_count,
                extra={
                    "correlation_id": str(correlation_id),
                    "total_events": len(events),
                    "empty_dedup_fields_count": empty_dedup_fields_count,
                },
            )

        if not unique_events:
            logger.debug(
                "All events were duplicates, skipping write",
                extra={
                    "correlation_id": str(correlation_id),
                    "total_events": len(events),
                },
            )
            return 0

        written = 0
        # Track dedup keys for events that were successfully persisted.
        # We accumulate them here and add to the dedup cache only after
        # the outer transaction commits.
        persisted_dedup_keys: list[str] = []
        try:
            async with self._pool.acquire() as conn:
                async with conn.transaction():
                    # Defense-in-depth: PostgreSQL SET LOCAL does not support
                    # $1 parameterized queries, so the value must be
                    # interpolated.  _statement_timeout_ms() returns a
                    # bounds-checked int clamped to [1, 600_000].  The
                    # explicit int() cast at the interpolation site is a
                    # second safety fence ensuring only a numeric literal
                    # can ever reach the SQL string, even if the return
                    # type of _statement_timeout_ms() were to change.
                    timeout_ms = self._statement_timeout_ms()
                    await conn.execute(
                        f"SET LOCAL statement_timeout = '{int(timeout_ms)}'"
                    )

                    for event_id, event in unique_events:
                        try:
                            # Use a SAVEPOINT so a per-row error does not
                            # abort the entire transaction.  asyncpg's nested
                            # conn.transaction() emits SAVEPOINT / RELEASE.
                            async with conn.transaction():
                                await conn.execute(
                                    """
                                    INSERT INTO llm_call_metrics (
                                        correlation_id, session_id, run_id, model_id,
                                        prompt_tokens, completion_tokens, total_tokens,
                                        estimated_cost_usd, latency_ms,
                                        usage_source, usage_is_estimated,
                                        usage_raw, input_hash,
                                        code_version, contract_version, source
                                    ) VALUES (
                                        $1, $2, $3, $4, $5, $6, $7, $8, $9,
                                        $10, $11, $12, $13, $14, $15, $16
                                    )
                                    ON CONFLICT (id) DO NOTHING
                                    """,
                                    _safe_uuid(event.get("correlation_id")),
                                    event.get("session_id") or None,
                                    event.get("run_id"),
                                    str(event.get("model_id", "unknown")),
                                    _safe_int(event.get("prompt_tokens")),
                                    _safe_int(event.get("completion_tokens")),
                                    _safe_int(event.get("total_tokens")),
                                    _safe_decimal(event.get("estimated_cost_usd")),
                                    _safe_numeric_or_none(event.get("latency_ms")),
                                    _resolve_usage_source(event),
                                    bool(event.get("usage_is_estimated", False)),
                                    _safe_jsonb(event.get("usage_raw")),
                                    str(event.get("input_hash", ""))[:71]
                                    or None,  # Truncate to VARCHAR(71): 'sha256-' (7) + hex (64) = 71 chars
                                    str(event.get("code_version", ""))[:64] or None,
                                    str(event.get("contract_version", ""))[:64] or None,
                                    str(event.get("reporting_source", ""))[:255]
                                    or None,
                                )
                            # SAVEPOINT released -- row is persisted within the
                            # outer transaction. Record the dedup key for
                            # post-commit cache insertion.
                            written += 1
                            persisted_dedup_keys.append(event_id)
                        except Exception:
                            logger.warning(
                                "Failed to insert call metric row, skipping",
                                exc_info=True,
                                extra={
                                    "correlation_id": str(correlation_id),
                                    "model_id": event.get("model_id"),
                                },
                            )

            # Outer transaction committed successfully. Now mark persisted
            # events in the dedup cache so they are skipped on replay.
            async with self._dedup_lock:
                for dedup_key in persisted_dedup_keys:
                    self._mark_seen(dedup_key)

            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            logger.debug(
                "Wrote call metrics batch",
                extra={
                    "correlation_id": str(correlation_id),
                    "written": written,
                    "total": len(events),
                    "deduplicated": len(events) - len(unique_events),
                },
            )

        except Exception as exc:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("write_call_metrics", correlation_id)
            logger.exception(
                "Failed to write call metrics batch",
                extra={
                    "correlation_id": str(correlation_id),
                    "total": len(events),
                    "error": str(exc),
                },
            )
            raise

        return written

    async def write_cost_aggregates(
        self,
        events: list[dict[str, object]],
        correlation_id: UUID | None = None,
    ) -> int:
        """Aggregate LLM call metrics into the llm_cost_aggregates table.

        For each event, computes aggregation keys across multiple dimensions
        (session, model, repo, pattern) and upserts one row per key per window
        (24h, 7d, 30d). Uses additive upsert: existing rows have their
        totals incremented.

        The ``estimated_coverage_pct`` is computed as a running weighted average
        based on the proportion of events where ``usage_is_estimated`` is True.

        Args:
            events: List of event dictionaries from ContractLlmCallMetrics.
            correlation_id: Correlation ID for tracing.

        Returns:
            Number of aggregate rows upserted.

        Raises:
            InfraUnavailableError: If the circuit breaker is open.
        """
        if not events:
            return 0

        if correlation_id is None:
            correlation_id = uuid4()

        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("write_cost_aggregates", correlation_id)

        # Filter duplicates. The "agg:" prefix ensures the aggregation dedup
        # cache entries are distinct from the call-metrics entries (see the
        # parallel comment in write_call_metrics). This means each event
        # consumes two cache slots (_MAX_DEDUP_CACHE_SIZE bounds total entries,
        # not per-event count), which is an acceptable trade-off to keep the
        # two write paths independently idempotent.
        #
        # NOTE: _is_duplicate() only checks the cache; it does NOT add entries.
        # Events are marked as seen via _mark_seen() only AFTER successful
        # database persistence to prevent data loss on write failures.
        unique_events: list[dict[str, object]] = []
        agg_dedup_keys: list[str] = []
        seen_in_batch: set[str] = set()
        empty_dedup_fields_count = 0
        async with self._dedup_lock:
            for event in events:
                event_id = _derive_stable_dedup_key(event)
                if _has_empty_dedup_fields(event):
                    empty_dedup_fields_count += 1
                dedup_key = f"agg:{event_id}"
                if not self._is_duplicate(dedup_key) and dedup_key not in seen_in_batch:
                    unique_events.append(event)
                    agg_dedup_keys.append(dedup_key)
                    seen_in_batch.add(dedup_key)

        if empty_dedup_fields_count > 0:
            logger.warning(
                "Aggregation batch contained %d event(s) with all dedup fallback "
                "fields empty; deduplication may not work correctly for these events",
                empty_dedup_fields_count,
                extra={
                    "correlation_id": str(correlation_id),
                    "total_events": len(events),
                    "empty_dedup_fields_count": empty_dedup_fields_count,
                },
            )

        if not unique_events:
            return 0

        # Build aggregation rows from events, then pre-aggregate rows
        # sharing the same (aggregation_key, window) composite key. Without
        # pre-aggregation, the SQL ON CONFLICT DO UPDATE clause computes
        # weighted averages against stale intermediate values when the same
        # key appears multiple times in the batch.
        raw_agg_rows = _build_aggregation_rows(unique_events)
        agg_rows = _pre_aggregate_rows(raw_agg_rows)

        if not agg_rows:
            return 0

        upserted = 0
        try:
            async with self._pool.acquire() as conn:
                async with conn.transaction():
                    # Defense-in-depth: PostgreSQL SET LOCAL does not support
                    # $1 parameterized queries, so the value must be
                    # interpolated.  _statement_timeout_ms() returns a
                    # bounds-checked int clamped to [1, 600_000].  The
                    # explicit int() cast at the interpolation site is a
                    # second safety fence ensuring only a numeric literal
                    # can ever reach the SQL string, even if the return
                    # type of _statement_timeout_ms() were to change.
                    timeout_ms = self._statement_timeout_ms()
                    await conn.execute(
                        f"SET LOCAL statement_timeout = '{int(timeout_ms)}'"
                    )

                    for row in agg_rows:
                        try:
                            # Use a SAVEPOINT so a per-row error does not
                            # abort the entire transaction.  asyncpg's nested
                            # conn.transaction() emits SAVEPOINT / RELEASE.
                            async with conn.transaction():
                                await conn.execute(
                                    """
                                    INSERT INTO llm_cost_aggregates (
                                        aggregation_key, window,
                                        total_cost_usd, total_tokens, call_count,
                                        estimated_coverage_pct
                                    ) VALUES ($1, $2::cost_aggregation_window, $3, $4, $5, $6)
                                    ON CONFLICT (aggregation_key, window)
                                    DO UPDATE SET
                                        total_cost_usd = llm_cost_aggregates.total_cost_usd + EXCLUDED.total_cost_usd,
                                        total_tokens = llm_cost_aggregates.total_tokens + EXCLUDED.total_tokens,
                                        call_count = llm_cost_aggregates.call_count + EXCLUDED.call_count,
                                        estimated_coverage_pct = (
                                            (llm_cost_aggregates.estimated_coverage_pct * llm_cost_aggregates.call_count
                                             + EXCLUDED.estimated_coverage_pct * EXCLUDED.call_count)
                                            / NULLIF(llm_cost_aggregates.call_count + EXCLUDED.call_count, 0)
                                        )
                                    """,
                                    row["aggregation_key"],
                                    row["window"],
                                    row["total_cost_usd"],
                                    row["total_tokens"],
                                    row["call_count"],
                                    row["estimated_coverage_pct"],
                                )
                            upserted += 1
                        except Exception:
                            logger.warning(
                                "Failed to upsert aggregate row, skipping",
                                exc_info=True,
                                extra={
                                    "correlation_id": str(correlation_id),
                                    "aggregation_key": row.get("aggregation_key"),
                                    "window": row.get("window"),
                                },
                            )

            # Outer transaction committed successfully. Now mark persisted
            # events in the dedup cache so they are skipped on replay.
            async with self._dedup_lock:
                for dedup_key in agg_dedup_keys:
                    self._mark_seen(dedup_key)

            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            logger.debug(
                "Wrote cost aggregates batch",
                extra={
                    "correlation_id": str(correlation_id),
                    "upserted": upserted,
                    "total_rows": len(agg_rows),
                    "events_processed": len(unique_events),
                },
            )

        except Exception as exc:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    "write_cost_aggregates", correlation_id
                )
            logger.exception(
                "Failed to write cost aggregates batch",
                extra={
                    "correlation_id": str(correlation_id),
                    "total_rows": len(agg_rows),
                    "error": str(exc),
                },
            )
            raise

        return upserted


# =============================================================================
# Module-level helper functions
# =============================================================================


def _derive_stable_dedup_key(event: dict[str, object]) -> str:
    """Derive a stable deduplication key from event fields.

    When ``input_hash`` is present and at least 8 characters long, it is used
    directly. Shorter values are considered unreliable (e.g., truncated or
    placeholder) and fall through to the composite hash path. Otherwise, a
    composite key is built from ``correlation_id``, ``model_id``, and
    ``created_at`` (falling back to ``session_id``) and hashed with SHA-256
    to produce a deterministic, replay-safe dedup key. This ensures events
    without ``input_hash`` can still be deduplicated on consumer replay.

    Note: If multiple events share the same (correlation_id, model_id) pair
    and lack created_at/session_id, they will produce identical dedup keys.
    Producers should always include created_at to avoid silent deduplication.

    Args:
        event: Event dictionary from ContractLlmCallMetrics.

    Returns:
        A stable string suitable for dedup cache lookup.
    """
    input_hash = str(event.get("input_hash", "")).strip()
    if len(input_hash) >= 8:
        return input_hash

    if input_hash:
        logger.debug(
            "input_hash too short (%d chars) to be a reliable dedup key; "
            "falling through to composite hash",
            len(input_hash),
        )

    # Build a composite key from stable event fields
    parts = [
        str(event.get("correlation_id", "")),
        str(event.get("model_id", "")),
        str(event.get("created_at", "")),
        str(event.get("session_id", "")),
    ]

    # Log at debug level per-event to avoid noise when a producer
    # systematically omits these fields.  Callers may emit a batch-level
    # summary warning instead.
    if all(p == "" for p in parts):
        logger.debug(
            "All dedup fallback fields are empty; deduplication may not work "
            "correctly for this event (correlation_id, model_id, "
            "created_at, session_id are all absent or empty)",
        )

    composite = "|".join(parts)
    return hashlib.sha256(composite.encode("utf-8")).hexdigest()


def _has_empty_dedup_fields(event: dict[str, object]) -> bool:
    """Check whether all dedup fallback fields are empty for an event.

    Used by batch-level callers to count events with unreliable dedup keys
    and emit a single summary warning per batch.

    Args:
        event: Event dictionary.

    Returns:
        True if correlation_id, model_id, created_at, and session_id are
        all absent or empty AND input_hash is also absent or too short.
    """
    input_hash = str(event.get("input_hash", "")).strip()
    if len(input_hash) >= 8:
        return False
    parts = [
        str(event.get("correlation_id", "")),
        str(event.get("model_id", "")),
        str(event.get("created_at", "")),
        str(event.get("session_id", "")),
    ]
    return all(p == "" for p in parts)


# Pre-compiled regex for control characters (C0 + C1 control codes).
_CONTROL_CHAR_RE: re.Pattern[str] = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def _sanitize_dimension_value(value: str) -> str:
    """Sanitize a dimension value for use in aggregation keys.

    Strips control characters and replaces colons with underscores so that
    the ``<prefix>:<value>`` aggregation key format remains unambiguous.

    Args:
        value: Raw dimension value (e.g., model_id, session_id).

    Returns:
        Sanitized string safe for use as the value portion of an
        aggregation key.
    """
    # Replace colons to avoid ambiguity with the prefix:value separator
    sanitized = str(value).replace(":", "_")
    # Remove control characters
    sanitized = _CONTROL_CHAR_RE.sub("", sanitized)
    return sanitized.strip()


def _build_aggregation_rows(
    events: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Build aggregation rows from a batch of events.

    For each event, generates aggregation keys for each applicable dimension
    (session, model, repo, pattern) and each window (24h, 7d, 30d).

    Args:
        events: List of event dictionaries.

    Returns:
        List of aggregation row dictionaries ready for upsert.
    """
    rows: list[dict[str, object]] = []

    for event in events:
        cost = _safe_decimal(event.get("estimated_cost_usd")) or Decimal("0")
        tokens = _safe_int_or_zero(event.get("total_tokens"))
        is_estimated = bool(event.get("usage_is_estimated", False))
        estimated_pct = Decimal("100.00") if is_estimated else Decimal("0.00")

        # Build aggregation keys for this event
        keys: list[str] = []

        # Session dimension
        session_id = event.get("session_id")
        if session_id:
            keys.append(
                f"{_KEY_PREFIX_SESSION}:{_sanitize_dimension_value(str(session_id))}"
            )

        # Model dimension (always present in ContractLlmCallMetrics)
        model_id = event.get("model_id")
        if model_id:
            keys.append(
                f"{_KEY_PREFIX_MODEL}:{_sanitize_dimension_value(str(model_id))}"
            )

        # Repo dimension (from extensions if available)
        extensions = event.get("extensions")
        if isinstance(extensions, dict):
            repo = extensions.get("repo")
            if repo:
                keys.append(
                    f"{_KEY_PREFIX_REPO}:{_sanitize_dimension_value(str(repo))}"
                )

            # Pattern dimension
            pattern_id = extensions.get("pattern_id")
            if pattern_id:
                keys.append(
                    f"{_KEY_PREFIX_PATTERN}:{_sanitize_dimension_value(str(pattern_id))}"
                )

        # Generate one row per key per window
        for key in keys:
            for window in AGGREGATION_WINDOWS:
                rows.append(
                    {
                        "aggregation_key": key[:512],  # VARCHAR(512) limit
                        "window": window,
                        "total_cost_usd": cost,
                        "total_tokens": tokens,
                        "call_count": 1,
                        "estimated_coverage_pct": estimated_pct,
                    }
                )

    return rows


def _pre_aggregate_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Pre-aggregate rows sharing the same (aggregation_key, window).

    When a single batch contains multiple events that produce identical
    aggregation keys (e.g., two calls to the same model in one batch),
    ``_build_aggregation_rows`` emits separate rows for each. Sending
    these as sequential INSERT ... ON CONFLICT DO UPDATE statements
    within the same transaction causes the weighted average formula
    to operate on intermediate (already-updated) values rather than
    the pre-batch baseline.

    This function merges duplicate composite keys in Python before
    hitting the database, producing at most one row per
    ``(aggregation_key, window)`` pair. Metrics are summed additively
    and ``estimated_coverage_pct`` is computed as a proper weighted
    average over the merged call counts.

    Args:
        rows: Aggregation rows from ``_build_aggregation_rows``.

    Returns:
        Deduplicated list of aggregation rows, one per composite key.
    """
    if not rows:
        return []

    merged: dict[tuple[str, str], dict[str, object]] = {}

    for row in rows:
        key = (str(row["aggregation_key"]), str(row["window"]))

        if key not in merged:
            # First occurrence -- shallow-copy to avoid mutating the input.
            merged[key] = dict(row)
        else:
            existing = merged[key]
            existing_cost = existing["total_cost_usd"]
            row_cost = row["total_cost_usd"]
            assert isinstance(existing_cost, Decimal)
            assert isinstance(row_cost, Decimal)
            existing["total_cost_usd"] = existing_cost + row_cost

            existing_tokens = existing["total_tokens"]
            row_tokens = row["total_tokens"]
            assert isinstance(existing_tokens, int)
            assert isinstance(row_tokens, int)
            existing["total_tokens"] = existing_tokens + row_tokens

            existing_count = existing["call_count"]
            row_count = row["call_count"]
            assert isinstance(existing_count, int)
            assert isinstance(row_count, int)

            # Weighted average of estimated_coverage_pct
            existing_pct = existing["estimated_coverage_pct"]
            row_pct = row["estimated_coverage_pct"]
            assert isinstance(existing_pct, Decimal)
            assert isinstance(row_pct, Decimal)

            total_count = existing_count + row_count
            if total_count > 0:
                existing["estimated_coverage_pct"] = (
                    existing_pct * existing_count + row_pct * row_count
                ) / total_count
            # else: keep existing pct (both counts are 0, degenerate case)

            existing["call_count"] = total_count

    return list(merged.values())


def _safe_uuid(value: object) -> UUID | None:
    """Safely convert a value to UUID, returning None on failure."""
    if value is None:
        return None
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (ValueError, AttributeError):
        return None


def _safe_int(value: object) -> int | None:
    """Safely convert a value to int, returning None on failure.

    Rejects NaN and Infinity float values which would raise ValueError
    on int() conversion.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return int(value)
    try:
        return int(str(value))
    except (ValueError, TypeError):
        return None


def _safe_int_or_zero(value: object) -> int:
    """Safely convert a value to int, returning 0 if None or unconvertible.

    Unlike ``_safe_int(x) or 0``, this correctly preserves a legitimate
    zero value returned by ``_safe_int`` instead of coalescing it to 0
    via the falsy ``or`` branch.
    """
    result = _safe_int(value)
    return result if result is not None else 0


def _safe_numeric_or_none(value: object) -> float | None:
    """Safely convert a value to float, returning None if missing or invalid.

    Used for nullable ``NUMERIC`` columns (e.g., ``latency_ms``) where:
    - A valid numeric value should be preserved with sub-millisecond precision
      (rounded to 2 decimal places to match the ``NUMERIC(10, 2)`` column).
    - Missing or invalid values should map to SQL ``NULL`` (via ``None``),
      rather than a default like ``0``, to distinguish "no data" from
      "zero latency".

    asyncpg maps Python ``None`` to SQL ``NULL`` transparently, so callers
    can pass the return value directly as a query parameter.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        fval = float(value)
        if not math.isfinite(fval):
            return None
        return round(fval, 2)
    try:
        fval = float(str(value))
        if not math.isfinite(fval):
            return None
        return round(fval, 2)
    except (ValueError, TypeError):
        return None


def _safe_decimal(value: object) -> Decimal | None:
    """Safely convert a value to Decimal, returning None on failure.

    Rejects NaN and Infinity values to prevent aggregation corruption
    in PostgreSQL NUMERIC columns (NaN + x = NaN).
    """
    if value is None:
        return None
    if isinstance(value, Decimal):
        if not value.is_finite():
            return None
        return value
    try:
        result = Decimal(str(value))
        if not result.is_finite():
            return None
        return result
    except Exception:
        return None


def _safe_jsonb(value: object) -> str | None:
    """Safely convert a value to a JSONB-compatible string."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        try:
            return json.dumps(value, default=str)
        except Exception:
            return None
    return None


def _resolve_usage_source(event: dict[str, object]) -> str:
    """Resolve the usage_source enum value from an event.

    The PostgreSQL enum ``usage_source_type`` has values: API, ESTIMATED, MISSING.
    The ContractLlmCallMetrics uses ``usage_normalized.source`` with values:
    api, estimated, missing (lowercase). We normalize to uppercase for the DB enum.

    Args:
        event: Event dictionary.

    Returns:
        One of 'API', 'ESTIMATED', or 'MISSING'.
    """
    # Check usage_normalized.source first
    normalized = event.get("usage_normalized")
    if isinstance(normalized, dict):
        source = normalized.get("source", "")
        if isinstance(source, str) and source.upper() in (
            "API",
            "ESTIMATED",
            "MISSING",
        ):
            return source.upper()

    # Fall back to usage_is_estimated flag
    if event.get("usage_is_estimated"):
        return "ESTIMATED"

    # Check if any token data is present
    if event.get("total_tokens") or event.get("prompt_tokens"):
        return "API"

    return "MISSING"


__all__ = [
    "AGGREGATION_WINDOWS",
    "WriterLlmCostAggregationPostgres",
]
