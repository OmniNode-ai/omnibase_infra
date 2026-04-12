# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# no-migration: delegation_events table already exists via migration 0007_delegation_events.sql (OMN-8512)
"""PostgreSQL writer for delegation projection.

Writes task-delegated events to the delegation_events table with
UPSERT-on-correlation_id semantics for idempotency.

Related Tickets:
    - OMN-8532: Add delegation projection consumer service
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from uuid import UUID, uuid4

import asyncpg

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.mixins import MixinAsyncCircuitBreaker

logger = logging.getLogger(__name__)

_MAX_DEDUP_CACHE_SIZE: int = 50_000

_UPSERT_SQL = """
INSERT INTO delegation_events (
    correlation_id,
    session_id,
    timestamp,
    task_type,
    delegated_to,
    model_name,
    delegated_by,
    quality_gate_passed,
    quality_gates_checked,
    quality_gates_failed,
    delegation_latency_ms,
    repo,
    is_shadow,
    llm_call_id
) VALUES (
    $1, $2, $3::timestamptz, $4, $5, $6, $7, $8, $9::jsonb, $10::jsonb, $11, $12, $13, $14
)
ON CONFLICT (correlation_id) DO UPDATE SET
    session_id = EXCLUDED.session_id,
    timestamp = EXCLUDED.timestamp,
    task_type = EXCLUDED.task_type,
    delegated_to = EXCLUDED.delegated_to,
    model_name = EXCLUDED.model_name,
    delegated_by = EXCLUDED.delegated_by,
    quality_gate_passed = EXCLUDED.quality_gate_passed,
    quality_gates_checked = EXCLUDED.quality_gates_checked,
    quality_gates_failed = EXCLUDED.quality_gates_failed,
    delegation_latency_ms = EXCLUDED.delegation_latency_ms,
    repo = EXCLUDED.repo,
    is_shadow = EXCLUDED.is_shadow,
    llm_call_id = EXCLUDED.llm_call_id
"""


class WriterDelegationProjectionPostgres(MixinAsyncCircuitBreaker):
    """PostgreSQL writer for delegation_events projection.

    UPSERT on correlation_id for idempotency. In-memory dedup cache
    bounds memory usage for replay scenarios.
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
        self._pool = pool
        self._query_timeout = query_timeout
        self._dedup_cache: OrderedDict[str, bool] = OrderedDict()

        self._init_circuit_breaker(
            threshold=circuit_breaker_threshold,
            reset_timeout=circuit_breaker_reset_timeout,
            service_name="delegation-projection-writer",
            transport_type=EnumInfraTransportType.DATABASE,
            half_open_successes=circuit_breaker_half_open_successes,
        )

    def _is_duplicate(self, correlation_id: str) -> bool:
        if correlation_id in self._dedup_cache:
            self._dedup_cache.move_to_end(correlation_id)
            return True
        return False

    def _mark_seen(self, correlation_id: str) -> None:
        self._dedup_cache[correlation_id] = True
        while len(self._dedup_cache) > _MAX_DEDUP_CACHE_SIZE:
            self._dedup_cache.popitem(last=False)

    async def write_events(
        self,
        events: list[dict[str, object]],
        correlation_id: UUID | None = None,
    ) -> int:
        """Write a batch of task-delegated events to delegation_events.

        Returns number of rows upserted (skips in-memory duplicates).

        Raises:
            InfraUnavailableError: If the circuit breaker is open.
        """
        if not events:
            return 0

        if correlation_id is None:
            correlation_id = uuid4()

        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("write_events", correlation_id)

        unique_events: list[tuple[str, dict[str, object]]] = []
        seen_in_batch: set[str] = set()
        for event in events:
            cid = str(event.get("correlation_id", ""))
            if not cid:
                continue
            if not self._is_duplicate(cid) and cid not in seen_in_batch:
                unique_events.append((cid, event))
                seen_in_batch.add(cid)

        if not unique_events:
            return 0

        written = 0
        persisted_keys: list[str] = []
        try:
            async with self._pool.acquire() as conn:
                async with conn.transaction():
                    for cid, ev in unique_events:
                        quality_gates_checked = ev.get("quality_gates_checked")
                        quality_gates_failed = ev.get("quality_gates_failed")
                        await conn.execute(
                            _UPSERT_SQL,
                            cid,
                            ev.get("session_id"),
                            ev.get("timestamp"),
                            str(ev.get("task_type", "")),
                            str(ev.get("delegated_to", "")),
                            str(ev.get("model_name", "")),
                            ev.get("delegated_by"),
                            bool(ev.get("quality_gate_passed", False)),
                            json.dumps(quality_gates_checked)
                            if quality_gates_checked is not None
                            else None,
                            json.dumps(quality_gates_failed)
                            if quality_gates_failed is not None
                            else None,
                            ev.get("delegation_latency_ms"),
                            ev.get("repo"),
                            bool(ev.get("is_shadow", False)),
                            ev.get("llm_call_id") or None,
                            timeout=self._query_timeout,
                        )
                        persisted_keys.append(cid)
                        written += 1

        except Exception:
            logger.exception(
                "Failed to write delegation_events batch",
                extra={
                    "correlation_id": str(correlation_id),
                    "count": len(unique_events),
                },
            )
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("write_events", correlation_id)
            raise

        for cid in persisted_keys:
            self._mark_seen(cid)

        async with self._circuit_breaker_lock:
            await self._reset_circuit_breaker()

        logger.debug(
            "Wrote %d delegation_events rows",
            written,
            extra={"correlation_id": str(correlation_id)},
        )
        return written
