# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# no-migration: migration 080_create_infra_routing_decisions already in this PR
"""PostgreSQL writer for infra routing decisions (OMN-8692).

Persists routing-decided events to infra_routing_decisions table.
"""

from __future__ import annotations

import json
import logging
from uuid import UUID, uuid4

import asyncpg

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    ModelInfraErrorContext,
    ModelTimeoutErrorContext,
    RuntimeHostError,
)
from omnibase_infra.mixins import MixinAsyncCircuitBreaker

logger = logging.getLogger(__name__)


class WriterInfraRoutingDecisionsPostgres(MixinAsyncCircuitBreaker):
    """PostgreSQL writer for infra routing decisions.

    Writes routing-decided events to the infra_routing_decisions table.
    UPSERT key: correlation_id (partial unique index — NULL correlation_ids
    are always inserted as new rows).
    """

    DEFAULT_QUERY_TIMEOUT_SECONDS: float = 30.0

    def __init__(
        self,
        pool: asyncpg.Pool,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_reset_timeout: float = 60.0,
        circuit_breaker_half_open_successes: int = 1,
        query_timeout: float | None = None,
    ) -> None:
        self._pool = pool
        self._query_timeout = query_timeout or self.DEFAULT_QUERY_TIMEOUT_SECONDS
        self._init_circuit_breaker(
            threshold=circuit_breaker_threshold,
            reset_timeout=circuit_breaker_reset_timeout,
            service_name="infra-routing-decisions-writer",
            transport_type=EnumInfraTransportType.DATABASE,
            half_open_successes=circuit_breaker_half_open_successes,
        )

    async def write_routing_decisions(
        self,
        events: list[dict[str, object]],
        correlation_id: UUID | None = None,
    ) -> int:
        """Write a batch of routing-decided events to infra_routing_decisions.

        Events with a correlation_id use ON CONFLICT DO UPDATE (upsert by
        correlation_id). Events without a correlation_id are always inserted.

        Returns the number of rows in the batch.
        """
        if not events:
            return 0

        op_correlation_id = correlation_id or uuid4()

        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(
                operation="write_routing_decisions",
                correlation_id=op_correlation_id,
            )

        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="write_routing_decisions",
            target_name="infra_routing_decisions",
            correlation_id=op_correlation_id,
        )

        upsert_sql = """
            INSERT INTO infra_routing_decisions (
                correlation_id,
                selected_provider, selected_tier, selected_model,
                selection_mode, fallback_indicator, is_fallback, reason,
                candidates_evaluated, candidate_providers,
                task_type, session_id, latency_ms
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            ON CONFLICT (correlation_id)
            WHERE correlation_id IS NOT NULL
            DO UPDATE SET
                selected_provider  = EXCLUDED.selected_provider,
                selected_tier      = EXCLUDED.selected_tier,
                selected_model     = EXCLUDED.selected_model,
                selection_mode     = EXCLUDED.selection_mode,
                fallback_indicator = EXCLUDED.fallback_indicator,
                is_fallback        = EXCLUDED.is_fallback,
                reason             = EXCLUDED.reason,
                candidates_evaluated = EXCLUDED.candidates_evaluated,
                candidate_providers = EXCLUDED.candidate_providers,
                task_type          = EXCLUDED.task_type,
                session_id         = EXCLUDED.session_id,
                latency_ms         = EXCLUDED.latency_ms,
                projected_at       = NOW()
        """

        def _safe_uuid(val: object) -> UUID | None:
            if val is None:
                return None
            if isinstance(val, UUID):
                return val
            try:
                return UUID(str(val))
            except (ValueError, AttributeError):
                return None

        def _safe_str(val: object, default: str = "") -> str:
            if val is None:
                return default
            return str(val)

        def _safe_bool(val: object, default: bool = False) -> bool:
            if isinstance(val, bool):
                return val
            if isinstance(val, int):
                return bool(val)
            return default

        def _safe_int(val: object, default: int = 0) -> int:
            if val is None:
                return default
            try:
                return int(str(val))
            except (TypeError, ValueError):
                return default

        def _safe_float_or_none(val: object) -> float | None:
            if val is None:
                return None
            try:
                return float(str(val))
            except (TypeError, ValueError):
                return None

        def _serialize_list(val: object) -> str | None:
            if val is None:
                return None
            if isinstance(val, (list, tuple)):
                return json.dumps(list(val))
            return None

        rows = [
            (
                _safe_uuid(e.get("correlation_id")),
                _safe_str(e.get("selected_provider")),
                _safe_str(e.get("selected_tier")),
                _safe_str(e.get("selected_model")),
                _safe_str(e.get("selection_mode"), "round_robin"),
                _safe_bool(e.get("fallback_indicator")),
                _safe_bool(e.get("is_fallback")),
                _safe_str(e.get("reason")),
                _safe_int(e.get("candidates_evaluated")),
                _serialize_list(e.get("candidate_providers")),
                e.get("task_type"),
                e.get("session_id"),
                _safe_float_or_none(e.get("latency_ms")),
            )
            for e in events
        ]

        try:
            async with self._pool.acquire() as conn:
                await conn.executemany(upsert_sql, rows)

            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            logger.debug(
                "Wrote infra routing decisions batch",
                extra={
                    "count": len(events),
                    "correlation_id": str(op_correlation_id),
                },
            )
            return len(events)

        except asyncpg.QueryCanceledError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="write_routing_decisions",
                    correlation_id=op_correlation_id,
                )
            raise InfraTimeoutError(
                "Write infra routing decisions timed out",
                context=ModelTimeoutErrorContext(
                    transport_type=context.transport_type,
                    operation=context.operation,
                    target_name=context.target_name,
                    correlation_id=context.correlation_id,
                    timeout_seconds=self._query_timeout,
                ),
            ) from e
        except asyncpg.PostgresConnectionError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="write_routing_decisions",
                    correlation_id=op_correlation_id,
                )
            raise InfraConnectionError(
                "Database connection failed during write_routing_decisions",
                context=context,
            ) from e
        except asyncpg.PostgresError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="write_routing_decisions",
                    correlation_id=op_correlation_id,
                )
            raise RuntimeHostError(
                f"Database error during write_routing_decisions: {type(e).__name__}",
                context=context,
            ) from e


__all__ = ["WriterInfraRoutingDecisionsPostgres"]
