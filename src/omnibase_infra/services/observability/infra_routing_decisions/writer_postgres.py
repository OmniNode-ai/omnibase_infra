# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# no-migration: migration 080_create_infra_routing_decisions already in this PR
"""PostgreSQL writer for infra routing decisions (OMN-8692).

Persists routing-decision events to infra_routing_decisions table.
"""

from __future__ import annotations

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
from omnibase_infra.services.observability.infra_routing_decisions.model_routing_decision_ingest import (
    ModelInfraRoutingDecisionIngest,
)

logger = logging.getLogger(__name__)


class WriterInfraRoutingDecisionsPostgres(MixinAsyncCircuitBreaker):
    """PostgreSQL writer for infra routing decisions.

    Writes routing-decision events to the infra_routing_decisions table.
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
        events: list[ModelInfraRoutingDecisionIngest],
        correlation_id: UUID | None = None,
    ) -> int:
        """Write a batch of routing-decision rows to infra_routing_decisions.

        Rows with a correlation_id use ON CONFLICT DO UPDATE (upsert by
        correlation_id). Rows without one are always inserted.

        OMN-16025: the INSERT names only the columns a routing decision actually
        carries. ``selection_mode``, ``fallback_indicator``, ``is_fallback``,
        ``candidates_evaluated``, ``candidate_providers``, ``session_id`` and
        ``latency_ms`` are NOT in the wire model, so they take the defaults
        migration 080 declares rather than being filled with a plausible
        constant -- a row asserting ``selection_mode='round_robin'`` about a
        contract-driven tier decision would be a fabricated fact, and this table
        exists to be read as evidence.

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

        context = ModelInfraErrorContext.with_correlation(
            correlation_id=op_correlation_id,
            transport_type=EnumInfraTransportType.DATABASE,
            operation="write_routing_decisions",
            target_name="infra_routing_decisions",
        )

        upsert_sql = """
            INSERT INTO infra_routing_decisions (
                correlation_id,
                selected_provider, selected_tier, selected_model,
                reason, task_type
            )
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (correlation_id)
            WHERE correlation_id IS NOT NULL
            DO UPDATE SET
                selected_provider  = EXCLUDED.selected_provider,
                selected_tier      = EXCLUDED.selected_tier,
                selected_model     = EXCLUDED.selected_model,
                reason             = EXCLUDED.reason,
                task_type          = EXCLUDED.task_type,
                projected_at       = NOW()
        """

        rows = [
            (
                event.correlation_id,
                event.selected_provider,
                event.selected_tier,
                event.selected_model,
                event.reason,
                event.task_type,
            )
            for event in events
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
