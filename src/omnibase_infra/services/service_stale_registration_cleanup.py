# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Stale registration cleanup service.

Detects and resets registration projections stuck in intermediate FSM states
past their deadlines. This handles the case where nodes registered under a
previous runtime version (pre-OMN-5132) that used the AWAITING_ACK handshake
flow. Since OMN-5132 eliminated the ACK round-trip, these nodes will never
receive an ACK and must be reset so the next introspection event triggers
direct-to-active re-registration.

The service transitions stale projections to ACK_TIMED_OUT, which is in the
_RETRIABLE_STATES set in RegistrationReducerService. This ensures:
  1. The node is no longer blocking new registrations
  2. The next introspection event will re-register the node as ACTIVE
  3. The transition is auditable (ACK_TIMED_OUT is a valid FSM state)

Usage:
    Run on startup or as a periodic maintenance task::

        cleanup = ServiceStaleRegistrationCleanup(pool)
        report = await cleanup.cleanup_stale_registrations()
        print(f"Reset {report.reset_count} stale registrations")

Related Tickets:
    - OMN-5821: Reset stale awaiting_ack registrations
    - OMN-5132: Direct-to-active registration FSM
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID, uuid4

import asyncpg

from omnibase_infra.enums import EnumInfraTransportType, EnumRegistrationState
from omnibase_infra.errors import (
    InfraConnectionError,
    ModelInfraErrorContext,
    RuntimeHostError,
)
from omnibase_infra.utils import sanitize_error_message

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class StaleCleanupReport:
    """Report from a stale registration cleanup run.

    Attributes:
        reset_count: Number of projections transitioned to ACK_TIMED_OUT.
        scanned_count: Number of projections examined.
        correlation_id: Correlation ID for tracing.
        executed_at: Timestamp of the cleanup run.
    """

    reset_count: int
    scanned_count: int
    correlation_id: UUID
    executed_at: datetime


class ServiceStaleRegistrationCleanup:
    """Service for cleaning up stale registration projections.

    Queries for projections in AWAITING_ACK or ACCEPTED state whose
    ack_deadline has passed, and transitions them to ACK_TIMED_OUT.

    This service performs direct SQL updates rather than going through the
    event-driven FSM, because the stale entries are artifacts of a prior
    runtime version and the normal event flow cannot reach them (no
    introspection events are being produced for these nodes).
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        """Initialize with an asyncpg connection pool.

        Args:
            pool: asyncpg connection pool for database access.
        """
        self._pool = pool

    async def cleanup_stale_registrations(
        self,
        correlation_id: UUID | None = None,
        domain: str = "registration",
    ) -> StaleCleanupReport:
        """Find and reset stale registrations past their ack deadline.

        Transitions projections in AWAITING_ACK or ACCEPTED state with
        expired ack_deadline to ACK_TIMED_OUT. This is a one-shot cleanup
        intended to run on startup.

        Args:
            correlation_id: Optional correlation ID for tracing.
            domain: Domain namespace (default: "registration").

        Returns:
            StaleCleanupReport with counts and metadata.

        Raises:
            InfraConnectionError: If database connection fails.
            RuntimeHostError: If query fails for other reasons.
        """
        corr_id = correlation_id or uuid4()
        now = datetime.now(UTC)
        ctx = ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="cleanup_stale_registrations",
            target_name="service.stale_registration_cleanup",
            correlation_id=corr_id,
        )

        stale_states = [
            EnumRegistrationState.AWAITING_ACK.value,
            EnumRegistrationState.ACCEPTED.value,
        ]

        # Count how many are stale
        count_sql = """
            SELECT count(*) FROM registration_projections
            WHERE domain = $1
              AND current_state = ANY($2::text[])
              AND ack_deadline IS NOT NULL
              AND ack_deadline < $3
        """

        # Update stale to ACK_TIMED_OUT
        update_sql = """
            UPDATE registration_projections
            SET current_state = $1,
                updated_at = $2
            WHERE domain = $3
              AND current_state = ANY($4::text[])
              AND ack_deadline IS NOT NULL
              AND ack_deadline < $5
        """

        try:
            async with self._pool.acquire() as conn:
                scanned_count = await conn.fetchval(
                    count_sql, domain, stale_states, now
                )
                scanned_count = scanned_count or 0

                if scanned_count == 0:
                    logger.info(
                        "No stale registrations found",
                        extra={
                            "correlation_id": str(corr_id),
                            "domain": domain,
                        },
                    )
                    return StaleCleanupReport(
                        reset_count=0,
                        scanned_count=0,
                        correlation_id=corr_id,
                        executed_at=now,
                    )

                result = await conn.execute(
                    update_sql,
                    EnumRegistrationState.ACK_TIMED_OUT.value,
                    now,
                    domain,
                    stale_states,
                    now,
                )

                # asyncpg returns "UPDATE N" string
                reset_count = int(result.split()[-1]) if result else 0

            logger.info(
                "Cleaned up stale registrations",
                extra={
                    "reset_count": reset_count,
                    "scanned_count": scanned_count,
                    "correlation_id": str(corr_id),
                    "domain": domain,
                    "target_state": EnumRegistrationState.ACK_TIMED_OUT.value,
                },
            )

            return StaleCleanupReport(
                reset_count=reset_count,
                scanned_count=scanned_count,
                correlation_id=corr_id,
                executed_at=now,
            )

        except asyncpg.PostgresConnectionError as e:
            raise InfraConnectionError(
                "Failed to connect for stale registration cleanup",
                context=ctx,
            ) from e

        except Exception as e:
            logger.warning(
                "Stale registration cleanup failed: %s",
                sanitize_error_message(e),
                extra={
                    "correlation_id": str(corr_id),
                    "error_type": type(e).__name__,
                },
            )
            raise RuntimeHostError(
                f"Stale registration cleanup failed: {type(e).__name__}",
                context=ctx,
            ) from e


__all__: list[str] = [
    "ServiceStaleRegistrationCleanup",
    "StaleCleanupReport",
]
