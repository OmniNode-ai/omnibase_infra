# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler for consumer health triage with graduated response.

Graduated response policy:
    1st occurrence  -> Slack WARNING notification
    2nd occurrence  -> Slack REPEATED notification
    3rd in 30 min   -> Emit restart command (gated by ENABLE_CONSUMER_AUTO_RESTART)
    Restart failure -> Create Linear ticket

All state is tracked in PostgreSQL tables:
    - consumer_health_triage: incident tracking by fingerprint
    - consumer_restart_state: restart rate limiting by consumer identity

Gated by ENABLE_CONSUMER_HEALTH_TRIAGE env var (default off).

Related Tickets:
    - OMN-5520: Create NodeConsumerHealthTriageEffect
    - OMN-5529: Runtime Health Event Pipeline (epic)
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.event_bus.topic_constants import TOPIC_CONSUMER_RESTART_CMD
from omnibase_infra.models.health.enum_consumer_incident_state import (
    EnumConsumerIncidentState,
)
from omnibase_infra.models.health.model_consumer_health_event import (
    ModelConsumerHealthEvent,
)
from omnibase_infra.models.health.model_consumer_restart_command import (
    ModelConsumerRestartCommand,
)

if TYPE_CHECKING:
    from aiokafka import AIOKafkaProducer
    from asyncpg import Pool

logger = logging.getLogger(__name__)

# Graduated response thresholds
_RESTART_THRESHOLD = 3  # 3rd occurrence triggers restart
_RESTART_WINDOW_MINUTES = 30  # Window for counting occurrences
_MAX_RESTARTS_PER_WINDOW = 2  # Max restarts in the window


class ModelTriageResult(BaseModel):
    """Result of a consumer health triage operation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fingerprint: str = Field(..., description="Event fingerprint that was triaged.")
    action: str = Field(
        ...,
        description="Action taken: 'slack_warning', 'slack_repeated', 'restart_command', 'linear_ticket', 'suppressed'.",
    )
    incident_state: str = Field(..., description="New incident state after triage.")
    occurrence_count: int = Field(
        ..., description="Total occurrences for this fingerprint."
    )


class HandlerConsumerHealthTriage:
    """Handler that applies graduated triage response to consumer health events.

    Dependencies (injected via constructor):
        - db_pool: asyncpg connection pool for PostgreSQL
        - producer: AIOKafkaProducer for emitting restart commands
        - slack_handler: Optional callable for Slack notifications
        - linear_handler: Optional callable for Linear ticket creation

    Feature Flags:
        - ENABLE_CONSUMER_HEALTH_TRIAGE: Master gate (default off)
        - ENABLE_CONSUMER_AUTO_RESTART: Restart command gate (default off)
    """

    def __init__(
        self,
        db_pool: Pool,
        producer: AIOKafkaProducer | None = None,
        *,
        slack_handler: Callable[[str], Awaitable[None]] | None = None,
        linear_handler: Callable[..., Awaitable[None]] | None = None,
    ) -> None:
        """Initialize the triage handler.

        Args:
            db_pool: asyncpg connection pool.
            producer: AIOKafkaProducer for restart commands (optional).
            slack_handler: Async callable accepting a message string.
            linear_handler: Async callable accepting title and description kwargs.
        """
        self._db_pool = db_pool
        self._producer = producer
        self._slack_handler = slack_handler
        self._linear_handler = linear_handler

    @staticmethod
    def is_enabled() -> bool:
        """Check if consumer health triage is enabled."""
        return os.environ.get(  # ONEX_EXCLUDE: env
            "ENABLE_CONSUMER_HEALTH_TRIAGE", ""
        ).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def is_auto_restart_enabled() -> bool:
        """Check if automated restart is enabled."""
        return os.environ.get(  # ONEX_EXCLUDE: env
            "ENABLE_CONSUMER_AUTO_RESTART", ""
        ).strip().lower() in {"1", "true", "yes", "on"}

    async def handle(self, event: ModelConsumerHealthEvent) -> ModelTriageResult:
        """Apply graduated triage to a consumer health event.

        Args:
            event: The consumer health event to triage.

        Returns:
            ModelTriageResult with action taken and new state.
        """
        if not self.is_enabled():
            return ModelTriageResult(
                fingerprint=event.fingerprint,
                action="suppressed",
                incident_state=EnumConsumerIncidentState.OPEN,
                occurrence_count=0,
            )

        # Upsert incident in consumer_health_triage table
        incident = await self._upsert_incident(event)
        occurrence_count = int(incident["occurrence_count"])

        # Graduated response logic
        if occurrence_count >= _RESTART_THRESHOLD and self.is_auto_restart_enabled():
            # Check restart rate limit before issuing restart
            can_restart = await self._check_restart_rate_limit(event)
            if can_restart:
                await self._emit_restart_command(event)
                await self._update_incident_state(
                    event.fingerprint, EnumConsumerIncidentState.RESTART_PENDING
                )
                return ModelTriageResult(
                    fingerprint=event.fingerprint,
                    action="restart_command",
                    incident_state=EnumConsumerIncidentState.RESTART_PENDING,
                    occurrence_count=occurrence_count,
                )
            else:
                # Rate limited — escalate to Linear ticket
                await self._create_linear_ticket(event, occurrence_count)
                await self._update_incident_state(
                    event.fingerprint, EnumConsumerIncidentState.TICKETED
                )
                return ModelTriageResult(
                    fingerprint=event.fingerprint,
                    action="linear_ticket",
                    incident_state=EnumConsumerIncidentState.TICKETED,
                    occurrence_count=occurrence_count,
                )
        elif occurrence_count >= 2:
            # 2nd+ occurrence — Slack repeated notification
            await self._send_slack_notification(event, repeated=True)
            return ModelTriageResult(
                fingerprint=event.fingerprint,
                action="slack_repeated",
                incident_state=EnumConsumerIncidentState.OPEN,
                occurrence_count=occurrence_count,
            )
        else:
            # 1st occurrence — Slack warning
            await self._send_slack_notification(event, repeated=False)
            return ModelTriageResult(
                fingerprint=event.fingerprint,
                action="slack_warning",
                incident_state=EnumConsumerIncidentState.OPEN,
                occurrence_count=occurrence_count,
            )

    async def _upsert_incident(
        self, event: ModelConsumerHealthEvent
    ) -> dict[str, object]:
        """Upsert incident row in consumer_health_triage table.

        Returns dict with at least 'occurrence_count'.
        """
        async with self._db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO consumer_health_triage (
                    fingerprint, consumer_id, consumer_group, topic,
                    event_type, severity, incident_state,
                    occurrence_count, first_seen_at, last_seen_at,
                    error_message, service_name, hostname, correlation_id
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, 1, NOW(), NOW(), $8, $9, $10, $11)
                ON CONFLICT (fingerprint) WHERE incident_state IN ('open', 'acknowledged')
                DO UPDATE SET
                    occurrence_count = consumer_health_triage.occurrence_count + 1,
                    last_seen_at = NOW(),
                    severity = EXCLUDED.severity,
                    error_message = EXCLUDED.error_message,
                    correlation_id = EXCLUDED.correlation_id
                RETURNING occurrence_count, incident_state
                """,
                event.fingerprint,
                event.consumer_identity,
                event.consumer_group,
                event.topic,
                event.event_type.value,
                event.severity.value,
                EnumConsumerIncidentState.OPEN.value,
                event.error_message,
                event.service_label,
                event.hostname,
                str(event.correlation_id),
            )
            if row is None:
                # Insert without ON CONFLICT (fingerprint has resolved/ticketed incident)
                row = await conn.fetchrow(
                    """
                    INSERT INTO consumer_health_triage (
                        fingerprint, consumer_id, consumer_group, topic,
                        event_type, severity, incident_state,
                        occurrence_count, first_seen_at, last_seen_at,
                        error_message, service_name, hostname, correlation_id
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, 1, NOW(), NOW(), $8, $9, $10, $11)
                    RETURNING occurrence_count, incident_state
                    """,
                    event.fingerprint,
                    event.consumer_identity,
                    event.consumer_group,
                    event.topic,
                    event.event_type.value,
                    event.severity.value,
                    EnumConsumerIncidentState.OPEN.value,
                    event.error_message,
                    event.service_label,
                    event.hostname,
                    str(event.correlation_id),
                )
            return (
                dict(row) if row else {"occurrence_count": 1, "incident_state": "open"}
            )

    async def _update_incident_state(
        self, fingerprint: str, state: EnumConsumerIncidentState
    ) -> None:
        """Update the incident state for a fingerprint."""
        async with self._db_pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE consumer_health_triage
                SET incident_state = $1, escalated_at = CASE WHEN $1 IN ('restart_pending', 'ticketed') THEN NOW() ELSE escalated_at END
                WHERE fingerprint = $2 AND incident_state NOT IN ('resolved')
                """,
                state.value,
                fingerprint,
            )

    async def _check_restart_rate_limit(self, event: ModelConsumerHealthEvent) -> bool:
        """Check if a restart is allowed for this consumer within the rate window.

        Returns True if restart is allowed.
        """
        async with self._db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT restart_count_30min, restart_window_start
                FROM consumer_restart_state
                WHERE consumer_id = $1 AND consumer_group = $2 AND topic = $3
                """,
                event.consumer_identity,
                event.consumer_group,
                event.topic,
            )
            if row is None:
                # No prior restarts — allowed
                return True

            window_start = row["restart_window_start"]
            count = row["restart_count_30min"]
            window_cutoff = datetime.now(UTC) - timedelta(
                minutes=_RESTART_WINDOW_MINUTES
            )

            if window_start < window_cutoff:
                # Window expired — reset
                return True

            return count < _MAX_RESTARTS_PER_WINDOW

    async def _emit_restart_command(self, event: ModelConsumerHealthEvent) -> None:
        """Emit a restart command to the consumer restart topic."""
        if self._producer is None:
            logger.warning("Cannot emit restart command: no producer available")
            return

        command = ModelConsumerRestartCommand(
            consumer_identity=event.consumer_identity,
            consumer_group=event.consumer_group,
            topic=event.topic,
            reason=f"Graduated triage: {event.event_type.value} occurred {_RESTART_THRESHOLD}+ times",
            fingerprint=event.fingerprint,
            correlation_id=event.correlation_id,
        )

        try:
            payload = json.dumps(command.model_dump(mode="json")).encode("utf-8")
            await self._producer.send(TOPIC_CONSUMER_RESTART_CMD, value=payload)
            logger.info(
                "Emitted restart command for consumer %s",
                event.consumer_identity,
            )

            # Update restart state
            async with self._db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO consumer_restart_state (
                        consumer_id, consumer_group, topic,
                        last_restart_at, restart_count_30min, restart_window_start
                    ) VALUES ($1, $2, $3, NOW(), 1, NOW())
                    ON CONFLICT (consumer_id, consumer_group, topic)
                    DO UPDATE SET
                        last_restart_at = NOW(),
                        restart_count_30min = CASE
                            WHEN consumer_restart_state.restart_window_start < NOW() - INTERVAL '30 minutes'
                            THEN 1
                            ELSE consumer_restart_state.restart_count_30min + 1
                        END,
                        restart_window_start = CASE
                            WHEN consumer_restart_state.restart_window_start < NOW() - INTERVAL '30 minutes'
                            THEN NOW()
                            ELSE consumer_restart_state.restart_window_start
                        END,
                        updated_at = NOW()
                    """,
                    event.consumer_identity,
                    event.consumer_group,
                    event.topic,
                )
        except Exception:  # noqa: BLE001 - best-effort restart command
            logger.warning(
                "Failed to emit restart command for %s",
                event.consumer_identity,
                exc_info=True,
            )

    async def _send_slack_notification(
        self, event: ModelConsumerHealthEvent, *, repeated: bool
    ) -> None:
        """Send a Slack notification for the health event."""
        if self._slack_handler is None:
            logger.debug("Slack handler not configured, skipping notification")
            return

        try:
            prefix = "REPEATED" if repeated else "WARNING"
            message = (
                f"[{prefix}] Consumer health: {event.event_type.value} "
                f"on {event.topic} ({event.consumer_identity})"
            )
            if event.error_message:
                message += f"\nError: {event.error_message}"
            await self._slack_handler(message)
        except Exception:  # noqa: BLE001 - best-effort notification
            logger.debug("Failed to send Slack notification", exc_info=True)

    async def _create_linear_ticket(
        self, event: ModelConsumerHealthEvent, occurrence_count: int
    ) -> None:
        """Create a Linear ticket for an unresolvable consumer health issue."""
        if self._linear_handler is None:
            logger.debug("Linear handler not configured, skipping ticket creation")
            return

        try:
            title = (
                f"Consumer health: {event.event_type.value} on {event.topic} "
                f"({occurrence_count} occurrences)"
            )
            description = (
                f"Consumer: {event.consumer_identity}\n"
                f"Group: {event.consumer_group}\n"
                f"Topic: {event.topic}\n"
                f"Event: {event.event_type.value}\n"
                f"Severity: {event.severity.value}\n"
                f"Occurrences: {occurrence_count}\n"
                f"Error: {event.error_message}\n"
                f"Fingerprint: {event.fingerprint}\n"
            )
            await self._linear_handler(title=title, description=description)
        except Exception:  # noqa: BLE001 - best-effort ticket creation
            logger.debug("Failed to create Linear ticket", exc_info=True)


__all__ = ["HandlerConsumerHealthTriage", "ModelTriageResult"]
