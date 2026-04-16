# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler for runtime error triage with first-match-wins rule engine.

Applies a configurable set of triage rules to runtime error events.
Rules are evaluated in priority order; first match determines the
triage action (suppress, alert, ticket).

Includes cross-layer correlation with Layer 1 consumer health incidents:
when a runtime error's logger_family matches a known Kafka consumer pattern,
the handler queries consumer_health_triage for active incidents with the
same consumer identity.

After triage, emits an ``error-triaged.v1`` event to Kafka for downstream
consumers (omnidash /runtime-errors dashboard).

Related Tickets:
    - OMN-5522: Create NodeRuntimeErrorTriageEffect
    - OMN-5529: Runtime Health Event Pipeline (epic)
    - OMN-5650: Error triage result emission
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.health.enum_runtime_error_category import (
    EnumRuntimeErrorCategory,
)
from omnibase_infra.models.health.model_runtime_error_event import (
    ModelRuntimeErrorEvent,
)
from omnibase_infra.nodes.node_runtime_error_triage_effect.models.model_triage_rule import (
    DEFAULT_TRIAGE_RULES,
    ModelTriageRule,
)

if TYPE_CHECKING:
    from asyncpg import Pool

    from omnibase_infra.protocols.protocol_event_bus_like import ProtocolEventBusLike

logger = logging.getLogger(__name__)

# Categories that may correlate with Layer 1 consumer health events
_KAFKA_CATEGORIES = {
    EnumRuntimeErrorCategory.KAFKA_CONSUMER,
    EnumRuntimeErrorCategory.KAFKA_PRODUCER,
}


class ModelRuntimeErrorTriageResult(BaseModel):
    """Result of a runtime error triage operation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fingerprint: str = Field(..., description="Event fingerprint that was triaged.")
    action: str = Field(
        ...,
        description="Action taken: 'suppress', 'alert', 'ticket'.",
    )
    matched_rule: str = Field(
        ..., description="Identifier of the triage rule that matched."
    )
    incident_state: str = Field(..., description="New incident state after triage.")
    occurrence_count: int = Field(
        ..., description="Total occurrences for this fingerprint."
    )
    correlated_consumer_fingerprint: str | None = Field(
        default=None,
        description="Fingerprint of correlated Layer 1 consumer health incident.",
    )


class HandlerRuntimeErrorTriage:
    """Handler that applies first-match-wins triage to runtime error events.

    Dependencies (injected via constructor):
        - db_pool: asyncpg connection pool for PostgreSQL
        - rules: Optional list of triage rules (defaults to DEFAULT_TRIAGE_RULES)
        - slack_handler: Optional callable for Slack notifications
        - linear_handler: Optional callable for Linear ticket creation
    """

    def __init__(
        self,
        db_pool: Pool | None = None,
        *,
        rules: list[ModelTriageRule] | None = None,
        slack_handler: Callable[[str], Awaitable[None]] | None = None,
        linear_handler: Callable[..., Awaitable[None]] | None = None,
        event_bus: ProtocolEventBusLike | None = None,
    ) -> None:
        """Initialize the triage handler.

        Args:
            db_pool: asyncpg connection pool. When None (auto-wired path),
                handle() returns a suppressed result with a warning log.
            rules: Triage rules in priority order. Defaults to DEFAULT_TRIAGE_RULES.
            slack_handler: Async callable accepting a message string.
            linear_handler: Async callable accepting title and description kwargs.
            event_bus: Optional event bus for emitting error-triaged events.
                If ``None``, triage events are not emitted (DB-only mode).
        """
        self._db_pool = db_pool
        self._rules = sorted(rules or DEFAULT_TRIAGE_RULES, key=lambda r: r.priority)
        self._slack_handler = slack_handler
        self._linear_handler = linear_handler
        self._event_bus = event_bus

        # Resolve triage topic lazily to avoid import cycles at module level
        self._triage_topic: str | None = None

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role: infrastructure handler for error triage."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Behavioral classification: side-effecting triage operation."""
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self, event: ModelRuntimeErrorEvent
    ) -> ModelRuntimeErrorTriageResult:
        """Apply first-match-wins triage to a runtime error event.

        Args:
            event: The runtime error event to triage.

        Returns:
            ModelRuntimeErrorTriageResult with action taken and metadata.
        """
        if self._db_pool is None:
            logger.warning(
                "HandlerRuntimeErrorTriage: db_pool not configured, suppressing event"
            )
            return ModelRuntimeErrorTriageResult(
                fingerprint=event.fingerprint,
                action="suppress",
                matched_rule="no-pool-fallback",
                incident_state="suppressed",
                occurrence_count=0,
            )

        # Find matching rule
        matched_rule = self._find_matching_rule(event)
        if matched_rule is None:
            # Should not happen with catch-all rule, but handle gracefully
            matched_rule = ModelTriageRule(name="fallback", action="alert")

        # Cross-layer correlation for Kafka-related errors
        correlated_fingerprint = None
        if event.error_category in _KAFKA_CATEGORIES:
            correlated_fingerprint = await self._correlate_with_layer1(event)

        # Upsert incident in runtime_error_triage table
        incident = await self._upsert_incident(
            event, matched_rule, correlated_fingerprint
        )
        occurrence_count = cast("int", incident["occurrence_count"])
        incident_state = cast("str", incident["incident_state"])

        # Execute the triage action and build result
        result: ModelRuntimeErrorTriageResult
        if matched_rule.action == "suppress":
            # Check if suppression window is active
            if incident_state == "suppressed":
                result = ModelRuntimeErrorTriageResult(
                    fingerprint=event.fingerprint,
                    action="suppress",
                    matched_rule=matched_rule.name,
                    incident_state=incident_state,
                    occurrence_count=occurrence_count,
                    correlated_consumer_fingerprint=correlated_fingerprint,
                )
            else:
                # First time or suppression expired — suppress and alert once
                await self._update_incident_state(event.fingerprint, "suppressed")
                await self._send_slack_notification(
                    event, matched_rule, correlated_fingerprint
                )
                result = ModelRuntimeErrorTriageResult(
                    fingerprint=event.fingerprint,
                    action="suppress",
                    matched_rule=matched_rule.name,
                    incident_state="suppressed",
                    occurrence_count=occurrence_count,
                    correlated_consumer_fingerprint=correlated_fingerprint,
                )

        elif matched_rule.action == "ticket":
            await self._create_linear_ticket(
                event, occurrence_count, correlated_fingerprint
            )
            await self._update_incident_state(event.fingerprint, "ticketed")
            result = ModelRuntimeErrorTriageResult(
                fingerprint=event.fingerprint,
                action="ticket",
                matched_rule=matched_rule.name,
                incident_state="ticketed",
                occurrence_count=occurrence_count,
                correlated_consumer_fingerprint=correlated_fingerprint,
            )

        else:
            # Default: alert via Slack
            await self._send_slack_notification(
                event, matched_rule, correlated_fingerprint
            )
            result = ModelRuntimeErrorTriageResult(
                fingerprint=event.fingerprint,
                action="alert",
                matched_rule=matched_rule.name,
                incident_state=incident_state,
                occurrence_count=occurrence_count,
                correlated_consumer_fingerprint=correlated_fingerprint,
            )

        # Emit triage result to Kafka for omnidash projection (OMN-5650)
        await self._emit_triage_event(result, event)

        return result

    def _resolve_triage_topic(self) -> str:
        """Lazily resolve the error-triaged topic string."""
        if self._triage_topic is None:
            from omnibase_infra.topics import topic_keys
            from omnibase_infra.topics.service_topic_registry import (
                ServiceTopicRegistry,
            )

            self._triage_topic = ServiceTopicRegistry.from_defaults().resolve(
                topic_keys.ERROR_TRIAGED
            )
        return self._triage_topic

    async def _emit_triage_event(
        self,
        result: ModelRuntimeErrorTriageResult,
        source_event: ModelRuntimeErrorEvent,
    ) -> None:
        """Emit an error-triaged event to Kafka for omnidash projection.

        Fire-and-forget: publish errors are logged but do not propagate,
        so triage results are still returned even if Kafka is unavailable.

        Args:
            result: The triage result to emit.
            source_event: The original runtime error event for correlation.
        """
        if self._event_bus is None:
            return

        now = datetime.now(UTC)
        payload = json.loads(result.model_dump_json())
        payload["triaged_at"] = now.isoformat()
        payload["source_error_category"] = source_event.error_category.value
        payload["source_severity"] = source_event.severity.value
        payload["source_logger_family"] = source_event.logger_family

        envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
            envelope_id=uuid4(),
            payload=payload,
            envelope_timestamp=now,
            correlation_id=source_event.correlation_id,
            source="runtime_error_triage",
        )

        try:
            topic = self._resolve_triage_topic()
            await self._event_bus.publish_envelope(
                envelope,  # type: ignore[arg-type]
                topic=topic,
            )
            logger.debug(
                "Emitted error-triaged event",
                extra={
                    "fingerprint": result.fingerprint,
                    "action": result.action,
                    "topic": topic,
                },
            )
        except Exception:  # noqa: BLE001 — never block triage on bus errors
            logger.warning(
                "Failed to emit error-triaged event (non-fatal)",
                extra={"fingerprint": result.fingerprint},
                exc_info=True,
            )

    def _find_matching_rule(
        self, event: ModelRuntimeErrorEvent
    ) -> ModelTriageRule | None:
        """Find the first matching triage rule for the event."""
        for rule in self._rules:
            if rule.matches(
                event.logger_family, event.error_category, event.raw_message
            ):
                return rule
        return None

    async def _correlate_with_layer1(self, event: ModelRuntimeErrorEvent) -> str | None:
        """Check for active Layer 1 consumer health incidents that may correlate.

        Queries consumer_health_triage for open incidents where the logger
        family or error message suggests the same consumer is affected.

        Returns the correlated fingerprint or None.
        """
        assert self._db_pool is not None  # guarded by handle() None-check
        try:
            async with self._db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT fingerprint
                    FROM consumer_health_triage
                    WHERE incident_state IN ('open', 'acknowledged', 'restart_pending')
                    AND last_seen_at > NOW() - INTERVAL '30 minutes'
                    ORDER BY last_seen_at DESC
                    LIMIT 1
                    """,
                )
                return row["fingerprint"] if row else None
        except Exception:  # noqa: BLE001 - cross-layer correlation is best-effort
            logger.debug("Failed to correlate with Layer 1", exc_info=True)
            return None

    async def _upsert_incident(
        self,
        event: ModelRuntimeErrorEvent,
        rule: ModelTriageRule,
        correlated_fingerprint: str | None,
    ) -> dict[str, object]:
        """Upsert incident row in runtime_error_triage table."""
        assert self._db_pool is not None  # guarded by handle() None-check
        async with self._db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO runtime_error_triage (
                    fingerprint, logger_name, error_category, severity,
                    incident_state, occurrence_count, message_template,
                    first_seen_at, last_seen_at,
                    correlated_consumer_fingerprint, service_name,
                    hostname, correlation_id
                ) VALUES ($1, $2, $3, $4, $5, 1, $6, NOW(), NOW(), $7, $8, $9, $10)
                ON CONFLICT (fingerprint) WHERE incident_state IN ('open', 'suppressed')
                DO UPDATE SET
                    occurrence_count = runtime_error_triage.occurrence_count + 1,
                    last_seen_at = NOW(),
                    severity = EXCLUDED.severity,
                    correlated_consumer_fingerprint = COALESCE(
                        EXCLUDED.correlated_consumer_fingerprint,
                        runtime_error_triage.correlated_consumer_fingerprint
                    ),
                    correlation_id = EXCLUDED.correlation_id
                RETURNING occurrence_count, incident_state
                """,
                event.fingerprint,
                event.logger_family,
                event.error_category.value,
                event.severity.value,
                "open",
                event.message_template,
                correlated_fingerprint,
                event.service_label,
                event.hostname,
                str(event.correlation_id),
            )
            if row is None:
                # Existing incident is ticketed/resolved — create new
                row = await conn.fetchrow(
                    """
                    INSERT INTO runtime_error_triage (
                        fingerprint, logger_name, error_category, severity,
                        incident_state, occurrence_count, message_template,
                        first_seen_at, last_seen_at,
                        correlated_consumer_fingerprint, service_name,
                        hostname, correlation_id
                    ) VALUES ($1, $2, $3, $4, 'open', 1, $5, NOW(), NOW(), $6, $7, $8, $9)
                    RETURNING occurrence_count, incident_state
                    """,
                    event.fingerprint,
                    event.logger_family,
                    event.error_category.value,
                    event.severity.value,
                    event.message_template,
                    correlated_fingerprint,
                    event.service_label,
                    event.hostname,
                    str(event.correlation_id),
                )
            return (
                dict(row) if row else {"occurrence_count": 1, "incident_state": "open"}
            )

    async def _update_incident_state(self, fingerprint: str, state: str) -> None:
        """Update the incident state for a fingerprint."""
        assert self._db_pool is not None  # guarded by handle() None-check
        async with self._db_pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE runtime_error_triage
                SET incident_state = $1,
                    escalated_at = CASE WHEN $1 IN ('ticketed') THEN NOW() ELSE escalated_at END
                WHERE fingerprint = $2 AND incident_state NOT IN ('resolved')
                """,
                state,
                fingerprint,
            )

    async def _send_slack_notification(
        self,
        event: ModelRuntimeErrorEvent,
        rule: ModelTriageRule,
        correlated_fingerprint: str | None,
    ) -> None:
        """Send a Slack notification for the runtime error event."""
        if self._slack_handler is None:
            logger.debug("Slack handler not configured, skipping notification")
            return

        try:
            message = (
                f"[{event.severity.value.upper()}] Runtime error: {event.error_category.value}\n"
                f"Logger: {event.logger_family}\n"
                f"Message: {event.raw_message[:200]}\n"
                f"Rule: {rule.name}"
            )
            if correlated_fingerprint:
                message += f"\nCorrelated with consumer health incident: {correlated_fingerprint}"
            await self._slack_handler(message)
        except Exception:  # noqa: BLE001 - best-effort notification
            logger.debug("Failed to send Slack notification", exc_info=True)

    async def _create_linear_ticket(
        self,
        event: ModelRuntimeErrorEvent,
        occurrence_count: int,
        correlated_fingerprint: str | None,
    ) -> None:
        """Create a Linear ticket for the runtime error."""
        if self._linear_handler is None:
            logger.debug("Linear handler not configured, skipping ticket creation")
            return

        try:
            title = (
                f"Runtime error: {event.error_category.value} in {event.logger_family} "
                f"({occurrence_count} occurrences)"
            )
            description = (
                f"Logger: {event.logger_family}\n"
                f"Category: {event.error_category.value}\n"
                f"Severity: {event.severity.value}\n"
                f"Message: {event.raw_message[:500]}\n"
                f"Fingerprint: {event.fingerprint}\n"
                f"Occurrences: {occurrence_count}\n"
            )
            if correlated_fingerprint:
                description += (
                    f"\nCorrelated consumer health incident: {correlated_fingerprint}"
                )
            if event.stack_trace:
                description += f"\n\nStack trace:\n```\n{event.stack_trace[:1000]}\n```"
            await self._linear_handler(title=title, description=description)
        except Exception:  # noqa: BLE001 - best-effort ticket creation
            logger.debug("Failed to create Linear ticket", exc_info=True)


__all__ = ["HandlerRuntimeErrorTriage", "ModelRuntimeErrorTriageResult"]
