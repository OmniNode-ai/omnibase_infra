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

The triage outcome is persisted to the ``runtime_error_triage`` table and
returned to the caller. The only event this node publishes is the escalation
event: an open incident that crosses the ``escalation`` rule declared in
``contract.yaml`` is moved to ``escalated`` in one atomic UPDATE and exactly one
``ModelRuntimeErrorEscalatedEvent`` is returned for the dispatcher to publish
on the contract-declared topic. The OMN-5650 ``error-triaged.v1`` emission stays
removed (OMN-17187).

Related Tickets:
    - OMN-5522: Create NodeRuntimeErrorTriageEffect
    - OMN-5529: Runtime Health Event Pipeline (epic)
    - OMN-17187: Remove the orphaned error-triaged.v1 emission
    - OMN-21029: Escalate incidents that cross the contract-declared rule
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.models.dispatch import ModelHandlerOutput
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.health.enum_runtime_error_category import (
    EnumRuntimeErrorCategory,
)
from omnibase_infra.models.health.model_runtime_error_event import (
    ModelRuntimeErrorEvent,
)
from omnibase_infra.nodes.node_runtime_error_triage_effect.models.model_escalation_rule import (
    ModelEscalationRule,
    load_escalation_rule,
)
from omnibase_infra.nodes.node_runtime_error_triage_effect.models.model_runtime_error_escalated_event import (
    ModelRuntimeErrorEscalatedEvent,
)
from omnibase_infra.nodes.node_runtime_error_triage_effect.models.model_triage_rule import (
    DEFAULT_TRIAGE_RULES,
    ModelTriageRule,
)

if TYPE_CHECKING:
    from asyncpg import Pool

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
    escalation: ModelRuntimeErrorEscalatedEvent | None = Field(
        default=None,
        description=(
            "Set only on the one occurrence that moved the incident to "
            "'escalated'; this is the event the dispatcher publishes."
        ),
    )


class HandlerRuntimeErrorTriage:
    """Handler that applies first-match-wins triage to runtime error events.

    Dependencies (injected via constructor):
        - db_pool: asyncpg connection pool for PostgreSQL
        - rules: Optional list of triage rules (defaults to DEFAULT_TRIAGE_RULES)
        - escalation_rule: Optional escalation rule (defaults to the ``escalation``
          block of this node's contract.yaml)
        - slack_handler: Optional callable for Slack notifications
        - linear_handler: Optional callable for Linear ticket creation
    """

    def __init__(
        self,
        db_pool: Pool | None = None,
        *,
        rules: list[ModelTriageRule] | None = None,
        escalation_rule: ModelEscalationRule | None = None,
        slack_handler: Callable[[str], Awaitable[None]] | None = None,
        linear_handler: Callable[..., Awaitable[None]] | None = None,
    ) -> None:
        """Initialize the triage handler.

        Args:
            db_pool: asyncpg connection pool. When None (auto-wired path),
                handle() returns a suppressed result with a warning log.
            rules: Triage rules in priority order. Defaults to DEFAULT_TRIAGE_RULES.
            escalation_rule: When an open incident escalates. Defaults to the
                contract-declared rule.
            slack_handler: Async callable accepting a message string.
            linear_handler: Async callable accepting title and description kwargs.
        """
        self._db_pool = db_pool
        self._rules = sorted(rules or DEFAULT_TRIAGE_RULES, key=lambda r: r.priority)
        self._escalation_rule = escalation_rule or load_escalation_rule()
        self._slack_handler = slack_handler
        self._linear_handler = linear_handler

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role: infrastructure handler for error triage."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Behavioral classification: side-effecting triage operation."""
        return EnumHandlerTypeCategory.EFFECT

    async def handle(self, event: ModelRuntimeErrorEvent) -> ModelHandlerOutput[None]:
        """Run triage as a DB-writing effect whose only bus output is an escalation.

        The persisted triage details are available from
        :meth:`triage_runtime_error` for direct callers. The auto-wired Kafka
        boundary receives an effect output carrying at most one event, the
        escalation event, which the contract declares in ``published_events``.
        Returning the details as a bare ``BaseModel`` would make them an
        undeliverable bus output, so they are never part of the output.
        """
        result = await self.triage_runtime_error(event)
        return ModelHandlerOutput.for_effect(
            input_envelope_id=event.event_id,
            correlation_id=event.correlation_id,
            handler_id=type(self).__name__,
            events=(result.escalation,) if result.escalation is not None else (),
        )

    async def triage_runtime_error(
        self, event: ModelRuntimeErrorEvent
    ) -> ModelRuntimeErrorTriageResult:
        """Apply first-match-wins triage to a runtime error event.

        Args:
            event: The runtime error event to triage.

        Returns:
            Persisted triage details for a direct caller. This value is not a
            Kafka output; :meth:`handle` publishes only its ``escalation``.
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
                escalation=await self._escalate_if_crossed(
                    event,
                    matched_rule,
                    incident_state=incident_state,
                    occurrence_count=occurrence_count,
                    correlated_fingerprint=correlated_fingerprint,
                ),
            )

        return result

    async def _escalate_if_crossed(
        self,
        event: ModelRuntimeErrorEvent,
        rule: ModelTriageRule,
        *,
        incident_state: str,
        occurrence_count: int,
        correlated_fingerprint: str | None,
    ) -> ModelRuntimeErrorEscalatedEvent | None:
        """Move the incident to ``escalated`` and build its one event.

        The decision is the pure :meth:`ModelEscalationRule.evaluate`. The
        write is a single conditional UPDATE, so of several consumers that see
        the rule crossed at once only one matches the ``open`` row and gets an
        event; every later occurrence sees ``escalated`` and decides no.
        """
        reason = self._escalation_rule.evaluate(
            incident_state=incident_state,
            occurrence_count=occurrence_count,
            severity=event.severity,
            action=rule.action,
        )
        if reason is None:
            return None
        assert self._db_pool is not None  # guarded by handle() None-check
        async with self._db_pool.acquire() as conn:
            claimed = await conn.fetchrow(
                """
                UPDATE runtime_error_triage
                SET incident_state = 'escalated', escalated_at = NOW()
                WHERE fingerprint = $1 AND incident_state = 'open'
                RETURNING id
                """,
                event.fingerprint,
            )
        if claimed is None:
            return None
        return ModelRuntimeErrorEscalatedEvent(
            correlation_id=event.correlation_id,
            fingerprint=event.fingerprint,
            logger_family=event.logger_family,
            error_category=event.error_category,
            severity=event.severity,
            message_template=event.message_template,
            occurrence_count=occurrence_count,
            reason=reason,
            matched_rule=rule.name,
            service_label=event.service_label,
            hostname=event.hostname,
            correlated_consumer_fingerprint=correlated_fingerprint or "",
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
                ON CONFLICT (fingerprint) WHERE incident_state IN ('open', 'suppressed', 'escalated')
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
