# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Pure-function Registration Reducer Service.

Encapsulates all four registration workflow decisions as pure functions.
ZERO I/O, ZERO imports of ProjectorShell or EventBus. All decisions return
a frozen ModelReducerDecision that the caller (handler) applies to the
outside world.

Decision Methods:
    - decide_introspection: New node or re-registration after retriable state
    - decide_ack: Acknowledgment processing for AWAITING_ACK / ACCEPTED nodes
    - decide_heartbeat: Liveness deadline extension for active nodes
    - decide_timeout: Ack timeout and liveness expiry detection

Thread / Coroutine Safety:
    This class is stateless beyond configuration. All methods are safe for
    concurrent invocation with different parameters.

Related Tickets:
    - OMN-888 (C1): Registration Orchestrator
    - OMN-889 (D1): Registration Reducer
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import datetime, timedelta
from urllib.parse import urlparse
from uuid import UUID, uuid4

from pydantic import BaseModel

from omnibase_core.models.reducer.model_intent import ModelIntent
from omnibase_infra.enums import EnumRegistrationState
from omnibase_infra.models.projection.model_registration_projection import (
    ModelRegistrationProjection,
)
from omnibase_infra.models.registration.commands.model_node_registration_acked import (
    ModelNodeRegistrationAcked,
)
from omnibase_infra.models.registration.events.model_node_became_active import (
    ModelNodeBecameActive,
)
from omnibase_infra.models.registration.events.model_node_liveness_expired import (
    ModelNodeLivenessExpired,
)
from omnibase_infra.models.registration.events.model_node_registration_accepted import (
    ModelNodeRegistrationAccepted,
)
from omnibase_infra.models.registration.events.model_node_registration_ack_received import (
    ModelNodeRegistrationAckReceived,
)
from omnibase_infra.models.registration.events.model_node_registration_ack_timed_out import (
    ModelNodeRegistrationAckTimedOut,
)
from omnibase_infra.models.registration.events.model_node_registration_initiated import (
    ModelNodeRegistrationInitiated,
)
from omnibase_infra.models.registration.model_node_introspection_event import (
    ModelNodeIntrospectionEvent,
)
from omnibase_infra.nodes.node_registration_orchestrator.models.model_projection_record import (
    ModelProjectionRecord,
)
from omnibase_infra.nodes.node_registration_orchestrator.models.model_reducer_context import (
    ModelReducerContext,
)
from omnibase_infra.nodes.node_registration_orchestrator.models.model_reducer_decision import (
    ModelReducerDecision,
)
from omnibase_infra.nodes.reducers.models.model_payload_consul_register import (
    ModelPayloadConsulRegister,
)
from omnibase_infra.nodes.reducers.models.model_payload_postgres_update_registration import (
    ModelPayloadPostgresUpdateRegistration,
    ModelRegistrationAckUpdate,
    ModelRegistrationHeartbeatUpdate,
)
from omnibase_infra.nodes.reducers.models.model_payload_postgres_upsert_registration import (
    ModelPayloadPostgresUpsertRegistration,
)

# States that allow re-registration (node can try again)
_RETRIABLE_STATES: frozenset[EnumRegistrationState] = frozenset(
    {
        EnumRegistrationState.LIVENESS_EXPIRED,
        EnumRegistrationState.REJECTED,
        EnumRegistrationState.ACK_TIMED_OUT,
    }
)

# States that block new registration (already in progress or active)
_BLOCKING_STATES: frozenset[EnumRegistrationState] = frozenset(
    {
        EnumRegistrationState.PENDING_REGISTRATION,
        EnumRegistrationState.ACCEPTED,
        EnumRegistrationState.AWAITING_ACK,
        EnumRegistrationState.ACK_RECEIVED,
        EnumRegistrationState.ACTIVE,
    }
)


def _no_op(reason: str) -> ModelReducerDecision:
    """Build a no-op decision with the given reason."""
    return ModelReducerDecision(action="no_op", reason=reason)


class RegistrationReducerService:
    """Pure-function service for registration workflow decisions.

    This service is the **authoritative decision-maker** for all registration
    state transitions. Handlers MUST delegate decisions here rather than
    implementing their own state logic. This ensures a single source of truth
    for the registration FSM, making it easier to audit, test, and reason
    about correctness.

    All four ``decide_*`` methods return a frozen ModelReducerDecision.
    No I/O is performed; no event bus, projector, or database access occurs.
    Callers are responsible for applying the returned events and intents.

    Args:
        ack_timeout_seconds: Timeout in seconds for node acknowledgment.
        liveness_interval_seconds: Interval in seconds for liveness deadline.
        liveness_window_seconds: Window in seconds for heartbeat liveness extension.
        consul_enabled: Whether to emit consul.register intents.
    """

    def __init__(
        self,
        ack_timeout_seconds: float = 30.0,
        liveness_interval_seconds: int = 60,
        liveness_window_seconds: float = 90.0,
        consul_enabled: bool = True,
    ) -> None:
        """Initialize the reducer service with timing and feature configuration.

        Args:
            ack_timeout_seconds: How long to wait for a node ack before timeout.
            liveness_interval_seconds: Initial liveness deadline offset from activation.
            liveness_window_seconds: Liveness deadline extension per heartbeat.
            consul_enabled: Whether to emit Consul registration intents.
        """
        self._ack_timeout_seconds = ack_timeout_seconds
        self._liveness_interval_seconds = liveness_interval_seconds
        self._liveness_window_seconds = liveness_window_seconds
        self._consul_enabled = consul_enabled

    @property
    def liveness_interval_seconds(self) -> int:
        """Return the configured liveness interval in seconds."""
        return self._liveness_interval_seconds

    # ------------------------------------------------------------------
    # Method 1: decide_introspection
    # ------------------------------------------------------------------

    def decide_introspection(
        self,
        projection: ModelRegistrationProjection | None,
        event: ModelNodeIntrospectionEvent,
        correlation_id: UUID,
        now: datetime,
    ) -> ModelReducerDecision:
        """Decide whether to initiate registration for an introspection event.

        Decision Logic:
            - No projection (new node) -> initiate
            - Retriable state (LIVENESS_EXPIRED, REJECTED, ACK_TIMED_OUT) -> initiate
            - Blocking state (PENDING, ACCEPTED, AWAITING_ACK, ACK_RECEIVED, ACTIVE) -> no-op

        When initiating, this method produces:
            - ModelNodeRegistrationInitiated event
            - ModelNodeRegistrationAccepted event
            - PostgreSQL upsert intent (state=AWAITING_ACK)
            - Consul register intent (if consul_enabled)

        Args:
            projection: Current registration projection, or None for new nodes.
            event: The node introspection event to process.
            correlation_id: Correlation ID for distributed tracing.
            now: Current time (injected, not generated).

        Returns:
            ModelReducerDecision with action="emit" or action="no_op".
        """
        # Determine whether to initiate registration
        should_initiate = False

        if projection is None:
            should_initiate = True
        else:
            current_state = projection.current_state
            if current_state in _RETRIABLE_STATES:
                should_initiate = True
            elif current_state in _BLOCKING_STATES:
                should_initiate = False

        if not should_initiate:
            state_label = (
                str(projection.current_state) if projection is not None else "unknown"
            )
            return _no_op(f"State {state_label} blocks new registration")

        node_id = event.node_id

        # Build registration attempt ID
        registration_attempt_id = uuid4()

        # Event 1: Registration initiated
        initiated_event = ModelNodeRegistrationInitiated(
            entity_id=node_id,
            node_id=node_id,
            correlation_id=correlation_id,
            causation_id=event.correlation_id,
            emitted_at=now,
            registration_attempt_id=registration_attempt_id,
        )

        # Event 2: Registration accepted (fast-forward to AWAITING_ACK)
        ack_deadline = now + timedelta(seconds=self._ack_timeout_seconds)
        accepted_event = ModelNodeRegistrationAccepted(
            entity_id=node_id,
            node_id=node_id,
            correlation_id=correlation_id,
            causation_id=event.correlation_id,
            emitted_at=now,
            ack_deadline=ack_deadline,
        )

        # Build intents
        intents: list[ModelIntent] = []

        # Intent 1: PostgreSQL upsert registration (state=AWAITING_ACK)
        node_type = event.node_type
        capabilities_data = (
            event.declared_capabilities.model_dump(mode="json")
            if event.declared_capabilities
            else {}
        )

        projection_record = ModelProjectionRecord(
            entity_id=node_id,
            domain="registration",
            current_state=EnumRegistrationState.AWAITING_ACK.value,
            node_type=node_type.value,
            data={
                "node_version": (
                    str(event.node_version) if event.node_version is not None else None
                ),
                "capabilities": capabilities_data,
                "contract_type": None,
                "intent_types": [],
                "protocols": [],
                "capability_tags": [],
                "contract_version": None,
                "ack_deadline": ack_deadline,
                "last_applied_event_id": registration_attempt_id,
                "registered_at": now,
                "updated_at": now,
                "correlation_id": correlation_id,
            },
        )

        postgres_payload = ModelPayloadPostgresUpsertRegistration(
            correlation_id=correlation_id,
            record=projection_record,
        )
        intents.append(
            ModelIntent(
                intent_type=postgres_payload.intent_type,
                target=f"postgres://node_registrations/{node_id}",
                payload=postgres_payload,
            )
        )

        # Intent 2: Consul service registration (conditional)
        if self._consul_enabled:
            consul_intent = self._build_consul_intent(
                event=event,
                node_id=node_id,
                correlation_id=correlation_id,
            )
            if consul_intent is not None:
                intents.append(consul_intent)

        return ModelReducerDecision(
            action="emit",
            new_state=EnumRegistrationState.AWAITING_ACK,
            events=(initiated_event, accepted_event),
            intents=tuple(intents),
            reason="Registration initiated",
        )

    # ------------------------------------------------------------------
    # Method 2: decide_ack
    # ------------------------------------------------------------------

    def decide_ack(
        self,
        projection: ModelRegistrationProjection | None,
        command: ModelNodeRegistrationAcked,
        correlation_id: UUID,
        now: datetime,
    ) -> ModelReducerDecision:
        """Decide whether to process a registration acknowledgment.

        Decision Logic:
            - No projection -> no-op (unknown node)
            - ACCEPTED or AWAITING_ACK -> emit AckReceived + BecameActive + UPDATE intent
            - ACK_RECEIVED or ACTIVE -> no-op (duplicate ack)
            - PENDING_REGISTRATION -> no-op (ack too early)
            - ACK_TIMED_OUT -> no-op (ack too late)
            - Terminal state -> no-op
            - Other -> no-op (unexpected state)

        Args:
            projection: Current registration projection, or None.
            command: The ack command from the node.
            correlation_id: Correlation ID for distributed tracing.
            now: Current time (injected, not generated).

        Returns:
            ModelReducerDecision with action="emit" or action="no_op".
        """
        if projection is None:
            return _no_op("Unknown node")

        current_state = projection.current_state
        node_id = command.node_id

        if current_state in {
            EnumRegistrationState.ACCEPTED,
            EnumRegistrationState.AWAITING_ACK,
        }:
            # Valid ack - build activation events
            liveness_deadline = now + timedelta(seconds=self._liveness_interval_seconds)

            ack_received = ModelNodeRegistrationAckReceived(
                entity_id=node_id,
                node_id=node_id,
                correlation_id=correlation_id,
                causation_id=command.command_id,
                emitted_at=now,
                liveness_deadline=liveness_deadline,
            )

            became_active = ModelNodeBecameActive(
                entity_id=node_id,
                node_id=node_id,
                correlation_id=correlation_id,
                causation_id=command.command_id,
                emitted_at=now,
                capabilities=projection.capabilities,
            )

            # Build UPDATE intent to transition projection to ACTIVE
            update_payload = ModelPayloadPostgresUpdateRegistration(
                correlation_id=correlation_id,
                entity_id=node_id,
                domain="registration",
                updates=ModelRegistrationAckUpdate(
                    current_state=EnumRegistrationState.ACTIVE.value,
                    liveness_deadline=liveness_deadline,
                    updated_at=now,
                ),
            )
            update_intent = ModelIntent(
                intent_type=update_payload.intent_type,
                target=f"postgres://node_registrations/{node_id}",
                payload=update_payload,
            )

            return ModelReducerDecision(
                action="emit",
                new_state=EnumRegistrationState.ACTIVE,
                events=(ack_received, became_active),
                intents=(update_intent,),
                reason="Ack received, transitioning to ACTIVE",
            )

        if current_state in {
            EnumRegistrationState.ACK_RECEIVED,
            EnumRegistrationState.ACTIVE,
        }:
            return _no_op("Duplicate ack")

        if current_state == EnumRegistrationState.PENDING_REGISTRATION:
            return _no_op("Ack too early")

        if current_state == EnumRegistrationState.ACK_TIMED_OUT:
            return _no_op("Ack too late")

        if current_state.is_terminal():
            return _no_op("Terminal state")

        return _no_op(f"Unexpected state: {current_state}")

    # ------------------------------------------------------------------
    # Method 3: decide_heartbeat
    # ------------------------------------------------------------------

    def decide_heartbeat(
        self,
        projection: ModelRegistrationProjection | None,
        node_id: UUID,
        heartbeat_timestamp: datetime,
        ctx: ModelReducerContext,
    ) -> ModelReducerDecision:
        """Decide whether to extend liveness deadline for a heartbeat.

        Heartbeats are processed even for non-ACTIVE nodes (with a warning
        logged by the caller), as this can happen during state transitions.

        Args:
            projection: Current registration projection, or None.
            node_id: UUID of the node sending the heartbeat.
            heartbeat_timestamp: Timestamp from the heartbeat event.
            ctx: Reducer context bundling correlation_id and now.

        Returns:
            ModelReducerDecision with action="emit" (UPDATE intent, no events)
            or action="no_op" if projection is None.
        """
        if projection is None:
            return _no_op("Unknown node")

        new_liveness_deadline = heartbeat_timestamp + timedelta(
            seconds=self._liveness_window_seconds
        )

        update_payload = ModelPayloadPostgresUpdateRegistration(
            correlation_id=ctx.correlation_id,
            entity_id=node_id,
            domain="registration",
            updates=ModelRegistrationHeartbeatUpdate(
                last_heartbeat_at=heartbeat_timestamp,
                liveness_deadline=new_liveness_deadline,
                updated_at=ctx.now,
            ),
        )
        update_intent = ModelIntent(
            intent_type=update_payload.intent_type,
            target=f"postgres://node_registrations/{node_id}",
            payload=update_payload,
        )

        return ModelReducerDecision(
            action="emit",
            new_state=None,
            events=(),
            intents=(update_intent,),
            reason="Heartbeat processed, liveness deadline extended",
        )

    # ------------------------------------------------------------------
    # Method 4: decide_timeout
    # ------------------------------------------------------------------

    def decide_timeout(
        self,
        overdue_ack_projections: Sequence[ModelRegistrationProjection],
        overdue_liveness_projections: Sequence[ModelRegistrationProjection],
        ctx: ModelReducerContext,
    ) -> ModelReducerDecision:
        """Decide which timeout events to emit for overdue projections.

        Scans two lists of overdue projections and emits timeout events
        for entities that pass deduplication checks.

        Args:
            overdue_ack_projections: Projections with overdue ack deadlines.
            overdue_liveness_projections: Projections with overdue liveness deadlines.
            ctx: Reducer context bundling correlation_id, now, and tick_id.

        Returns:
            ModelReducerDecision with timeout events, or no_op if none detected.
        """
        events: list[BaseModel] = []

        # Check ack timeouts
        for projection in overdue_ack_projections:
            if not projection.needs_ack_timeout_event(ctx.now):
                continue
            ack_deadline = projection.ack_deadline
            if ack_deadline is None:
                # Defensive: needs_ack_timeout_event() guarantees ack_deadline is set
                continue

            events.append(
                ModelNodeRegistrationAckTimedOut(
                    entity_id=projection.entity_id,
                    node_id=projection.entity_id,
                    correlation_id=ctx.correlation_id,
                    causation_id=ctx.tick_id,
                    emitted_at=ctx.now,
                    deadline_at=ack_deadline,
                )
            )

        # Check liveness expirations
        for projection in overdue_liveness_projections:
            if not projection.needs_liveness_timeout_event(ctx.now):
                continue

            events.append(
                ModelNodeLivenessExpired(
                    entity_id=projection.entity_id,
                    node_id=projection.entity_id,
                    correlation_id=ctx.correlation_id,
                    causation_id=ctx.tick_id,
                    emitted_at=ctx.now,
                    last_heartbeat_at=projection.last_heartbeat_at,
                )
            )

        if not events:
            return _no_op("No timeouts detected")

        return ModelReducerDecision(
            action="emit",
            new_state=None,
            events=tuple(events),
            intents=(),
            reason=(
                f"Detected {len(events)} timeout(s): "
                f"{sum(1 for e in events if isinstance(e, ModelNodeRegistrationAckTimedOut))} ack, "
                f"{sum(1 for e in events if isinstance(e, ModelNodeLivenessExpired))} liveness"
            ),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_tool_name(name: str) -> str:
        """Sanitize tool name for use in Consul tags.

        Converts free-form text into stable, Consul-safe identifiers.

        Transformation rules:
            1. Convert to lowercase
            2. Replace non-alphanumeric characters with dashes
            3. Collapse multiple consecutive dashes into one
            4. Remove leading/trailing dashes
            5. Truncate to 63 characters (Consul tag limit)

        Args:
            name: Raw tool name or description text.

        Returns:
            Sanitized string suitable for Consul tags (lowercase, alphanumeric
            with dashes, max 63 chars).
        """
        sanitized = re.sub(r"[^a-zA-Z0-9]+", "-", name.lower())
        sanitized = sanitized.strip("-")
        if not sanitized:
            return "unnamed"
        return sanitized[:63]

    def _build_consul_intent(
        self,
        event: ModelNodeIntrospectionEvent,
        node_id: UUID,
        correlation_id: UUID,
    ) -> ModelIntent | None:
        """Build a Consul service registration intent from introspection event.

        Extracts service name, tags, address, and port from the event to
        construct a ModelPayloadConsulRegister intent payload.

        Args:
            event: The node introspection event.
            node_id: UUID of the node being registered.
            correlation_id: Correlation ID for distributed tracing.

        Returns:
            ModelIntent for Consul registration, or None if consul is disabled.
        """
        node_type = event.node_type
        service_name = f"onex-{node_type.value}"
        service_id = f"onex-{node_type.value}-{node_id}"

        # Build tags
        tags: list[str] = ["onex", f"node-type:{node_type.value}"]

        # Add MCP tags for orchestrators with MCP config enabled
        mcp_config = (
            event.declared_capabilities.mcp
            if event.declared_capabilities is not None
            else None
        )
        if node_type.value == "orchestrator" and mcp_config is not None:
            if mcp_config.expose:
                mcp_tool_name_raw = mcp_config.tool_name
                if not mcp_tool_name_raw:
                    node_name = event.metadata.description if event.metadata else None
                    mcp_tool_name_raw = node_name or service_name
                mcp_tool_name = self._sanitize_tool_name(mcp_tool_name_raw)
                tags.extend(["mcp-enabled", f"mcp-tool:{mcp_tool_name}"])

        # Extract address and port from endpoints if available
        address: str | None = None
        port: int | None = None
        endpoints = event.endpoints
        if endpoints:
            health_url = endpoints.get("health") or endpoints.get("api")
            if health_url:
                try:
                    parsed = urlparse(health_url)
                    if parsed.hostname:
                        address = parsed.hostname
                    if parsed.port:
                        port = parsed.port
                except ValueError:
                    pass

        consul_payload = ModelPayloadConsulRegister(
            correlation_id=correlation_id,
            node_id=str(node_id),
            service_id=service_id,
            service_name=service_name,
            tags=tags,
            address=address,
            port=port,
            event_bus_config=event.event_bus,  # Pass through from introspection event
        )

        return ModelIntent(
            intent_type=consul_payload.intent_type,
            target=f"consul://service/{service_name}",
            payload=consul_payload,
        )


__all__: list[str] = ["RegistrationReducerService"]
