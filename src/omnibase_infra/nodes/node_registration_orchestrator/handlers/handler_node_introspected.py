# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler for NodeIntrospectionEvent - canonical registration trigger.

This handler processes NodeIntrospectionEvent payloads from nodes announcing
their presence in the cluster. It queries the projection for current state
and returns intents for the effect layer to execute.

Decision Logic:
    The handler emits NodeRegistrationInitiated when:
    - No projection exists (new node)
    - State is LIVENESS_EXPIRED (re-registration after death)
    - State is REJECTED (retry after rejection)
    - State is ACK_TIMED_OUT (retry after timeout)

    The handler does NOT emit when:
    - State is PENDING_REGISTRATION (already processing)
    - State is ACCEPTED (already accepted, waiting for ack)
    - State is AWAITING_ACK (already waiting for ack)
    - State is ACK_RECEIVED (already acknowledged)
    - State is ACTIVE (already active - heartbeat should be used)

Intent-Based Architecture (OMN-2050):
    This handler performs ZERO direct I/O. Instead, it returns ModelIntent
    objects that the effect layer executes via IntentExecutor:

    - ModelPayloadPostgresUpsertRegistration: Persists the projection
    - ModelPayloadConsulRegister: Registers with Consul for service discovery

    This follows the ONEX four-node pattern where handlers (COMPUTE) produce
    intents, and the EFFECT layer executes them.

Coroutine Safety:
    This handler is stateless and coroutine-safe for concurrent calls
    with different event instances.

Related Tickets:
    - OMN-888 (C1): Registration Orchestrator
    - OMN-944 (F1): Registration Projection Schema
    - OMN-892: 2-Way Registration E2E Integration Test
    - OMN-2050: Wire MessageDispatchEngine as single consumer path
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timedelta
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict

from omnibase_core.enums import EnumMessageCategory, EnumNodeKind
from omnibase_core.models.dispatch.model_handler_output import ModelHandlerOutput
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.models.reducer.model_intent import ModelIntent
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
    EnumRegistrationState,
)
from omnibase_infra.errors import ModelInfraErrorContext
from omnibase_infra.models.registration.events.model_node_registration_initiated import (
    ModelNodeRegistrationInitiated,
)
from omnibase_infra.models.registration.model_node_introspection_event import (
    ModelNodeIntrospectionEvent,
)
from omnibase_infra.nodes.reducers.models.model_payload_consul_register import (
    ModelPayloadConsulRegister,
)
from omnibase_infra.nodes.reducers.models.model_payload_postgres_upsert_registration import (
    ModelPayloadPostgresUpsertRegistration,
)
from omnibase_infra.projectors.projection_reader_registration import (
    ProjectionReaderRegistration,
)
from omnibase_infra.utils import validate_timezone_aware_with_context

logger = logging.getLogger(__name__)


class ModelProjectionRecord(BaseModel):
    """Minimal model for wrapping projection record dicts.

    Uses extra='allow' so all dict keys are preserved as extra fields.
    This allows the record to be stored as SerializeAsAny[BaseModel] in
    ModelPayloadPostgresUpsertRegistration while retaining all data.
    SerializeAsAny ensures model_dump() serializes all extra fields.
    """

    model_config = ConfigDict(extra="allow", frozen=True)


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


class HandlerNodeIntrospected:
    """Handler for NodeIntrospectionEvent - canonical registration trigger.

    This handler processes introspection events from nodes announcing
    themselves to the cluster. It queries the current projection state
    and decides whether to initiate a new registration workflow.

    Intent-Based Output (OMN-2050):
        When initiating registration, the handler returns ModelIntent objects
        for the effect layer to execute:

        1. Handler decides to initiate registration
        2. Returns ModelPayloadPostgresUpsertRegistration intent
        3. Returns ModelPayloadConsulRegister intent (if applicable)
        4. Returns ModelNodeRegistrationInitiated event

        The effect layer (IntentExecutor) handles persistence and
        service discovery registration. This handler performs ZERO direct I/O.

    State Decision Matrix:
        | Current State       | Action                          |
        |---------------------|----------------------------------|
        | None (new node)     | Emit NodeRegistrationInitiated   |
        | LIVENESS_EXPIRED    | Emit NodeRegistrationInitiated   |
        | REJECTED            | Emit NodeRegistrationInitiated   |
        | ACK_TIMED_OUT       | Emit NodeRegistrationInitiated   |
        | PENDING_REGISTRATION| No-op (already processing)       |
        | ACCEPTED            | No-op (waiting for ack)          |
        | AWAITING_ACK        | No-op (waiting for ack)          |
        | ACK_RECEIVED        | No-op (transitioning to active)  |
        | ACTIVE              | No-op (use heartbeat instead)    |

    Attributes:
        _projection_reader: Reader for registration projection state.
        _ack_timeout_seconds: Timeout for node acknowledgment (default: 30s).

    Example:
        >>> from datetime import datetime, UTC
        >>> from uuid import uuid4
        >>> handler = HandlerNodeIntrospected(projection_reader)
        >>> output = await handler.handle(envelope)
        >>> # output.intents contains ModelIntent objects for effect layer
        >>> # output.events contains ModelNodeRegistrationInitiated
    """

    # Default timeout for node acknowledgment (30 seconds)
    DEFAULT_ACK_TIMEOUT_SECONDS: float = 30.0

    def __init__(
        self,
        projection_reader: ProjectionReaderRegistration,
        ack_timeout_seconds: float | None = None,
    ) -> None:
        """Initialize the handler with a projection reader.

        Args:
            projection_reader: Reader for querying registration projection state.
            ack_timeout_seconds: Timeout in seconds for node acknowledgment.
                Default: 30 seconds. Used to calculate ack_deadline in intents.
        """
        self._projection_reader = projection_reader
        self._ack_timeout_seconds = (
            ack_timeout_seconds
            if ack_timeout_seconds is not None
            else self.DEFAULT_ACK_TIMEOUT_SECONDS
        )

    @property
    def handler_id(self) -> str:
        """Unique identifier for this handler."""
        return "handler-node-introspected"

    @property
    def category(self) -> EnumMessageCategory:
        """Message category this handler processes."""
        return EnumMessageCategory.EVENT

    @property
    def message_types(self) -> set[str]:
        """Set of message type names this handler can process."""
        return {"ModelNodeIntrospectionEvent"}

    @property
    def node_kind(self) -> EnumNodeKind:
        """Node kind this handler belongs to."""
        return EnumNodeKind.ORCHESTRATOR

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role classification for this handler.

        Returns NODE_HANDLER because this handler processes node-level
        introspection events (not infrastructure plumbing).
        """
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Behavioral classification for this handler.

        Returns COMPUTE because this handler performs pure decision logic:
        reads projection state and emits intents for the effect layer.
        No direct I/O is performed (OMN-2050).
        """
        return EnumHandlerTypeCategory.COMPUTE

    def _sanitize_tool_name(self, name: str) -> str:
        """Sanitize tool name for use in Consul tags.

        Converts free-form text (like descriptions) into stable, Consul-safe
        identifiers. This ensures consistent service discovery matching.

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

        Example:
            >>> handler._sanitize_tool_name("My Cool Tool (v2.0)")
            'my-cool-tool-v2-0'
            >>> handler._sanitize_tool_name("  Spaces & Special!Chars  ")
            'spaces-special-chars'
        """
        # Replace non-alphanumeric with dash, lowercase
        sanitized = re.sub(r"[^a-zA-Z0-9]+", "-", name.lower())
        # Remove leading/trailing dashes
        sanitized = sanitized.strip("-")
        # Truncate to Consul tag limit (63 chars is common limit for DNS labels)
        return sanitized[:63]

    async def handle(
        self,
        envelope: ModelEventEnvelope[ModelNodeIntrospectionEvent],
    ) -> ModelHandlerOutput[object]:
        """Process introspection event and decide on registration.

        Queries the current projection state for the node and decides
        whether to emit a NodeRegistrationInitiated event and intents
        for the effect layer to execute.

        Returns ModelHandlerOutput with:
        - events: (ModelNodeRegistrationInitiated,) if registration initiated
        - intents: (postgres_upsert_intent, consul_register_intent) for effect layer

        Args:
            envelope: Event envelope containing ModelNodeIntrospectionEvent payload.

        Returns:
            ModelHandlerOutput with events and intents for effect layer execution.

        Raises:
            RuntimeHostError: If projection query fails (propagated).
            ProtocolConfigurationError: If envelope timestamp is naive (no timezone info).
        """
        start_time = time.perf_counter()

        # Extract from envelope
        event = envelope.payload
        now: datetime = envelope.envelope_timestamp
        correlation_id: UUID = envelope.correlation_id or uuid4()

        # Validate timezone-awareness for time injection pattern
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.RUNTIME,
            operation="handle_introspection_event",
            target_name="handler.node_introspected",
            correlation_id=correlation_id,
        )
        validate_timezone_aware_with_context(now, ctx)

        node_id = event.node_id

        # Query current projection state
        projection = await self._projection_reader.get_entity_state(
            entity_id=node_id,
            domain="registration",
            correlation_id=correlation_id,
        )

        # Decision: Should we initiate registration?
        should_initiate = False
        current_state: EnumRegistrationState | None = None

        if projection is None:
            # New node - initiate registration
            should_initiate = True
            logger.info(
                "New node detected, initiating registration",
                extra={
                    "node_id": str(node_id),
                    "correlation_id": str(correlation_id),
                },
            )
        else:
            current_state = projection.current_state

            if current_state in _RETRIABLE_STATES:
                # Retriable state - allow re-registration
                should_initiate = True
                logger.info(
                    "Node in retriable state, initiating re-registration",
                    extra={
                        "node_id": str(node_id),
                        "current_state": str(current_state),
                        "correlation_id": str(correlation_id),
                    },
                )
            elif current_state in _BLOCKING_STATES:
                # Blocking state - no-op
                should_initiate = False
                logger.debug(
                    "Node in blocking state, skipping registration",
                    extra={
                        "node_id": str(node_id),
                        "current_state": str(current_state),
                        "correlation_id": str(correlation_id),
                    },
                )

        if not should_initiate:
            processing_time_ms = (time.perf_counter() - start_time) * 1000
            return ModelHandlerOutput(
                input_envelope_id=envelope.envelope_id,
                correlation_id=correlation_id,
                handler_id=self.handler_id,
                node_kind=self.node_kind,
                events=(),
                intents=(),
                projections=(),
                result=None,
                processing_time_ms=processing_time_ms,
                timestamp=now,
            )

        # Build NodeRegistrationInitiated event
        registration_attempt_id = uuid4()
        initiated_event = ModelNodeRegistrationInitiated(
            entity_id=node_id,
            node_id=node_id,
            correlation_id=correlation_id,
            causation_id=event.correlation_id,  # Link to triggering event
            emitted_at=now,  # Use injected time for consistency
            registration_attempt_id=registration_attempt_id,
        )

        # Build intents for effect layer execution (OMN-2050)
        intents: list[ModelIntent] = []

        # Intent 1: PostgreSQL upsert registration
        ack_deadline = now + timedelta(seconds=self._ack_timeout_seconds)
        node_type = event.node_type
        node_version = event.node_version
        capabilities = event.declared_capabilities
        capabilities_json = capabilities.model_dump_json() if capabilities else "{}"

        projection_record = ModelProjectionRecord.model_validate(
            {
                "entity_id": str(node_id),
                "domain": "registration",
                "current_state": EnumRegistrationState.PENDING_REGISTRATION.value,
                "node_type": node_type.value,
                "node_version": str(node_version) if node_version else None,
                "capabilities": capabilities_json,
                "contract_type": None,
                "intent_types": [],
                "protocols": [],
                "capability_tags": [],
                "contract_version": None,
                "ack_deadline": ack_deadline.isoformat(),
                "last_applied_event_id": str(registration_attempt_id),
                "registered_at": now.isoformat(),
                "updated_at": now.isoformat(),
                "correlation_id": str(correlation_id),
            }
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

        # Intent 2: Consul service registration
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
            mcp_expose = getattr(mcp_config, "expose", False)
            if mcp_expose:
                mcp_tool_name_raw = getattr(mcp_config, "tool_name", None)
                if not mcp_tool_name_raw:
                    node_name = event.metadata.description if event.metadata else None
                    mcp_tool_name_raw = node_name or service_name
                mcp_tool_name = self._sanitize_tool_name(mcp_tool_name_raw)
                tags.extend(["mcp-enabled", f"mcp-tool:{mcp_tool_name}"])

        consul_payload = ModelPayloadConsulRegister(
            correlation_id=correlation_id,
            service_id=service_id,
            service_name=service_name,
            tags=tags,
        )

        intents.append(
            ModelIntent(
                intent_type=consul_payload.intent_type,
                target=f"consul://service/{service_name}",
                payload=consul_payload,
            )
        )

        logger.info(
            "Emitting NodeRegistrationInitiated with %d intents",
            len(intents),
            extra={
                "node_id": str(node_id),
                "registration_attempt_id": str(initiated_event.registration_attempt_id),
                "correlation_id": str(correlation_id),
                "intent_types": [
                    getattr(i.payload, "intent_type", "unknown") for i in intents
                ],
            },
        )

        processing_time_ms = (time.perf_counter() - start_time) * 1000
        return ModelHandlerOutput(
            input_envelope_id=envelope.envelope_id,
            correlation_id=correlation_id,
            handler_id=self.handler_id,
            node_kind=self.node_kind,
            events=(initiated_event,),
            intents=tuple(intents),
            projections=(),
            result=None,
            processing_time_ms=processing_time_ms,
            timestamp=now,
        )


__all__: list[str] = ["HandlerNodeIntrospected"]
