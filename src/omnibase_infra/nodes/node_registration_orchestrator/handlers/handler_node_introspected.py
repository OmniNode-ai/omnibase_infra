# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler for NodeIntrospectionEvent - canonical registration trigger.

This handler processes NodeIntrospectionEvent payloads from nodes announcing
their presence in the cluster. It queries the projection for current state
and emits NodeRegistrationInitiated if the node is new or needs to retry.

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

Projection Persistence:
    When the handler initiates registration, it persists the projection to
    PostgreSQL BEFORE returning events. This ensures read models are consistent
    before downstream processing. The projection is written with state
    PENDING_REGISTRATION.

    If no projector is provided, the handler operates in read-only mode and
    only emits events without persisting the projection. This mode is useful
    for testing or when projection persistence is handled elsewhere.

Consul Registration (Dual Registration):
    When the handler initiates registration and a ConsulHandler is provided,
    it registers the node with Consul for service discovery AFTER persisting
    to PostgreSQL. This enables dual registration:
    - PostgreSQL: Projection state for orchestrator FSM
    - Consul: Service discovery for runtime lookup

    If ConsulHandler is not provided or Consul registration fails, the handler
    logs the failure but continues (PostgreSQL is the source of truth).

Coroutine Safety:
    This handler is stateless and coroutine-safe for concurrent calls
    with different event instances.

Related Tickets:
    - OMN-888 (C1): Registration Orchestrator
    - OMN-944 (F1): Registration Projection Schema
    - OMN-892: 2-Way Registration E2E Integration Test
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING
from urllib.parse import urlparse
from uuid import UUID, uuid4

from omnibase_infra.enums import EnumRegistrationState

if TYPE_CHECKING:
    from omnibase_core.types import JsonType
    from pydantic import BaseModel

    from omnibase_infra.handlers import ConsulHandler
    from omnibase_infra.projectors import ProjectorRegistration
from omnibase_infra.models.registration.events.model_node_registration_initiated import (
    ModelNodeRegistrationInitiated,
)
from omnibase_infra.models.registration.model_node_introspection_event import (
    ModelNodeIntrospectionEvent,
)
from omnibase_infra.projectors.projection_reader_registration import (
    ProjectionReaderRegistration,
)
from omnibase_infra.utils import sanitize_error_message

logger = logging.getLogger(__name__)


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

    Projection Persistence:
        When the handler initiates registration, it persists the projection
        BEFORE returning events. This ordering is critical:

        1. Handler decides to initiate registration
        2. Projection is persisted with state PENDING_REGISTRATION
        3. Events are returned for publishing

        This ensures that `ProjectionReaderRegistration.get_entity_state()`
        returns the updated projection immediately after event processing.

    Consul Registration (Dual Registration):
        When a ConsulHandler is configured, the handler registers the node
        with Consul AFTER persisting to PostgreSQL. This enables service
        discovery via Consul while maintaining PostgreSQL as source of truth.

        Service naming convention:
            - service_name: `onex-{node_type}` (matches ONEX convention)
            - service_id: `onex-{node_type}-{node_id}` (unique identifier)

        If Consul registration fails, the handler logs the error but continues
        (PostgreSQL persistence is the source of truth).

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
        _projector: Optional projector for persisting state transitions.
        _consul_handler: Optional Consul handler for service discovery registration.
        _ack_timeout_seconds: Timeout for node acknowledgment (default: 30s).

    Example:
        >>> from datetime import datetime, UTC
        >>> from uuid import uuid4
        >>> # Use explicit timestamps (time injection pattern) - not datetime.now()
        >>> now = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)
        >>> # With projector and Consul for full dual registration
        >>> handler = HandlerNodeIntrospected(
        ...     projection_reader,
        ...     projector=projector,
        ...     consul_handler=consul_handler,
        ... )
        >>> events = await handler.handle(
        ...     event=introspection_event,
        ...     now=now,
        ...     correlation_id=uuid4(),
        ... )
        >>> if events:
        ...     assert isinstance(events[0], ModelNodeRegistrationInitiated)
    """

    # Default timeout for node acknowledgment (30 seconds)
    DEFAULT_ACK_TIMEOUT_SECONDS: float = 30.0

    def __init__(
        self,
        projection_reader: ProjectionReaderRegistration,
        projector: ProjectorRegistration | None = None,
        ack_timeout_seconds: float | None = None,
        consul_handler: ConsulHandler | None = None,
    ) -> None:
        """Initialize the handler with a projection reader and optional components.

        Args:
            projection_reader: Reader for querying registration projection state.
            projector: Optional projector for persisting state transitions.
                If None, the handler operates in read-only mode (useful for testing).
            ack_timeout_seconds: Timeout in seconds for node acknowledgment.
                Default: 30 seconds. Used to calculate ack_deadline when persisting.
            consul_handler: Optional ConsulHandler for Consul service registration.
                If provided, nodes will be registered with Consul for service discovery.
                If None or not initialized, Consul registration is skipped.
        """
        self._projection_reader = projection_reader
        self._projector = projector
        self._consul_handler = consul_handler
        self._ack_timeout_seconds = (
            ack_timeout_seconds
            if ack_timeout_seconds is not None
            else self.DEFAULT_ACK_TIMEOUT_SECONDS
        )

    @property
    def has_projector(self) -> bool:
        """Check if projector is configured for projection persistence."""
        return self._projector is not None

    @property
    def has_consul_handler(self) -> bool:
        """Check if ConsulHandler is configured for Consul registration."""
        return self._consul_handler is not None

    async def handle(
        self,
        event: ModelNodeIntrospectionEvent,
        now: datetime,
        correlation_id: UUID,
    ) -> list[BaseModel]:
        """Process introspection event and decide on registration.

        Queries the current projection state for the node and decides
        whether to emit a NodeRegistrationInitiated event to start
        the registration workflow.

        When initiating registration with a projector configured, the handler:
        1. Persists the projection with state PENDING_REGISTRATION
        2. Returns the NodeRegistrationInitiated event

        This ordering ensures projections are readable before events are published.

        Args:
            event: The introspection event from the node.
            now: Injected current time (used for timestamps and ack_deadline).
            correlation_id: Correlation ID for distributed tracing.

        Returns:
            List containing ModelNodeRegistrationInitiated if registration
            should be initiated, empty list otherwise.

        Raises:
            RuntimeHostError: If projection query or persist fails (propagated).
            InfraConnectionError: If database connection fails during persist.
            InfraTimeoutError: If database operation times out.
            ValueError: If now is naive (no timezone info).
        """
        # Validate timezone-awareness for time injection pattern
        if now.tzinfo is None:
            raise ValueError(
                "now must be timezone-aware. Use datetime.now(UTC) or "
                "datetime(..., tzinfo=timezone.utc) instead of naive datetime."
            )

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
            return []

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

        # CRITICAL: Persist projection BEFORE returning events
        # This ensures read models are consistent before downstream processing
        if self._projector is not None:
            # Calculate ack deadline
            ack_deadline = now + timedelta(seconds=self._ack_timeout_seconds)

            # Extract node type and version from introspection event
            # node_type is EnumNodeKind (enum with values: effect, compute, reducer, orchestrator)
            node_type = event.node_type
            node_version = event.node_version

            # Use declared_capabilities directly from the introspection event
            # ModelNodeIntrospectionEvent.declared_capabilities is already ModelNodeCapabilities
            capabilities = event.declared_capabilities

            await self._projector.persist_state_transition(
                entity_id=node_id,
                domain="registration",
                new_state=EnumRegistrationState.PENDING_REGISTRATION,
                node_type=node_type.value,
                node_version=node_version,
                capabilities=capabilities,
                event_id=registration_attempt_id,
                now=now,
                ack_deadline=ack_deadline,
                correlation_id=correlation_id,
            )

            logger.info(
                "Projection persisted for registration initiation",
                extra={
                    "node_id": str(node_id),
                    "new_state": EnumRegistrationState.PENDING_REGISTRATION.value,
                    "ack_deadline": ack_deadline.isoformat(),
                    "correlation_id": str(correlation_id),
                },
            )
        else:
            logger.debug(
                "No projector configured, skipping projection persistence",
                extra={
                    "node_id": str(node_id),
                    "correlation_id": str(correlation_id),
                },
            )

        # Register with Consul for service discovery (dual registration)
        # This happens AFTER PostgreSQL persistence (source of truth)
        await self._register_with_consul(
            node_id=node_id,
            node_type=event.node_type.value,
            endpoints=event.endpoints,
            correlation_id=correlation_id,
        )

        logger.info(
            "Emitting NodeRegistrationInitiated",
            extra={
                "node_id": str(node_id),
                "registration_attempt_id": str(initiated_event.registration_attempt_id),
                "correlation_id": str(correlation_id),
            },
        )

        return [initiated_event]

    async def _register_with_consul(
        self,
        node_id: UUID,
        node_type: str,
        endpoints: dict[str, str] | None,
        correlation_id: UUID,
    ) -> None:
        """Register node with Consul for service discovery.

        Registers the node as a Consul service with:
        - service_name: `onex-{node_type}` (ONEX convention for service discovery)
        - service_id: `onex-{node_type}-{node_id}` (unique identifier)
        - tags: [`onex`, `node-type:{node_type}`]
        - address/port: Extracted from endpoints if available

        This method is idempotent - re-registering the same service_id updates it.
        Errors are logged but not propagated (PostgreSQL is source of truth).

        Args:
            node_id: Node UUID for service naming.
            node_type: ONEX node type (effect, compute, reducer, orchestrator).
            endpoints: Optional dict of endpoint URLs from introspection event.
            correlation_id: Correlation ID for tracing.
        """
        if self._consul_handler is None:
            logger.debug(
                "No ConsulHandler configured, skipping Consul registration",
                extra={
                    "node_id": str(node_id),
                    "correlation_id": str(correlation_id),
                },
            )
            return

        # Build service name following ONEX convention
        # Format: onex-{node_type} (e.g., onex-effect, onex-compute)
        service_name = f"onex-{node_type}"
        service_id = f"onex-{node_type}-{node_id}"

        # Extract address and port from endpoints if available
        address: str | None = None
        port: int | None = None
        if endpoints:
            # Try health endpoint first, then api
            health_url = endpoints.get("health") or endpoints.get("api")
            if health_url:
                # Parse URL to extract host and port
                # URL format: http://host:port/path
                try:
                    parsed = urlparse(health_url)
                    if parsed.hostname:
                        address = parsed.hostname
                    else:
                        # URL parsed but no hostname extracted - log for troubleshooting
                        # NOTE: Don't log raw URL - may contain credentials
                        endpoint_type = "health" if endpoints.get("health") else "api"
                        logger.debug(
                            "URL parsed but no hostname extracted from %s endpoint",
                            endpoint_type,
                            extra={
                                "node_id": str(node_id),
                                "endpoint_type": endpoint_type,
                                "has_scheme": bool(parsed.scheme),
                                "has_netloc": bool(parsed.netloc),
                                "correlation_id": str(correlation_id),
                            },
                        )
                    if parsed.port:
                        port = parsed.port
                    else:
                        # URL parsed but no port extracted - log for troubleshooting
                        # This is common for URLs using default ports (80/443)
                        endpoint_type = "health" if endpoints.get("health") else "api"
                        logger.debug(
                            "URL parsed but no port extracted from %s endpoint "
                            "(may use default port)",
                            endpoint_type,
                            extra={
                                "node_id": str(node_id),
                                "endpoint_type": endpoint_type,
                                "hostname_extracted": address is not None,
                                "correlation_id": str(correlation_id),
                            },
                        )
                except ValueError as e:
                    # urlparse raises ValueError for malformed URLs
                    # If parsing fails, continue without address/port
                    # NOTE: Don't log raw URL - may contain credentials (e.g., user:pass@host)
                    # Use sanitize_error_message to safely log error details
                    sanitized_error = sanitize_error_message(e)
                    endpoint_type = "health" if endpoints.get("health") else "api"
                    logger.debug(
                        "URL parsing failed for %s endpoint: %s",
                        endpoint_type,
                        sanitized_error,
                        extra={
                            "node_id": str(node_id),
                            "endpoint_type": endpoint_type,
                            "correlation_id": str(correlation_id),
                        },
                    )

        # Build Consul registration payload
        consul_payload: dict[str, JsonType] = {
            "name": service_name,
            "service_id": service_id,
            "tags": ["onex", f"node-type:{node_type}"],
        }
        if address:
            consul_payload["address"] = address
        if port:
            consul_payload["port"] = port

        try:
            # Build envelope for ConsulHandler.execute()
            envelope: dict[str, JsonType] = {
                "operation": "consul.register",
                "payload": consul_payload,
                "correlation_id": str(correlation_id),
                "envelope_id": str(uuid4()),
            }

            await self._consul_handler.execute(envelope)

            logger.info(
                "Node registered with Consul for service discovery",
                extra={
                    "node_id": str(node_id),
                    "service_name": service_name,
                    "service_id": service_id,
                    "correlation_id": str(correlation_id),
                },
            )

        except Exception as e:
            # Log error but don't propagate - PostgreSQL is source of truth
            # Consul registration is best-effort for service discovery
            # Use sanitize_error_message to avoid logging sensitive data
            # NOTE: Do NOT use exc_info=True here - stack traces may contain
            # connection strings, credentials, or other sensitive information
            sanitized_error = sanitize_error_message(e)
            logger.warning(
                "Consul registration failed (non-fatal): %s (error_type=%s)",
                sanitized_error,
                type(e).__name__,
                extra={
                    "node_id": str(node_id),
                    "service_name": service_name,
                    "correlation_id": str(correlation_id),
                    "error_type": type(e).__name__,
                },
            )


__all__: list[str] = ["HandlerNodeIntrospected"]
