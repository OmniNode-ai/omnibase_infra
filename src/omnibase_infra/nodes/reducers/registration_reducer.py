# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Canonical Registration Reducer following ProtocolReducer pattern.

This reducer replaces the legacy NodeDualRegistrationReducer (887 lines)
with a pure function implementation (~80 lines) that follows the canonical
ONEX reducer pattern defined in DESIGN_TWO_WAY_REGISTRATION_ARCHITECTURE.md.

Performance Characteristics:
    This reducer is designed for high-performance event processing with the
    following targets:

    - reduce() processing: <300ms per event (target)
    - Intent building: <50ms per intent (includes Consul + PostgreSQL)
    - Idempotency check: <1ms

    Performance is logged when thresholds are exceeded. These thresholds are
    configurable via module-level constants (PERF_THRESHOLD_*).

    Typical performance on standard hardware:
    - Simple introspection events: ~0.1-1ms
    - Complex events with full metadata: ~1-5ms
    - Hash-based event ID derivation: ~0.01ms

Circuit Breaker Considerations:
    This reducer does NOT require a circuit breaker because:

    1. **Pure Function Pattern**: Reducers are pure functions - they perform
       NO I/O operations. All external interactions are delegated to the
       Effect layer via emitted intents.

    2. **No Transient Failures**: Without I/O, there are no transient failures
       to recover from. Circuit breakers are designed for I/O resilience.

    3. **Deterministic Behavior**: Given the same state and event, the reducer
       always produces the same output. There's no "retry" semantic.

    4. **Effect Layer Responsibility**: Circuit breakers should be implemented
       in the Effect layer nodes (ConsulAdapter, PostgresAdapter) that actually
       perform the external I/O operations.

    See CLAUDE.md "Dispatcher Resilience Pattern" section for the general
    principle: "Dispatchers own their own resilience."

Architecture:
    - Pure function: reduce(state, event) -> ModelReducerOutput
    - No internal state - state passed in and returned
    - No I/O - emits intents for Effect layer
    - Deterministic - same inputs produce same outputs

Key Differences from Legacy:
    - No FSM machinery (FSM was for tracking async I/O, which pure reducers don't do)
    - No initialize()/shutdown() lifecycle methods
    - No internal metrics tracking (metrics are output-level concerns)
    - State is external and immutable (ModelRegistrationState)
    - Uses omnibase_core's ModelReducerOutput[T] for standard output format

State Persistence Strategy:
    This reducer follows the ONEX pure reducer pattern where the reducer itself
    performs NO I/O. State persistence is handled externally by the Runtime and
    Projector components.

    **HOW STATE IS STORED (PostgreSQL via Projector Layer):**

    The reducer does NOT persist state directly. Instead:

    1. Reducer returns ModelReducerOutput containing the new state
    2. Runtime extracts the result (ModelRegistrationState) from the output
    3. Runtime invokes Projector.persist() to write state to PostgreSQL
    4. State is stored in the ``node_registrations`` table with fields matching
       the ModelRegistrationState model plus tracking fields (last_event_offset)

    PostgreSQL Schema (conceptual)::

        node_registrations:
            node_id            UUID PRIMARY KEY
            status             VARCHAR(20)  -- 'idle', 'pending', 'partial', 'complete', 'failed'
            consul_confirmed   BOOLEAN
            postgres_confirmed BOOLEAN
            last_processed_event_id  UUID
            failure_reason     VARCHAR(50)
            last_event_offset  BIGINT       -- For idempotent updates
            updated_at         TIMESTAMP

    **HOW STATE IS RETRIEVED (Before reduce() is Called):**

    Before calling reduce(), the orchestrator/runtime loads current state:

    1. Orchestrator receives NodeIntrospectionEvent from Kafka
    2. Orchestrator extracts entity_id (node_id) from event envelope
    3. Orchestrator queries projection via ProtocolProjectionReader::

           state = await projection_reader.get_projection(
               entity_type="registration",
               entity_id=node_id
           )
           if state is None:
               state = ModelRegistrationState()  # Initial idle state

    4. Orchestrator invokes reducer: output = reducer.reduce(state, event)
    5. Orchestrator passes output to Runtime for persistence and publishing

    **STATE FLOW (Complete Round-Trip):**

    ::

        +--------------+
        | Kafka Event  |  NodeIntrospectionEvent
        +------+-------+
               |
               v
        +------------------+
        |   Orchestrator   |  1. Receives event
        |                  |  2. Loads state from PostgreSQL via ProtocolProjectionReader
        +--------+---------+
                 | state + event
                 v
        +------------------+
        |     Reducer      |  3. reduce(state, event) -> ModelReducerOutput
        |   (THIS CLASS)   |     - Pure computation, no I/O
        |                  |     - Returns new state + intents
        +--------+---------+
                 | ModelReducerOutput
                 v
        +------------------+
        |     Runtime      |  4. Extracts result (new state)
        |                  |  5. Invokes Projector.persist() - SYNCHRONOUS
        |                  |  6. Waits for persist acknowledgment
        +--------+---------+
                 | persist()
                 v
        +------------------+
        |   PostgreSQL     |  7. State written to node_registrations table
        |   (Projection)   |     - Idempotent via last_event_offset check
        +--------+---------+
                 | ack
                 v
        +------------------+
        |     Runtime      |  8. AFTER persist acks, publish intents to Kafka
        |                  |     - Ordering guarantee: persist BEFORE publish
        +--------+---------+
                 | publish intents
                 v
        +------------------+
        |  Kafka (intents) |  9. Intents available for Effect layer consumption
        +------------------+

    **ORDERING GUARANTEE (Critical for Consistency):**

    Per ticket F0 (Projector Execution Model) in ONEX_RUNTIME_REGISTRATION_TICKET_PLAN.md:

    - Projections are PERSISTED to PostgreSQL BEFORE intents are PUBLISHED to Kafka
    - This ensures read models are consistent before downstream processing
    - Effects can safely assume projection state is current when they execute
    - No race conditions where effects execute before state is visible

    **IDEMPOTENCY (Safe Replay via last_processed_event_id):**

    The state model tracks ``last_processed_event_id`` to enable safe replay:

    1. Each event has a unique event_id (correlation_id or generated UUID)
    2. Before processing, reducer calls state.is_duplicate_event(event_id)
    3. If duplicate, reducer returns current state unchanged with no intents
    4. PostgreSQL projection also tracks last_event_offset for offset-based idempotency

    This handles crash scenarios:

    - If system crashes after persist but before Kafka ack, event is redelivered
    - Reducer detects duplicate via last_processed_event_id match
    - No duplicate intents are emitted
    - System converges to correct state

    See also: Ticket B3 (Idempotency Guard) for runtime-level idempotency.

Intent Emission:
    The reducer emits ModelIntent objects (reducer-layer intents) that wrap
    the typed infrastructure intents:
    - consul.register: Consul service registration
    - postgres.upsert_registration: PostgreSQL record upsert

    The payload contains the serialized typed intent for Effect layer execution.

Confirmation Event Flow:
    This section documents how confirmation events flow from Effect layer back to
    this reducer, completing the registration workflow cycle.

    1. INITIAL FLOW (Introspection -> Intents):

        +----------------+     +-----------+     +------------------+
        | Node emits     | --> | Reducer   | --> | Intents emitted  |
        | Introspection  |     | processes |     | to Kafka         |
        | Event          |     | event     |     | (consul.register,|
        +----------------+     +-----------+     | postgres.upsert) |
                                                 +------------------+
                                                          |
                                                          v
                                             +------------------------+
                                             | Runtime routes intents |
                                             | to Effect layer nodes  |
                                             +------------------------+

    2. EFFECT LAYER EXECUTION:

        +-------------------+     +------------------+     +------------------+
        | ConsulAdapter     | --> | Execute intent   | --> | Publish          |
        | (Effect Node)     |     | (register svc)   |     | confirmation     |
        +-------------------+     +------------------+     | event to Kafka   |
                                                          +------------------+
                                                                   |
        +-------------------+     +------------------+             |
        | PostgresAdapter   | --> | Execute intent   | ------------+
        | (Effect Node)     |     | (upsert record)  |             |
        +-------------------+     +------------------+             v
                                                          +------------------+
                                                          | Confirmation     |
                                                          | events on Kafka  |
                                                          +------------------+

    3. CONFIRMATION EVENT FLOW (Back to Reducer):

        +-------------------+     +------------------+     +-------------------+
        | Kafka topic:      | --> | Runtime routes   | --> | Reducer processes |
        | onex.registration.|     | confirmation     |     | confirmation via  |
        | events            |     | to reducer       |     | reduce_confirm()  |
        +-------------------+     +------------------+     +-------------------+
                                                                   |
                                                                   v
                                                          +-------------------+
                                                          | State transitions:|
                                                          | pending -> partial|
                                                          | partial -> complete|
                                                          +-------------------+

    4. CONFIRMATION EVENT TYPES:

        - consul.registered: Confirmation from ConsulAdapter that service
          was successfully registered in Consul. Published to:
          onex.registration.events (or onex.<domain>.events)

          Payload includes:
            - correlation_id: Links back to original introspection event
            - service_id: The registered Consul service ID
            - success: bool indicating registration outcome
            - error: Optional error message if failed

        - postgres.registration_upserted: Confirmation from PostgresAdapter
          that registration record was successfully upserted. Published to:
          onex.registration.events

          Payload includes:
            - correlation_id: Links back to original introspection event
            - node_id: The registered node ID
            - success: bool indicating upsert outcome
            - error: Optional error message if failed

    5. STATE TRANSITION DIAGRAM:

        +-------+   introspection   +---------+
        | idle  | ----------------> | pending |
        +-------+                   +---------+
           ^                         |       |
           |       consul confirmed  |       | postgres confirmed
           |       (first)          v       v (first)
           |                  +---------+
           |                  | partial |
           |                  +---------+
           |                    |       |
           |   remaining        |       | error received
           |   confirmed        v       v
           |              +---------+ +---------+
           +---reset------| complete| | failed  |---reset---+
                          +---------+ +---------+           |
                                                            v
                                                      +-------+
                                                      | idle  |
                                                      +-------+

        Transitions:
        - idle -> pending: On introspection event (emits intents)
        - pending -> partial: First confirmation received (consul OR postgres)
        - pending -> failed: Error confirmation received
        - partial -> complete: Second confirmation received (both confirmed)
        - partial -> failed: Error confirmation for remaining backend
        - any -> failed: Validation or backend error
        - failed -> idle: Reset event (allows retry after failure)
        - complete -> idle: Reset event (allows re-registration)

    6. IMPLEMENTATION NOTE - reduce_confirmation():

        The reduce_confirmation() method (to be implemented) will handle
        confirmation events. It uses the same pure reducer pattern:

            def reduce_confirmation(
                self,
                state: ModelRegistrationState,
                confirmation: ModelRegistrationConfirmation,
            ) -> ModelReducerOutput[ModelRegistrationState]:
                '''Process confirmation event from Effect layer.'''
                # Validate confirmation matches current node_id
                # Transition state based on confirmation type:
                #   - consul.registered -> with_consul_confirmed()
                #   - postgres.registration_upserted -> with_postgres_confirmed()
                #   - error -> with_failure()
                # Return new state with no intents (confirmations don't emit new intents)

        The confirmation event model should include:
            - event_type: "consul.registered" | "postgres.registration_upserted"
            - correlation_id: UUID linking to original introspection
            - node_id: UUID of the registered node
            - success: bool
            - error_message: str | None
            - timestamp: datetime

    7. IDEMPOTENCY:

        Confirmation events are also subject to idempotency:
        - Duplicate confirmations (same event_id) are skipped
        - Confirmations for wrong node_id are rejected
        - Re-confirmations after complete/failed are no-ops

    8. TIMEOUT HANDLING:

        Per DESIGN_TWO_WAY_REGISTRATION_ARCHITECTURE.md, timeouts are owned
        by the Orchestrator layer, not the Reducer:

        - Orchestrator tracks pending registrations with deadlines
        - Orchestrator consumes RuntimeTick events for timeout evaluation
        - Orchestrator emits RegistrationTimedOut events when deadline passes
        - Reducer folds RegistrationTimedOut as a failure confirmation

Related:
    - NodeDualRegistrationReducer: Legacy 887-line implementation (deprecated)
    - DESIGN_TWO_WAY_REGISTRATION_ARCHITECTURE.md: Architecture design
    - MESSAGE_DISPATCH_ENGINE.md: How runtime routes events to reducers
    - OMN-889: Infrastructure MVP - ModelNodeIntrospectionEvent
    - OMN-912: ModelIntent typed payloads
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from datetime import UTC, datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, field_validator

from omnibase_core.enums import EnumNodeKind, EnumReductionType, EnumStreamingMode
from omnibase_core.models.reducer.model_intent import ModelIntent
from omnibase_core.nodes import ModelReducerOutput
from omnibase_infra.enums import EnumConfirmationEventType
from omnibase_infra.models.registration import (
    ModelNodeIntrospectionEvent,
    ModelNodeRegistrationRecord,
)
from omnibase_infra.nodes.reducers.models.model_payload_consul_register import (
    ModelPayloadConsulRegister,
)
from omnibase_infra.nodes.reducers.models.model_payload_postgres_upsert_registration import (
    ModelPayloadPostgresUpsertRegistration,
)
from omnibase_infra.nodes.reducers.models.model_registration_confirmation import (
    ModelRegistrationConfirmation,
)
from omnibase_infra.nodes.reducers.models.model_registration_state import (
    ModelRegistrationState,
)

# =============================================================================
# Performance Threshold Constants (in milliseconds)
#
# These constants define the performance targets for the RegistrationReducer.
# When processing time exceeds these thresholds, a warning is logged to help
# identify performance regressions or unusually complex events.
#
# These are intentionally generous for production use (300ms target) because:
# 1. Pure reducers should be fast (no I/O)
# 2. Most events complete in <5ms
# 3. Threshold alerts indicate something unusual (test environment, GC pause, etc.)
#
# Environment Variable Configuration:
#   ONEX_PERF_THRESHOLD_REDUCE_MS - reduce() processing threshold (default: 300.0)
#   ONEX_PERF_THRESHOLD_INTENT_BUILD_MS - intent building threshold (default: 50.0)
#   ONEX_PERF_THRESHOLD_IDEMPOTENCY_CHECK_MS - idempotency check threshold (default: 1.0)
#
# Example usage:
#   export ONEX_PERF_THRESHOLD_REDUCE_MS=100.0  # Stricter threshold for production
#   export ONEX_PERF_THRESHOLD_REDUCE_MS=1000.0  # Relaxed threshold for dev/CI
# =============================================================================

# Target processing time for reduce() method (<300ms per event)
# This is the primary performance metric for the reducer.
PERF_THRESHOLD_REDUCE_MS: float = float(
    os.getenv("ONEX_PERF_THRESHOLD_REDUCE_MS", "300.0")
)

# Target processing time for intent building (<50ms per intent)
# Consul and PostgreSQL intent construction should be fast.
PERF_THRESHOLD_INTENT_BUILD_MS: float = float(
    os.getenv("ONEX_PERF_THRESHOLD_INTENT_BUILD_MS", "50.0")
)

# Target processing time for idempotency check (<1ms)
# Simple UUID comparison should be nearly instant.
PERF_THRESHOLD_IDEMPOTENCY_CHECK_MS: float = float(
    os.getenv("ONEX_PERF_THRESHOLD_IDEMPOTENCY_CHECK_MS", "1.0")
)

# Logger for performance warnings and validation errors
_logger = logging.getLogger(__name__)


# =============================================================================
# Validation Error Types
# =============================================================================

ValidationErrorCode = Literal[
    "missing_node_id",
    "missing_node_type",
    "invalid_node_type",
]

# Sentinel value for "not set" state
_SENTINEL_STR: str = ""


class ModelValidationResult(BaseModel):
    """Result of event validation with detailed error information.

    This Pydantic model replaces the previous dataclass implementation
    to comply with ONEX requirements for Pydantic-based data structures.

    This model uses sentinel values (empty string) instead of nullable unions
    to minimize union count in the codebase (OMN-1004).

    Sentinel Values:
        - Empty string ("") for field_name and error_message means "not set"
        - None for error_code (unavoidable for Literal type safety)
        - Use ``has_field_name``, ``has_error_message`` to check

    Constructor API:
        Constructors accept ``None`` for string fields and convert to sentinel.

    Attributes:
        is_valid: Whether the event passed validation.
        error_code: Distinct code identifying the validation failure (if any).
        field_name: Name of the field that failed validation. Empty string if not set.
        error_message: Human-readable error message for logging. Empty string if not set.

    .. versionchanged:: 0.7.0
        Refactored to use sentinel values for string fields (OMN-1004).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    is_valid: bool
    error_code: ValidationErrorCode | None = None
    field_name: str = _SENTINEL_STR
    error_message: str = _SENTINEL_STR

    # ---- Validators for None-to-Sentinel Conversion ----
    @field_validator("field_name", "error_message", mode="before")
    @classmethod
    def _convert_none_to_str_sentinel(cls, v: object) -> str:
        """Convert None to empty string sentinel for API convenience."""
        if v is None:
            return _SENTINEL_STR
        if isinstance(v, str):
            return v
        return str(v)

    # ---- Sentinel Check Properties ----
    @property
    def has_field_name(self) -> bool:
        """Check if field_name is set (not empty string)."""
        return self.field_name != _SENTINEL_STR

    @property
    def has_error_message(self) -> bool:
        """Check if error_message is set (not empty string)."""
        return self.error_message != _SENTINEL_STR

    @classmethod
    def success(cls) -> ModelValidationResult:
        """Create a successful validation result."""
        return cls(is_valid=True)

    @classmethod
    def failure(
        cls,
        error_code: ValidationErrorCode,
        field_name: str,
        error_message: str,
    ) -> ModelValidationResult:
        """Create a failed validation result with error details."""
        return cls(
            is_valid=False,
            error_code=error_code,
            field_name=field_name,
            error_message=error_message,
        )


# TODO(OMN-889): Complete pure reducer implementation - add reduce_confirmation() method
class RegistrationReducer:
    """Pure reducer for node registration workflow.

    Follows ProtocolReducer pattern:
    - reduce(state, event) -> ModelReducerOutput
    - Pure function, no side effects
    - Emits intents for Consul and PostgreSQL registration

    This is a stateless class - all state is passed in and returned via
    ModelRegistrationState. The class exists to group related pure functions.

    Event Processing Methods:
        This reducer handles two categories of events:

        1. reduce(state, introspection_event) -> Processes initial node introspection,
           emits registration intents for Effect layer execution.

        2. reduce_confirmation(state, confirmation_event) -> Processes confirmation
           events from Effect layer, updates state to partial/complete/failed.
           (See module docstring section 6 for implementation details.)

    Complete Event Cycle:
        1. Node publishes introspection event to Kafka
        2. Runtime routes introspection to this reducer via reduce()
        3. Reducer emits intents (consul.register, postgres.upsert_registration)
        4. Runtime publishes intents to Kafka intent topics
        5. Effect layer nodes (ConsulAdapter, PostgresAdapter) consume intents
        6. Effect nodes execute I/O and publish confirmation events to Kafka
        7. Runtime routes confirmation events back to this reducer
        8. Reducer updates state: pending -> partial -> complete

    Topic Subscriptions:
        The reducer node subscribes to:
        - onex.registration.events (or onex.<domain>.events)

        This includes both introspection events and confirmation events.
        The reduce() method dispatches to the appropriate handler based on
        event type.

    Example:
        >>> from uuid import uuid4
        >>> from omnibase_infra.models.registration import ModelNodeIntrospectionEvent
        >>> from omnibase_infra.nodes.reducers import RegistrationReducer
        >>> from omnibase_infra.nodes.reducers.models import ModelRegistrationState
        >>>
        >>> reducer = RegistrationReducer()
        >>> state = ModelRegistrationState()  # Initial idle state
        >>> event = ModelNodeIntrospectionEvent(
        ...     node_id=uuid4(),
        ...     node_type="effect",
        ...     node_version="1.0.0",
        ...     endpoints={"health": "http://localhost:8080/health"},
        ... )
        >>> output = reducer.reduce(state, event)
        >>> print(output.result.status)  # "pending"
        >>> print(len(output.intents))   # 2 (Consul + PostgreSQL)
    """

    def reduce(
        self,
        state: ModelRegistrationState,
        event: ModelNodeIntrospectionEvent,
    ) -> ModelReducerOutput[ModelRegistrationState]:
        """Pure reduce function: state + event -> new_state + intents.

        Processes a node introspection event and emits registration intents
        for both Consul and PostgreSQL backends. The returned output contains
        the new state and any intents to be executed by the Effect layer.

        This is PHASE 1 of the confirmation event flow:
            1. Node publishes introspection event -> Runtime routes here
            2. This method processes event -> Emits intents
            3. Runtime publishes intents to Kafka -> Effect layer executes
            4. Effect layer publishes confirmations -> reduce_confirmation() handles

        Idempotency:
            If the event has already been processed (based on event_id), the
            reducer returns immediately with the current state and no intents.

        Validation:
            If the event fails validation (e.g., missing node_id), the reducer
            transitions to failed state with no intents.

        Args:
            state: Current registration state (immutable).
            event: Node introspection event to process.

        Returns:
            ModelReducerOutput containing new_state and intents tuple.
            The result field contains the new ModelRegistrationState.
            The intents field contains registration intents for Effect layer.
        """
        start_time = time.perf_counter()

        # =====================================================================
        # CONFIRMATION FLOW STEP 1: Receive introspection event from Kafka
        # This event was published by a node during startup/discovery.
        # The Runtime (MessageDispatchEngine) routed it here based on:
        #   - Topic: onex.registration.events (or similar)
        #   - Message type: ModelNodeIntrospectionEvent
        # =====================================================================

        # Resolve event ID for idempotency.
        # CRITICAL: We use a deterministic derivation when correlation_id is absent.
        # Using uuid4() here would break idempotency because replayed events would
        # get different IDs each time, making duplicate detection impossible.
        # Instead, we derive a UUID from the event's content hash (node_id + node_type
        # + timestamp), ensuring the same event always produces the same ID.
        event_id = event.correlation_id or self._derive_deterministic_event_id(event)

        # Idempotency guard - skip if we've already processed this event
        if state.is_duplicate_event(event_id):
            return self._build_output(
                state=state,
                intents=(),
                processing_time_ms=0.0,
                items_processed=0,
            )

        # Validate event - failures transition to failed state with no intents
        # Note: Validation errors are logged with sanitized context (no PII/secrets)
        # but the output intentionally uses a generic failure_reason to avoid
        # exposing internal validation logic to external consumers.
        validation_result = self._validate_event(event)
        if not validation_result.is_valid:
            # Log validation failure with sanitized context for diagnostics
            # SECURITY: Only field presence booleans and error codes are logged
            _logger.warning(
                "Event validation failed",
                extra={
                    "error_code": validation_result.error_code,
                    "field_name": validation_result.field_name,
                    "error_message": validation_result.error_message,
                    "correlation_id": str(event_id),
                    "has_node_id": event.node_id is not None,
                    "has_node_type": hasattr(event, "node_type")
                    and event.node_type is not None,
                },
            )
            new_state = state.with_failure("validation_failed", event_id)
            return self._build_output(
                state=new_state,
                intents=(),
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
                items_processed=0,
            )

        # =====================================================================
        # CONFIRMATION FLOW STEP 2: Build intents for Effect layer
        # These intents describe the desired I/O operations:
        #   - consul.register: Register service in Consul
        #   - postgres.upsert_registration: Upsert record in PostgreSQL
        #
        # The correlation_id is propagated to enable confirmation tracking.
        # When Effect nodes complete, they publish confirmation events with
        # this correlation_id, allowing this reducer to match confirmations
        # to the original introspection event.
        # =====================================================================

        correlation_id = event.correlation_id or event_id
        consul_intent = self._build_consul_intent(event, correlation_id)
        postgres_intent = self._build_postgres_intent(event, correlation_id)

        # Collect non-None intents
        intents: tuple[ModelIntent, ...] = tuple(
            intent for intent in [consul_intent, postgres_intent] if intent is not None
        )

        # =====================================================================
        # CONFIRMATION FLOW STEP 3: Transition to pending state
        # State: idle -> pending
        #
        # After this method returns:
        #   - Runtime publishes intents to Kafka (onex.registration.intents)
        #   - Effect nodes (ConsulAdapter, PostgresAdapter) consume intents
        #   - Effect nodes execute I/O and publish confirmation events
        #   - Runtime routes confirmation events to reduce_confirmation()
        #   - reduce_confirmation() transitions: pending -> partial -> complete
        # =====================================================================

        new_state = state.with_pending_registration(event.node_id, event_id)

        processing_time_ms = (time.perf_counter() - start_time) * 1000

        # Performance logging: warn if processing exceeds threshold
        if processing_time_ms > PERF_THRESHOLD_REDUCE_MS:
            _logger.warning(
                "Reducer processing time exceeded threshold",
                extra={
                    "processing_time_ms": processing_time_ms,
                    "threshold_ms": PERF_THRESHOLD_REDUCE_MS,
                    "node_type": event.node_type.value,
                    "intent_count": len(intents),
                    "correlation_id": str(correlation_id),
                },
            )

        return self._build_output(
            state=new_state,
            intents=intents,
            processing_time_ms=processing_time_ms,
            items_processed=1,
        )

    def _is_valid(self, event: ModelNodeIntrospectionEvent) -> bool:
        """Validate introspection event for processing.

        Convenience wrapper around _validate_event() that returns a simple bool.
        Use _validate_event() when detailed error information is needed.

        Args:
            event: Introspection event to validate.

        Returns:
            True if the event is valid for processing, False otherwise.
        """
        return self._validate_event(event).is_valid

    def _validate_event(
        self, event: ModelNodeIntrospectionEvent
    ) -> ModelValidationResult:
        """Validate introspection event with detailed error information.

        Validates that required fields are present for registration workflow.
        Returns a ValidationResult with distinct error codes for each failure
        scenario, enabling proper logging and diagnostics.

        **Validation Rules:**
        - node_id: Must be present (required for registration identity)
        - node_type: Must be present and valid (required for service categorization)

        **Error Codes:**
        - missing_node_id: node_id is None
        - missing_node_type: node_type attribute is missing or None
        - invalid_node_type: node_type is not a valid ONEX node type

        **Security Note:**
        Error messages are logged server-side but the reducer's output uses
        a generic "validation_failed" reason to avoid exposing internal
        validation logic to external consumers.

        Args:
            event: Introspection event to validate.

        Returns:
            ValidationResult with is_valid=True if valid, or detailed error
            information if validation failed.
        """
        # Validate node_id: required for registration identity
        if event.node_id is None:
            return ModelValidationResult.failure(
                error_code="missing_node_id",
                field_name="node_id",
                error_message="node_id is required for registration identity",
            )

        # Validate node_type: must be present
        if not hasattr(event, "node_type") or event.node_type is None:
            return ModelValidationResult.failure(
                error_code="missing_node_type",
                field_name="node_type",
                error_message="node_type is required for service categorization",
            )

        # Validate node_type value is valid ONEX type
        # Use EnumNodeKind values (excluding RUNTIME_HOST which is not a registration type)
        valid_node_types = {
            EnumNodeKind.EFFECT.value,
            EnumNodeKind.COMPUTE.value,
            EnumNodeKind.REDUCER.value,
            EnumNodeKind.ORCHESTRATOR.value,
        }
        if event.node_type.value not in valid_node_types:
            return ModelValidationResult.failure(
                error_code="invalid_node_type",
                field_name="node_type",
                error_message=(
                    f"node_type must be one of: {', '.join(sorted(valid_node_types))}"
                ),
            )

        return ModelValidationResult.success()

    def _derive_deterministic_event_id(
        self, event: ModelNodeIntrospectionEvent
    ) -> UUID:
        """Derive a deterministic event ID from event content.

        When an event lacks a correlation_id, we must derive a stable identifier
        from its content to preserve idempotency guarantees. Using uuid4() would
        break idempotency because replayed events would get different IDs.

        The derived ID uses a SHA-256 hash of the event's identifying fields:
        - node_id: Unique node identifier
        - node_type: Node archetype (effect, compute, reducer, orchestrator)
        - timestamp: Event creation timestamp (ISO format for stability)

        This ensures:
        1. Same event content always produces the same ID
        2. Different events produce different IDs (collision-resistant)
        3. ID format is compatible with existing UUID-based tracking

        Args:
            event: The introspection event to derive an ID from.

        Returns:
            A deterministic UUID derived from the event's content.
        """
        # Build a canonical string from the event's identifying fields.
        # Using ISO format for timestamp ensures string stability across serialization.
        # The pipe delimiter prevents ambiguity between field values.
        canonical_content = (
            f"{event.node_id}|{event.node_type.value}|{event.timestamp.isoformat()}"
        )

        # Compute SHA-256 hash and convert to UUID format.
        # SHA-256 provides strong collision resistance for content-derived IDs.
        content_hash = hashlib.sha256(canonical_content.encode("utf-8")).hexdigest()

        # Take first 32 hex chars (128 bits) and format as UUID.
        # Insert hyphens in standard UUID format: 8-4-4-4-12
        uuid_hex = content_hash[:32]
        uuid_str = (
            f"{uuid_hex[:8]}-{uuid_hex[8:12]}-{uuid_hex[12:16]}-"
            f"{uuid_hex[16:20]}-{uuid_hex[20:32]}"
        )

        return UUID(uuid_str)

    def _build_consul_intent(
        self,
        event: ModelNodeIntrospectionEvent,
        correlation_id: UUID,
    ) -> ModelIntent | None:
        """Build Consul registration intent (pure, no I/O).

        Creates a ModelIntent that describes the desired Consul service
        registration. The Effect layer is responsible for executing this intent.

        Args:
            event: Introspection event containing node data.
            correlation_id: Correlation ID for tracing.

        Returns:
            ModelIntent with intent_type="extension" (dual-layer routing).
            The routing key "consul.register" is in payload.intent_type.
        """
        service_id = f"onex-{event.node_type.value}-{event.node_id}"
        service_name = f"onex-{event.node_type.value}"
        tags = [
            f"node_type:{event.node_type.value}",
            f"node_version:{event.node_version}",
        ]

        # Build health check configuration if health endpoint is provided
        health_endpoint = event.endpoints.get("health") if event.endpoints else None
        health_check: dict[str, str] | None = None
        if health_endpoint:
            health_check = {
                "HTTP": health_endpoint,
                "Interval": "10s",
                "Timeout": "5s",
            }

        # Build typed Consul registration payload (implements ProtocolIntentPayload)
        consul_payload = ModelPayloadConsulRegister(
            correlation_id=correlation_id,
            service_id=service_id,
            service_name=service_name,
            tags=tags,
            health_check=health_check,
            event_bus_config=event.event_bus,  # Pass through from introspection event
        )

        # ModelIntent.payload expects ProtocolIntentPayload, which our model implements
        return ModelIntent(
            intent_type="extension",
            target=f"consul://service/{service_name}",
            payload=consul_payload,
        )

    def _build_postgres_intent(
        self,
        event: ModelNodeIntrospectionEvent,
        correlation_id: UUID,
    ) -> ModelIntent | None:
        """Build PostgreSQL upsert intent (pure, no I/O).

        Creates a ModelIntent that describes the desired PostgreSQL record
        upsert. The Effect layer is responsible for executing this intent.

        Args:
            event: Introspection event containing node data.
            correlation_id: Correlation ID for tracing.

        Returns:
            ModelIntent with intent_type="extension" (dual-layer routing).
            The routing key "postgres.upsert_registration" is in payload.intent_type.
        """
        now = datetime.now(UTC)

        # Build the registration record using strongly-typed models
        # event.declared_capabilities and event.metadata are already typed as
        # ModelNodeCapabilities and ModelNodeMetadata respectively
        record = ModelNodeRegistrationRecord(
            node_id=event.node_id,
            node_type=event.node_type,
            node_version=event.node_version,
            capabilities=event.declared_capabilities,
            endpoints=dict(event.endpoints) if event.endpoints else {},
            metadata=event.metadata,
            health_endpoint=(
                event.endpoints.get("health") if event.endpoints else None
            ),
            registered_at=now,
            updated_at=now,
        )

        # Build typed PostgreSQL upsert payload (implements ProtocolIntentPayload)
        postgres_payload = ModelPayloadPostgresUpsertRegistration(
            correlation_id=correlation_id,
            record=record,
        )

        # ModelIntent.payload expects ProtocolIntentPayload, which our model implements
        return ModelIntent(
            intent_type="extension",
            target=f"postgres://node_registrations/{event.node_id}",
            payload=postgres_payload,
        )

    # =========================================================================
    # CONFIRMATION EVENT HANDLING (PHASE 2 of the event flow)
    #
    # The following method will handle confirmation events from Effect layer.
    # It is documented here as a stub to show the complete event flow.
    #
    # Follow-up Ticket: OMN-996 (Implement Confirmation Event Handling)
    #   https://linear.app/omninode/issue/OMN-996
    #
    # Prerequisites:
    #   - [DONE] ModelRegistrationConfirmation model defined
    #     See: omnibase_infra.nodes.reducers.models.model_registration_confirmation
    #   - Effect layer confirmation event publishing implemented
    #   - Tests added for confirmation event handling
    #
    # See module docstring section 6 for detailed implementation notes.
    # =========================================================================

    def reduce_confirmation(
        self,
        state: ModelRegistrationState,
        confirmation: ModelRegistrationConfirmation,
    ) -> ModelReducerOutput[ModelRegistrationState]:
        """Process confirmation event from Effect layer.

        Not yet implemented. See OMN-996 for tracking.

        Args:
            state: Current registration state (immutable).
            confirmation: Confirmation event from Effect layer.

        Returns:
            ModelReducerOutput with new state and no intents.

        Raises:
            NotImplementedError: Always raised until implementation is complete.
        """
        raise NotImplementedError(
            "reduce_confirmation() is not yet implemented. "
            "See ticket OMN-996: https://linear.app/omninode/issue/OMN-996"
        )

    # TODO(OMN-996): Implement reduce_confirmation() using ModelRegistrationConfirmation
    # Ticket: https://linear.app/omninode/issue/OMN-996
    # Status: Backlog - Phase 2 of dual registration event flow
    #
    # Scope: Process confirmation events from Effect layer (Consul/PostgreSQL)
    # to complete state transitions: pending -> partial -> complete
    #
    def reduce_reset(
        self,
        state: ModelRegistrationState,
        reset_event_id: UUID,
    ) -> ModelReducerOutput[ModelRegistrationState]:
        """Process a reset event to recover from failed or complete states.

        This method allows the FSM to recover from terminal states (failed, complete)
        and return to idle, enabling retry workflows after failures.

        State Validation:
            Reset is ONLY allowed from terminal states (failed, complete). Attempting
            to reset from in-flight states (pending, partial) or idle will result in
            a failed state with failure_reason="invalid_reset_state".

            This validation prevents accidental loss of in-flight registration state.
            If a reset is attempted while registration is in progress (pending/partial),
            the Consul or PostgreSQL confirmations could be lost, leaving the system
            in an inconsistent state.

        Use Cases:
            - Retry after registration failure (consul_failed, postgres_failed)
            - Re-register a node after deregistration
            - Manual recovery triggered by operator

        State Transitions:
            - failed -> idle: Clears failure, enables retry
            - complete -> idle: Enables re-registration
            - idle -> failed: Invalid reset (already idle)
            - pending -> failed: Invalid reset (would lose in-flight state)
            - partial -> failed: Invalid reset (would lose in-flight state)

        Idempotency:
            Reset events are subject to the same idempotency checks as other events.
            If the reset_event_id matches last_processed_event_id, no transition occurs.

        Args:
            state: Current registration state (immutable).
            reset_event_id: UUID of the reset event triggering this transition.

        Returns:
            ModelReducerOutput with new state and no intents.
            - If reset allowed: new state is idle
            - If reset not allowed: new state is failed with
              failure_reason="invalid_reset_state"

        Example:
            >>> from uuid import uuid4
            >>> from omnibase_infra.nodes.reducers import RegistrationReducer
            >>> from omnibase_infra.nodes.reducers.models import ModelRegistrationState
            >>>
            >>> reducer = RegistrationReducer()
            >>> # Reset from failed state succeeds
            >>> failed_state = ModelRegistrationState(
            ...     status="failed",
            ...     failure_reason="consul_failed"
            ... )
            >>> output = reducer.reduce_reset(failed_state, uuid4())
            >>> output.result.status
            'idle'
            >>>
            >>> # Reset from pending state fails (would lose in-flight state)
            >>> pending_state = ModelRegistrationState(status="pending")
            >>> output = reducer.reduce_reset(pending_state, uuid4())
            >>> output.result.status
            'failed'
            >>> output.result.failure_reason
            'invalid_reset_state'
        """
        start_time = time.perf_counter()

        # Idempotency guard
        if state.is_duplicate_event(reset_event_id):
            return self._build_output(
                state=state,
                intents=(),
                processing_time_ms=0.0,
                items_processed=0,
            )

        # Validate state allows reset - only terminal states (failed, complete)
        # can be reset. Resetting from pending or partial would lose in-flight
        # registration state, potentially causing inconsistency between Consul
        # and PostgreSQL.
        if not state.can_reset():
            # Not in a resettable state - transition to failed with clear error
            # This prevents accidental loss of in-flight registration state.
            new_state = state.with_failure("invalid_reset_state", reset_event_id)
            return self._build_output(
                state=new_state,
                intents=(),
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
                items_processed=1,  # We processed the event (it caused a state change)
            )

        # Perform the reset transition
        new_state = state.with_reset(reset_event_id)

        processing_time_ms = (time.perf_counter() - start_time) * 1000

        return self._build_output(
            state=new_state,
            intents=(),  # Reset emits no intents
            processing_time_ms=processing_time_ms,
            items_processed=1,
        )

    def _build_output(
        self,
        state: ModelRegistrationState,
        intents: tuple[ModelIntent, ...],
        processing_time_ms: float,
        items_processed: int,
    ) -> ModelReducerOutput[ModelRegistrationState]:
        """Build standardized ModelReducerOutput.

        Creates the output model with all required fields for the
        omnibase_core reducer output contract.

        Args:
            state: New registration state to return.
            intents: Tuple of ModelIntent objects to emit.
            processing_time_ms: Time taken to process the event.
            items_processed: Number of events processed (0 or 1).

        Returns:
            ModelReducerOutput containing the state and intents.
        """
        return ModelReducerOutput(
            result=state,
            operation_id=uuid4(),
            reduction_type=EnumReductionType.MERGE,
            processing_time_ms=processing_time_ms,
            items_processed=items_processed,
            conflicts_resolved=0,
            streaming_mode=EnumStreamingMode.BATCH,
            batches_processed=1,
            intents=intents,
        )


__all__ = [
    "PERF_THRESHOLD_IDEMPOTENCY_CHECK_MS",
    "PERF_THRESHOLD_INTENT_BUILD_MS",
    # Performance threshold constants (for tests and monitoring)
    "PERF_THRESHOLD_REDUCE_MS",
    # Validation types (for tests and custom validators)
    "ModelValidationResult",
    "RegistrationReducer",
    "ValidationErrorCode",
]
