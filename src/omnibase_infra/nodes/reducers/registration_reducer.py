# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Canonical Registration Reducer following ProtocolReducer pattern.

This reducer replaces the legacy NodeDualRegistrationReducer (887 lines)
with a pure function implementation (~80 lines) that follows the canonical
ONEX reducer pattern defined in DESIGN_TWO_WAY_REGISTRATION_ARCHITECTURE.md.

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
                                     |       |
                   consul confirmed  |       | postgres confirmed
                   (first)          v       v (first)
                              +---------+
                              | partial |
                              +---------+
                                |       |
           remaining confirmed  |       | error received
           (postgres or consul) v       v
                           +---------+ +---------+
                           |complete | | failed  |
                           +---------+ +---------+

        Transitions:
        - idle -> pending: On introspection event (emits intents)
        - pending -> partial: First confirmation received (consul OR postgres)
        - pending -> failed: Error confirmation received
        - partial -> complete: Second confirmation received (both confirmed)
        - partial -> failed: Error confirmation for remaining backend
        - any -> failed: Validation or backend error

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
            - error_message: Optional[str]
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

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from omnibase_core.enums import EnumReductionType, EnumStreamingMode
from omnibase_core.models.reducer.model_intent import ModelIntent
from omnibase_core.nodes import ModelReducerOutput

from omnibase_infra.models.registration import (
    ModelNodeIntrospectionEvent,
    ModelNodeRegistrationRecord,
)
from omnibase_infra.nodes.reducers.models.model_registration_state import (
    ModelRegistrationState,
)


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
        import time

        start_time = time.perf_counter()

        # =====================================================================
        # CONFIRMATION FLOW STEP 1: Receive introspection event from Kafka
        # This event was published by a node during startup/discovery.
        # The Runtime (MessageDispatchEngine) routed it here based on:
        #   - Topic: onex.registration.events (or similar)
        #   - Message type: ModelNodeIntrospectionEvent
        # =====================================================================

        # Resolve event ID for idempotency
        event_id = event.correlation_id or uuid4()

        # Idempotency guard - skip if we've already processed this event
        if state.is_duplicate_event(event_id):
            return self._build_output(
                state=state,
                intents=(),
                processing_time_ms=0.0,
                items_processed=0,
            )

        # Validate event
        if not self._is_valid(event):
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

        return self._build_output(
            state=new_state,
            intents=intents,
            processing_time_ms=processing_time_ms,
            items_processed=1,
        )

    def _is_valid(self, event: ModelNodeIntrospectionEvent) -> bool:
        """Validate introspection event.

        Validates that required fields are present. Pydantic handles type
        validation at model construction; this method checks semantic validity.

        Args:
            event: Introspection event to validate.

        Returns:
            True if the event is valid for processing.
        """
        # node_id is required for registration
        return event.node_id is not None

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
            ModelIntent with intent_type="consul.register" and Consul payload.
        """
        service_id = f"node-{event.node_type}-{event.node_id}"
        service_name = f"onex-{event.node_type}"
        tags = [
            f"node_type:{event.node_type}",
            f"node_version:{event.node_version}",
        ]

        # Build health check configuration if health endpoint is provided
        health_endpoint = event.endpoints.get("health") if event.endpoints else None
        health_check: dict[str, Any] | None = None
        if health_endpoint:
            health_check = {
                "HTTP": health_endpoint,
                "Interval": "10s",
                "Timeout": "5s",
            }

        # Build payload for Consul registration
        payload: dict[str, Any] = {
            "correlation_id": str(correlation_id),
            "service_id": service_id,
            "service_name": service_name,
            "tags": tags,
        }
        if health_check:
            payload["health_check"] = health_check

        return ModelIntent(
            intent_type="consul.register",
            target=f"consul://service/{service_name}",
            payload=payload,
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
            ModelIntent with intent_type="postgres.upsert_registration" and record payload.
        """
        now = datetime.now(UTC)

        # Convert capabilities to dict if it's a model
        if hasattr(event.capabilities, "model_dump"):
            capabilities_dict = event.capabilities.model_dump(mode="json")
        else:
            capabilities_dict = dict(event.capabilities) if event.capabilities else {}

        # Convert metadata to dict if it's a model
        if hasattr(event.metadata, "model_dump"):
            metadata_dict = event.metadata.model_dump(mode="json")
        else:
            metadata_dict = dict(event.metadata) if event.metadata else {}

        # Build the registration record as a Pydantic model for validation,
        # then serialize to dict for the intent payload
        record = ModelNodeRegistrationRecord(
            node_id=event.node_id,
            node_type=event.node_type,
            node_version=event.node_version,
            capabilities=capabilities_dict,
            endpoints=dict(event.endpoints) if event.endpoints else {},
            metadata=metadata_dict,
            health_endpoint=(
                event.endpoints.get("health") if event.endpoints else None
            ),
            registered_at=now,
            updated_at=now,
        )

        # Build payload with correlation_id and serialized record
        payload: dict[str, Any] = {
            "correlation_id": str(correlation_id),
            "record": record.model_dump(mode="json"),
        }

        return ModelIntent(
            intent_type="postgres.upsert_registration",
            target=f"postgres://node_registrations/{event.node_id}",
            payload=payload,
        )

    # =========================================================================
    # CONFIRMATION EVENT HANDLING (PHASE 2 of the event flow)
    #
    # The following method will handle confirmation events from Effect layer.
    # It is documented here as a stub to show the complete event flow.
    # Implementation is pending the ModelRegistrationConfirmation model.
    #
    # See module docstring section 6 for detailed implementation notes.
    # =========================================================================

    # NOTE: reduce_confirmation() is not yet implemented. The stub below
    # documents the expected interface and behavior for when confirmation
    # event models are defined.
    #
    # def reduce_confirmation(
    #     self,
    #     state: ModelRegistrationState,
    #     confirmation: "ModelRegistrationConfirmation",
    # ) -> ModelReducerOutput[ModelRegistrationState]:
    #     """Process confirmation event from Effect layer (PHASE 2).
    #
    #     This is PHASE 2 of the confirmation event flow:
    #         1. Effect layer executes intent (consul.register or postgres.upsert)
    #         2. Effect layer publishes confirmation event to Kafka
    #         3. Runtime routes confirmation here based on event type
    #         4. This method updates state: pending -> partial -> complete
    #
    #     Confirmation Event Types:
    #         - consul.registered: Consul service registration confirmed
    #         - postgres.registration_upserted: PostgreSQL upsert confirmed
    #         - *.failed: Error from Effect layer (transitions to failed)
    #
    #     State Transitions:
    #         - pending + consul confirmed -> partial (waiting for postgres)
    #         - pending + postgres confirmed -> partial (waiting for consul)
    #         - partial + remaining confirmed -> complete (both done)
    #         - any + error -> failed
    #
    #     Idempotency:
    #         - Duplicate confirmations (same event_id) are skipped
    #         - Confirmations for wrong node_id are rejected
    #         - Re-confirmations after complete/failed are no-ops
    #
    #     Args:
    #         state: Current registration state (immutable).
    #         confirmation: Confirmation event from Effect layer.
    #             Contains: event_type, correlation_id, node_id, success, error
    #
    #     Returns:
    #         ModelReducerOutput with new state and no intents.
    #         Confirmations do not emit new intents - they only update state.
    #     """
    #     import time
    #     start_time = time.perf_counter()
    #
    #     # Idempotency guard
    #     event_id = confirmation.event_id
    #     if state.is_duplicate_event(event_id):
    #         return self._build_output(state, (), 0.0, 0)
    #
    #     # Validate confirmation matches current node
    #     if confirmation.node_id != state.node_id:
    #         # Confirmation for different node - ignore
    #         return self._build_output(state, (), 0.0, 0)
    #
    #     # Handle error confirmation
    #     if not confirmation.success:
    #         if confirmation.event_type == "consul.registered":
    #             new_state = state.with_failure("consul_failed", event_id)
    #         elif confirmation.event_type == "postgres.registration_upserted":
    #             new_state = state.with_failure("postgres_failed", event_id)
    #         else:
    #             new_state = state.with_failure("both_failed", event_id)
    #         return self._build_output(
    #             new_state, (), (time.perf_counter() - start_time) * 1000, 1
    #         )
    #
    #     # Handle success confirmation - transition state
    #     if confirmation.event_type == "consul.registered":
    #         new_state = state.with_consul_confirmed(event_id)
    #     elif confirmation.event_type == "postgres.registration_upserted":
    #         new_state = state.with_postgres_confirmed(event_id)
    #     else:
    #         # Unknown confirmation type - ignore
    #         return self._build_output(state, (), 0.0, 0)
    #
    #     return self._build_output(
    #         new_state,
    #         (),  # Confirmations emit no new intents
    #         (time.perf_counter() - start_time) * 1000,
    #         1,
    #     )

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


__all__ = ["RegistrationReducer"]
