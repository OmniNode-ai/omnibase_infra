# DESIGN: Event-Driven Orchestration and Reducer Pattern
## Canonical Workflow Architecture for ONEX

    Version: 2.1.0
    Status: Design (Canonical, Public)
    Created: 2025-12-18
    Updated: 2025-12-19
    Scope: This document defines the foundational workflow architecture for ONEX.
           All future workflows (registration, discovery, health, provisioning, etc.)
           MUST conform to this pattern.

---

## Linear Ticket References

    Canonical Plan: ONEX_RUNTIME_REGISTRATION_TICKET_PLAN.md

    Foundation:
        A1  Terminology & Layering      OMN-931
        A2  Execution Shapes            OMN-933
        A2a Message Envelope            OMN-936
        A2b Topic Taxonomy              OMN-939
        A3  Registration Trigger        OMN-943

    Runtime:
        B1  Dispatch Engine             OMN-934
        B1a Message Type Registry       OMN-937
        B2  Handler Output Model        OMN-941
        B3  Idempotency Guard           OMN-945
        B4  Handler Context             OMN-948
        B5  Correlation Propagation     OMN-951
        B6  Runtime Scheduler           OMN-953

    Orchestrator:
        C0  Projection Reader           OMN-930
        C1  Registration Orchestrator   OMN-888 (In Progress)
        C2  Durable Timeout Handling    OMN-932
        C3  Command-Based Registration  OMN-935

    Reducer:
        D1  Registration Reducer        OMN-889 (In Review)
        D2  FSM Contract                OMN-938
        D3  Reducer Test Suite          OMN-942

    Effects:
        E1  Registry Effect             OMN-890 (Done)
        E2  Compensation & Retry        OMN-946
        E3  Dead Letter Queue           OMN-949

    Projection:
        F0  Projector Execution Model   OMN-940
        F1  Registration Schema         OMN-944
        F2  Snapshot Publishing         OMN-947

    Testing:
        G1  Reducer Tests               OMN-950
        G2  Orchestrator Tests          OMN-952
        G3  E2E Integration Tests       OMN-915
        G4  Effect Idempotency Tests    OMN-954
        G5  Chaos & Replay Tests        OMN-955

    Migration:
        H1  Refactor Plan               OMN-956
        H2  Migration Checklist         OMN-957

---

## 0. Design Principles

    1) Events represent facts.
        - Events describe things that have occurred.
        - Events may trigger workflow progression.

    2) Decisions are explicit.
        - Orchestrators make decisions and emit decision events.
        - Reducers do not make decisions.

    3) Reducers are pure.
        - No I/O, no network access, no clock reads.
        - Deterministic behavior: identical inputs produce identical outputs.

    4) Time is owned by orchestrators.
        - Time-based behavior (timeouts, expiry) is implemented by emitting events.
        - Reducers do not evaluate time.

    5) I/O is isolated.
        - External I/O is executed only by effect handlers.
        - Business logic does not exist in effect nodes.

    6) Ordering is deterministic.
        - Ordering is based on partition offsets or monotonic sequence numbers.
        - Wall-clock timestamps are for audit purposes only.

    7) Idempotency is required.
        - All handlers must be safe under at-least-once delivery.

---

## 1. Terminology

    Command:
        - A request directed to a decision-making component.
        - Requires authentication and authorization.
        - Example: RegisterNodeRequested.

    Event:
        - A statement of fact emitted by the component that owns the fact.
        - Used to describe state changes and trigger workflows.
        - Example: NodeIntrospected, NodeRegistrationAccepted.

    Reducer:
        - A pure function that folds events into state.
        - Produces deterministic outputs such as projections and intents.

    Intent:
        - A deterministic instruction describing an external I/O action.
        - Consumed only by effect nodes.

    Handler:
        - A runtime unit that processes a single message.
        - Returns outputs to be published by the runtime.

    Orchestrator:
        - A node responsible for coordinating workflows.
        - Consumes commands and/or events.
        - Emits decision events.
        - Does not perform external I/O.

    Effect Node:
        - A node responsible for executing external I/O.
        - Consumes intents.
        - Contains no business decision logic.

    Projector:
        - Persists reducer outputs into materialized read models.
        - Tracks offsets and ensures idempotent writes.

---

## 2. Canonical Runtime Architecture

### 2.1 Logical Planes

    Ingestion Plane:
        - Consumes messages from Kafka.
        - Dispatches messages to handlers.
        - Enforces idempotency at the handler boundary.

    Decision Plane:
        - Orchestrator handlers consume commands and events.
        - Emit decision events.
        - Own time-based workflow behavior.

    State Plane:
        - Reducers fold events into state.
        - Emit projections and intents.
        - Projectors persist projections.

    Execution Plane:
        - Effect handlers consume intents.
        - Execute external I/O.
        - Optionally emit operation result events.

### 2.2 Data Flow

        Commands and Events
                |
                v
        Runtime Dispatcher
                |
                v
        Orchestrator Handlers
                |
                v
        Immutable Event Log
                |
                v
        Reducer Handlers
          |            |
          v            v
      Projections    Intents
          |            |
          v            v
      Projector    Effect Handlers
                        |
                        v
                  External Systems

---

## 3. Handler Design

### 3.1 Handler Contract

    Handlers process a single message and return outputs.
    The runtime is responsible for publishing outputs.

    Code:

        from typing import Protocol
        from pydantic import BaseModel
        from uuid import UUID

        class ModelHandlerContext(BaseModel):
            correlation_id: UUID
            topic: str
            partition: int
            offset: int
            received_at_epoch_ms: int
            authenticated_principal: str | None = None
            principal_claims: dict | None = None

        class ProtocolHandler(Protocol):
            async def handle(
                self,
                message: BaseModel,
                ctx: ModelHandlerContext,
            ) -> "ModelHandlerOutput":
                ...

        class ModelHandlerOutput(BaseModel):
            events: list[BaseModel] = []
            intents: list[BaseModel] = []
            projections: list[BaseModel] = []
            metrics: dict[str, float] = {}
            logs: list[str] = []

### 3.2 Runtime Dispatch Rules

    - Message routing is determined by topic and message type.
    - Handlers do not publish messages directly.
    - All outputs are returned to the runtime for publication.
    - Side effects outside declared outputs are prohibited.

---

## 4. Topics and Ordering

### 4.1 Required Topics

    - onex.<domain>.commands
    - onex.<domain>.events
    - onex.<domain>.intents
    - onex.<domain>.snapshots (optional)
    - onex.node.lifecycle.events

### 4.2 Partitioning Rules

    - Partition keys must represent the entity whose ordering matters.
    - All workflow logic assumes per-entity ordered delivery.

### 4.3 Cleanup Policies

    - commands: delete
    - events: delete (immutable log)
    - intents: delete
    - snapshots: compact,delete

---

## 5. Message Model

### 5.1 Envelope Fields

    All messages include the following fields:

        from pydantic import BaseModel, ConfigDict
        from uuid import UUID
        from datetime import datetime

        class ModelEnvelope(BaseModel):
            model_config = ConfigDict(frozen=True, extra="forbid")

            message_id: UUID
            correlation_id: UUID
            causation_id: UUID | None = None
            emitted_at: datetime
            entity_id: UUID

    Rule:
        - causation_id references the message_id of the immediate causal message.

---

## 6. Event-Driven Orchestration Pattern

### 6.1 Core Pattern

    - Nodes emit events describing facts (e.g., introspection, heartbeat).
    - Orchestrators consume relevant events and commands.
    - Orchestrators emit decision events.
    - Reducers fold decision events into state.
    - Reducers emit intents for required I/O.
    - Effect nodes execute intents.

### 6.2 Registration Workflow Example

    1) Node emits NodeIntrospected (event).
    2) Registration orchestrator consumes NodeIntrospected.
    3) Orchestrator emits NodeRegistrationAccepted or NodeRegistrationRejected.
    4) Reducer folds registration events and emits:
        - ConsulRegisterIntent
        - PostgresUpsertRegistrationIntent
    5) Effect handlers execute I/O.
    6) Node emits NodeRegistrationAcked (command).
    7) Orchestrator emits NodeRegistrationAckReceived and NodeBecameActive.

    Both event-triggered and command-triggered workflows follow this structure.

---

## 7. Reducer Pattern

### 7.1 Responsibilities

    - Consume events only.
    - Fold events into deterministic state.
    - Emit projections and intents deterministically.

### 7.2 Interface

    Code:

        from typing import Protocol
        from pydantic import BaseModel

        class ProtocolReducer(Protocol):
            def reduce(
                self,
                state: BaseModel,
                event: BaseModel,
            ) -> "ModelReduceOutput":
                ...

        class ModelReduceOutput(BaseModel):
            new_state: BaseModel
            projections: list[BaseModel] = []
            intents: list[BaseModel] = []

### 7.3 State Ownership

    - Reducers own the finite state machine.
    - Reducers do not evaluate time or perform validation logic.

---

## 8. Orchestrator Pattern

### 8.1 Responsibilities

    - Consume events and commands.
    - Validate and coordinate workflows.
    - Emit decision events.
    - Manage time-based behavior via emitted events.
    - Read current state from projections.

### 8.2 Timeout Handling

    Timeouts must be durable and restart-safe.

    Recommended approach:
        - Store deadlines in projections.
        - Periodically query for overdue entities.
        - Emit timeout events after validating current state.

---

## 9. Effect Pattern

### 9.1 Responsibilities

    - Consume intents.
    - Execute external I/O.
    - Apply retries and circuit breakers.
    - Emit operation result events only when required.

    Effects do not:
        - Decide workflow transitions.
        - Modify domain state directly.

---

## 10. Projector Pattern

### 10.1 Responsibilities

    - Persist projections.
    - Track last processed offset.
    - Ensure idempotent writes.

    Example:

        UPDATE node_registrations
        SET
            state = $state,
            updated_at = $updated_at,
            last_event_offset = $offset
        WHERE
            node_id = $node_id
            AND (last_event_offset IS NULL OR $offset > last_event_offset);

---

## 11. Handler Wiring Summary

### Orchestrator Node

    Subscribes to:
        - <domain>.commands
        - <domain>.events
        - node.lifecycle.events

### Reducer Node

    Subscribes to:
        - <domain>.events

### Effect Node

    Subscribes to:
        - <domain>.intents

### Projector

    Applies reducer projection outputs to storage.

---

## 12. Testing Requirements

    Reducers:
        - Event-sequence determinism tests.
        - No command handling.

    Orchestrators:
        - Event/command to decision-event tests.
        - Durable timeout behavior.

    Effects:
        - Intent execution tests.
        - Idempotency and retry behavior.

    Integration:
        - End-to-end workflows.
        - Restart and duplicate delivery scenarios.

---

## 13. Summary

    This document defines the canonical ONEX workflow architecture:

        - Event-driven orchestration
        - Explicit decision boundaries
        - Pure reducers
        - Isolated I/O
        - Deterministic state and ordering

    All future workflows MUST follow this structure to ensure
    correctness, observability, and long-term maintainability.
