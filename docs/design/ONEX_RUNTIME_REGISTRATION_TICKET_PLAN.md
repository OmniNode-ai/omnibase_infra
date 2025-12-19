# ONEX Runtime & Two-Way Registration — Canonical Ticket Plan

    Status: Canonical
    Audience: Internal + Public
    Purpose:
        This ticket set defines the authoritative execution, orchestration, reducer, handler,
        projector, and effect model for ONEX. All future workflows MUST conform.

---

## Global Architectural Constraints (Apply to ALL Tickets)

    1. Reducers fold EVENTS only
        - Reducers never fold commands
        - Reducers never read clocks
        - Reducers use event.emitted_at only

    2. Orchestrators own workflow and time
        - Orchestrators emit EVENTS only
        - Orchestrators receive now: datetime injected by runtime
        - Orchestrators NEVER perform I/O

    3. Handlers return outputs; runtime publishes
        - Handlers do not publish
        - Runtime is the only publisher

    4. Effects execute I/O only
        - No business logic
        - Idempotent by default
        - Retry-safe

    5. Deterministic ordering
        - Per-entity ordering uses partition offsets or monotonic sequence
        - Wall-clock timestamps are audit-only

    6. No versioned directories
        - No v1_0_0, v2, etc.
        - Versioning is logical, not structural

    7. Handler vs Node terminology
        - Handlers are execution units within a node
        - Nodes host handlers; handlers do not own lifecycle or messaging

---

## A. Foundation (P0)

### A1. Canonical Layering and Terminology
    Priority: P0
    Description:
        Establish authoritative definitions for:
            - Event
            - Command
            - Intent
            - Reducer
            - Orchestrator
            - Effect
            - Handler
            - Projector
            - Runtime
    Acceptance:
        - Single document referenced by all workflow docs
        - No conflicting definitions remain

### A2. Canonical Execution Shapes
    Priority: P0
    Description:
        Define the only allowed execution shapes and forbid everything else.

        Allowed:
            - Event -> Orchestrator
            - Command -> Orchestrator (gated flows only)
            - Event -> Reducer
            - Reducer -> Projections + Intents
            - Intent -> Effect
            - Handler -> Output -> Runtime (publish happens here)

        Forbidden examples:
            - Command -> Reducer
            - Reducer -> I/O
            - Orchestrator -> I/O
            - Effect -> workflow decisions
            - Runtime -> domain decisions
    Acceptance:
        - Shapes are explicit, diagrammed, and referenced by all components
        - Implement "Execution Shape Validator" (CI gate) that fails if:
            - reducer handler returns events
            - orchestrator handler returns intents or projections
            - effect handler returns projections
            - any handler publishes directly (publishing must be runtime-only)
            - any reducer references injected time or system time
        - Execution Shape Validator fails if message category and topic category disagree:
            - events must only be read from <domain>.events topics
            - commands must only be read from <domain>.commands topics
            - intents must only be read from <domain>.intents topics
        - Add routing coverage check: all message types must be registered in B1a, startup fails fast if unmapped
        - Provide at least one "known bad" test case proving validator catches violations
        - Runtime MUST reject handler output that violates declared handler type
          (orchestrator, reducer, effect), even if the code compiles

### A2a. Canonical Message Envelope
    Priority: P0
    Description:
        Define and enforce a single envelope model for commands, events, and intents.

        Required fields:
            - message_id
            - correlation_id
            - causation_id (nullable)
            - emitted_at
            - entity_id

        Rules:
            - causation_id references the immediate parent message_id
            - entity_id is the partition key and identity anchor for ordering
            - For registration domain: entity_id = node_id
              (removes partition-key ambiguity for registration workflows)
    Deliverables:
        - omnibase_core/models/common/model_envelope.py
        - validation helpers (schema-level)
    Acceptance:
        - All registration messages compile with strict models
        - Unit tests reject missing or extra fields
    Implementation Note:
        - Implementation sequencing: A2 and A2a can be built in parallel after A1
        - A2 enforcement can require envelope presence once A2a lands

### A2b. Canonical Topic Taxonomy
    Priority: P0
    Description:
        Standardize topic naming and semantics across all domains.

        Required topics:
            - onex.<domain>.commands
            - onex.<domain>.events
            - onex.<domain>.intents
            - onex.<domain>.snapshots (optional)

        Rules:
            - Partition key MUST be entity_id
            - events topics are immutable logs (delete-only, no compaction)
            - snapshots topics may be compacted
    Deliverables:
        - docs/standards/onex_topic_taxonomy.md
        - initial topic manifest for registration
    Acceptance:
        - Registration workflow fully conforms
        - Runtime routing uses topic taxonomy, not ad hoc strings
        - Default configs documented and enforced:
            - events topics: cleanup.policy=delete only
            - snapshots topics: cleanup.policy=compact,delete
            - commands topics: retention defaults documented
            - intents topics: retention defaults documented
        - Topic creation scripts enforce defaults
        - Registration domain topics created with defaults verified in tests
        - Partition key must be envelope.entity_id, validated by routing

### A3. Canonical Registration Trigger Decision
    Priority: P0
    Decision:
        Canonical trigger:
            - NodeIntrospected (EVENT)
        Optional trigger (non-canonical, gated):
            - RegisterNodeRequested (COMMAND)

        Rationale:
            - Canonical path stays aligned with event-driven registration
            - Command path is reserved for administrative or exceptional flows
    Acceptance:
        - Canonical path documented as default
        - Optional path clearly labeled and isolated

---

## B. Runtime (P0)

### B1. Runtime Dispatch Engine
    Priority: P0
    Description:
        Route messages to handlers based on:
            - topic category (commands/events/intents)
            - message type
    Acceptance:
        - Deterministic routing
        - Runtime performs publishing of handler outputs only
        - Runtime does not infer workflow meaning

### B1a. Message Type Registry
    Priority: P0
    Description:
        Central registry mapping:
            - message type -> handler implementation(s)
            - topic category constraints (what message types can appear where)

        Domain ownership enforcement:
            - registration.* messages cannot be handled by non-registration handlers
            - cross-domain consumption requires explicit opt-in
            - Domain ownership is derived from:
                - topic name (onex.<domain>.*) AND
                - message type registry entry domain
            - Runtime rejects mismatches between topic domain and message type domain
    Acceptance:
        - Startup-time validation that fails fast before consumers start processing messages
        - Missing handler mappings fail fast
        - Runtime rejects mismatches between topic domain and message type domain

### B2. Handler Output Model
    Priority: P0
    Description:
        Standardize handler outputs.

        Output fields:
            - events[]
            - intents[]
            - projections[]
            - metrics
            - logs

        Rules:
            - Orchestrator handlers output events only
            - Reducer handlers output projections + intents only
            - Effect handlers output optional result events only
            - Runtime publishes outputs in deterministic order:
                1) projections
                2) intents
                3) events
            - Within each category, preserve handler-returned order
    Acceptance:
        - Runtime publishes all outputs
        - Handlers do not publish directly

### B3. Idempotency Guard
    Priority: P0
    Description:
        Enforce idempotent processing at the runtime boundary.

        Implementation:
            - Primary: Postgres (durable)
            - Optional: Valkey cache layer (P2, if throughput demands)
    Acceptance:
        - Duplicate message_id is safely ignored
        - Verified under at-least-once delivery
        - Replay-safe behavior demonstrated
        - Idempotency key strategy:
            - key: (domain, message_id) OR message_id if globally unique
            - store enforces uniqueness at key level

### B4. Handler Context (Time Injection)
    Priority: P0
    Description:
        Add now: datetime to handler context.

        Rules:
            - Orchestrators MAY use now for deadlines and timeouts
            - Effects MAY use now for retries/metrics
            - Reducers MUST ignore now and never depend on it
    Acceptance:
        - No handler reads system clock
        - Tests enforce no direct time reads
        - Define distinct context models:
            - ModelOrchestratorContext(now: datetime, ...)
            - ModelEffectContext(now: datetime, ...)
            - ModelReducerContext(...) # does not include now
        - Runtime dispatch enforces context type: reducers never receive now, orchestrators/effects do
        - Add tests proving reducers cannot access now via typing and runtime dispatch

### B5. Correlation and Causation Propagation Enforcement
    Priority: P0
    Description:
        Enforce workflow traceability by validating correlation and causation chains.

        Rules:
            - All messages within a workflow share correlation_id
            - Every produced message has causation_id = parent.message_id
            - causation chains are local (no skipping ancestors)
    Acceptance:
        - Validation rejects broken propagation
        - Integration test verifies full chain for registration

### B6. Runtime Scheduler
    Priority: P0
    Description:
        Emit RuntimeTick events on configurable cadence for timeout evaluation.

        Responsibilities:
            - Emit RuntimeTick(now) to internal topic or schedule hook
            - Own time injection for the system
            - Single source of truth for "now" across orchestrators

        Naming note:
            - Runtime emits RuntimeTick (infrastructure concern)
            - Orchestrators derive timeout decisions (domain concern)
    Dependencies: B4
    Acceptance:
        - RuntimeTick emitted on configurable interval
        - Orchestrators consume RuntimeTick for timeout evaluation
        - now field in RuntimeTick matches handler context injection

---

## C. Orchestrator (P0)

### C0. Projection Reader Interface
    Priority: P0
    Description:
        Orchestrators read current state using projections only.

        Deliverables:
            - ProtocolProjectionReader
            - Registration-specific query methods
    Acceptance:
        - No topic scanning for state
        - All orchestration decisions are projection-backed

### C1. Registration Orchestrator (Event-Driven)
    Priority: P0
    Dependencies: B6
    Description:
        Orchestrate registration using EVENTS (canonical) and COMMANDS (gated), emitting EVENTS only.

        Consumes:
            - NodeIntrospected (EVENT)  [canonical trigger]
            - NodeRegistrationAcked (COMMAND)
            - RuntimeTick (internal runtime schedule event or trigger)

        Emits (decision events):
            - NodeRegistrationInitiated
            - NodeRegistrationAccepted
            - NodeRegistrationRejected
            - NodeRegistrationAckTimedOut
            - NodeRegistrationAckReceived
            - NodeBecameActive
            - NodeLivenessExpired

        Note on naming:
            - NodeRegistrationInitiated is emitted from NodeIntrospected to represent
              the start of a registration attempt without implying command-driven semantics.
    Acceptance:
        - Emits events only
        - Performs no I/O
        - Uses projection reader for state decisions
        - Uses injected now for deadlines

### C2. Durable Timeout Handling
    Priority: P0
    Description:
        Ack and liveness timeouts must survive restarts.

        Requirements:
            - Deadlines stored in projections for restart safety
            - Orchestrator periodically queries for overdue entities
            - Orchestrator emits timeout decision events
    Acceptance:
        - Restart-safe behavior verified via integration test
        - No in-memory-only deadlines
        - Runtime scheduler emits RuntimeTick on configurable cadence
        - Orchestrator uses injected now from RuntimeTick for timeout evaluation
        - Canonical approach: emitted_at markers for timeout dedupe
        - Projections store per-timeout-type emission markers:
            - ack_timeout_emitted_at (nullable)
            - liveness_timeout_emitted_at (nullable)
        - Orchestrator emits timeout events only if marker is absent
        - Integration test: restart orchestrator after deadline passed, verify exactly one timeout event emitted

### C3. Optional Command-Based Registration (Gated)
    Priority: P2
    Description:
        Support RegisterNodeRequested for non-canonical flows (admin, manual, external nodes).

        Rules:
            - Must be explicitly enabled
            - Must not replace canonical NodeIntrospected flow
    Acceptance:
        - Clearly documented as non-default

---

## D. Reducer (P0)

### D1. Registration Reducer (Event-Only, Emits Intents and Projections)
    Priority: P0
    Description:
        Pure FSM reducer that folds decision events and deterministically emits:
            - Updated state
            - Registration projections
            - I/O intents for effects

        Folds:
            - NodeRegistrationInitiated
            - NodeRegistrationAccepted
            - NodeRegistrationRejected
            - NodeRegistrationAckTimedOut
            - NodeRegistrationAckReceived
            - NodeBecameActive
            - NodeLivenessExpired

        Emits:
            - ConsulRegisterIntent
            - PostgresUpsertRegistrationIntent
            - RegistrationProjection
    Acceptance:
        - Event-only folding enforced (commands rejected or ignored)
        - Deterministic intent emission
        - No time reads (uses event.emitted_at)
        - Reducer output ordering is deterministic:
            - projections are emitted before intents
            - intents are ordered deterministically within a single event fold

### D2. FSM Contract
    Priority: P0
    Description:
        Formal definition of registration state machine.
    Acceptance:
        - Reducer transitions match contract exactly

### D3. Reducer Test Suite
    Priority: P0
    Acceptance:
        - Same input event sequence produces same outputs
        - Command folding is forbidden and tested

---

## E. Effects (P0/P1)

### E1. Registry Effect (I/O Only)
    Priority: P0
    Description:
        Consume intents and execute external I/O.

        Consumes:
            - ConsulRegisterIntent
            - PostgresUpsertRegistrationIntent

        Rules:
            - No workflow decisions
            - Idempotent
            - Retry-safe
    Acceptance:
        - Duplicate intents cause no harmful side effects
        - Circuit breaker behavior verified
        - All intents include: intent_id (alias message_id), entity_id, registration_id where relevant
        - Effect idempotency key strategy documented:
            - primary: intent_id
            - secondary (natural key): (entity_id, intent_type, registration_id)
        - Tests prove executing same intent twice yields no additional side effects
        - Natural key conflict handling:
            - If intent_id differs but natural key matches, effect treats as duplicate
            - Exception: payload differs in a way requiring explicit conflict handling

### E2. Compensation and Retry Policy
    Priority: P1
    Description:
        Define retry, backoff, and compensation strategy for partial failures.

        Note:
            - Promote to P0 only if correctness requires compensation before proceeding

        Non-goal:
            - Compensation does NOT imply rollback of emitted domain events
            - Compensation applies only to external side effects (I/O operations)
    Acceptance:
        - Written policy and tests for failure cases exist

### E3. Dead Letter Queue Handling
    Priority: P1
    Description:
        Define DLQ behavior for permanently failing intents.
    Acceptance:
        - DLQ topic configured
        - Alerting integrated
        - No silent drops

---

## F. Projection and Storage (P0)

### F0. Projector Execution Model
    Priority: P0
    Dependencies: B2, A2a
    Note: Moved earlier because orchestrator correctness depends on projection semantics
    Description:
        Define who persists projections and how ordering is enforced.

        Decision:
            - Reducer returns projections
            - Runtime invokes projector to persist projections
            - Projector enforces ordering and idempotency

        Ordering rules:
            - Per-entity monotonic application based on (partition, offset) or sequence
            - Reject stale updates

        Idempotency layers (both required):
            - Runtime idempotency (B3): prevents duplicate handler execution
            - Projector idempotency (F0): prevents stale/out-of-order projection writes
    Acceptance:
        - Offset-aware idempotent writes implemented
        - Restart-safe projection rebuild process defined

### F1. Registration Projection Schema
    Priority: P0
    Acceptance:
        - Stores:
            - current state
            - deadlines (ack_deadline, liveness_deadline)
            - last_applied_event_id (message_id)
            - last_applied_offset (canonical)
            - last_applied_sequence (optional, only if using non-Kafka transports)
        - Supports orchestration queries efficiently
        - Makes idempotency and replay correctness auditable

### F2. Snapshot Publishing
    Priority: P1
    Description:
        Optional compacted snapshots for read optimization.
    Acceptance:
        - Snapshots do not replace immutable event logs

---

## G. Testing (P0/P2)

### G1. Reducer Tests
    Priority: P0

### G2. Orchestrator Tests
    Priority: P0
    Acceptance:
        - Emits events only
        - No I/O
        - Uses injected now, not system clock

### G3. End-to-End Integration Tests
    Priority: P0
    Dependencies: G2, E1, F1
    Note: E2E tests require effects and projections to be implemented
    Acceptance:
        - Full handshake happy path
        - Restart mid-flight with pending deadlines
        - Duplicate delivery behavior
        - System behavior verified when:
            - orchestrator crashes after emitting acceptance
            - reducer and effect restart independently
            - no duplicate activation occurs

### G4. Effect Idempotency and Retry Tests
    Priority: P0
    Acceptance:
        - Duplicate intent safe
        - Circuit breaker verified
        - Backoff policy validated

### G5. Chaos and Replay Tests
    Priority: P2

---

## H. Migration (P1)

### H1. Legacy Component Refactor Plan
    Priority: P1
    Description:
        Identify legacy code paths and refactor into:
            - orchestrator handlers
            - reducer handlers
            - effect handlers
            - projector persistence
    Acceptance:
        - No big-bang rewrite required
        - Explicit cutover points defined
        - No dual-write paths allowed:
            - legacy and canonical systems must not both emit authoritative events
            - cutover is event-boundary based, not code-path based

### H2. Migration Checklist
    Priority: P1
    Acceptance:
        - Explicit ticket ordering
        - Partial migration hazards documented

---

## Dependency Order (Authoritative)

    A1 -> A2 -> A2a -> A2b -> A3
        ↓
    B1 -> B1a -> B2 -> B3 -> B4 -> B5 -> B6
        ↓
    C0 -> C1 (depends on B6) -> C2
        ↓
    D1 -> D2 -> D3
        ↓
    E1 -> E2 -> E3
        ↓
    F0 (depends on B2, A2a) -> F1 -> F2
        ↓
    G1 -> G2 -> G3 (depends on G2, E1, F1) -> G4 -> G5
        ↓
    H1 -> H2

---

    Tickets at the same dependency level may be executed in parallel if they
    do not introduce contradictory contracts or shared-interface churn.

    This document is the canonical baseline for all future workflows.

---

## Appendix: Why This Model

    Why reducers are pure:
        - Deterministic replay guarantees identical state reconstruction
        - Testable without mocking external systems
        - Enables event sourcing and temporal queries

    Why orchestrators own time:
        - Timeouts become explicit domain decisions, not infrastructure magic
        - Restart-safe: deadlines persist in projections, not memory
        - Testable: inject time, verify decisions

    Why runtime publishes:
        - Single point of delivery guarantee enforcement
        - Handlers remain pure functions returning data
        - Enables batching, ordering, and transactional semantics

    Why events are immutable:
        - Audit trail is complete and trustworthy
        - Replay is deterministic
        - No "undo" semantics prevent corruption

    This model optimizes for correctness under failure,
    not convenience under success.

    Convenience features must be layered without weakening invariants.
