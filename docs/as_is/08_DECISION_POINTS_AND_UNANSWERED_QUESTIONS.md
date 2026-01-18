> **Navigation**: [Home](../index.md) > [As-Is](INDEX.md) > Decision Points

## Decision points and unanswered questions (As-Is)

This document is the **parking lot** for all open questions that must be resolved before we
switch to PLAN mode and commit to an “ideal implementation.”

The goal is to avoid accidentally baking assumptions into a plan.

### A) Runtime planes: which “runtime” is responsible for what?

- **A1**: Do we keep **two runtimes** long-term?
  - Core runtime router (`ModelOnexEnvelope` + `EnvelopeRouter`)
  - Infra runtime host (`operation`-string envelopes + `RuntimeHostProcess`)

- **A2**: If we keep both, what is the *formal boundary*?
  - What messages enter Core runtime vs Infra runtime host?
  - Is one “workflow/domain plane” and the other “I/O execution plane”?

- **A3**: If we converge, which surface becomes canonical for dispatch?
  - Canonical message shape: `ModelOnexEnvelope` vs infra operation envelope vs SPI envelope/consume APIs
  - Canonical routing key: `EnumHandlerType` vs operation prefix strings

### B) Envelope/message canon: what is “the” canonical wrapper per plane?

Core currently has multiple envelope/message abstractions. We need explicit guidance on:

- **B1**: Which wrapper is canonical for *inter-node / inter-service* messages?
  - `ModelOnexEnvelope`
  - `ModelEventEnvelope[T]`
  - Event bus topic/key/value headers messages

- **B2**: Do we standardize on one envelope model, or define “envelope families” by use case?

- **B3**: If `ModelOnexEnvelope` is canonical, what is the mapping to/from:
  - event bus messages (`topic/key/value/headers`)
  - workflow event sourcing messages (sequence/idempotency)
  - infra operation envelopes

### C) Topic strategy and naming: what is canonical?

- **C1**: Do we adopt a canonical topic convention (e.g., `onex.<domain>.(commands|events|intents)`),
  or keep multiple naming schemes per plane?
- **C2**: Who owns topic creation/config (Core vs Infra vs deployment tooling)?
- **C3**: Partitioning key standard:
  - What is the canonical partition key field (entity_id? instance_id? node_id?) for each domain?

### D) Ordering, idempotency, and offsets: where is the source of truth?

- **D1**: Which components are responsible for idempotency?
  - handler boundary (infra runtime host)
  - projector/state plane (if/when introduced)
  - effect execution plane

- **D2**: Do we adopt SPI’s workflow event sourcing concepts (sequence_number, idempotency_key) as canonical,
  or keep the simpler event bus offset-based ordering?

- **D3**: What is the canonical “event log” model?
  - Kafka topic(s) as immutable log?
  - snapshot topic(s)?
  - projector offsets persisted where?

### E) Reducer and orchestrator semantics: purity and statefulness

- **E1**: Reducer purity definition:
  - Are reducers “pure functions” in the strong sense, or “no external I/O” but may read local contracts/files?
  - (Infra3 `NodeDualRegistrationReducer` currently loads FSM contracts from disk.)

- **E2**: Orchestrator state model:
  - Core orchestrator is MVP-stateful. Do we accept that in the ideal design or require externalized state?

- **E3**: Do orchestrators read from projections/materialized state? If yes, what is the projection system?

### F) Intent system: one intent type system or multiple?

- **F1**: Core “closed set” intents vs extension intents:
  - When do we use each?
  - Do infra/workflow intents belong in Core, SPI, or Infra?

- **F2**: Do effects emit “result events”? If so, what’s the canonical schema and routing?

### G) Ownership boundaries: Core vs SPI vs Infra changes

- **G1**: Which repo “owns” the workflow substrate?
  - Core currently has NodeEffect/NodeReducer/NodeOrchestrator + EnvelopeRouter
  - SPI has workflow orchestration protocols
  - Infra has runtime host and bus implementations

- **G2**: What is allowed to change in Core/SPI under the current policy?
  - Bugfix/tech-debt only (Core)
  - Stable (SPI)
  - If ideal architecture requires Core/SPI changes, what is the decision gate?

### H) Two-way registration specifics (to decide after substrate questions)

- **H1**: What is the canonical “registration” domain state machine and where does it live?
  - reducer-owned FSM vs orchestrator-owned decisions vs mixed

- **H2**: What are the canonical facts/events vs decisions?
  - (Requires agreement on message/envelope + topic strategy first.)
