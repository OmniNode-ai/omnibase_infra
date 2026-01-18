> **Navigation**: [Home](../index.md) > As-Is Architecture

## As-Is Architecture Map (Core + SPI + Infra3)

This folder documents the **current (“as-is”) architecture shape** across:

- `omnibase_core` (active, bugfix/tech-debt only)
- `omnibase_spi` (stable, contracts/protocols)
- `omnibase_infra3` (stable infra implementations)

It is intentionally **descriptive, not prescriptive**: it captures how things work today,
what the primary interfaces look like, and where there are parallel concepts that can be
confused (e.g., different “envelope” models and different “runtime” dispatch loops).

### Documents

- `01_LAYERING_AND_TERMINOLOGY.md`
  - Vocabulary map, repo layering, “same word, different meaning” pitfalls.
- `02_NODE_EXECUTION_SHAPES.md`
  - Node kinds (effect/compute/reducer/orchestrator), base classes, `process()` shapes,
    contract-driven patterns, and service wrappers.
- `03_MESSAGING_AND_ENVELOPES.md`
  - Event/message/envelope models and their fields across Core/SPI/Infra.
- `04_EVENT_BUS_SHAPES.md`
  - Core event bus protocol, Infra3 implementations, SPI extensions (envelope and workflow buses).
- `05_RUNTIME_DISPATCH_SHAPES.md`
  - Core `EnvelopeRouter` vs Infra3 `RuntimeHostProcess`: what they route, how, and why.
- `06_TWO_WAY_REGISTRATION_AS_IS_TRACE.md`
  - End-to-end trace of the current 2-way registration workflow as implemented today.
- `07_INTERFACE_CROSSWALK.md`
  - A cross-repo table mapping concepts to the concrete types/files in Core/SPI/Infra.
- `08_DECISION_POINTS_AND_UNANSWERED_QUESTIONS.md`
  - Central list of decisions/unknowns to resolve before any PLAN-mode implementation plan.
