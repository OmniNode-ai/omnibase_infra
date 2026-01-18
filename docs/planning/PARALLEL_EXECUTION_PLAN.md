> **Navigation**: [Home](../index.md) > [Planning](README.md) > Parallel Execution Plan

# Parallelizable Execution Plan for ONEX Runtime & Registration Tickets

**Resources**: 5 omnibase_core repos + 4 omnibase_infra repos = 9 parallel workers
**Strategy**: Wave-based execution maximizing parallelization within dependency constraints

---

## Complete Ticket Mapping (Code → Linear ID)

### Section A: Foundation (P0)
| Code | Linear | Title | Repo |
|------|--------|-------|------|
| A1 | [OMN-931](https://linear.app/omninode/issue/OMN-931) | Canonical Layering and Terminology | core |
| A2 | [OMN-933](https://linear.app/omninode/issue/OMN-933) | Canonical Execution Shapes | core |
| A2a | [OMN-936](https://linear.app/omninode/issue/OMN-936) | Canonical Message Envelope | core |
| A2b | [OMN-939](https://linear.app/omninode/issue/OMN-939) | Canonical Topic Taxonomy | core |
| A3 | [OMN-943](https://linear.app/omninode/issue/OMN-943) | Registration Trigger Decision | core |
| A2-val | [OMN-958](https://linear.app/omninode/issue/OMN-958) | Execution Shape Validator CI Gate | infra |

### Section B: Runtime (P0)
| Code | Linear | Title | Repo |
|------|--------|-------|------|
| B1 | [OMN-934](https://linear.app/omninode/issue/OMN-934) | Runtime Dispatch Engine | core |
| B1a | [OMN-937](https://linear.app/omninode/issue/OMN-937) | Message Type Registry | core |
| B2 | [OMN-941](https://linear.app/omninode/issue/OMN-941) | Handler Output Model | core |
| B3 | [OMN-945](https://linear.app/omninode/issue/OMN-945) | Idempotency Guard | core |
| B4 | [OMN-948](https://linear.app/omninode/issue/OMN-948) | Handler Context (Time Injection) | core |
| B5 | [OMN-951](https://linear.app/omninode/issue/OMN-951) | Correlation/Causation Propagation | core |
| B6 | [OMN-953](https://linear.app/omninode/issue/OMN-953) | Runtime Scheduler | core |

### Section C: Orchestrator (P0)
| Code | Linear | Title | Repo |
|------|--------|-------|------|
| C0 | [OMN-930](https://linear.app/omninode/issue/OMN-930) | Projection Reader Interface | infra |
| C1 | [OMN-888](https://linear.app/omninode/issue/OMN-888) | Registration Orchestrator | infra |
| C2 | [OMN-932](https://linear.app/omninode/issue/OMN-932) | Durable Timeout Handling | infra |
| C3 | [OMN-935](https://linear.app/omninode/issue/OMN-935) | Command-Based Registration (P2) | infra |

### Section D: Reducer (P0)
| Code | Linear | Title | Repo |
|------|--------|-------|------|
| D1 | [OMN-889](https://linear.app/omninode/issue/OMN-889) | Registration Reducer | infra |
| D2 | [OMN-938](https://linear.app/omninode/issue/OMN-938) | FSM Contract | core |
| D3 | [OMN-942](https://linear.app/omninode/issue/OMN-942) | Reducer Test Suite | infra |

### Section E: Effects (P0/P1)
| Code | Linear | Title | Repo |
|------|--------|-------|------|
| E1 | [OMN-890](https://linear.app/omninode/issue/OMN-890) | Registry Effect (I/O Only) | infra |
| E2 | [OMN-946](https://linear.app/omninode/issue/OMN-946) | Compensation/Retry Policy (P1) | infra |
| E3 | [OMN-949](https://linear.app/omninode/issue/OMN-949) | Dead Letter Queue (P1) | infra |

### Section F: Projection (P0/P1)
| Code | Linear | Title | Repo |
|------|--------|-------|------|
| F0 | [OMN-940](https://linear.app/omninode/issue/OMN-940) | Projector Execution Model | core/infra |
| F1 | [OMN-944](https://linear.app/omninode/issue/OMN-944) | Registration Projection Schema | infra |
| F2 | [OMN-947](https://linear.app/omninode/issue/OMN-947) | Snapshot Publishing (P1) | infra |

### Section G: Testing (P0/P2)
| Code | Linear | Title | Repo |
|------|--------|-------|------|
| G1 | [OMN-950](https://linear.app/omninode/issue/OMN-950) | Reducer Tests | infra |
| G2 | [OMN-952](https://linear.app/omninode/issue/OMN-952) | Orchestrator Tests | infra |
| G3 | [OMN-915](https://linear.app/omninode/issue/OMN-915) | E2E Integration Tests (Mocked) | infra |
| G3-real | [OMN-892](https://linear.app/omninode/issue/OMN-892) | E2E Integration Tests (Real Infra) | infra |
| G4 | [OMN-954](https://linear.app/omninode/issue/OMN-954) | Effect Idempotency Tests | infra |
| G5 | [OMN-955](https://linear.app/omninode/issue/OMN-955) | Chaos and Replay Tests (P2) | infra |

### Section H: Migration (P1)
| Code | Linear | Title | Repo |
|------|--------|-------|------|
| H1 | [OMN-956](https://linear.app/omninode/issue/OMN-956) | Legacy Refactor Plan | infra |
| H2 | [OMN-957](https://linear.app/omninode/issue/OMN-957) | Migration Checklist | infra |

---

## Wave Execution Plan

### Wave 1: Foundation (4 parallel tasks)
**Duration**: Start immediately
**Repos**: 4 omnibase_core

| Repo | Ticket | Linear | Description |
|------|--------|--------|-------------|
| core-1 | **A1** | [OMN-931](https://linear.app/omninode/issue/OMN-931) | Canonical Terminology |
| core-2 | **A2** | [OMN-933](https://linear.app/omninode/issue/OMN-933) | Execution Shapes |
| core-3 | **A2a** | [OMN-936](https://linear.app/omninode/issue/OMN-936) | Message Envelope |
| core-4 | **A2b** | [OMN-939](https://linear.app/omninode/issue/OMN-939) | Topic Taxonomy |

---

### Wave 2: Runtime Core + Foundation Complete (9 parallel tasks)
**Duration**: After A1 complete
**Repos**: 5 omnibase_core + 4 omnibase_infra

| Repo | Ticket | Linear | Description |
|------|--------|--------|-------------|
| core-1 | **B1** | [OMN-934](https://linear.app/omninode/issue/OMN-934) | Runtime Dispatch Engine |
| core-2 | **B1a** | [OMN-937](https://linear.app/omninode/issue/OMN-937) | Message Type Registry |
| core-3 | **B2** | [OMN-941](https://linear.app/omninode/issue/OMN-941) | Handler Output Model |
| core-4 | **B3** | [OMN-945](https://linear.app/omninode/issue/OMN-945) | Idempotency Guard |
| core-5 | **B4** | [OMN-948](https://linear.app/omninode/issue/OMN-948) | Handler Context |
| infra-1 | **A3** | [OMN-943](https://linear.app/omninode/issue/OMN-943) | Registration Trigger |
| infra-2 | **D2** | [OMN-938](https://linear.app/omninode/issue/OMN-938) | FSM Contract |
| infra-3 | **F1** | [OMN-944](https://linear.app/omninode/issue/OMN-944) | Projection Schema |
| infra-4 | **A2-val** | [OMN-958](https://linear.app/omninode/issue/OMN-958) | Shape Validator |

---

### Wave 3: Runtime Complete + Build Start (9 parallel tasks)
**Duration**: After B4 complete
**Repos**: 5 omnibase_core + 4 omnibase_infra

| Repo | Ticket | Linear | Description |
|------|--------|--------|-------------|
| core-1 | **B5** | [OMN-951](https://linear.app/omninode/issue/OMN-951) | Correlation/Causation |
| core-2 | **B6** | [OMN-953](https://linear.app/omninode/issue/OMN-953) | Runtime Scheduler |
| core-3 | **F0** | [OMN-940](https://linear.app/omninode/issue/OMN-940) | Projector Model |
| core-4 | *(integration)* | - | Integrate B1-B4 |
| core-5 | *(integration)* | - | Integration testing |
| infra-1 | **C0** | [OMN-930](https://linear.app/omninode/issue/OMN-930) | Projection Reader |
| infra-2 | **D1** | [OMN-889](https://linear.app/omninode/issue/OMN-889) | Registration Reducer |
| infra-3 | **E1** | [OMN-890](https://linear.app/omninode/issue/OMN-890) | Registry Effect |
| infra-4 | **G1** | [OMN-950](https://linear.app/omninode/issue/OMN-950) | Reducer Tests |

---

### Wave 4: Orchestrator + Testing (8 parallel tasks)
**Duration**: After B6, C0 complete
**Repos**: 4 omnibase_core + 4 omnibase_infra

| Repo | Ticket | Linear | Description |
|------|--------|--------|-------------|
| core-1 | *(release prep)* | - | Prepare 0.5.x release |
| core-2 | *(documentation)* | - | API documentation |
| core-3 | *(integration)* | - | Full runtime tests |
| core-4 | *(standby)* | - | Support infra |
| infra-1 | **C1** | [OMN-888](https://linear.app/omninode/issue/OMN-888) | Registration Orchestrator |
| infra-2 | **C2** | [OMN-932](https://linear.app/omninode/issue/OMN-932) | Durable Timeouts |
| infra-3 | **D3** | [OMN-942](https://linear.app/omninode/issue/OMN-942) | Reducer Test Suite |
| infra-4 | **G2** | [OMN-952](https://linear.app/omninode/issue/OMN-952) | Orchestrator Tests |

---

### Wave 5: Integration + P1 Tickets (9 parallel tasks)
**Duration**: After Wave 4
**Repos**: 5 omnibase_core + 4 omnibase_infra

| Repo | Ticket | Linear | Description |
|------|--------|--------|-------------|
| core-1 | *(release)* | - | Release 0.5.x |
| core-2 | *(support)* | - | Support infra |
| core-3 | *(support)* | - | Bug fixes |
| core-4 | *(support)* | - | Performance |
| core-5 | *(support)* | - | Documentation |
| infra-1 | **G3** | [OMN-915](https://linear.app/omninode/issue/OMN-915) | E2E Tests (Mocked) |
| infra-2 | **G4** | [OMN-954](https://linear.app/omninode/issue/OMN-954) | Effect Idempotency |
| infra-3 | **E2** | [OMN-946](https://linear.app/omninode/issue/OMN-946) | Compensation Policy |
| infra-4 | **E3** | [OMN-949](https://linear.app/omninode/issue/OMN-949) | Dead Letter Queue |

---

### Wave 6: Polish + Migration (6 parallel tasks)
**Duration**: After Wave 5
**Repos**: 2 omnibase_core + 4 omnibase_infra

| Repo | Ticket | Linear | Description |
|------|--------|--------|-------------|
| core-1 | *(maintenance)* | - | Bug fixes |
| core-2 | *(maintenance)* | - | Documentation |
| infra-1 | **C3** | [OMN-935](https://linear.app/omninode/issue/OMN-935) | Command Registration (P2) |
| infra-2 | **F2** | [OMN-947](https://linear.app/omninode/issue/OMN-947) | Snapshot Publishing |
| infra-3 | **H1** | [OMN-956](https://linear.app/omninode/issue/OMN-956) | Legacy Refactor Plan |
| infra-4 | **H2** | [OMN-957](https://linear.app/omninode/issue/OMN-957) | Migration Checklist |

---

### Wave 7: Chaos Testing + Final (4 parallel tasks)
**Duration**: After Wave 6
**Repos**: 4 omnibase_infra

| Repo | Ticket | Linear | Description |
|------|--------|--------|-------------|
| infra-1 | **G5** | [OMN-955](https://linear.app/omninode/issue/OMN-955) | Chaos Tests (P2) |
| infra-2 | **G3-real** | [OMN-892](https://linear.app/omninode/issue/OMN-892) | E2E (Real Infra) |
| infra-3 | *(polish)* | - | Performance opt |
| infra-4 | *(polish)* | - | Final validation |

---

## Gantt-Style Timeline

```
Wave 1: ████████ Foundation [OMN-931, 933, 936, 939]
Wave 2: ████████████████ Runtime + Foundation [OMN-934, 937, 941, 945, 948, 943, 938, 944, 958]
Wave 3: ████████████████ Runtime Complete [OMN-951, 953, 940, 930, 889, 890, 950]
Wave 4: ████████████████ Orchestrator [OMN-888, 932, 942, 952]
Wave 5: ████████████████ Integration [OMN-915, 954, 946, 949]
Wave 6: ████████ Polish [OMN-935, 947, 956, 957]
Wave 7: ████ Chaos [OMN-955, 892]
```

---

## Critical Path (Linear IDs)

```
OMN-931 → OMN-933 → OMN-934 → OMN-937 → OMN-941 → OMN-945 → OMN-948 → OMN-951 → OMN-953 → OMN-888 → OMN-932 → OMN-915 → OMN-954 → OMN-955
   A1   →   A2    →   B1    →   B1a   →   B2    →   B3    →   B4    →   B5    →   B6    →   C1    →   C2    →   G3    →   G4    →   G5
```

---

## Quick Reference: Max Parallelization

| Wave | Tasks | Core | Infra | Key Linear Tickets |
|------|-------|------|-------|-------------------|
| 1 | 4 | 4 | 0 | OMN-931, 933, 936, 939 |
| 2 | 9 | 5 | 4 | OMN-934, 937, 941, 945, 948, 943, 938, 944, 958 |
| 3 | 9 | 5 | 4 | OMN-951, 953, 940, 930, 889, 890, 950 |
| 4 | 8 | 4 | 4 | OMN-888, 932, 942, 952 |
| 5 | 9 | 5 | 4 | OMN-915, 954, 946, 949 |
| 6 | 6 | 2 | 4 | OMN-935, 947, 956, 957 |
| 7 | 4 | 0 | 4 | OMN-955, 892 |

---

## Blockers Before Starting

1. ✅ omnibase_core 0.5.3 release ([PR #216](https://github.com/OmniNode-ai/omnibase_core/pull/216))
2. ⏳ [OMN-959](https://linear.app/omninode/issue/OMN-959) - Update omnibase_infra pyproject.toml to `omnibase-core = "^0.5.0"`
3. 🚀 Start Wave 1 with [OMN-931](https://linear.app/omninode/issue/OMN-931) (Terminology)
