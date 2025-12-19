# Parallelizable Execution Plan for ONEX Runtime & Registration Tickets

**Resources**: omnibase_core + omnibase_infra + omnibase_spi repos for parallel workers
**Strategy**: Wave-based execution maximizing parallelization within dependency constraints

---

## Complete Ticket Mapping (Code → Linear ID)

### Section A: Foundation (P0)
| Code | Linear | Title | Repo | Depends On | Status |
|------|--------|-------|------|------------|--------|
| A1 | [OMN-931](https://linear.app/omninode/issue/OMN-931) | Canonical Layering and Terminology | core | *(none - root)* | ✅ Done |
| A2 | [OMN-933](https://linear.app/omninode/issue/OMN-933) | Canonical Execution Shapes | core | A1 | ✅ Done |
| A2a | [OMN-936](https://linear.app/omninode/issue/OMN-936) | Canonical Message Envelope | core | A2 | ✅ Done |
| A2b | [OMN-939](https://linear.app/omninode/issue/OMN-939) | Canonical Topic Taxonomy | infra | A2a | ✅ Done |
| A3 | [OMN-943](https://linear.app/omninode/issue/OMN-943) | Registration Trigger Decision | core | A2b | ⏳ |
| A2-val | [OMN-958](https://linear.app/omninode/issue/OMN-958) | Execution Shape Validator CI Gate | infra | A2 | ⏳ |

### Section B: Runtime (P0)
| Code | Linear | Title | Repo | Depends On |
|------|--------|-------|------|------------|
| B1 | [OMN-934](https://linear.app/omninode/issue/OMN-934) | Runtime Dispatch Engine | infra | A3 |
| B1a | [OMN-937](https://linear.app/omninode/issue/OMN-937) | Message Type Registry | infra | B1 |
| B2 | [OMN-941](https://linear.app/omninode/issue/OMN-941) | Handler Output Model | core | B1a |
| B3 | [OMN-945](https://linear.app/omninode/issue/OMN-945) | Idempotency Guard | infra | B2 |
| B4 | [OMN-948](https://linear.app/omninode/issue/OMN-948) | Handler Context (Time Injection) | core | B3 |
| B5 | [OMN-951](https://linear.app/omninode/issue/OMN-951) | Correlation/Causation Propagation | infra | B4 |
| B6 | [OMN-953](https://linear.app/omninode/issue/OMN-953) | Runtime Scheduler | infra | B4 |

### Section C: Orchestrator (P0)
| Code | Linear | Title | Repo | Depends On |
|------|--------|-------|------|------------|
| C0 | [OMN-930](https://linear.app/omninode/issue/OMN-930) | Projection Reader Interface | spi | B5 |
| C1 | [OMN-888](https://linear.app/omninode/issue/OMN-888) | Registration Orchestrator | infra | C0, B6 |
| C2 | [OMN-932](https://linear.app/omninode/issue/OMN-932) | Durable Timeout Handling | infra | C1 |
| C3 | [OMN-935](https://linear.app/omninode/issue/OMN-935) | Command-Based Registration (P2) | infra | C2 |

### Section D: Reducer (P0)
| Code | Linear | Title | Repo | Depends On |
|------|--------|-------|------|------------|
| D2 | [OMN-938](https://linear.app/omninode/issue/OMN-938) | FSM Contract | core | OMN-912 ✅, OMN-913 ✅ |
| D1 | [OMN-889](https://linear.app/omninode/issue/OMN-889) | Registration Reducer | infra | D2 |
| D3 | [OMN-942](https://linear.app/omninode/issue/OMN-942) | Reducer Test Suite | infra | D1, D2 |

### Section E: Effects (P0/P1)
| Code | Linear | Title | Repo | Depends On |
|------|--------|-------|------|------------|
| E1 | [OMN-890](https://linear.app/omninode/issue/OMN-890) | Registry Effect (I/O Only) | infra | D1 |
| E2 | [OMN-946](https://linear.app/omninode/issue/OMN-946) | Compensation/Retry Policy (P1) | infra | E1 |
| E3 | [OMN-949](https://linear.app/omninode/issue/OMN-949) | Dead Letter Queue (P1) | infra | E1 |

### Section F: Projection (P0/P1)
| Code | Linear | Title | Repo | Depends On |
|------|--------|-------|------|------------|
| F0 | [OMN-940](https://linear.app/omninode/issue/OMN-940) | Projector Execution Model | spi | B2, A2a |
| F1 | [OMN-944](https://linear.app/omninode/issue/OMN-944) | Registration Projection Schema | infra | F0 |
| F2 | [OMN-947](https://linear.app/omninode/issue/OMN-947) | Snapshot Publishing (P1) | infra | F1 |

### Section G: Testing (P0/P2)
| Code | Linear | Title | Repo | Depends On |
|------|--------|-------|------|------------|
| G1 | [OMN-950](https://linear.app/omninode/issue/OMN-950) | Reducer Tests | infra | D3 |
| G2 | [OMN-952](https://linear.app/omninode/issue/OMN-952) | Orchestrator Tests | infra | C1 |
| G3 | [OMN-915](https://linear.app/omninode/issue/OMN-915) | E2E Integration Tests (Mocked) | infra | C1, D1, E1 |
| G3-real | [OMN-892](https://linear.app/omninode/issue/OMN-892) | E2E Integration Tests (Real Infra) | infra | G3 |
| G4 | [OMN-954](https://linear.app/omninode/issue/OMN-954) | Effect Idempotency Tests | infra | E1 |
| G5 | [OMN-955](https://linear.app/omninode/issue/OMN-955) | Chaos and Replay Tests (P2) | infra | G4 |

### Section H: Migration (P1)
| Code | Linear | Title | Repo | Depends On |
|------|--------|-------|------|------------|
| H1 | [OMN-956](https://linear.app/omninode/issue/OMN-956) | Legacy Refactor Plan | infra | G4 |
| H2 | [OMN-957](https://linear.app/omninode/issue/OMN-957) | Migration Checklist | infra | H1 |

---

## Wave Execution Plan

### Wave 1: Foundation (4 parallel tasks) ✅ COMPLETE
**Duration**: Start immediately
**Repos**: 3 omnibase_core + 1 omnibase_infra

| Repo | Ticket | Linear | Description | Status |
|------|--------|--------|-------------|--------|
| core-1 | **A1** | [OMN-931](https://linear.app/omninode/issue/OMN-931) | Canonical Terminology | ✅ Done |
| core-2 | **A2** | [OMN-933](https://linear.app/omninode/issue/OMN-933) | Execution Shapes | ✅ Done |
| core-3 | **A2a** | [OMN-936](https://linear.app/omninode/issue/OMN-936) | Message Envelope | ✅ Done |
| infra-1 | **A2b** | [OMN-939](https://linear.app/omninode/issue/OMN-939) | Topic Taxonomy | ✅ Done |

---

### Wave 2: Runtime Core + Foundation Complete (9 parallel tasks)
**Duration**: After A1 complete
**Repos**: 2 omnibase_core + 6 omnibase_infra + 1 omnibase_spi

| Repo | Ticket | Linear | Description |
|------|--------|--------|-------------|
| infra-1 | **B1** | [OMN-934](https://linear.app/omninode/issue/OMN-934) | Runtime Dispatch Engine |
| infra-2 | **B1a** | [OMN-937](https://linear.app/omninode/issue/OMN-937) | Message Type Registry |
| core-1 | **B2** | [OMN-941](https://linear.app/omninode/issue/OMN-941) | Handler Output Model |
| infra-3 | **B3** | [OMN-945](https://linear.app/omninode/issue/OMN-945) | Idempotency Guard |
| core-2 | **B4** | [OMN-948](https://linear.app/omninode/issue/OMN-948) | Handler Context |
| core-3 | **A3** | [OMN-943](https://linear.app/omninode/issue/OMN-943) | Registration Trigger |
| core-4 | **D2** | [OMN-938](https://linear.app/omninode/issue/OMN-938) | FSM Contract |
| infra-5 | **F1** | [OMN-944](https://linear.app/omninode/issue/OMN-944) | Projection Schema |
| infra-4 | **A2-val** | [OMN-958](https://linear.app/omninode/issue/OMN-958) | Shape Validator |

---

### Wave 3: Runtime Complete + Build Start (9 parallel tasks)
**Duration**: After B4 complete
**Repos**: 2 omnibase_spi + 4 omnibase_infra

| Repo | Ticket | Linear | Description |
|------|--------|--------|-------------|
| infra-1 | **B5** | [OMN-951](https://linear.app/omninode/issue/OMN-951) | Correlation/Causation |
| infra-2 | **B6** | [OMN-953](https://linear.app/omninode/issue/OMN-953) | Runtime Scheduler |
| spi-1 | **F0** | [OMN-940](https://linear.app/omninode/issue/OMN-940) | Projector Model |
| spi-2 | **C0** | [OMN-930](https://linear.app/omninode/issue/OMN-930) | Projection Reader |
| infra-3 | **D1** | [OMN-889](https://linear.app/omninode/issue/OMN-889) | Registration Reducer |
| infra-4 | **E1** | [OMN-890](https://linear.app/omninode/issue/OMN-890) | Registry Effect |
| infra-5 | **G1** | [OMN-950](https://linear.app/omninode/issue/OMN-950) | Reducer Tests |
| *(integration)* | - | - | Integrate B1-B4 |
| *(integration)* | - | - | Integration testing |

---

### Wave 4: Orchestrator + Testing (4 tickets + support tasks)
**Duration**: After B6, C0 complete
**Repos**: 4 omnibase_infra (+ core support)

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

### Wave 5: Integration + P1 Tickets (4 tickets + support tasks)
**Duration**: After Wave 4
**Repos**: 4 omnibase_infra (+ core support)

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

### Wave 6: Polish + Migration (4 tickets + support tasks)
**Duration**: After Wave 5
**Repos**: 4 omnibase_infra (+ core maintenance)

| Repo | Ticket | Linear | Description |
|------|--------|--------|-------------|
| core-1 | *(maintenance)* | - | Bug fixes |
| core-2 | *(maintenance)* | - | Documentation |
| infra-1 | **C3** | [OMN-935](https://linear.app/omninode/issue/OMN-935) | Command Registration (P2) |
| infra-2 | **F2** | [OMN-947](https://linear.app/omninode/issue/OMN-947) | Snapshot Publishing |
| infra-3 | **H1** | [OMN-956](https://linear.app/omninode/issue/OMN-956) | Legacy Refactor Plan |
| infra-4 | **H2** | [OMN-957](https://linear.app/omninode/issue/OMN-957) | Migration Checklist |

---

### Wave 7: Chaos Testing + Final (2 tickets + polish tasks)
**Duration**: After Wave 6
**Repos**: 2 omnibase_infra (+ polish)

| Repo | Ticket | Linear | Description |
|------|--------|--------|-------------|
| infra-1 | **G5** | [OMN-955](https://linear.app/omninode/issue/OMN-955) | Chaos Tests (P2) |
| infra-2 | **G3-real** | [OMN-892](https://linear.app/omninode/issue/OMN-892) | E2E (Real Infra) |
| infra-3 | *(polish)* | - | Performance opt |
| infra-4 | *(polish)* | - | Final validation |

---

## Gantt-Style Timeline

```
Wave 1: ████████ Foundation [OMN-931, 933, 936, 939] ✅ COMPLETE
Wave 2: ████████████████ Runtime + Foundation [OMN-934, 937, 941, 945, 948, 943, 938, 944, 958]
Wave 3: ████████████████ Runtime Complete [OMN-951, 953, 940, 930, 889, 890, 950]
Wave 4: ████████████████ Orchestrator [OMN-888, 932, 942, 952]
Wave 5: ████████████████ Integration [OMN-915, 954, 946, 949]
Wave 6: ████████ Polish [OMN-935, 947, 956, 957]
Wave 7: ████ Chaos [OMN-955, 892]
```

---

## Critical Path (Linear IDs)

### Main Runtime Path
```
OMN-931 → OMN-933 → OMN-936 → OMN-939 → OMN-943 → OMN-934 → OMN-937 → OMN-941 → OMN-945 → OMN-948
   A1   →   A2    →   A2a   →   A2b   →   A3    →   B1    →   B1a   →   B2    →   B3    →   B4
                                                                                            ↓
                                                                              ┌─────────────┴─────────────┐
                                                                              ↓                           ↓
                                                                          OMN-951                     OMN-953
                                                                            B5                          B6
                                                                              ↓                           ↓
                                                                          OMN-930 ────────────────────────┤
                                                                            C0                            ↓
                                                                              └──────────────────────> OMN-888 → OMN-932
                                                                                                         C1       C2
```

### Reducer/Effect Path (parallel to B-series after A2)
```
OMN-912 ✅ ─┐
           ├──> OMN-938 → OMN-889 → OMN-890 → OMN-954 → OMN-955
OMN-913 ✅ ─┘     D2        D1        E1        G4        G5
```

### Projection Path (parallel after B2)
```
OMN-941 (B2) + OMN-936 (A2a) → OMN-940 → OMN-944
                                 F0        F1
```

---

## Quick Reference: Max Parallelization

| Wave | Tasks | Core | Infra | SPI | Key Linear Tickets |
|------|-------|------|-------|-----|-------------------|
| 1 | 4 | 3 | 1 | 0 | OMN-931, 933, 936, 939 |
| 2 | 9 | 4 | 5 | 0 | OMN-934, 937, 941, 945, 948, 943, 938, 944, 958 |
| 3 | 7 | 0 | 5 | 2 | OMN-951, 953, 940, 930, 889, 890, 950 |
| 4 | 4 | 0 | 4 | 0 | OMN-888, 932, 942, 952 |
| 5 | 4 | 0 | 4 | 0 | OMN-915, 954, 946, 949 |
| 6 | 4 | 0 | 4 | 0 | OMN-935, 947, 956, 957 |
| 7 | 2 | 0 | 2 | 0 | OMN-955, 892 |

---

## Blockers Before Starting

1. ✅ omnibase_core 0.5.3 release ([PR #216](https://github.com/OmniNode-ai/omnibase_core/pull/216))
2. ⏳ [OMN-959](https://linear.app/omninode/issue/OMN-959) - Update omnibase_infra pyproject.toml to `omnibase-core = "^0.5.0"`
3. ✅ Wave 1 COMPLETE - All 4 foundation tickets done (OMN-931, 933, 936, 939)
4. 🚀 **NOW**: Start Wave 2 - Runtime Core + Foundation Complete
