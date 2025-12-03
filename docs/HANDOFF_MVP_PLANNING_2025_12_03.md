# Handoff: MVP Planning Session - omnibase_infra

**Date**: 2025-12-03
**Session Focus**: MVP Proposed Work Issues document enhancement and Linear setup
**Status**: Document complete, Linear ticket creation pending

---

## Session Summary

This session focused on transforming the MVP Proposed Work Issues document from a dense technical spec into a comprehensive, navigable, and actionable planning document.

### What Was Accomplished

#### 1. Document Enhancements (41 Total Improvements)

**Round 1 (10 suggestions)**:
- Core/Infra boundary clarifications
- Contract semantics (stability warnings, handler uniqueness)
- Event bus requirements (ordering guarantees, response patterns)
- RuntimeHostProcess clarifications (error handling, concurrency model)
- Testing improvements (determinism, performance requirements)
- Handler API sharpness (async requirements, lifecycle)
- Deployment design (resource targets, production warnings)
- Topic naming schema (validation rules, examples)
- Scope boundary notes (Do Not Implement sections)
- Minor micro-fixes (fail-fast rules, threading prohibition)

**Round 2 (18 suggestions)**:
- Executive Summary (5 bullet points)
- Glossary (9 terms defined)
- FAQ section (5 common questions)
- 8 new architectural invariants
- Prohibited fields section
- Cross-version compatibility rules
- Error taxonomy improvements (InvalidOperationError, ContractValidationError)
- Handler contracts (operations, fields, response shapes)
- SQL injection protection details
- Handler implementation guidelines
- NodeRuntime registration order and envelope immutability
- InMemoryEventBus memory bounds and test features
- KafkaEventBus slow consumer handling
- Contract evolution roadmap
- Testing infrastructure (deterministic helpers)
- Dockerfile ENTRYPOINT documentation
- K8s HPA template with multi-instance warning
- Naming conventions table
- Risk items table (7 risks with mitigations)
- Maintainability guidelines (LOC limits, docstrings)
- Future-proofing section

**Round 3 (13 suggestions)**:
- "Why This Document Exists" statement
- Error envelope requirements (9-field table + JSON example)
- Correlation ID ownership rules
- Contract loader fail-fast rules (9 conditions)
- Blocking I/O rules table (12 operations)
- Handler operation naming convention
- Expected first PRs (12-entry onboarding guide)
- Handler lifecycle state machine (ASCII diagram)
- Missing handler behavior documentation
- Testing without Docker guide
- Example node contracts (effect + compute)
- Topic naming examples (12 valid/invalid examples)

**Round 4 (4 high-impact additions)**:
- Map of Abstractions (ASCII diagram showing Core→SPI→Infra)
- MVP Constraints Checklist (12 things = MVP works)
- Failure Examples (6 concrete code snippets of what NOT to do)
- Minimum Reference Contract (smallest working YAML)
- How to Read This Document (role-based navigation)

#### 2. Linear Setup

**Labels Created**:
| Label | Color | ID |
|-------|-------|-----|
| `mvp` | Blue (#0ea5e9) | `afae1c79-1608-4150-aa9a-a21f97fc31e8` |
| `beta` | Purple (#8b5cf6) | `3a95919f-97a6-42ad-9083-0f51e9968375` |
| `production` | Green (#22c55e) | `6908456d-63fd-406d-a6e2-deef4ba25c3d` |

**Existing Labels Available**:
- `omnibase_infra` (red)
- `omnibase_core` (green)
- `omnibase_spi` (pink)
- `testing`, `docs`, `security`, `event bus`, `db`, `infra`, etc.

**Team**: Omninode (`9bdff6a3-f4ef-4ff7-b29a-6c4cf44371e6`)
**Project**: MVP - OmniNode Platform Foundation (`e44ddbf4-b4c7-40dc-84fa-f402ec27b38e`)

---

## Pending Work

### 1. Linear Issue Templates

User requested standardized issue templates in Linear for cross-project consistency. The template format (based on omnibase_core tickets) is:

```markdown
## Description

{description}

## TDD Approach

**TDD: {Required|Optional|None}** - {reason}

## Acceptance Criteria

- [ ] {criterion 1}
- [ ] {criterion 2}
- [ ] mypy --strict passes
- [ ] pyright passes

## Phase

Phase {N}: {Phase Name}

## Dependencies

* {dependency or "None"}

## Reference

See `docs/MVP_PROPOSED_WORK_ISSUES.md` for full context.
```

**Action Required**: Linear doesn't have native issue templates. Options:
1. Create a Linear document with template text for copy/paste
2. Use automation (Linear API) to enforce template structure
3. Create GitHub issue templates that sync to Linear

### 2. Linear Ticket Creation

54 issues ready to be created in Linear:

| Milestone | Count | Labels |
|-----------|-------|--------|
| MVP (v0.1.0) | 24 | `mvp`, `omnibase_infra` |
| Beta (v0.2.0) | 22 | `beta`, `omnibase_infra` |
| Production (v0.3.0) | 8 | `production`, `omnibase_infra` |

**Issue Breakdown by Phase**:
- Phase 0: CI Guardrails (2 MVP, 2 Beta)
- Phase 1: Core Types (9 MVP, 2 Beta) - in omnibase_core
- Phase 2: SPI Updates (3 MVP) - in omnibase_spi
- Phase 3: Infrastructure (8 MVP, 10 Beta)
- Phase 4: Testing (2 MVP, 5 Beta, 3 Prod)
- Phase 5: Deployment (2 MVP, 3 Beta, 2 Prod)

---

## Key Files Modified

| File | Status | Description |
|------|--------|-------------|
| `docs/MVP_PROPOSED_WORK_ISSUES.md` | Updated | ~2800 lines, comprehensive MVP spec |
| `docs/MVP_EXECUTION_PLAN.md` | Unchanged | Original execution plan |

---

## Document Structure (MVP_PROPOSED_WORK_ISSUES.md)

```
1. Why This Document Exists
2. Executive Summary
3. Glossary (9 terms)
4. Naming Conventions
5. FAQ (5 questions)
6. Map of Abstractions (ASCII diagram)
7. Milestone Overview (MVP/Beta/Production)
8. MVP Constraints Checklist (12 items)
9. Simplified Contract Format
10. Topic Naming Schema
11. Contract Evolution Roadmap
12. Non-Goals
13. MVP Scope Boundaries (Do Not Implement)
14. Beta Scope Boundaries
15. Architectural Invariants Checklist
16. Correlation ID Ownership Rules
17. Blocking I/O Rules
18. Failure Examples (6 code snippets)
19. Phase 0-5 Issues (54 total)
20. Handler Lifecycle State Machine
21. Testing Infrastructure
22. Expected First PRs (12-entry onboarding)
23. Execution Order (dependency graphs)
24. Naming Conventions
25. MVP Risk Items (7 risks)
26. Maintainability Guidelines
27. Future-Proofing
28. Success Metrics
29. Issue Creation Guidelines
30. Example Node Contracts
31. Minimum Reference Contract
32. How to Read This Document
33. Document Organization (Future)
```

---

## Next Steps for Continuation

1. **Decide on Linear template approach** (document vs automation vs GitHub sync)
2. **Create 54 Linear tickets** using the MCP tool
3. **Review the updated MVP document** for any final adjustments
4. **Consider splitting document** into multiple files as suggested in "Document Organization"

---

## Commands for Quick Context

```bash
# View the updated MVP document
cat /Users/jonah/Code/omnibase_infra/docs/MVP_PROPOSED_WORK_ISSUES.md

# Check Linear labels
# Use mcp__linear-server__list_issue_labels

# Create issues (when ready)
# Use mcp__linear-server__create_issue with project "MVP - OmniNode Platform Foundation"
```

---

**Session Duration**: Extended planning session
**Polymorphic Agents Used**: 14 (across 4 rounds of parallel execution)
**Total Suggestions Integrated**: 41
