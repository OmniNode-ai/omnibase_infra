Extra additional cash# Event Naming Convention Consolidation Analysis

**⚠️ MIGRATION COMPLETE - ARCHIVED DOCUMENT**

This document provided the analysis and migration plan for consolidating event naming from the old suffix-based pattern (`NodeGenerationRequestedEvent`) to the new ONEX v2.0 compliant prefix-based pattern (`ModelEventNodeGenerationRequested`).

**Migration Status**: ✅ COMPLETE (October 2025)
**Result**: All events migrated to `ModelEvent*` prefix pattern
**Documentation**: See `docs/events/SCHEMA_VERSIONING.md` for current event naming standards

---

## Executive Summary (Historical)

Two competing event naming patterns existed in omninode_bridge:

1. **Old Style (Suffix-based)**: `NodeGenerationRequestedEvent`, `GenerationMetricsRecordedEvent`
   - Location: `src/omninode_bridge/events/models/codegen_events.py` (611 lines)
   - Usage: **~45+ locations** (more widely used)
   - Components: Metrics Reducer, E2E Integration Tests

2. **New Style (Prefix-based)**: `ModelEventNodeGenerationRequested`, `ModelEventGenerationMetricsRecorded`
   - Location: `src/omninode_bridge/events/codegen/model_event_node_generation.py` (524 lines)
   - Usage: **~26+ locations** (newer components)
   - Components: Orchestrator, CLI, CLI Tests

**Recommendation**: Migrate to **New Style (Model* prefix)** as it aligns with:
- ONEX v2.0 compliance standards (explicitly mentioned in code)
- Intent events already using `ModelEventPublishIntent` pattern
- Consistent with omnibase_core naming conventions (`Model*` prefix)

---

## Detailed Analysis

### 1. Old Style Events (Suffix-based)

**File**: `src/omninode_bridge/events/models/codegen_events.py`

**Event Classes** (13 total):
```python
# Envelope
OnexEnvelopeV1

# Node Generation Workflow (5)
NodeGenerationRequestedEvent
NodeGenerationStartedEvent
NodeGenerationStageCompletedEvent
NodeGenerationCompletedEvent
NodeGenerationFailedEvent

# Metrics Aggregation (1)
GenerationMetricsRecordedEvent

# Pattern Storage (2)
PatternStorageRequestedEvent
PatternStoredEvent

# Intelligence Gathering (2)
IntelligenceQueryRequestedEvent
IntelligenceQueryCompletedEvent

# Orchestration (2)
OrchestratorCheckpointReachedEvent
OrchestratorCheckpointResponseEvent
```

**Exported From**:
- `src/omninode_bridge/events/__init__.py` (main events module)

**Usage Locations** (45+ occurrences):

1. **NodeCodegenMetricsReducer** (Primary Consumer)
   - `src/omninode_bridge/nodes/codegen_metrics_reducer/v1_0_0/node.py`
     - Lines 36-42: Import statements
     - Line 48-50: Type union definition
     - Lines 261-265: Event deserialization
   - `src/omninode_bridge/nodes/codegen_metrics_reducer/v1_0_0/aggregator.py`
     - Lines 26-30: Import statements
     - Lines 138-160: Type hints and event processing
   - `src/omninode_bridge/nodes/codegen_metrics_reducer/v1_0_0/tests/test_aggregator.py`
     - Lines 13-17: Import statements
     - Lines 40-73: Fixture creation functions
     - Multiple test methods using these events (~30 occurrences)

2. **E2E Integration Tests**
   - `tests/integration/e2e/test_codegen_event_system_e2e.py`
     - Lines 32-46: Import statements
     - Throughout test methods for event validation

**Export Pattern**:
```python
# In src/omninode_bridge/events/__init__.py
from omninode_bridge.events.models.codegen_events import (
    NodeGenerationRequestedEvent,
    NodeGenerationStartedEvent,
    # ... etc
)
```

---

### 2. New Style Events (Prefix-based)

**File**: `src/omninode_bridge/events/codegen/model_event_node_generation.py`

**Event Classes** (12 total - same logical events, different naming):
```python
# Node Generation Workflow (5)
ModelEventNodeGenerationRequested
ModelEventNodeGenerationStarted
ModelEventNodeGenerationStageCompleted
ModelEventNodeGenerationCompleted
ModelEventNodeGenerationFailed

# Metrics Aggregation (1)
ModelEventGenerationMetricsRecorded

# Pattern Storage (2)
ModelEventPatternStorageRequested
ModelEventPatternStored

# Intelligence Gathering (2)
ModelEventIntelligenceQueryRequested
ModelEventIntelligenceQueryCompleted

# Orchestration (2)
ModelEventOrchestratorCheckpointReached
ModelEventOrchestratorCheckpointResponse
```

**Exported From**:
- `src/omninode_bridge/events/codegen/__init__.py` (codegen-specific module)

**Usage Locations** (26+ occurrences):

1. **NodeCodegenOrchestrator** (Primary Publisher)
   - `src/omninode_bridge/nodes/codegen_orchestrator/v1_0_0/node.py`
     - Lines 28-32: Import statements
     - Line 249: Event creation and publishing
   - `src/omninode_bridge/nodes/codegen_orchestrator/v1_0_0/workflow.py`
     - Lines 31-35: Import statements
     - Line 687: Event creation and publishing

2. **CLI Components**
   - `src/omninode_bridge/cli/codegen/commands/generate.py`
     - Line 15: Import statement
     - Line 100: Event creation
   - `src/omninode_bridge/cli/codegen/client/kafka_client.py`
     - Line 17: Import statement
     - Line 68: Type hint and event publishing
   - `src/omninode_bridge/cli/codegen/protocols.py`
     - Line 12: Import statement
     - Line 31: Protocol method signature

3. **CLI Tests**
   - `tests/cli/codegen/conftest.py`
     - Line 13: Import statement
     - Line 26: Type hint
     - Line 40: Method signature
   - `tests/cli/codegen/test_cli_components.py`
     - Line 14: Import statement
     - Lines 49, 61: Event creation in tests
   - `tests/cli/codegen/test_cli_error_paths.py`
     - Line 40: Import statement
     - Line 43: Event creation in test

**Export Pattern**:
```python
# In src/omninode_bridge/events/codegen/__init__.py
from .model_event_node_generation import (
    ModelEventNodeGenerationRequested,
    ModelEventNodeGenerationStarted,
    # ... etc
)
```

---

### 3. Supporting Evidence for New Style

**Intent Events Already Using Model* Prefix**:
```python
# In src/omninode_bridge/events/models/intent_events.py
class ModelEventPublishIntent(EventBase):
    """Intent to publish an event to Kafka."""
    ...

class ModelIntentExecutionResult(EventBase):
    """Result of executing an intent."""
    ...
```

**ONEX v2.0 Compliance Documentation**:
From `src/omninode_bridge/events/codegen/model_event_node_generation.py`:
```python
"""
ONEX v2.0 Compliance:
- Model prefix naming: ModelEvent*
- Pydantic v2 validation
- UUID correlation tracking
- Timestamp tracking for all events
"""
```

**Omnibase Core Conventions**:
The codebase uses `Model*` prefix throughout:
- `ModelContainer`
- `ModelContractOrchestrator`
- `ModelContractReducer`
- `ModelOnexError`

---

## Impact Analysis

### Files Requiring Changes

#### High Priority (Direct Event Usage)
1. **Metrics Reducer Node** (3 files, ~45 occurrences)
   - `src/omninode_bridge/nodes/codegen_metrics_reducer/v1_0_0/node.py`
   - `src/omninode_bridge/nodes/codegen_metrics_reducer/v1_0_0/aggregator.py`
   - `src/omninode_bridge/nodes/codegen_metrics_reducer/v1_0_0/tests/test_aggregator.py`

2. **E2E Integration Tests** (1 file, ~15 occurrences)
   - `tests/integration/e2e/test_codegen_event_system_e2e.py`

3. **Main Events Module** (1 file, ~26 exports)
   - `src/omninode_bridge/events/__init__.py`

#### Medium Priority (Documentation References)
4. **Documentation Files** (3+ files)
   - `docs/events/SCHEMA_VERSIONING.md` (~15 references)
   - `TEST_REPORT_CODEGEN_MVP.md` (~4 references)
   - `POLY-12-SUMMARY.md` (~1 reference)
   - `POLY_9_FINAL_SUMMARY.md` (~1 reference)
   - `POLY_9_COMPLETION_REPORT.md` (~1 reference)

#### Low Priority (Deletion)
5. **Old Event Definitions** (1 file, 611 lines)
   - `src/omninode_bridge/events/models/codegen_events.py` (DELETE after migration)

**Total Files**: 8 primary code files + 5 documentation files = **13 files**

---

## Migration Strategy

### Phase 1: Preparation (No Code Changes)
1. Create feature branch: `refactor/consolidate-event-naming`
2. Run full test suite to establish baseline: `pytest -v`
3. Create backup of old event definitions for reference
4. Document current import patterns for rollback if needed

### Phase 2: Update Event Consumers (Metrics Reducer)
**Goal**: Update metrics reducer to use new event names

**Files to Modify**:
1. `src/omninode_bridge/nodes/codegen_metrics_reducer/v1_0_0/node.py`
   ```python
   # OLD:
   from omninode_bridge.events.models.codegen_events import (
       NodeGenerationCompletedEvent,
       NodeGenerationStartedEvent,
   )

   # NEW:
   from omninode_bridge.events.codegen import (
       ModelEventNodeGenerationCompleted,
       ModelEventNodeGenerationStarted,
   )

   # Update type hints:
   # OLD: NodeGenerationStartedEvent | NodeGenerationCompletedEvent
   # NEW: ModelEventNodeGenerationStarted | ModelEventNodeGenerationCompleted

   # Update deserialization:
   # OLD: return NodeGenerationStartedEvent(**event_data)
   # NEW: return ModelEventNodeGenerationStarted(**event_data)
   ```

2. `src/omninode_bridge/nodes/codegen_metrics_reducer/v1_0_0/aggregator.py`
   ```python
   # Update imports
   # Update type hints in method signatures
   # Update docstrings
   ```

3. `src/omninode_bridge/nodes/codegen_metrics_reducer/v1_0_0/tests/test_aggregator.py`
   ```python
   # Update imports
   # Update fixture return types
   # Update event instantiation in tests (~30 occurrences)
   ```

**Validation**:
```bash
pytest src/omninode_bridge/nodes/codegen_metrics_reducer/v1_0_0/tests/ -v
```

### Phase 3: Update E2E Integration Tests
**Files to Modify**:
1. `tests/integration/e2e/test_codegen_event_system_e2e.py`
   ```python
   # OLD:
   from omninode_bridge.events.models.codegen_events import (
       NodeGenerationRequestedEvent,
       NodeGenerationStartedEvent,
       # ...
   )

   # NEW:
   from omninode_bridge.events.codegen import (
       ModelEventNodeGenerationRequested,
       ModelEventNodeGenerationStarted,
       # ...
   )

   # Update all event instantiations throughout test methods
   ```

**Validation**:
```bash
pytest tests/integration/e2e/test_codegen_event_system_e2e.py -v
```

### Phase 4: Update Main Events Module
**Files to Modify**:
1. `src/omninode_bridge/events/__init__.py`
   ```python
   # Option A: Re-export from new location with new names
   from omninode_bridge.events.codegen import (
       ModelEventNodeGenerationRequested,
       ModelEventNodeGenerationStarted,
       # ... etc
   )

   # Option B: Provide backward compatibility aliases (if needed)
   NodeGenerationRequestedEvent = ModelEventNodeGenerationRequested
   NodeGenerationStartedEvent = ModelEventNodeGenerationStarted
   # ... etc

   # Update __all__ list to export new names
   ```

**Recommendation**: Use Option A (clean break) since this is an MVP foundation codebase.

### Phase 5: Update Documentation
**Files to Modify**:
1. `docs/events/SCHEMA_VERSIONING.md`
   - Update all example code snippets
   - Update class name references

2. `TEST_REPORT_CODEGEN_MVP.md`
   - Update test result references

3. `POLY-12-SUMMARY.md`, `POLY_9_FINAL_SUMMARY.md`, `POLY_9_COMPLETION_REPORT.md`
   - Update historical references (optional, mark as legacy)

### Phase 6: Remove Old Event Definitions
**Files to Delete**:
1. `src/omninode_bridge/events/models/codegen_events.py`
   ```bash
   git rm src/omninode_bridge/events/models/codegen_events.py
   ```

**Files to Update**:
1. `src/omninode_bridge/events/models/__init__.py`
   - Remove exports of deleted events

**Validation**:
```bash
# Verify no remaining imports
grep -r "from.*codegen_events import" src/ tests/
# Should return no results

# Run full test suite
pytest -v
```

### Phase 7: Final Validation & Cleanup
1. **Run Full Test Suite**:
   ```bash
   pytest -v --cov=src/omninode_bridge --cov-report=term-missing
   ```

2. **Check for Stragglers**:
   ```bash
   # Search for old naming pattern
   grep -r "NodeGeneration.*Event" --include="*.py" src/ tests/
   # Should only find in documentation/comments
   ```

3. **Update CLAUDE.md**:
   - Add note about consolidated event naming
   - Update any examples using old events

4. **Git Commit**:
   ```bash
   git add -A
   git commit -m "refactor: Consolidate event naming to ModelEvent* prefix (ONEX v2.0)

   - Migrate all events to ModelEvent* prefix for ONEX v2.0 compliance
   - Update NodeCodegenMetricsReducer to use new event names
   - Update E2E integration tests with new event names
   - Update main events module exports
   - Remove old codegen_events.py (611 lines)
   - Update documentation to reflect new naming

   Breaking change: Old event names (NodeGeneration*Event) removed.
   All consumers updated in this commit.

   Total: 13 files changed, ~90 occurrences updated"
   ```

---

## Risk Assessment

### Low Risk
- **Test Coverage**: High coverage (92.8%) provides safety net
- **Scope**: Limited to event definitions and consumers
- **Isolation**: Events are data models with no complex logic

### Medium Risk
- **Import Updates**: ~90 occurrences across 8 files need updating
- **Test Updates**: Multiple test files need synchronization

### Mitigation Strategies
1. **Atomic Migration**: Complete all updates in single commit
2. **Comprehensive Testing**: Run full test suite after each phase
3. **Validation Script**: Create script to verify no old imports remain
4. **Documentation**: Update inline comments and docstrings

---

## Timeline Estimate

Assuming careful, methodical approach with validation at each step:

- **Phase 1** (Preparation): 15 minutes
- **Phase 2** (Metrics Reducer): 45 minutes (3 files, ~45 occurrences)
- **Phase 3** (E2E Tests): 30 minutes (1 file, ~15 occurrences)
- **Phase 4** (Main Module): 15 minutes (1 file, exports update)
- **Phase 5** (Documentation): 30 minutes (5 files, reference updates)
- **Phase 6** (Deletion): 15 minutes (cleanup + verification)
- **Phase 7** (Final Validation): 30 minutes (full test suite + review)

**Total Estimated Time**: ~3 hours

---

## Alternative Approach: Gradual Migration

If immediate full migration is too risky, consider:

1. **Keep Both Styles** temporarily
2. **Add Deprecation Warnings** to old style
3. **Migrate One Component at a Time** over multiple PRs
4. **Remove Old Style** after 2-3 sprints

**Pros**: Lower risk, easier to roll back
**Cons**: Temporary inconsistency, technical debt accumulation

**Recommendation**: Full migration is preferred for MVP foundation codebase.

---

## Success Criteria

✅ All tests pass (501 tests, 92.8% coverage maintained)
✅ No old event names (`NodeGeneration*Event`) in code imports
✅ All documentation updated with new naming
✅ Clean git history with single atomic commit
✅ Zero backward compatibility aliases (clean break)

---

## Conclusion

**Recommended Action**: Proceed with **full migration to ModelEvent* prefix**.

**Rationale**:
1. Aligns with ONEX v2.0 compliance standards
2. Consistent with omnibase_core conventions
3. Intent events already use Model* prefix
4. MVP foundation allows breaking changes
5. Limited scope (~13 files) makes migration tractable
6. High test coverage provides safety net

**Next Steps**:
1. Review and approve this migration plan
2. Create feature branch
3. Execute Phase 1-7 migration
4. Submit PR for review
5. Merge after validation

---

**Generated**: 2025-10-23
**Author**: Claude Code Agent (Polymorphic)
**Context**: Event naming consolidation analysis for omninode_bridge MVP
