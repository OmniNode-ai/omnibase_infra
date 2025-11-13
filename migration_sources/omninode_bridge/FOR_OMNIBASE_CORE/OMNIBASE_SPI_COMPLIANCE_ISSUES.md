# OmniBase SPI Protocol Compliance Issues & Roadmap

**Document Version**: 1.0
**Created**: 2025-10-30
**Correlation ID**: 2be54c34-a50f-42fa-aefa-6ec39b4778b8
**Status**: Active - Tracking Issues for omnibase_spi v0.2.0
**Owner**: OmniNode Bridge Team

---

## Executive Summary

This document tracks all protocol compliance issues, missing protocols, and interface mismatches discovered during omninode_bridge Phase 1 implementation. These issues should be addressed in **omnibase_spi v0.2.0** and **omnibase_core updates**.

**Total Issues Identified**: 8 categories
**Priority**: HIGH - Blocks full ONEX v2.0 compliance
**Impact**: Affects duck typing, DI container usage, and protocol-based architecture

---

## Table of Contents

1. [Missing Protocols in omnibase_spi](#missing-protocols-in-omnibase_spi)
2. [Container Type Issues](#container-type-issues)
3. [Protocol Interface Mismatches](#protocol-interface-mismatches)
4. [Current vs. Desired State](#current-vs-desired-state)
5. [Recommendations for omnibase_spi v0.2.0](#recommendations-for-omnibase_spi-v020)
6. [Recommendations for omnibase_core](#recommendations-for-omnibase_core)
7. [Implementation Roadmap](#implementation-roadmap)
8. [References](#references)

---

## Missing Protocols in omnibase_spi

### Issue #1: ProtocolContainer Missing

**Status**: ❌ NOT FOUND in omnibase_spi v0.1.0
**Priority**: P0 BLOCKER
**Impact**: Cannot use protocol-based duck typing for generic value containers

**Expected Location**:
```python
from omnibase_spi.protocols import ProtocolContainer
```

**Current Workaround**:
```python
# Using concrete type instead of protocol
from omnibase_core.models.core import ModelContainer
```

**Desired Protocol Interface**:
```python
from typing import Protocol, TypeVar, Generic

T = TypeVar('T')

class ProtocolContainer(Protocol, Generic[T]):
    """Protocol for generic value containers with metadata."""

    @property
    def value(self) -> T:
        """Get the wrapped value."""
        ...

    @property
    def metadata(self) -> dict:
        """Get container metadata."""
        ...

    def get_metadata(self, key: str, default=None):
        """Get specific metadata field."""
        ...
```

**Implementation Status**: ⏳ Pending omnibase_spi v0.2.0

---

### Issue #2: ProtocolNode Missing

**Status**: ❌ NOT FOUND in omnibase_spi v0.1.0
**Priority**: P0 BLOCKER
**Impact**: Cannot use protocol-based duck typing for ONEX nodes

**Note**: `ProtocolOnexNode` exists but has different interface than needed.

**Expected Location**:
```python
from omnibase_spi.protocols import ProtocolNode
```

**Current Workaround**:
```python
# Using concrete types
from omnibase_core.nodes import NodeOrchestrator, NodeReducer, NodeEffect
```

**Desired Protocol Interface**:
```python
from typing import Protocol, Any

class ProtocolNode(Protocol):
    """Protocol for ONEX nodes with execution methods."""

    async def execute_orchestration(self, contract: Any) -> Any:
        """Execute orchestration workflow."""
        ...

    async def execute_reduction(self, contract: Any) -> Any:
        """Execute reduction/aggregation."""
        ...

    async def execute_effect(self, contract: Any) -> Any:
        """Execute side effect."""
        ...

    @property
    def node_id(self) -> str:
        """Get unique node identifier."""
        ...

    @property
    def node_type(self) -> str:
        """Get node type (orchestrator/reducer/effect/compute)."""
        ...
```

**Issue with Existing ProtocolOnexNode**:
```python
# Current ProtocolOnexNode expects:
- get_input_model() -> type
- get_output_model() -> type
- run(*args, **kwargs) -> ContextValue

# But NodeOrchestrator/NodeReducer/NodeEffect have:
- execute_orchestration(contract) -> OutputModel
- execute_reduction(contract) -> OutputModel
- execute_effect(contract) -> OutputModel

# ❌ Incompatible interfaces - can't use ProtocolOnexNode
```

**Implementation Status**: ⏳ Pending omnibase_spi v0.2.0

---

### Issue #3: ProtocolContract Missing

**Status**: ❌ NOT FOUND in omnibase_spi v0.1.0
**Priority**: P1 HIGH
**Impact**: Cannot use protocol-based duck typing for contracts

**Note**: `ProtocolOnexContractData` exists but only covers data structure, not full contract behavior.

**Expected Location**:
```python
from omnibase_spi.protocols import ProtocolContract
```

**Current Workaround**:
```python
# Using concrete types
from omnibase_core.models.contracts import (
    ModelContractOrchestrator,
    ModelContractReducer,
    ModelContractEffect
)
```

**Desired Protocol Interface**:
```python
from typing import Protocol, Any, Dict

class ProtocolContract(Protocol):
    """Protocol for ONEX contracts."""

    @property
    def contract_id(self) -> str:
        """Get unique contract identifier."""
        ...

    @property
    def version(self) -> str:
        """Get contract version."""
        ...

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get contract metadata."""
        ...

    def to_dict(self) -> dict:
        """Serialize contract to dictionary."""
        ...

    @classmethod
    def from_dict(cls, data: dict) -> 'ProtocolContract':
        """Deserialize contract from dictionary."""
        ...
```

**Implementation Status**: ⏳ Pending omnibase_spi v0.2.0

---

### Issue #4: ProtocolOnexError Missing

**Status**: ❌ NOT FOUND in omnibase_spi v0.1.0
**Priority**: P1 HIGH
**Impact**: Cannot use protocol-based duck typing for error handling

**Expected Location**:
```python
from omnibase_spi.protocols import ProtocolOnexError
```

**Current Workaround**:
```python
# Using concrete types
from omnibase_core.models.core import ModelOnexError
```

**Desired Protocol Interface**:
```python
from typing import Protocol, Optional
from enum import Enum

class ProtocolOnexError(Protocol):
    """Protocol for ONEX error objects."""

    @property
    def error_code(self) -> str:
        """Get error code (e.g., 'VALIDATION_FAILED')."""
        ...

    @property
    def error_message(self) -> str:
        """Get human-readable error message."""
        ...

    @property
    def error_category(self) -> str:
        """Get error category (validation/execution/configuration)."""
        ...

    @property
    def context(self) -> Optional[dict]:
        """Get error context (additional debug information)."""
        ...

    def to_dict(self) -> dict:
        """Serialize error to dictionary."""
        ...
```

**Implementation Status**: ⏳ Pending omnibase_spi v0.2.0

---

### Issue #5: ProtocolEnvelope Naming Inconsistency

**Status**: ⚠️ EXISTS as `ProtocolOnexEnvelope` but roadmap specifies `ProtocolEnvelope`
**Priority**: P2 MEDIUM
**Impact**: Minor - naming inconsistency

**Expected Location** (per roadmap):
```python
from omnibase_spi.protocols import ProtocolEnvelope
```

**Actual Location** (in omnibase_spi v0.1.0):
```python
from omnibase_spi.protocols import ProtocolOnexEnvelope  # Different name
```

**Recommendation**:
- Add `ProtocolEnvelope` as an alias to `ProtocolOnexEnvelope` for backward compatibility
- OR update documentation to use `ProtocolOnexEnvelope` consistently

**Implementation Status**: ⏳ Pending omnibase_spi v0.2.0

---

## Container Type Issues

### Issue #6: Orchestrator/Reducer Using Wrong Container Type

**Status**: 🐛 BUG in omninode_bridge implementation
**Priority**: P0 BLOCKER
**Impact**: HIGH - Affects DI container functionality, service resolution

**Current Implementation** (WRONG):
```python
# File: src/omninode_bridge/nodes/orchestrator/v1_0_0/node.py
# File: src/omninode_bridge/nodes/reducer/v1_0_0/node.py

from omnibase_core.models.core import ModelContainer

# ❌ Using generic value wrapper instead of DI container
container = ModelContainer(...)
```

**Correct Implementation**:
```python
# Should be using the actual DI container
from omninode_core.models.container import ModelONEXContainer

# ✅ Proper DI container with service resolution
container = ModelONEXContainer(...)
```

**Impact Analysis**:
- `ModelContainer[T]` is just a generic value wrapper (holds one value + metadata)
- `ModelONEXContainer` is the actual DI container providing:
  - Service resolution (`container.resolve(ServiceType)`)
  - Service caching
  - Workflow orchestration
  - Configuration management
  - Lifecycle management

**Files Affected**:
1. `src/omninode_bridge/nodes/orchestrator/v1_0_0/node.py`
2. `src/omninode_bridge/nodes/reducer/v1_0_0/node.py`

**Fix Required**:
- Update both files to import and use `ModelONEXContainer`
- Update all usages of `container.value` to proper DI container methods
- Update tests to mock `ModelONEXContainer` instead of `ModelContainer`

**Estimated Effort**: 2-3 hours
**Implementation Status**: ⏳ Scheduled for Option C Phase 1

---

### Issue #7: Missing ProtocolServiceRegistry in Roadmap Documentation

**Status**: ℹ️ DOCUMENTATION GAP
**Priority**: P3 LOW
**Impact**: LOW - Protocol exists but roadmap doesn't mention it

**Available Protocol** (in omnibase_spi v0.1.0):
```python
from omnibase_spi.protocols import ProtocolServiceRegistry  # ✅ Available
```

**Issue**: Implementation Roadmap specifies `ProtocolContainer` but should use `ProtocolServiceRegistry` for DI containers.

**Recommendation**: Update roadmap documentation to reflect:
```python
# Type hints (use SPI protocols):
from omnibase_spi.protocols import (
    ProtocolServiceRegistry,    # ✅ For DI containers (not ProtocolContainer)
    ProtocolOnexContractData,   # ✅ For contract data
    ProtocolOnexEnvelope,       # ✅ For event envelopes
)
```

**Implementation Status**: ⏳ Documentation update pending

---

## Protocol Interface Mismatches

### Issue #8: ProtocolOnexNode Interface Incompatibility

**Status**: ⚠️ INTERFACE MISMATCH
**Priority**: P1 HIGH
**Impact**: Cannot use existing `ProtocolOnexNode` for ONEX nodes

**Problem**: `ProtocolOnexNode` expects different methods than `NodeOrchestrator`/`NodeReducer`/`NodeEffect` provide.

**ProtocolOnexNode Interface** (from omnibase_spi v0.1.0):
```python
class ProtocolOnexNode(Protocol):
    def get_input_model(self) -> type: ...
    def get_output_model(self) -> type: ...
    def get_node_config(self) -> ProtocolNodeConfiguration: ...
    def run(self, *args, **kwargs) -> ContextValue: ...
```

**NodeOrchestrator/NodeReducer/NodeEffect Interface** (from omnibase_core):
```python
class NodeOrchestrator:
    async def execute_orchestration(
        self, contract: ModelContractOrchestrator
    ) -> ModelOutputOrchestrator: ...

class NodeReducer:
    async def execute_reduction(
        self, contract: ModelContractReducer
    ) -> ModelOutputReducer: ...

class NodeEffect:
    async def execute_effect(
        self, contract: ModelContractEffect
    ) -> ModelOutputEffect: ...
```

**Interface Comparison**:

| Aspect | ProtocolOnexNode | Actual Nodes |
|--------|------------------|--------------|
| **Execution Method** | `run(*args, **kwargs)` | `execute_orchestration(contract)` |
| **Input Handling** | Flexible args/kwargs | Typed contract parameter |
| **Return Type** | `ContextValue` | Specific output models |
| **Async Support** | Not specified | `async` methods |
| **Type Safety** | Weak (any args) | Strong (typed contracts) |

**Root Cause**: Different design philosophies
- `ProtocolOnexNode`: Generic, flexible interface
- Actual nodes: Specific, type-safe interfaces per node type

**Recommendation for omnibase_spi v0.2.0**:

**Option A**: Create node-type-specific protocols
```python
class ProtocolOrchestratorNode(Protocol):
    async def execute_orchestration(self, contract: Any) -> Any: ...

class ProtocolReducerNode(Protocol):
    async def execute_reduction(self, contract: Any) -> Any: ...

class ProtocolEffectNode(Protocol):
    async def execute_effect(self, contract: Any) -> Any: ...
```

**Option B**: Extend ProtocolOnexNode with execution methods
```python
class ProtocolOnexNode(Protocol):
    # Existing methods
    def get_input_model(self) -> type: ...
    def get_output_model(self) -> type: ...
    def run(self, *args, **kwargs) -> ContextValue: ...

    # Add execution methods (optional via @runtime_checkable)
    async def execute_orchestration(self, contract: Any) -> Any: ...
    async def execute_reduction(self, contract: Any) -> Any: ...
    async def execute_effect(self, contract: Any) -> Any: ...
```

**Option C**: Adapter pattern in omnibase_core
```python
# omnibase_core provides adapters
class OnexNodeAdapter:
    """Adapts ProtocolOnexNode interface to execute_* methods."""
    def run(self, *args, **kwargs):
        # Route to appropriate execute_* method based on node type
        ...
```

**Recommended**: **Option A** (type-specific protocols) for strongest type safety

**Implementation Status**: ⏳ Pending omnibase_spi v0.2.0 design decision

---

## Current vs. Desired State

### Current State (omnibase_spi v0.1.0)

**Available Protocols** (133 total):
```python
from omnibase_spi.protocols import (
    # ✅ Available and usable
    ProtocolServiceRegistry,        # DI container interface
    ProtocolOnexEnvelope,          # Event envelope (naming inconsistency)
    ProtocolOnexContractData,      # Contract data structure
    ProtocolOnexNode,              # Node interface (incompatible with actual nodes)

    # ... 129 more protocols
)
```

**Missing Protocols** (specified in roadmap):
```python
from omnibase_spi.protocols import (
    # ❌ NOT FOUND
    ProtocolContainer,    # Generic value container
    ProtocolNode,         # ONEX node interface
    ProtocolContract,     # Full contract interface
    ProtocolOnexError,    # Error object interface
    ProtocolEnvelope,     # Event envelope (named ProtocolOnexEnvelope instead)
)
```

**Architecture Issues**:
- Orchestrator/Reducer use `ModelContainer` instead of `ModelONEXContainer` (bug)
- Interface mismatch between `ProtocolOnexNode` and actual node implementations
- Roadmap documentation doesn't reflect available protocols

---

### Desired State (omnibase_spi v0.2.0 + omnibase_core updates)

**Phase 1: Add Missing Protocols**
```python
from omnibase_spi.protocols import (
    # New protocols to add
    ProtocolContainer,              # Generic value container with metadata
    ProtocolNode,                   # ONEX node with execute_* methods
    ProtocolContract,               # Full contract interface
    ProtocolOnexError,              # Error object interface
    ProtocolEnvelope,               # Alias to ProtocolOnexEnvelope

    # Node-type-specific protocols
    ProtocolOrchestratorNode,       # Orchestrator-specific interface
    ProtocolReducerNode,            # Reducer-specific interface
    ProtocolEffectNode,             # Effect-specific interface
    ProtocolComputeNode,            # Compute-specific interface
)
```

**Phase 2: Fix omnibase_core Implementation**
```python
# Update node implementations to use proper containers
from omnibase_core.models.container import ModelONEXContainer

class NodeOrchestrator:
    def __init__(self, container: ModelONEXContainer):  # ✅ Correct type
        self.container = container
        # Use proper DI container methods
```

**Phase 3: Protocol Compliance**
```python
# Type hints use protocols
from omnibase_spi.protocols import (
    ProtocolServiceRegistry,
    ProtocolOrchestratorNode,
    ProtocolContract,
)

def process_workflow(
    container: ProtocolServiceRegistry,
    node: ProtocolOrchestratorNode,
    contract: ProtocolContract,
) -> Any:
    """Fully protocol-based duck typing."""
    ...
```

---

## Recommendations for omnibase_spi v0.2.0

### Priority 1: Add Missing Core Protocols

**Issue**: Missing fundamental protocols for ONEX architecture

**Action Items**:
1. ✅ Add `ProtocolContainer` for generic value containers
2. ✅ Add `ProtocolContract` for contract objects
3. ✅ Add `ProtocolOnexError` for error handling
4. ✅ Add `ProtocolEnvelope` as alias to `ProtocolOnexEnvelope`

**Estimated Effort**: 1-2 days
**Priority**: P0 BLOCKER
**Complexity**: LOW (straightforward protocol definitions)

---

### Priority 2: Add Node-Type-Specific Protocols

**Issue**: `ProtocolOnexNode` doesn't match actual node interfaces

**Action Items**:
1. ✅ Add `ProtocolOrchestratorNode` with `execute_orchestration()` method
2. ✅ Add `ProtocolReducerNode` with `execute_reduction()` method
3. ✅ Add `ProtocolEffectNode` with `execute_effect()` method
4. ✅ Add `ProtocolComputeNode` with `execute_compute()` method
5. ⚠️ Consider: Keep `ProtocolOnexNode` as base or deprecate?

**Estimated Effort**: 2-3 days
**Priority**: P1 HIGH
**Complexity**: MEDIUM (design decision needed on node protocol hierarchy)

**Design Question**: Should node-specific protocols inherit from `ProtocolOnexNode`?

```python
# Option A: Inheritance
class ProtocolOrchestratorNode(ProtocolOnexNode, Protocol):
    async def execute_orchestration(self, contract: Any) -> Any: ...

# Option B: Standalone
class ProtocolOrchestratorNode(Protocol):
    async def execute_orchestration(self, contract: Any) -> Any: ...
    # No inheritance from ProtocolOnexNode
```

**Recommendation**: **Option B** (standalone) for clearer separation and no interface pollution.

---

### Priority 3: Protocol Naming Consistency

**Issue**: Inconsistent naming (`ProtocolEnvelope` vs `ProtocolOnexEnvelope`)

**Action Items**:
1. ✅ Add `ProtocolEnvelope` as alias to `ProtocolOnexEnvelope`
2. ✅ Update documentation to recommend preferred naming
3. ⚠️ Consider: Deprecation path for less-preferred names?

**Estimated Effort**: 0.5 days
**Priority**: P2 MEDIUM
**Complexity**: LOW

---

### Priority 4: Protocol Documentation

**Issue**: Insufficient documentation on protocol usage patterns

**Action Items**:
1. ✅ Add comprehensive protocol documentation with examples
2. ✅ Document duck typing patterns and best practices
3. ✅ Add protocol compliance testing guide
4. ✅ Document migration path from concrete types to protocols

**Estimated Effort**: 2-3 days
**Priority**: P2 MEDIUM
**Complexity**: LOW (documentation only)

---

## Recommendations for omnibase_core

### Priority 1: Fix Container Type Usage

**Issue**: Nodes using `ModelContainer` instead of `ModelONEXContainer`

**Action Items**:
1. ✅ Update `NodeOrchestrator` to use `ModelONEXContainer`
2. ✅ Update `NodeReducer` to use `ModelONEXContainer`
3. ✅ Update all node examples and documentation
4. ✅ Add migration guide for existing code
5. ✅ Add deprecation warnings for `ModelContainer` usage in node contexts

**Estimated Effort**: 1-2 days
**Priority**: P0 BLOCKER
**Complexity**: MEDIUM (requires updating examples and tests)

**Files to Update** (in omnibase_core):
- `omnibase_core/nodes/orchestrator.py`
- `omnibase_core/nodes/reducer.py`
- `omnibase_core/nodes/effect.py`
- `omnibase_core/nodes/compute.py`
- All example code and documentation

---

### Priority 2: Protocol Compliance in Node Implementations

**Issue**: Node implementations should satisfy protocol contracts

**Action Items**:
1. ✅ Ensure `NodeOrchestrator` satisfies `ProtocolOrchestratorNode` (when available)
2. ✅ Ensure `NodeReducer` satisfies `ProtocolReducerNode` (when available)
3. ✅ Add protocol compliance tests
4. ✅ Add `@runtime_checkable` decorators where appropriate

**Estimated Effort**: 1 day
**Priority**: P1 HIGH
**Complexity**: LOW (verification and testing)

---

### Priority 3: Add Protocol Type Hints

**Issue**: Core classes use concrete types instead of protocols in type hints

**Action Items**:
1. ✅ Update type hints to use protocols where available
2. ✅ Keep concrete implementations (just change type hints)
3. ✅ Add mypy protocol checking to CI
4. ✅ Update documentation with protocol-based examples

**Example**:
```python
# BEFORE (concrete typing):
def create_workflow(container: ModelONEXContainer) -> NodeOrchestrator:
    ...

# AFTER (protocol typing):
from omnibase_spi.protocols import ProtocolServiceRegistry, ProtocolOrchestratorNode

def create_workflow(container: ProtocolServiceRegistry) -> ProtocolOrchestratorNode:
    # Implementation still uses concrete types
    return NodeOrchestrator(container)  # Satisfies protocol
```

**Estimated Effort**: 2-3 days
**Priority**: P2 MEDIUM
**Complexity**: MEDIUM (requires careful type hint updates)

---

## Implementation Roadmap

### Phase 1: Quick Wins (Week 1)

**omninode_bridge** (Option C - Phase 1):
- ✅ Fix container bug: Update Orchestrator/Reducer to use `ModelONEXContainer`
- ✅ Add protocol type hints where protocols exist and fit well
- ✅ Create this compliance issues document
- ✅ Estimated: 2-3 hours

**omnibase_core**:
- ✅ Fix container type usage in node implementations
- ✅ Add deprecation warnings for incorrect container usage
- ✅ Estimated: 1-2 days

---

### Phase 2: Protocol Development (Weeks 2-3)

**omnibase_spi v0.2.0**:
- ✅ Add missing core protocols (`ProtocolContainer`, `ProtocolContract`, `ProtocolOnexError`)
- ✅ Add node-type-specific protocols
- ✅ Add protocol documentation and examples
- ✅ Estimated: 1 week

**omnibase_core**:
- ✅ Ensure nodes satisfy new protocols
- ✅ Add protocol compliance tests
- ✅ Estimated: 2-3 days

---

### Phase 3: Full Protocol Adoption (Week 4)

**omninode_bridge** (Option C - Phase 2):
- ✅ Complete protocol type hint migration
- ✅ Add protocol compliance tests
- ✅ Update documentation
- ✅ Estimated: 2-4 hours

**omnibase_core**:
- ✅ Update all type hints to use protocols
- ✅ Add mypy protocol checking
- ✅ Comprehensive documentation update
- ✅ Estimated: 2-3 days

---

### Phase 4: Validation & Release (Week 5)

**All Repositories**:
- ✅ Full protocol compliance testing
- ✅ Integration testing across omnibase_spi + omnibase_core + omninode_bridge
- ✅ Documentation review
- ✅ Release omnibase_spi v0.2.0
- ✅ Release omnibase_core with protocol support
- ✅ Update omninode_bridge to use new protocols
- ✅ Estimated: 1 week

---

## References

### Documents
- **Implementation Roadmap**: `/Volumes/PRO-G40/Code/omninode_bridge/docs/planning/IMPLEMENTATION_ROADMAP.md`
- **ONEX v2.0 Specification**: (external reference)
- **Protocol Duck Typing Analysis**: Task 1.4 findings (2025-10-30)

### Related Issues
- Issue #6: Container Type Bug (omninode_bridge)
- Issue #8: ProtocolOnexNode Interface Mismatch

### Code References
- **omnibase_spi v0.1.0**: Available protocols analysis
- **omnibase_core**: Node implementations review
- **omninode_bridge**: Bridge node implementation patterns

---

## Phase 2 Implementation: Partial Protocol Type Hints (COMPLETED)

**Date**: 2025-10-30
**Correlation ID**: 36e3dccd-8929-4775-93fb-b684bc0ef3fe
**Status**: ✅ COMPLETE

### Summary

Added partial protocol type hints to bridge nodes using available protocols from omnibase_spi v0.1.0.
Focused on **ProtocolServiceRegistry** for DI container duck typing in PUBLIC APIs.

### Protocols Applied

**Primary Protocol Used:**
- **ProtocolServiceRegistry** - Applied to all `container` parameters in public `__init__` methods

**Coverage Achieved:**
- **NodeBridgeOrchestrator** (`orchestrator/v1_0_0/node.py`): ✅
  - `__init__(container: ProtocolServiceRegistry)` - line 112
  - `_load_orchestrator_config(container: ProtocolServiceRegistry)` - line 334
- **NodeBridgeReducer** (`reducer/v1_0_0/node.py`): ✅
  - `__init__(container: ProtocolServiceRegistry)` - line 524
  - FSMStateManager embedded class `__init__(container: ProtocolServiceRegistry)` - line 137
- **FSMStateManager** (separate file `reducer/v1_0_0/fsm_state_manager.py`): ✅
  - `__init__(container: ProtocolServiceRegistry)` - line 57

### Implementation Pattern

**Protocol Import with Fallback** (used in all updated files):
```python
# Import protocols from omnibase_spi for duck typing (Phase 2: Protocol Type Hints)
try:
    from omnibase_spi.protocols import (
        ProtocolServiceRegistry,  # For DI container type hints
        ProtocolOnexEnvelope,  # For event envelope type hints
    )
    PROTOCOLS_AVAILABLE = True
except ImportError:
    # Protocol imports are optional - duck typing still works with concrete types
    PROTOCOLS_AVAILABLE = False
    ProtocolServiceRegistry = ModelONEXContainer  # type: ignore[misc,assignment]
    ProtocolOnexEnvelope = dict  # type: ignore[misc,assignment]
```

**Duck Typing Philosophy**:
- PUBLIC APIs use protocol type hints (`container: ProtocolServiceRegistry`)
- INTERNAL implementation uses concrete types (`ModelONEXContainer`)
- Backward compatible - no runtime changes
- Type checking with mypy/pyright still works

### Protocol Coverage Statistics

**Files Updated**: 3
**Type Hints Updated**: 5
**Protocol Coverage**: ~35% (partial by design)

**Protocols Used**:
- ✅ ProtocolServiceRegistry: 5 usages (container parameters)
- ⏳ ProtocolOnexEnvelope: Imported but not yet applied (future phase)
- ❌ ProtocolOnexContractData: Not applicable to current public APIs

### Protocols Not Applied (and Why)

**ProtocolOnexEnvelope**:
- Not applied: Event publishing methods are internal, not public APIs
- Reason: Duck typing philosophy - use protocols only for PUBLIC APIs
- Future: May apply if event publishing becomes public API

**ProtocolOnexContractData**:
- Not applied: Contract parameters use specific subcontract types
- Reason: Strong typing preferred for contract validation
- Future: May consider if generic contract handling is needed

**ProtocolContainer** (missing from omnibase_spi v0.1.0):
- Cannot apply: Protocol doesn't exist yet
- Tracked in: Issue #1 above
- Future: Will apply when available in omnibase_spi v0.2.0

### Testing & Validation

**Type Checking**: ✅ Passed
- mypy: No errors (protocols satisfy duck typing)
- pyright: No errors (protocol aliases work correctly)

**Tests**: ✅ All passed (no regressions)
- Unit tests: Passed (backward compatible)
- Integration tests: Passed (concrete types still work)

**Backward Compatibility**: ✅ Confirmed
- Code works with omnibase_spi v0.1.0 (uses protocols)
- Code works without omnibase_spi (uses aliases to concrete types)
- No runtime changes - pure type hint updates

### Documentation Updates

**Files Updated**:
1. `orchestrator/v1_0_0/node.py` - Added protocol docstrings to `__init__` methods
2. `reducer/v1_0_0/node.py` - Added protocol docstrings to `__init__` methods
3. `reducer/v1_0_0/fsm_state_manager.py` - Added protocol docstrings to `__init__`
4. `FOR_OMNIBASE_CORE/OMNIBASE_SPI_COMPLIANCE_ISSUES.md` - Added Phase 2 completion section

**Docstring Pattern**:
```python
Args:
    container: DI container with service resolution (uses ProtocolServiceRegistry
              for duck typing - any object with get_service/register_service methods)

Note:
    Uses ProtocolServiceRegistry protocol type hint for PUBLIC API duck typing.
    Internal implementation still uses concrete ModelONEXContainer from omnibase_core.
    This enables flexibility while maintaining type safety.
```

### Recommendations for Future Work

**Phase 3 (Future)**:
1. Apply ProtocolOnexEnvelope to event publishing methods (if made public)
2. Consider ProtocolOnexContractData for generic contract handling
3. Apply missing protocols when available in omnibase_spi v0.2.0
4. Extend protocol coverage to other nodes (registry, database adapter, etc.)

**Lessons Learned**:
1. **Partial protocol adoption works well** - 35% coverage provides value without over-engineering
2. **Duck typing philosophy is effective** - PUBLIC APIs with protocols, INTERNAL with concrete types
3. **Fallback pattern is essential** - Protocol aliases ensure backward compatibility
4. **Docstrings matter** - Clear documentation helps developers understand protocol usage

### Timeline

- **Phase 1 (Container Bug Fix)**: 2 hours - ✅ COMPLETE
- **Phase 2 (Protocol Type Hints)**: 2 hours - ✅ COMPLETE
- **Total Implementation**: 4 hours

---

## Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-10-30 | 1.0 | Initial compliance issues document | Protocol Analysis Team |
| 2025-10-30 | 1.1 | Added Phase 2 completion (protocol type hints) | Protocol Analysis Team |

---

**End of Document**
