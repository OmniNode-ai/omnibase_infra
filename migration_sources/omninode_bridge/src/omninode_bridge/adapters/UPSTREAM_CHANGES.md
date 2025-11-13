# Upstream Changes Tracking

This document tracks all changes needed in **omnibase_spi** and **omnibase_core** to eliminate the need for this adapter module.

**Goal**: Enable native protocol compliance so that omninode_bridge can use omnibase_core classes directly with type-safe duck typing.

**Migration Trigger**: Once omnibase_spi v0.2.0+ and omnibase_core v0.2.0+ are released with these changes, this adapter module can be removed.

---

## Changes Needed in omnibase_spi

### v0.2.0 (Next Release) - Protocol Definitions

**Priority**: HIGH (blocks adapter removal)

**Location**: `omnibase_spi/protocols/`

#### 1. Add `ProtocolContainer`

**File**: `omnibase_spi/protocols/protocol_container.py`

```python
from typing import Any, Protocol, runtime_checkable

@runtime_checkable
class ProtocolContainer(Protocol):
    """Protocol for dependency injection containers."""

    def get_service(self, name: str) -> Any: ...
    def register_service(self, name: str, service: Any) -> None: ...

    @property
    def config(self) -> dict: ...
```

**Rationale**: Enables duck typing for any DI container implementation without forcing inheritance.

---

#### 2. Add `ProtocolNode`

**File**: `omnibase_spi/protocols/protocol_node.py`

```python
from typing import Any, Protocol, runtime_checkable

@runtime_checkable
class ProtocolNode(Protocol):
    """Protocol for ONEX nodes (all types)."""

    async def process(self, input_data: Any) -> Any: ...
    def get_contract(self) -> Any: ...
```

**Rationale**: Provides unified interface across all node types (Effect, Compute, Reducer, Orchestrator).

---

#### 3. Add `ProtocolOnexError`

**File**: `omnibase_spi/protocols/protocol_onex_error.py`

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class ProtocolOnexError(Protocol):
    """Protocol for ONEX errors."""

    code: str
    message: str
    details: dict
```

**Rationale**: Standardizes error handling across ONEX ecosystem.

---

#### 4. Add `ProtocolContract`

**File**: `omnibase_spi/protocols/protocol_contract.py`

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class ProtocolContract(Protocol):
    """Protocol for ONEX contracts."""

    name: str
    version: str
    node_type: str

    def validate(self) -> bool: ...
    def to_dict(self) -> dict: ...
```

**Rationale**: Enables contract duck typing for validation and serialization.

---

#### 5. Update `__init__.py`

**File**: `omnibase_spi/protocols/__init__.py`

```python
"""Protocol definitions for ONEX ecosystem."""

from .protocol_container import ProtocolContainer
from .protocol_contract import ProtocolContract
from .protocol_node import ProtocolNode
from .protocol_onex_error import ProtocolOnexError

__all__ = [
    "ProtocolContainer",
    "ProtocolNode",
    "ProtocolOnexError",
    "ProtocolContract",
]
```

---

## Changes Needed in omnibase_core

### v0.2.0 (Next Release) - Protocol Implementation

**Priority**: HIGH (blocks adapter removal)

#### 1. ModelONEXContainer Protocol Compliance

**File**: `omnibase_core/models/container/model_onex_container.py`

**Changes**:

```python
from omnibase_spi.protocols import ProtocolContainer

@runtime_checkable
class ModelONEXContainer(ProtocolContainer):
    """ONEX dependency injection container."""

    # ADD: register_service() method
    def register_service(self, name: str, service: Any) -> None:
        """Register a service in the container.

        Args:
            name: Service identifier
            service: Service instance to register
        """
        if not hasattr(self, '_services'):
            self._services = {}
        self._services[name] = service

    # UPDATE: get_service() signature for duck typing
    def get_service(self, name: str) -> Any:
        """Get service by name (duck typing compatible).

        Args:
            name: Service identifier

        Returns:
            Service instance

        Note:
            For protocol type-based lookup, use get_service_typed()
        """
        return self._services.get(name)

    # KEEP: Existing protocol-based method
    def get_service_typed(self, protocol_type: type[T], service_name: str | None = None) -> T:
        """Get service with protocol type validation.

        This is the existing method for type-safe service lookup.
        """
        # ... existing implementation ...

    # ADD: Explicit config property if not present
    @property
    def config(self) -> dict:
        """Get container configuration."""
        return self._config
```

**Rationale**:
- Makes ModelONEXContainer protocol-compliant
- Preserves existing type-safe `get_service_typed()` method
- Adds duck typing-compatible `get_service()` and `register_service()`

---

#### 2. NodeOrchestrator Protocol Compliance

**File**: `omnibase_core/nodes/node_orchestrator.py`

**Changes**:

```python
from omnibase_spi.protocols import ProtocolNode

@runtime_checkable
class NodeOrchestrator(ProtocolNode):
    """Base class for Orchestrator nodes."""

    # ADD: Unified process() method
    async def process(self, input_data: Any) -> Any:
        """Process input data (unified interface).

        Delegates to execute_orchestration for backward compatibility.

        Args:
            input_data: Input contract or data

        Returns:
            Processed output
        """
        return await self.execute_orchestration(input_data)

    # KEEP: Existing method for backward compatibility
    async def execute_orchestration(self, contract: ModelContractOrchestrator) -> ModelContractOrchestrator:
        """Execute orchestration logic."""
        # ... existing implementation ...

    # ADD: Contract storage and retrieval
    def __init__(self, container: ProtocolContainer):
        super().__init__(container)
        self._contract = None  # Will be set during initialization

    def get_contract(self) -> Any:
        """Retrieve node's contract definition."""
        return self._contract

    def set_contract(self, contract: Any) -> None:
        """Store node's contract definition."""
        self._contract = contract
```

**Rationale**:
- Adds unified `process()` interface for all node types
- Maintains backward compatibility with `execute_orchestration()`
- Enables contract introspection

---

#### 3. NodeReducer Protocol Compliance

**File**: `omnibase_core/nodes/node_reducer.py`

**Changes**: Same pattern as NodeOrchestrator:
- Add `process()` → delegates to `execute_reduction()`
- Add `get_contract()` / `set_contract()`
- Implement `ProtocolNode`

---

#### 4. NodeEffect Protocol Compliance

**File**: `omnibase_core/nodes/node_effect.py`

**Changes**: Same pattern as NodeOrchestrator:
- Add `process()` → delegates to `execute_effect()`
- Add `get_contract()` / `set_contract()`
- Implement `ProtocolNode`

---

#### 5. NodeCompute Protocol Compliance

**File**: `omnibase_core/nodes/node_compute.py`

**Changes**: Same pattern as NodeOrchestrator:
- Add `process()` → delegates to `execute_compute()`
- Add `get_contract()` / `set_contract()`
- Implement `ProtocolNode`

---

#### 6. ModelOnexError Protocol Compliance

**File**: `omnibase_core/errors/model_onex_error.py`

**Changes**:

```python
from omnibase_spi.protocols import ProtocolOnexError

@runtime_checkable
class ModelOnexError(Exception, ProtocolOnexError):
    """ONEX error with protocol compliance."""

    # Ensure these attributes exist
    code: str
    message: str
    details: dict
```

**Rationale**: Makes error handling protocol-compliant.

---

## Migration Path

Once upstream changes are merged:

### Step 1: Update Dependencies

**File**: `pyproject.toml`

```toml
[tool.poetry.dependencies]
omnibase-core = "^0.2.0"  # Changed from ^0.1.0
omnibase-spi = "^0.2.0"   # Changed from ^0.1.0
```

### Step 2: Update Imports

**Search/Replace across codebase:**

```python
# BEFORE (with adapters)
from omninode_bridge.adapters import (
    AdapterModelContainer,
    AdapterNodeOrchestrator,
    AdapterNodeEffect,
    ProtocolContainer,
    ProtocolNode,
)

# AFTER (native omnibase_core)
from omnibase_spi.protocols import (
    ProtocolContainer,
    ProtocolNode,
)
from omnibase_core.models.container.model_onex_container import ModelONEXContainer
from omnibase_core.nodes.node_orchestrator import NodeOrchestrator
from omnibase_core.nodes.node_effect import NodeEffect
```

### Step 3: Update Type Hints

```python
# BEFORE
container: ProtocolContainer = AdapterModelContainer.create(config={...})
node: ProtocolNode = AdapterNodeOrchestrator(container)

# AFTER
container: ProtocolContainer = ModelONEXContainer(config={...})
node: ProtocolNode = NodeOrchestrator(container)
```

### Step 4: Remove Adapter Module

```bash
# Delete adapter module
rm -rf src/omninode_bridge/adapters/

# Update .gitignore if needed
echo "# Adapters removed after omnibase_core v0.2.0 migration" >> .gitignore
```

### Step 5: Validate

```bash
# Run type checking
poetry run mypy src/omninode_bridge/

# Run tests
poetry run pytest tests/

# Verify no adapter imports remain
grep -r "from.*adapters import" src/omninode_bridge/
```

---

## Benefits After Migration

1. **Cleaner Code**: Direct use of omnibase_core classes without adapters
2. **Better Type Safety**: Native protocol support with mypy validation
3. **Reduced Maintenance**: No adapter module to maintain
4. **Ecosystem Consistency**: All omninode projects use same patterns
5. **Performance**: Eliminates adapter overhead (minimal, but present)

---

## Timeline

| Phase | Timeframe | Status |
|-------|-----------|--------|
| Adapter Implementation | 2025-10-30 | ✅ Complete |
| omnibase_spi v0.2.0 Proposal | Week 1 | ⏳ Pending |
| omnibase_core v0.2.0 Proposal | Week 1 | ⏳ Pending |
| Upstream Review & Merge | Weeks 2-4 | ⏳ Pending |
| Release omnibase_spi v0.2.0 | Week 4 | ⏳ Pending |
| Release omnibase_core v0.2.0 | Week 5 | ⏳ Pending |
| Migrate omninode_bridge | Week 6 | ⏳ Pending |
| Remove Adapter Module | Week 6 | ⏳ Pending |

---

## Questions & Decisions

### Q1: Should `process()` replace type-specific methods?

**Decision**: NO - Add `process()` as a unified interface but keep existing methods (`execute_orchestration`, `execute_effect`, etc.) for backward compatibility.

**Rationale**:
- Maintains backward compatibility with existing code
- Allows gradual migration to unified interface
- Both interfaces can coexist

---

### Q2: Should `get_service()` signature change in ModelONEXContainer?

**Current**: `get_service(protocol_type: type[T], service_name: str | None) -> T`

**Proposal**: Add overload for duck typing
```python
@overload
def get_service(self, name: str) -> Any: ...

@overload
def get_service(self, protocol_type: type[T], service_name: str | None = None) -> T: ...
```

**Decision**: RECOMMENDED - Use method overloading to support both patterns.

---

### Q3: Should protocol compliance be enforced at runtime?

**Decision**: YES - Use `@runtime_checkable` decorator on all protocols.

**Rationale**:
- Enables `isinstance()` checks for protocols
- Helps with debugging and validation
- Minimal runtime overhead

---

## Contact & Discussion

**Owner**: OmniNode Bridge Team
**Upstream Repos**:
- [omnibase_spi](https://github.com/omnibase/omnibase_spi) *(placeholder)*
- [omnibase_core](https://github.com/omnibase/omnibase_core) *(placeholder)*

**Discussion**: Create issues in respective repos with title:
- `[RFC] Protocol-based interfaces for duck typing (v0.2.0)`

---

**Last Updated**: 2025-10-30
**Status**: Adapter implementation complete, awaiting upstream proposals
