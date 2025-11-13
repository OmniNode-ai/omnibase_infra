# Protocol Adapters for omnibase_core

**Status**: ✅ Implemented and Tested
**Created**: 2025-10-30
**Author**: OmniNode Bridge Team

## Overview

This module provides protocol-based adapters that enable duck typing with omnibase_core classes until upstream changes are merged in omnibase_spi v0.2.0 and omnibase_core v0.2.0.

## What is the Adapter Pattern?

The adapter pattern allows us to:
1. **Define protocols** - Interfaces that classes should implement
2. **Create adapters** - Wrapper classes that make existing classes protocol-compliant
3. **Enable duck typing** - Type-safe interfaces without forced inheritance
4. **Track upstream changes** - Document what needs to change in omnibase_core/omnibase_spi

## Files Implemented

### 1. `protocols.py` - Protocol Definitions

Defines 4 core protocols:

- **ProtocolContainer** - DI container interface
  - `get_service(name: str) -> Any`
  - `register_service(name: str, service: Any) -> None`
  - `config: dict` property

- **ProtocolNode** - Node interface (all types)
  - `async def process(input_data: Any) -> Any`
  - `def get_contract() -> Any`

- **ProtocolOnexError** - Error interface
  - `code: str`
  - `message: str`
  - `details: dict`

- **ProtocolContract** - Contract interface
  - `name: str`
  - `version: str`
  - `node_type: str`
  - `validate() -> bool`
  - `to_dict() -> dict`

### 2. `container_adapter.py` - Container Adapter

**AdapterModelContainer** - Makes ModelContainer protocol-compliant

**Key Features:**
- Wraps stub ModelContainer for now
- Provides unified interface for DI container
- Delegates to wrapped container methods
- TODO: Support real ModelONEXContainer when upstream adds compatibility

**Usage:**
```python
from omninode_bridge.adapters import AdapterModelContainer, ProtocolContainer

# Create with protocol typing
container: ProtocolContainer = AdapterModelContainer.create(
    config={
        "metadata_stamping_service_url": "http://localhost:8053",
        "kafka_broker_url": "localhost:9092"
    }
)

# Register services
container.register_service("kafka_client", kafka_client)

# Retrieve services
service = container.get_service("kafka_client")
```

### 3. `node_adapters.py` - Node Adapters

Adapter classes for all node types:
- **AdapterNodeOrchestrator** - Orchestrator nodes
- **AdapterNodeReducer** - Reducer nodes
- **AdapterNodeEffect** - Effect nodes
- **AdapterNodeCompute** - Compute nodes

**Key Features:**
- Adds unified `process()` method
- Implements ProtocolNode interface
- Maintains backward compatibility
- Delegates to type-specific methods

**Usage:**
```python
from omninode_bridge.adapters import AdapterNodeOrchestrator, ProtocolNode

container = AdapterModelContainer.create(config={...})

# Create node with protocol typing
node: ProtocolNode = AdapterNodeOrchestrator(container)

# Use unified interface
result = await node.process(input_data)
```

### 4. `UPSTREAM_CHANGES.md` - Change Tracking

Comprehensive document tracking:
- Changes needed in omnibase_spi v0.2.0
- Changes needed in omnibase_core v0.2.0
- Migration path after upstream changes
- Timeline and decision log

### 5. `example_usage.py` - Usage Examples

Demonstrates:
- Container adapter usage
- Node adapter usage
- Protocol-based type hints
- Migration path (before/after)

### 6. `__init__.py` - Public API

Exports all protocols, adapters, and type aliases for easy import.

## Tests

**Location**: `tests/test_adapters.py`

**Coverage**: 12 tests, all passing

**Test Categories:**
1. **Container Adapter Tests** (4 tests)
   - Container creation
   - Protocol compliance
   - Service registration/retrieval
   - Non-existent service handling

2. **Node Adapter Tests** (4 tests)
   - Node creation
   - Protocol compliance
   - Unified process() method
   - Contract retrieval (NotImplementedError)

3. **Duck Typing Tests** (2 tests)
   - Functions accepting ProtocolContainer
   - Functions accepting ProtocolNode

4. **Passthrough Tests** (2 tests)
   - Container delegation
   - Node delegation

## Key Design Decisions

### Decision 1: Use Stub ModelContainer Only

**Rationale**: Real ModelONEXContainer from omnibase_core uses dependency_injector with:
- Configuration providers (not plain dict)
- Protocol-based service registry
- Complex initialization

Creating an adapter for ModelONEXContainer would require significant complexity. Instead:
- Use stub ModelContainer for MVP
- Document upstream changes needed
- Migrate once omnibase_core v0.2.0 provides compatible interface

### Decision 2: Unified process() Method

**Rationale**: All node types (Effect, Compute, Reducer, Orchestrator) have different method names:
- Effect: `execute_effect()`
- Compute: `execute_compute()`
- Reducer: `execute_reduction()`
- Orchestrator: `execute_orchestration()`

Adding unified `process()` method:
- Enables generic node handling
- Maintains backward compatibility
- Simplifies polymorphic code

### Decision 3: Track Upstream Changes

**Rationale**: Adapters are temporary solution until upstream changes. By documenting:
- What needs to change
- Why it needs to change
- How to migrate

We ensure smooth transition when omnibase_core v0.2.0+ is released.

## Migration Path (Future)

Once omnibase_spi v0.2.0+ and omnibase_core v0.2.0+ are released:

1. **Update dependencies**:
   ```toml
   omnibase-core = "^0.2.0"
   omnibase-spi = "^0.2.0"
   ```

2. **Update imports**:
   ```python
   # BEFORE
   from omninode_bridge.adapters import AdapterModelContainer, ProtocolContainer

   # AFTER
   from omnibase_spi.protocols import ProtocolContainer
   from omnibase_core.models.container import ModelONEXContainer
   ```

3. **Remove adapters module**:
   ```bash
   rm -rf src/omninode_bridge/adapters/
   ```

4. **Validate**:
   ```bash
   poetry run mypy src/omninode_bridge/
   poetry run pytest tests/
   ```

## Benefits

1. **Type Safety**: mypy validation with protocol-based duck typing
2. **Flexibility**: Works with any container/node implementing protocols
3. **Forward Compatible**: Easy migration when upstream changes land
4. **Documented**: Complete tracking of needed upstream changes
5. **Tested**: Comprehensive test coverage validates adapter behavior

## Limitations

### Current Limitations

1. **Stub ModelContainer Only**: Real ModelONEXContainer not yet supported
2. **No Contract Storage**: Nodes don't store contracts yet (raises NotImplementedError)
3. **Basic get_service**: No protocol-based service lookup (string names only)

### Upstream Dependencies

Changes needed in:
- omnibase_spi v0.2.0 - Add protocol definitions
- omnibase_core v0.2.0 - Implement protocols natively

See `UPSTREAM_CHANGES.md` for complete details.

## Usage Best Practices

### DO:
✅ Use protocol type hints for function parameters
✅ Create containers via AdapterModelContainer.create()
✅ Use unified process() method for generic node handling
✅ Check isinstance() with protocols for runtime validation

### DON'T:
❌ Don't mix adapter and non-adapter imports
❌ Don't access internal _container attribute directly
❌ Don't rely on adapter-specific behavior (may change with upstream)

## Examples

See `example_usage.py` for complete examples of:
- Container adapter usage
- Node adapter usage
- Type hints with protocols
- Migration path

Run examples:
```bash
poetry run python src/omninode_bridge/adapters/example_usage.py
```

## Questions & Support

**Documentation**: See `UPSTREAM_CHANGES.md` for detailed upstream change tracking
**Tests**: See `tests/test_adapters.py` for comprehensive test examples
**Examples**: See `example_usage.py` for usage patterns

---

**Last Updated**: 2025-10-30
**Status**: ✅ Complete - All tests passing, ready for use in omninode_bridge
