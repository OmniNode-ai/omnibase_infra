# ONEX Implementation Guide

**Version**: 2.0.0
**Status**: ✅ Canonical Reference
**Last Updated**: 2025-10-01
**Purpose**: Comprehensive guide for implementing ONEX nodes and node groups

---

## 📖 What to Read First

**New to ONEX?** Start here:
1. Read this guide's [Quick Start](#quick-start) section
2. Review [ONEX_QUICK_REFERENCE.md](ONEX_QUICK_REFERENCE.md) for patterns and examples
3. Explore [examples/](examples/) directory for real implementations

**Looking for specific info?**
- **Directory structure** → [Structure Guide](#directory-structure)
- **Base classes** → [Composed Base Classes](#composed-base-classes)
- **Naming patterns** → [Naming Conventions](#naming-conventions)
- **Shared resources** → [SHARED_RESOURCE_VERSIONING.md](SHARED_RESOURCE_VERSIONING.md)
- **Quick patterns** → [ONEX_QUICK_REFERENCE.md](ONEX_QUICK_REFERENCE.md)

---

## 🚀 Quick Start

### 3-Tier Base Class System

ONEX provides **three levels** of base contracts for progressive enhancement:

```python
# Level 1: MINIMAL (advanced use cases)
from omnibase_core.models.contracts import ModelContractEffect

# Level 2: STANDARD (recommended for most nodes) ⭐
from omnibase_core.models.contracts import ModelContractEffectStandard

# Level 3: FULL (complex infrastructure nodes)
from omnibase_core.models.contracts import ModelContractEffectFull
```

**Use Standard for 90% of nodes** - includes common operational patterns like service resolution, health monitoring, performance tracking, and configuration management.

### Creating Your First Node

```python
#!/usr/bin/env python3
"""My Effect Node - Production-ready with standard patterns."""

from pathlib import Path
from omnibase.constants.contract_constants import CONTRACT_FILENAME
from omnibase.core.node_base import NodeBase
from omnibase.core.node_effect import NodeEffect
from omnibase_core.models.core import ModelOnexContainer
from omnibase_core.models.contracts import ModelContractEffectStandard

from .models.model_input_state import ModelMyNodeInputState
from .models.model_output_state import ModelMyNodeOutputState


class ToolMyNodeProcessor(NodeEffect):
    """
    My production Effect node using Standard composition.

    Automatically includes:
    - Service resolution (dependency injection)
    - Health monitoring
    - Performance tracking
    - Configuration management
    - Request/response patterns
    """

    def __init__(self, container: ModelOnexContainer) -> None:
        super().__init__(container)
        # Your initialization here

    async def execute_effect(
        self, contract: ModelContractEffectStandard
    ) -> ModelMyNodeOutputState:
        """Main processing method."""
        # Your business logic here
        pass


def main():
    """One-line main function."""
    return NodeBase(Path(__file__).parent / CONTRACT_FILENAME)


if __name__ == "__main__":
    main()
```

---

## 📁 Directory Structure

### Minimum Viable Structure

**Use this** for new nodes (production-ready):

```
<node_group>/                           # e.g., "canary"
├── __init__.py                         # Group package
├── README.md                           # Main documentation
│
├── deployment/                         # Deployment configs
│   ├── docker-compose.<group>.yml
│   └── *.env files
│
└── <node_name>/                        # e.g., "my_tool"
    ├── __init__.py
    └── v1_0_0/
        ├── __init__.py
        ├── node.py                     # ONLY node class + main()
        │
        ├── contract.yaml               # Main interface
        ├── node_config.yaml            # Runtime config
        ├── deployment_config.yaml      # Deployment config
        │
        ├── contracts/                  # YAML subcontracts
        │   ├── contract_actions.yaml
        │   ├── contract_cli.yaml
        │   ├── contract_examples.yaml
        │   └── contract_models.yaml
        │
        └── models/                     # Node-specific models
            ├── __init__.py
            ├── model_input_state.py
            └── model_output_state.py
```

### Maximum Recommended Structure

**Evolve to this** as needs arise (best practices):

```
<node_group>/                           # e.g., "canary"
│
├── __init__.py
├── README.md
├── API_REFERENCE.md
├── compatibility.yaml                  # Version compatibility matrix
│
├── shared/                             # LAZY: Only when 2+ nodes share
│   ├── models/                         # Independent versioning
│   │   ├── v1/                         # Major version 1 (stable)
│   │   │   ├── __init__.py
│   │   │   └── model_*.py
│   │   └── v2/                         # Major version 2 (breaking changes)
│   │       ├── __init__.py
│   │       └── model_*.py
│   └── protocols/                      # Shared protocols (if needed)
│       ├── v1/
│       └── v2/
│
├── tests/                              # Group-level integration tests
│   ├── integration/
│   │   └── test_node_interactions.py
│   └── fixtures/
│
├── deployment/
│   ├── docker-compose.<group>.yml
│   └── *.env files
│
└── <node_name>/                        # e.g., "my_tool"
    ├── __init__.py
    └── v1_0_0/
        ├── README.md                   # Node documentation
        ├── CHANGELOG.md                # Version history
        ├── node.py                     # ONLY node class + main()
        │
        ├── contract.yaml               # Main interface
        ├── node_config.yaml
        ├── deployment_config.yaml
        ├── state_transitions.yaml      # State machine (if needed)
        ├── workflow_testing.yaml       # Testing workflows (if needed)
        ├── security_config.yaml        # Security (Effect nodes)
        │
        ├── contracts/                  # YAML subcontracts
        │   ├── contract_actions.yaml
        │   ├── contract_cli.yaml
        │   ├── contract_examples.yaml
        │   ├── contract_models.yaml
        │   └── contract_validation.yaml
        │
        ├── models/                     # Node-specific models
        │   ├── __init__.py
        │   ├── model_input_state.py
        │   ├── model_output_state.py
        │   └── enum_*.py
        │
        ├── protocols/                  # Node-specific protocols
        │   ├── __init__.py
        │   └── protocol_<node>.py
        │
        ├── tests/                      # Node unit tests
        │   ├── unit/
        │   │   └── test_node.py
        │   └── fixtures/
        │
        └── mock_configurations/        # Testing mocks (optional)
            ├── event_bus_mock_behaviors.yaml
            ├── llm_mock_responses.yaml
            └── uuid_mock_behaviors.yaml
```

---

## 🎯 Core Principles

### 1. Composed Base Classes (Recommended Approach) ⭐

**Use pre-composed base classes** that aggregate common subcontract patterns:

#### Three Levels of Composition

```
ModelContract{Type} (minimal)           ← Advanced use cases
    ↓
ModelContract{Type}Standard (common)    ← 90% of nodes ⭐
    ↓
ModelContract{Type}Full (complete)      ← Complex infrastructure
```

#### When to Use Each Level

| Level | Use When | Example |
|-------|----------|---------|
| **Minimal** | Custom subcontract composition needed | Specialized nodes |
| **Standard** ⭐ | Most production nodes | API clients, file processors |
| **Full** | Complex infrastructure nodes | Database connectors, message brokers |

**Recommendation**: Start with **Standard**, downgrade to Minimal only if you need custom composition.

### 2. Lazy Promotion for Shared Resources

**Don't create `shared/` upfront**. Follow this progression:

```
Phase 1: Model in node
node_1/v1_0_0/models/model_data.py

Phase 2: Second node needs it → Promote to shared/v1/
shared/models/v1/model_data.py
node_1/v1_0_0/  # updates imports
node_2/v1_0_0/  # uses shared version

Phase 3: Breaking change needed → Create v2
shared/models/v1/model_data.py  # Old version (frozen)
shared/models/v2/model_data.py  # New version (breaking changes)
```

**Promotion Criteria** (ALL must be true):
1. ✅ Actually used by 2+ consumers (not "might be")
2. ✅ Same semantic meaning across consumers
3. ✅ Same version lifecycle requirements
4. ✅ Detected by duplication analysis (not speculative)

See [SHARED_RESOURCE_VERSIONING.md](SHARED_RESOURCE_VERSIONING.md) for complete details.

### 3. Protocols: Hybrid Approach

**Both locations are valid** based on scope:

| Protocol Scope | Location | Example |
|----------------|----------|---------|
| Node-specific | `node/v1_0_0/protocols/` | `protocol_my_node.py` |
| Shared (2+ nodes) | `shared/protocols/v1/` | `protocol_common.py` |
| Framework-wide | `omnibase_core/protocols/` | `ProtocolOnexNode` |

**Start node-local**, promote when actually shared.

### 4. Container Type: ModelOnexContainer Only

**Always use ModelOnexContainer** (proper Pydantic container):

```python
# ✅ CORRECT
from omnibase_core.models.core import ModelOnexContainer

class MyNode(NodeEffect):
    def __init__(self, container: ModelOnexContainer) -> None:
        super().__init__(container)
```

```python
# ❌ WRONG - Legacy technical debt
from omnibase.core.onex_container import ONEXContainer
```

### 5. Node.py Purity

**node.py contains ONLY**:
- ✅ One node class (Effect/Compute/Reducer/Orchestrator)
- ✅ main() function (one-liner)
- ✅ Class-level constants (if needed)
- ❌ NO other classes
- ❌ NO enums (use `models/enum_*.py`)
- ❌ NO helper functions (use separate modules)

### 6. Independent Node Versioning

✅ Each node has `v1_0_0/`, `v2_0_0/`, etc.
✅ Nodes evolve independently
✅ Use `compatibility.yaml` to track which versions work together
❌ NO group-level versioning (breaks independence)

---

## 🎨 Composed Base Classes

### Overview

Pre-composed base classes provide **zero-boilerplate** operational patterns:

```python
# Instead of manually composing subcontracts...
class ModelContractEffect(ModelContractBase):
    service_resolution: ModelServiceResolutionSubcontract | None = None
    health_check: ModelHealthCheckSubcontract | None = None
    performance_monitoring: ModelPerformanceMonitoringSubcontract | None = None
    configuration: ModelConfigurationSubcontract | None = None
    # ... etc (boilerplate!)

# Use pre-composed Standard for common patterns:
class ModelContractEffectStandard(ModelContractEffect):
    # All common subcontracts included with sensible defaults!
```

### Effect Node Compositions

#### ModelContractEffectStandard ⭐
**Common patterns for typical Effect nodes**

```python
from omnibase_core.models.contracts import ModelContractEffectStandard

class ToolProductionAPIClient(NodeEffect):
    """
    Production Effect node with standard operational patterns.

    Automatically includes:
    - Service resolution (dependency injection)
    - Health monitoring
    - Performance tracking
    - Configuration management
    - Request/response patterns
    """
```

**Included Subcontracts**:
- ✅ `service_resolution` - Service discovery and DI
- ✅ `health_check` - Health monitoring
- ✅ `performance_monitoring` - Performance metrics
- ✅ `configuration` - Configuration management
- ✅ `request_response` - Request/response patterns
- Plus inherited: `event_type`, `caching`, `routing`

#### ModelContractEffectFull
**All applicable Effect subcontracts**

```python
from omnibase_core.models.contracts import ModelContractEffectFull

class ToolDatabaseConnector(NodeEffect):
    """
    Complex infrastructure node with all operational capabilities.

    Includes Standard features PLUS:
    - External dependencies tracking
    - Runtime introspection
    - State management
    - FSM patterns
    """
```

**Adds to Standard**:
- ✅ `external_dependencies` - External dependency management
- ✅ `introspection` - Runtime introspection
- ✅ `state_management` - Advanced state management (optional)
- ✅ `fsm` - Finite state machine patterns (optional)

### Compute Node Compositions

#### ModelContractComputeStandard ⭐
**Common patterns for typical Compute nodes**

```python
from omnibase_core.models.contracts import ModelContractComputeStandard

class ToolDataTransformer(NodeCompute):
    """
    Standard Compute node with performance optimization patterns.

    Includes:
    - Caching (critical for pure functions)
    - Performance monitoring
    - Configuration management
    - Health checks
    """
```

**Included Subcontracts**:
- ✅ `caching` - Result caching for pure computations
- ✅ `performance_monitoring` - Computation performance tracking
- ✅ `configuration` - Algorithm configuration
- ✅ `health_check` - Computation health monitoring

#### ModelContractComputeFull
**All applicable Compute subcontracts**

**Adds to Standard**:
- ✅ `service_resolution` - External data source resolution (optional)
- ✅ `request_response` - Request/response patterns (optional)
- ✅ `introspection` - Runtime introspection (optional)

### Reducer Node Compositions

#### ModelContractReducerStandard ⭐
**Common patterns for typical Reducer nodes**

```python
from omnibase_core.models.contracts import ModelContractReducerStandard

class ToolDataAggregator(NodeReducer):
    """
    Standard Reducer node with aggregation patterns.

    Includes:
    - Aggregation (core reducer functionality)
    - State management
    - Caching
    - Performance monitoring
    """
```

**Included Subcontracts**:
- ✅ `aggregation` - Data aggregation strategies
- ✅ `state_management` - Aggregation state management
- ✅ `caching` - Aggregation result caching
- ✅ `performance_monitoring` - Aggregation performance tracking

### Orchestrator Node Compositions

#### ModelContractOrchestratorStandard ⭐
**Common patterns for typical Orchestrator nodes**

```python
from omnibase_core.models.contracts import ModelContractOrchestratorStandard

class ToolWorkflowCoordinator(NodeOrchestrator):
    """
    Standard Orchestrator node with workflow coordination.

    Includes:
    - Workflow coordination (core orchestrator functionality)
    - Routing
    - Service resolution
    - Health monitoring
    - Event coordination
    """
```

**Included Subcontracts**:
- ✅ `workflow_coordination` - Multi-node workflow coordination
- ✅ `routing` - Node routing and load balancing
- ✅ `service_resolution` - Node and service discovery
- ✅ `event_type` - Event-driven orchestration
- ✅ `health_check` - Orchestration health monitoring

#### ModelContractOrchestratorFull
**All applicable Orchestrator subcontracts**

**Adds to Standard**:
- ✅ `fsm` - State machine-based workflow control
- ✅ `state_management` - Workflow state management
- ✅ `performance_monitoring` - Orchestration performance tracking
- ✅ `configuration` - Workflow configuration management

### Subcontract-to-Node Type Matrix

| Subcontract | Effect | Compute | Reducer | Orchestrator |
|------------|--------|---------|---------|--------------|
| **ServiceResolution** | ✅ Standard | Optional | Optional | ✅ Standard |
| **HealthCheck** | ✅ Standard | ✅ Standard | ✅ Standard | ✅ Standard |
| **PerformanceMonitoring** | ✅ Standard | ✅ Standard | ✅ Standard | Optional |
| **Configuration** | ✅ Standard | ✅ Standard | Optional | Optional |
| **RequestResponse** | ✅ Standard | Optional | ❌ | ❌ |
| **Caching** | Optional | ✅ Standard | ✅ Standard | ❌ |
| **EventType** | Optional | ❌ | ❌ | ✅ Standard |
| **Routing** | Optional | ❌ | ❌ | ✅ Standard |
| **WorkflowCoordination** | ❌ | ❌ | ❌ | ✅ Standard |
| **Aggregation** | ❌ | ❌ | ✅ Standard | ❌ |
| **StateManagement** | Optional | ❌ | ✅ Standard | Optional |
| **FSM** | Optional | ❌ | ❌ | Optional |
| **ExternalDependencies** | Optional | Optional | ❌ | ❌ |
| **Introspection** | Optional | Optional | ❌ | ❌ |

**Legend**:
- ✅ Standard: Included in `{Type}Standard` composition
- Optional: Available in `{Type}Full` or can be added manually
- ❌: Not applicable to this node type

---

## 📝 Naming Conventions

### File Naming

| Type | Pattern | Example |
|------|---------|---------|
| Nodes | `node_<name>_<type>.py` | `node_database_writer_effect.py` |
| Models | `model_<name>.py` | `model_task_data.py` |
| Enums | `enum_<name>.py` | `enum_task_status.py` |
| Contracts | `model_contract_<type>.py` | `model_contract_effect.py` |
| Subcontracts | `model_<type>_subcontract.py` | `model_fsm_subcontract.py` |
| Protocols | `protocol_<name>.py` | `protocol_event_bus.py` |

### Class Naming

| Type | Pattern | Example |
|------|---------|---------|
| Nodes | `Node<Name><Type>` | `NodeDatabaseWriterEffect` |
| Models | `Model<Name>` | `ModelTaskData` |
| Enums | `Enum<Name>` | `EnumTaskStatus` |
| Contracts | `ModelContract<Type>` | `ModelContractEffect` |
| Subcontracts | `Model<Type>Subcontract` | `ModelFSMSubcontract` |
| Protocols | `Protocol<Name>` | `ProtocolEventBus` |

**Key Point**: Naming is **SUFFIX-based** - the type comes LAST:
- `NodeDatabaseWriter**Effect**` (not `NodeEffectDatabaseWriter`)
- `node_database_writer_**effect**.py` (not `node_effect_database_writer.py`)

---

## 🔧 Framework Components

### DO NOT DUPLICATE

These are imported from `omnibase_core`:

#### Base Contracts
```python
from omnibase_core.models.contracts import (
    ModelContractBase,
    ModelContractEffect,
    ModelContractCompute,
    ModelContractReducer,
    ModelContractOrchestrator,
)
```

#### Composed Base Contracts ⭐
```python
from omnibase_core.models.contracts import (
    ModelContractEffectStandard,
    ModelContractEffectFull,
    ModelContractComputeStandard,
    ModelContractComputeFull,
    ModelContractReducerStandard,
    ModelContractOrchestratorStandard,
    ModelContractOrchestratorFull,
)
```

#### Subcontracts
```python
from omnibase_core.models.contracts.subcontracts import (
    ModelFSMSubcontract,
    ModelEventTypeSubcontract,
    ModelAggregationSubcontract,
    ModelStateManagementSubcontract,
    ModelRoutingSubcontract,
    ModelCachingSubcontract,
    ModelServiceResolutionSubcontract,
    ModelHealthCheckSubcontract,
    ModelPerformanceMonitoringSubcontract,
    ModelConfigurationSubcontract,
    ModelRequestResponseSubcontract,
    ModelExternalDependenciesSubcontract,
    ModelIntrospectionSubcontract,
    ModelWorkflowCoordinationSubcontract,
)
```

#### Container
```python
from omnibase_core.models.core import ModelOnexContainer
```

---

## ❌ Anti-Patterns

### 1. Premature Shared Resources

```
# ❌ WRONG - Creating shared/ upfront
node_group/
├── shared/
│   └── models/v1/      # Created "just in case"
│       └── model_*.py  # No nodes use it yet

# ✅ CORRECT - Start with node-level models
node_group/
└── node_1/
    └── v1_0_0/
        └── models/
            └── model_*.py  # Only promote when 2+ nodes need it
```

### 2. Using Minimal When Standard Would Work

```python
# ❌ WRONG - Manual composition when Standard exists
from omnibase_core.models.contracts import ModelContractEffect

class MyNode(NodeEffect):
    # Manually adding all standard subcontracts...
    # (boilerplate repetition!)

# ✅ CORRECT - Use Standard composition
from omnibase_core.models.contracts import ModelContractEffectStandard

class MyNode(NodeEffect):
    # All standard subcontracts included automatically!
```

### 3. Multiple Classes in node.py

```python
# ❌ WRONG - Multiple classes in node.py
class MyDataModel(BaseModel):      # Should be in models/
    pass

class MyEnum(Enum):                 # Should be in models/
    pass

class MyNode(NodeEffect):           # Only this should be in node.py
    pass

# ✅ CORRECT - One class in node.py
# node.py
from .models.model_data import MyDataModel
from .models.enum_status import MyEnum

class MyNode(NodeEffect):
    pass
```

### 4. Using Legacy Container

```python
# ❌ WRONG - Legacy container (technical debt)
from omnibase.core.onex_container import ONEXContainer

class MyNode(NodeEffect):
    def __init__(self, container: ONEXContainer) -> None:
        pass

# ✅ CORRECT - Proper Pydantic container
from omnibase_core.models.core import ModelOnexContainer

class MyNode(NodeEffect):
    def __init__(self, container: ModelOnexContainer) -> None:
        pass
```

### 5. Group-Level Versioning

```
# ❌ WRONG - Version at group level
node_group/
└── v1_0_0/         # Breaks independent node evolution
    ├── node_1/
    └── node_2/

# ✅ CORRECT - Version per node
node_group/
├── node_1/
│   └── v1_0_0/    # Independent versioning
└── node_2/
    └── v2_0_0/    # Can be different version
```

### 6. Premature Protocol Promotion

```
# ❌ WRONG - Moving protocol to omnibase_core prematurely
omnibase_core/protocols/
└── protocol_experimental.py  # Only one node uses it!

# ✅ CORRECT - Keep in node until actually shared
node/v1_0_0/protocols/
└── protocol_experimental.py  # Promote when 2+ nodes need it
```

---

## 🔄 Migration Strategy

### From Current to Best Practices

**Phase 1: Adopt Composed Base Classes**
1. ✅ Update imports to use `ModelContract{Type}Standard`
2. ✅ Remove manual subcontract composition
3. ✅ Update contract YAML with `composed_type: "standard"`
4. ✅ Test that all subcontracts work as expected

**Phase 2: Add Structure**
1. 🆕 Add `compatibility.yaml` at group level
2. 🆕 Add `README.md` + `CHANGELOG.md` per node
3. 🆕 Add `tests/` directories (explicit structure)
4. 🆕 Add node-level documentation

**Phase 3: Lazy Promotion (when needed)**
1. ⏸️ Monitor for duplicate models/protocols
2. ⏸️ Create `shared/models/v1/` when 2nd node needs it
3. ⏸️ Create `shared/models/v2/` when breaking changes needed
4. ⏸️ Promote protocols to `omnibase_core` when truly framework-wide

### Migration Scripts

**Required tooling**:
1. `scripts/detect_duplicate_models.py` - Find models to promote
2. `scripts/validate_compatibility.py` - Check version matrix
3. `scripts/generate_docs.py` - Auto-gen from contracts
4. `scripts/migrate_to_standard.py` - Convert to Standard base classes

---

## 🛠️ Tooling Support

### Duplication Detection
```bash
# Find models/protocols that should be promoted
python scripts/detect_duplicate_models.py --group canary
```

### Compatibility Validation
```bash
# Validate version compatibility matrix
python scripts/validate_compatibility.py --all
```

### Documentation Generation
```bash
# Auto-generate API_REFERENCE.md from contracts
python scripts/generate_docs.py --group canary --format markdown
```

### CLI Tools
```bash
# Create new node with Standard template
onex create node --type effect --template standard --name my_tool

# Validate node structure
onex validate node --path ./my_tool/v1_0_0/

# Promote model to shared
onex promote model --model model_data --to shared/v1/
```

---

## 📚 References

**Related Documentation**:
- [ONEX_QUICK_REFERENCE.md](ONEX_QUICK_REFERENCE.md) - Patterns and examples
- [SHARED_RESOURCE_VERSIONING.md](SHARED_RESOURCE_VERSIONING.md) - Versioning strategy
- [examples/](examples/) - Real implementation examples

**Framework Code**:
- `omnibase_core/models/contracts/` - Base and composed contracts
- `omnibase_core/models/contracts/subcontracts/` - All subcontracts
- `omnibase_core/models/core/` - ModelOnexContainer
- `omnibase_core/protocols/` - Framework-wide protocols

**Reference Implementations**:
- `omnibase_3/src/omnibase/tools/canary/` - Production node group

---

## 📋 Quick Checklist

### Creating a New Node

- [ ] Choose correct node type (Effect/Compute/Reducer/Orchestrator)
- [ ] Use **Standard** composed base class (not minimal)
- [ ] Follow file naming: `node_<name>_<type>.py`
- [ ] Follow class naming: `Node<Name><Type>`
- [ ] Use ModelOnexContainer (not ONEXContainer)
- [ ] One class per node.py (no enums, no helpers)
- [ ] Create contract.yaml with all required fields
- [ ] Keep models node-local (promote when 2+ nodes need)
- [ ] Keep protocols node-local (promote when actually shared)
- [ ] Add comprehensive tests
- [ ] Document in README.md and CHANGELOG.md

### Before Promoting to Shared

- [ ] Actually used by 2+ nodes (not "might be")
- [ ] Same semantic meaning across consumers
- [ ] Same version lifecycle requirements
- [ ] Detected by duplication analysis
- [ ] Use `shared/models/v1/` (major version only)
- [ ] Update imports in all consuming nodes
- [ ] Add tests for shared resource

---

**Status**: ✅ Canonical Reference
**Version**: 2.0.0
**Last Updated**: 2025-10-01
**Key Feature**: Composed base classes (Standard/Full) as recommended approach
