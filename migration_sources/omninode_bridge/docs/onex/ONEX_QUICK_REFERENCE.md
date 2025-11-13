# ONEX Quick Reference

**Version**: 2.0.0
**Purpose**: Fast lookup for patterns, naming, and templates
**For comprehensive guide**: See [ONEX_GUIDE.md](ONEX_GUIDE.md)

---

## 🏗️ 4-Node Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   EFFECT    │───▶│   COMPUTE   │───▶│   REDUCER   │───▶│ORCHESTRATOR │
│   (Input)   │    │ (Process)   │    │(Aggregate)  │    │(Coordinate) │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

**Unidirectional Data Flow**: EFFECT → COMPUTE → REDUCER → ORCHESTRATOR

---

## 📝 Naming Conventions (SUFFIX-based)

### Files
- Nodes: `node_<name>_<type>.py` → `node_database_writer_effect.py`
- Models: `model_<name>.py` → `model_task_data.py`
- Enums: `enum_<name>.py` → `enum_task_status.py`
- Protocols: `protocol_<name>.py` → `protocol_event_bus.py`

### Classes
- Nodes: `Node<Name><Type>` → `NodeDatabaseWriterEffect`
- Models: `Model<Name>` → `ModelTaskData`
- Enums: `Enum<Name>` → `EnumTaskStatus`
- Protocols: `Protocol<Name>` → `ProtocolEventBus`

**Key**: Type comes LAST (suffix-based), not first!

---

## 🎨 Node Types Quick Reference

### NodeEffect
**Purpose**: Side effects (I/O, database, API calls)
**Method**: `async def execute_effect(self, contract: ModelContractEffect) -> Any`
**Use for**: Database writes, file operations, API calls, event publishing

### NodeCompute
**Purpose**: Pure computations (no side effects)
**Method**: `async def execute_compute(self, contract: ModelContractCompute) -> Any`
**Use for**: Data transformations, calculations, filtering, validation

### NodeReducer
**Purpose**: Data aggregation and state reduction
**Method**: `async def execute_reduction(self, contract: ModelContractReducer) -> Any`
**Use for**: Aggregation, statistics, state merging, report generation

### NodeOrchestrator
**Purpose**: Workflow coordination
**Method**: `async def execute_orchestration(self, contract: ModelContractOrchestrator) -> Any`
**Use for**: Multi-step workflows, pipeline coordination, dependency resolution

---

## 🚀 Quick Start Templates

### Effect Node (Standard) ⭐

```python
#!/usr/bin/env python3
"""My Effect Node."""

from pathlib import Path
from omnibase.constants.contract_constants import CONTRACT_FILENAME
from omnibase.core.node_base import NodeBase
from omnibase.core.node_effect import NodeEffect
from omnibase_core.models.core import ModelOnexContainer
from omnibase_core.models.contracts import ModelContractEffectStandard

from .models.model_input_state import ModelMyInputState
from .models.model_output_state import ModelMyOutputState


class ToolMyProcessor(NodeEffect):
    """
    My Effect node with standard operational patterns.

    Includes: service resolution, health check, performance monitoring,
    configuration, request/response patterns.
    """

    def __init__(self, container: ModelOnexContainer) -> None:
        super().__init__(container)

    async def execute_effect(
        self, contract: ModelContractEffectStandard
    ) -> ModelMyOutputState:
        """Execute side effect with transaction support."""
        async with self.transaction_manager.begin():
            result = await self._perform_operation(contract)
            return ModelMyOutputState(success=True, data=result)


def main():
    return NodeBase(Path(__file__).parent / CONTRACT_FILENAME)


if __name__ == "__main__":
    main()
```

### Compute Node (Standard)

```python
#!/usr/bin/env python3
"""My Compute Node."""

from omnibase.core.node_compute import NodeCompute
from omnibase_core.models.core import ModelOnexContainer
from omnibase_core.models.contracts import ModelContractComputeStandard

from .models.model_input_state import ModelMyInputState
from .models.model_output_state import ModelMyOutputState


class ToolMyTransform(NodeCompute):
    """
    My Compute node with caching and performance monitoring.

    Includes: caching, performance monitoring, configuration, health check.
    """

    def __init__(self, container: ModelOnexContainer) -> None:
        super().__init__(container)

    async def execute_compute(
        self, contract: ModelContractComputeStandard
    ) -> ModelMyOutputState:
        """Execute computation with caching."""
        # Check cache
        cached = self.cache.get(contract.cache_key)
        if cached:
            return cached

        # Compute
        result = self._transform_data(contract.input_data)

        # Cache result
        self.cache.set(contract.cache_key, result)
        return ModelMyOutputState(success=True, data=result)


def main():
    return NodeBase(Path(__file__).parent / CONTRACT_FILENAME)


if __name__ == "__main__":
    main()
```

### Reducer Node (Standard)

```python
#!/usr/bin/env python3
"""My Reducer Node."""

from omnibase.core.node_reducer import NodeReducer
from omnibase_core.models.core import ModelOnexContainer
from omnibase_core.models.contracts import ModelContractReducerStandard

from .models.model_input_state import ModelMyInputState
from .models.model_output_state import ModelMyOutputState


class ToolMyAggregator(NodeReducer):
    """
    My Reducer node with aggregation patterns.

    Includes: aggregation, state management, caching, performance monitoring.
    """

    def __init__(self, container: ModelOnexContainer) -> None:
        super().__init__(container)

    async def execute_reduction(
        self, contract: ModelContractReducerStandard
    ) -> ModelMyOutputState:
        """Execute reduction with aggregation."""
        aggregated_data = []

        # Process stream
        async for item in contract.input_stream:
            aggregated_data.append(self._process_item(item))

        # Aggregate
        result = self._aggregate(aggregated_data)
        return ModelMyOutputState(success=True, data=result)


def main():
    return NodeBase(Path(__file__).parent / CONTRACT_FILENAME)


if __name__ == "__main__":
    main()
```

### Orchestrator Node (Standard)

```python
#!/usr/bin/env python3
"""My Orchestrator Node."""

from omnibase.core.node_orchestrator import NodeOrchestrator
from omnibase_core.models.core import ModelOnexContainer
from omnibase_core.models.contracts import ModelContractOrchestratorStandard

from .models.model_input_state import ModelMyInputState
from .models.model_output_state import ModelMyOutputState


class ToolMyCoordinator(NodeOrchestrator):
    """
    My Orchestrator node with workflow coordination.

    Includes: workflow coordination, routing, service resolution,
    event handling, health monitoring.
    """

    def __init__(self, container: ModelOnexContainer) -> None:
        super().__init__(container)

    async def execute_orchestration(
        self, contract: ModelContractOrchestratorStandard
    ) -> ModelMyOutputState:
        """Execute orchestration with workflow coordination."""
        # Coordinate workflow
        results = await self._coordinate_workflow(contract)
        return ModelMyOutputState(success=True, data=results)


def main():
    return NodeBase(Path(__file__).parent / CONTRACT_FILENAME)


if __name__ == "__main__":
    main()
```

---

## 📦 Base Class Levels

### Choose Your Level

```python
# MINIMAL (advanced use cases - custom composition)
from omnibase_core.models.contracts import ModelContractEffect

# STANDARD (90% of nodes - recommended) ⭐
from omnibase_core.models.contracts import ModelContractEffectStandard

# FULL (complex infrastructure nodes)
from omnibase_core.models.contracts import ModelContractEffectFull
```

### What Each Level Includes

| Feature | Minimal | Standard | Full |
|---------|---------|----------|------|
| **Basic contract fields** | ✅ | ✅ | ✅ |
| **Service resolution** | ❌ | ✅ | ✅ |
| **Health monitoring** | ❌ | ✅ | ✅ |
| **Performance tracking** | ❌ | ✅ | ✅ |
| **Configuration mgmt** | ❌ | ✅ | ✅ |
| **Request/response** | ❌ | ✅ | ✅ |
| **External dependencies** | ❌ | ❌ | ✅ |
| **Runtime introspection** | ❌ | ❌ | ✅ |
| **State management** | ❌ | ❌ | ✅ |
| **FSM patterns** | ❌ | ❌ | ✅ |

---

## 📄 Contract YAML Template

### Minimal Contract
```yaml
name: my_operation
version: 1.0.0
description: "My operation description"
node_type: EFFECT  # or COMPUTE, REDUCER, ORCHESTRATOR

io_operations:
  - operation_type: "file_write"
    path: "/data/output.json"
```

### Standard Contract
```yaml
name: my_operation
version: 1.0.0
description: "My operation description"
node_type: EFFECT
composed_type: "standard"  # ⭐ Triggers Standard composition

io_operations:
  - operation_type: "api_call"
    endpoint: "${SERVICE_URL}/api/data"

service_resolution:
  service_name: "data_api"
  discovery_method: "dns"

health_check:
  endpoint: "/health"
  interval_seconds: 30

performance_monitoring:
  enable_metrics: true
  sample_rate: 1.0
```

### Full Orchestrator Contract
```yaml
name: my_workflow
version: 1.0.0
description: "Complex workflow orchestration"
node_type: ORCHESTRATOR
composed_type: "full"  # ⭐ All capabilities

workflow_coordination:
  max_concurrent_workflows: 100
  execution_timeout_seconds: 300

routing:
  strategy: "round_robin"
  load_balancing: true

fsm:
  initial_state: "pending"
  states:
    - name: "pending"
      transitions: ["processing"]
    - name: "processing"
      transitions: ["completed", "failed"]
```

---

## 🎯 Decision Trees

### Which Base Class Level?

```
Need custom subcontract composition?
  YES → Use MINIMAL
  NO  → ↓

Is this a complex infrastructure node?
  YES → Use FULL (e.g., database connector, message broker)
  NO  → ↓

Use STANDARD ⭐ (90% of nodes)
```

### Where Should This Resource Live?

```
Is it a MODEL or PROTOCOL?
  ↓
Used by 2+ nodes?
  NO  → Keep in node/v1_0_0/models/ or protocols/
  YES → Same semantic meaning?
          NO  → Keep separate
          YES → ↓

Used by nodes in SAME group?
  YES → Promote to shared/models/v1/ or protocols/v1/
  NO  → Used by nodes in DIFFERENT groups?
          YES → Promote to project/shared/models/v1/
```

### Protocol Location?

```
Node-specific? → node/v1_0_0/protocols/
Shared (2+ nodes)? → shared/protocols/v1/
Framework-wide? → omnibase_core/protocols/
```

---

## ✅ Best Practices Checklist

### Creating a New Node

- [ ] Choose correct node type (Effect/Compute/Reducer/Orchestrator)
- [ ] Use **Standard** base class (not Minimal unless needed)
- [ ] Use `ModelOnexContainer` (not ONEXContainer)
- [ ] Follow SUFFIX-based naming (`Node<Name><Type>`)
- [ ] One class per node.py (no enums, no helpers)
- [ ] Create contract.yaml with `composed_type: "standard"`
- [ ] Keep models node-local initially (lazy promotion)
- [ ] Keep protocols node-local initially (lazy promotion)
- [ ] Implement proper error handling and logging
- [ ] Add UUID correlation tracking
- [ ] Write comprehensive tests

### Before Promoting to Shared

- [ ] Resource used by 2+ nodes (not "might be")
- [ ] Same semantic meaning across consumers
- [ ] Duplication detected by tooling
- [ ] Use `shared/models/v1/` or `shared/protocols/v1/`
- [ ] Update imports in all consuming nodes
- [ ] Add tests for shared resource

### Contract YAML

- [ ] Include `name`, `version`, `description`, `node_type`
- [ ] Add `composed_type: "standard"` for Standard base class
- [ ] Define all required subcontracts
- [ ] Use environment variables for config (`${VAR_NAME}`)
- [ ] Add validation rules if needed
- [ ] Document expected inputs/outputs

---

## 🚫 Common Mistakes

### ❌ Don't
- Use Minimal when Standard would work
- Use ONEXContainer (legacy)
- Put multiple classes in node.py
- Create shared/ upfront
- Use semantic versioning for shared resources (v1_0_0 → use v1, v2)
- Promote resources prematurely
- Break unidirectional flow
- Skip contract validation

### ✅ Do
- Use Standard for 90% of nodes
- Use ModelOnexContainer
- One class per node.py
- Lazy promotion to shared/
- Major versioning for shared (v1, v2, v3)
- Promote when 2+ nodes actually need it
- Follow unidirectional data flow
- Validate contracts

---

## 📚 Quick Imports Reference

### Standard Node Imports
```python
# Node bases
from omnibase.core.node_effect import NodeEffect
from omnibase.core.node_compute import NodeCompute
from omnibase.core.node_reducer import NodeReducer
from omnibase.core.node_orchestrator import NodeOrchestrator

# Standard contracts (recommended) ⭐
from omnibase_core.models.contracts import (
    ModelContractEffectStandard,
    ModelContractComputeStandard,
    ModelContractReducerStandard,
    ModelContractOrchestratorStandard,
)

# Container
from omnibase_core.models.core import ModelOnexContainer

# Node base for main()
from omnibase.core.node_base import NodeBase
from omnibase.constants.contract_constants import CONTRACT_FILENAME
```

### Full Node Imports
```python
# Full contracts (complex infrastructure)
from omnibase_core.models.contracts import (
    ModelContractEffectFull,
    ModelContractComputeFull,
    ModelContractOrchestratorFull,
)
```

### Minimal Node Imports
```python
# Minimal contracts (custom composition)
from omnibase_core.models.contracts import (
    ModelContractEffect,
    ModelContractCompute,
    ModelContractReducer,
    ModelContractOrchestrator,
)

# Individual subcontracts
from omnibase_core.models.contracts.subcontracts import (
    ModelFSMSubcontract,
    ModelEventTypeSubcontract,
    ModelCachingSubcontract,
    # ... etc
)
```

---

## 🔗 Related Documentation

- **[ONEX_GUIDE.md](ONEX_GUIDE.md)** - Comprehensive implementation guide
- **[SHARED_RESOURCE_VERSIONING.md](SHARED_RESOURCE_VERSIONING.md)** - Versioning strategy
- **[examples/](examples/)** - Real implementation examples

---

**Status**: ✅ Quick Reference
**Version**: 2.0.0
**Last Updated**: 2025-10-01
