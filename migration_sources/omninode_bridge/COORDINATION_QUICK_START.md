# Coordination System Quick Start Guide

**Phase 4 Weeks 3-4 Integration - Complete**

## 🚀 Quick Start

### Basic Usage (5 minutes)

```python
from omninode_bridge.agents.coordination import (
    CoordinationOrchestrator,
    ThreadSafeState,
)
from omninode_bridge.agents.metrics import MetricsCollector

# 1. Setup
state = ThreadSafeState()
metrics = MetricsCollector()
await metrics.start()

# 2. Create orchestrator
orchestrator = CoordinationOrchestrator(
    state=state,
    metrics_collector=metrics
)

# 3. Coordinate workflow
result = await orchestrator.coordinate_workflow(
    workflow_id="my-workflow",
    agent_assignments={
        "agent-1": {
            "objective": "Generate models",
            "tasks": ["parse", "generate"],
        }
    }
)

# 4. Check results
print(f"✓ Workflow completed in {result['duration_ms']}ms")
print(f"✓ Contexts distributed: {result['contexts_distributed']}")
```

## 📦 What's Integrated

The `CoordinationOrchestrator` integrates **4 coordination components**:

1. **SignalCoordinator** - Agent communication
2. **SmartRoutingOrchestrator** - Task routing
3. **ContextDistributor** - Context packaging
4. **DependencyResolver** - Dependency resolution

## 🎯 Common Use Cases

### 1. Simple Workflow (No Dependencies)

```python
result = await orchestrator.coordinate_workflow(
    workflow_id="simple-workflow",
    agent_assignments={
        "agent-1": {"objective": "Task 1", "tasks": ["subtask-1"]},
        "agent-2": {"objective": "Task 2", "tasks": ["subtask-2"]},
    }
)
```

### 2. Workflow with Dependencies

```python
result = await orchestrator.coordinate_workflow(
    workflow_id="dependent-workflow",
    agent_assignments={
        "model-gen": {
            "objective": "Generate models",
            "tasks": ["parse", "generate"],
        },
        "validator-gen": {
            "objective": "Generate validators",
            "tasks": ["validate"],
            "dependencies": ["model-gen"],  # Waits for model-gen
        }
    }
)
```

### 3. Workflow with Shared Intelligence

```python
result = await orchestrator.coordinate_workflow(
    workflow_id="intelligent-workflow",
    agent_assignments={
        "agent-1": {...},
        "agent-2": {...},
    },
    shared_intelligence={
        "patterns": ["singleton", "factory"],
        "conventions": {"naming": "snake_case"},
    }
)
```

### 4. Get Agent Context

```python
context = await orchestrator.get_agent_context(
    coordination_id=result["coordination_id"],
    agent_id="agent-1"
)

print(f"Tasks: {context.assignment.tasks}")
print(f"Objective: {context.assignment.objective}")
```

### 5. Signal Agent Completion

```python
await orchestrator.signal_agent_completion(
    coordination_id=result["coordination_id"],
    agent_id="agent-1",
    result_summary={
        "status": "completed",
        "items_generated": 5,
        "quality_score": 0.95,
    }
)
```

### 6. Check Dependency Status

```python
is_resolved = await orchestrator.check_dependency_status(
    coordination_id=result["coordination_id"],
    dependency_id="model-gen-complete"
)
```

### 7. Get Coordination Metrics

```python
metrics = orchestrator.get_coordination_metrics(
    coordination_id=result["coordination_id"]
)

print(f"Signals sent: {metrics['signal_metrics']['total_signals_sent']}")
print(f"Contexts distributed: {metrics['context_metrics']['contexts_distributed']}")
```

## 🔧 Configuration Options

### Enable/Disable Components

```python
orchestrator = CoordinationOrchestrator(
    state=state,
    metrics_collector=metrics,
    enable_routing=True,  # Enable smart routing
    enable_dependency_resolution=True,  # Enable dependencies
)
```

### With Agent Registry (for Routing)

```python
from omninode_bridge.agents.registry import AgentRegistry

registry = AgentRegistry(state=state, metrics_collector=metrics)

orchestrator = CoordinationOrchestrator(
    state=state,
    metrics_collector=metrics,
    agent_registry=registry,  # Required for routing
    enable_routing=True,
)
```

## 📊 Performance Targets

| Component | Target | Status |
|-----------|--------|--------|
| Signal Propagation | <100ms | ✅ |
| Context Distribution | <200ms/agent | ✅ |
| Dependency Resolution | <2s total | ✅ |
| Routing Decision | <5ms | ✅ |
| **Full Workflow** | **<2s** | **✅** |

## 🧪 Example Workflows

### Code Generation Workflow

```python
result = await orchestrator.coordinate_workflow(
    workflow_id="codegen-session-1",
    agent_assignments={
        "model-gen": {
            "objective": "Generate Pydantic models",
            "tasks": ["parse_contract", "generate_models"],
            "input_data": {"contract_path": "./contract.yaml"}
        },
        "validator-gen": {
            "objective": "Generate validators",
            "tasks": ["generate_validators"],
            "dependencies": ["model-gen"],
        },
        "test-gen": {
            "objective": "Generate tests",
            "tasks": ["generate_tests"],
            "dependencies": ["model-gen", "validator-gen"],
        }
    },
    shared_intelligence={
        "patterns": ["singleton", "factory"],
        "conventions": {
            "naming": "snake_case",
            "max_line_length": 88,
        }
    }
)
```

## 🎓 Advanced Usage

### Direct Component Access

```python
# Access components directly if needed
signal_coordinator = orchestrator.signal_coordinator
context_distributor = orchestrator.context_distributor
dependency_resolver = orchestrator.dependency_resolver
routing_orchestrator = orchestrator.routing_orchestrator
```

### Custom Resource Allocation

```python
from omninode_bridge.agents.coordination import ResourceAllocation

orchestrator = CoordinationOrchestrator(
    state=state,
    metrics_collector=metrics,
    default_resource_allocation=ResourceAllocation(
        max_memory_mb=1024,
        max_cpu_cores=4,
        max_execution_time_seconds=300,
    )
)
```

### Subscribe to Signals

```python
async for signal in orchestrator.signal_coordinator.subscribe_to_signals(
    coordination_id=result["coordination_id"],
    agent_id="my-agent",
    signal_types=["agent_completed", "dependency_resolved"]
):
    print(f"Received signal: {signal.signal_type}")
```

## 📚 Full Example

See complete working example:
- **File**: `examples/coordination_integration_example.py`
- **Run**: `python examples/coordination_integration_example.py`

## 🔍 Key Files

- **Orchestrator**: `src/omninode_bridge/agents/coordination/orchestrator.py`
- **Signals**: `src/omninode_bridge/agents/coordination/signals.py`
- **Routing**: `src/omninode_bridge/agents/coordination/routing.py`
- **Context**: `src/omninode_bridge/agents/coordination/context_distribution.py`
- **Dependencies**: `src/omninode_bridge/agents/coordination/dependency_resolution.py`

## ✅ Success Criteria

- [x] All 4 components integrated
- [x] 183/183 tests passing
- [x] No circular dependencies
- [x] Clean public API
- [x] Integration example provided
- [x] ONEX v2.0 compliant
- [x] Performance targets met

## 🆘 Troubleshooting

### Import Error
```python
# Make sure to use the full import path
from omninode_bridge.agents.coordination import CoordinationOrchestrator
```

### Routing Not Available
```python
# Routing requires agent registry
registry = AgentRegistry(state=state, metrics_collector=metrics)
orchestrator = CoordinationOrchestrator(
    state=state,
    metrics_collector=metrics,
    agent_registry=registry,  # Required!
    enable_routing=True,
)
```

### Dependency Resolution Timeout
```python
# Adjust timeout in dependency configuration
dependency = Dependency(
    dependency_id="my-dep",
    dependency_type=DependencyType.AGENT_COMPLETION,
    target="agent-1",
    timeout=300,  # 5 minutes instead of default 2 minutes
)
```

## 📖 Further Reading

- **Integration Summary**: `COORDINATION_INTEGRATION_SUMMARY.md`
- **Signal System Guide**: `docs/architecture/COORDINATION_SIGNAL_SYSTEM.md`
- **Routing Guide**: `docs/guides/ROUTING_ORCHESTRATION_GUIDE.md`
- **Context Guide**: `docs/guides/CONTEXT_DISTRIBUTION_GUIDE.md`
- **Dependency Guide**: `docs/architecture/DEPENDENCY_RESOLUTION_IMPLEMENTATION.md`

## 🎉 Ready to Use!

The coordination system is **production-ready** and fully integrated. Start with the basic usage above and explore the examples for more advanced scenarios.

**Next Steps**:
1. Try the basic example above
2. Run `python examples/coordination_integration_example.py`
3. Explore the integration summary for detailed documentation
4. Check component-specific guides for advanced features
