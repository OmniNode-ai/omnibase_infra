# Dependency Resolution System

Production-ready dependency resolution for multi-agent coordination in parallel code generation workflows.

## Quick Start

```python
from omninode_bridge.agents.coordination import (
    DependencyResolver,
    Dependency,
    DependencyType,
    SignalCoordinatorStub,  # Replace with real SignalCoordinator when available
)
from omninode_bridge.agents.metrics import MetricsCollector
from omninode_bridge.agents.coordination import ThreadSafeState

# Initialize
signal_coordinator = SignalCoordinatorStub()  # or SignalCoordinator()
metrics = MetricsCollector(kafka_enabled=False, postgres_enabled=False)
state = ThreadSafeState()

resolver = DependencyResolver(
    signal_coordinator=signal_coordinator,
    metrics_collector=metrics,
    state=state,
)

# Create dependency
dependency = Dependency(
    dependency_id="model_gen_complete",
    dependency_type=DependencyType.AGENT_COMPLETION,
    target="agent-model-generator",
    timeout=120,
    metadata={"agent_id": "agent-model-generator"},
)

# Mark agent as completed (simulate)
await signal_coordinator.mark_agent_completed(
    coordination_id="coord-123",
    agent_id="agent-model-generator",
)

# Resolve dependency
result = await resolver.resolve_dependency("coord-123", dependency)
print(f"Success: {result.success}, Duration: {result.duration_ms}ms")
```

## Three Dependency Types

### 1. Agent Completion

Wait for another agent to complete its work.

```python
dependency = Dependency(
    dependency_id="model_complete",
    dependency_type=DependencyType.AGENT_COMPLETION,
    target="agent-model-generator",
    timeout=120,
    metadata={
        "agent_id": "agent-model-generator",
        "completion_event": "completion",  # Optional
        "require_success": True,  # Optional
    },
)
```

**Use Cases**:
- Validator generator waits for model generator
- Test generator waits for validator generator
- Sequential code generation steps

### 2. Resource Availability

Check if a resource (database, API, file) is available.

```python
dependency = Dependency(
    dependency_id="db_available",
    dependency_type=DependencyType.RESOURCE_AVAILABILITY,
    target="database_connection",
    timeout=60,
    metadata={
        "resource_id": "database_connection",
        "resource_type": "database",
        "check_interval_ms": 100,  # Check every 100ms
        "availability_threshold": 1.0,  # 100% available
    },
)

# Mark resource as available
await resolver.mark_resource_available("database_connection", available=True)
```

**Use Cases**:
- Wait for database connection
- Wait for API endpoint availability
- Wait for template files to be loaded
- Wait for external service health

### 3. Quality Gate

Wait for a quality gate to pass a threshold.

```python
dependency = Dependency(
    dependency_id="coverage_passed",
    dependency_type=DependencyType.QUALITY_GATE,
    target="coverage_gate",
    timeout=60,
    metadata={
        "gate_id": "coverage_gate",
        "gate_type": "coverage",
        "threshold": 0.8,  # 80% coverage required
        "check_interval_ms": 500,  # Check every 500ms
    },
)

# Update quality gate score
await resolver.update_quality_gate_score("coverage_gate", score=0.85)
```

**Use Cases**:
- Wait for test coverage threshold
- Wait for linting checks to pass
- Wait for type checking to complete
- Wait for security scans to pass

## Resolve Multiple Dependencies

```python
agent_context = {
    "agent_id": "agent-test-generator",
    "dependencies": [
        {
            "id": "model_complete",
            "type": "agent_completion",
            "target": "agent-model-generator",
            "timeout": 120,
            "metadata": {"agent_id": "agent-model-generator"},
        },
        {
            "id": "validator_complete",
            "type": "agent_completion",
            "target": "agent-validator-generator",
            "timeout": 120,
            "metadata": {"agent_id": "agent-validator-generator"},
        },
        {
            "id": "coverage_passed",
            "type": "quality_gate",
            "target": "coverage_gate",
            "timeout": 60,
            "metadata": {"gate_id": "coverage_gate", "threshold": 0.8},
        },
    ],
}

# Resolve all dependencies sequentially
success = await resolver.resolve_agent_dependencies("coord-123", agent_context)
if success:
    print("All dependencies resolved!")
else:
    print("Dependency resolution failed")
```

## Dependency Status

```python
# Get dependency status
status = resolver.get_dependency_status("coord-123", "model_complete")
print(f"Status: {status['status']}")
print(f"Resolved: {status['resolved']}")
print(f"Type: {status['dependency_type']}")

# Get pending count
pending = resolver.get_pending_dependencies_count("coord-123")
print(f"Pending dependencies: {pending}")
```

## Error Handling

```python
from omninode_bridge.agents.coordination import (
    DependencyTimeoutError,
    DependencyResolutionError,
)

try:
    result = await resolver.resolve_dependency(coordination_id, dependency)
except DependencyTimeoutError as e:
    print(f"Timeout: {e.timeout_seconds}s")
    print(f"Dependency: {e.dependency_id}")
except DependencyResolutionError as e:
    print(f"Resolution failed: {e.error_message}")
```

## Performance

- **Single dependency**: <100ms (immediate resolution)
- **Multiple dependencies**: <2s for 10+ dependencies
- **Scalability**: 100+ dependencies per coordination session
- **Async non-blocking**: All checks use `asyncio.sleep()`

## Configuration

```python
resolver = DependencyResolver(
    signal_coordinator=signal_coordinator,
    metrics_collector=metrics,
    state=state,
    max_concurrent_resolutions=10,  # Max concurrent dependency checks
)
```

## Integration with SignalCoordinator

Currently uses `SignalCoordinatorStub` for testing. When SignalCoordinator (Component 1) is complete:

```python
# Current (stub)
from omninode_bridge.agents.coordination.signal_coordinator_stub import SignalCoordinatorStub
signal_coordinator = SignalCoordinatorStub()

# Future (real implementation)
from omninode_bridge.agents.coordination import SignalCoordinator
signal_coordinator = SignalCoordinator(...)
```

The `ISignalCoordinator` interface is already defined and implemented by both stub and real versions.

## Testing

```bash
# Run all dependency resolution tests
pytest tests/unit/agents/coordination/test_dependency_resolution.py -v

# Run performance tests only
pytest tests/unit/agents/coordination/test_dependency_resolution.py::TestPerformance -v
```

## Architecture

```
DependencyResolver
├── resolve_agent_dependencies()  # Main entry point
│   └── resolve_dependency()       # Resolve single dependency
│       ├── _wait_for_agent_completion()
│       ├── _check_resource_availability()
│       └── _wait_for_quality_gate()
├── mark_resource_available()      # Helper: Update resource status
├── update_quality_gate_score()    # Helper: Update gate score
└── get_dependency_status()        # Query: Get dependency status
```

## Files

- `dependency_models.py`: Data models (Dependency, DependencyType, etc.)
- `dependency_resolution.py`: Main resolver implementation
- `signal_coordinator_stub.py`: Stub for testing (temporary)
- `exceptions.py`: DependencyResolutionError, DependencyTimeoutError

## See Also

- [Implementation Summary](../../../docs/architecture/DEPENDENCY_RESOLUTION_IMPLEMENTATION.md)
- [Thread-Safe State](./thread_safe_state.py)
- [Metrics Collector](../metrics/collector.py)
- [Signal Coordination](./signals.py) (coming soon)
