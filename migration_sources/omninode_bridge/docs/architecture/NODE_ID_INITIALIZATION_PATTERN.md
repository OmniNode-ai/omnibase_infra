# Node ID Initialization Pattern - ONEX v2.0 Bridge Nodes

## Summary

All ONEX bridge nodes properly initialize `node_id` before introspection system initialization. This document describes the verified pattern and best practices.

## ✅ Verified Pattern (All Nodes Compliant)

### Correct Initialization Order

```python
class NodeBridgeOrchestrator(NodeOrchestrator, HealthCheckMixin, NodeIntrospectionMixin):
    def __init__(self, container: ModelONEXContainer) -> None:
        # 1. FIRST: Call parent constructor (sets node_id)
        super().__init__(container)

        # 2. Node-specific configuration
        self.metadata_stamping_service_url = container.config.get(...)
        self.kafka_broker_url = container.config.get(...)

        # 3. Initialize health checks (optional, doesn't require node_id)
        self.initialize_health_checks()

        # 4. Initialize introspection (uses node_id from parent)
        self.initialize_introspection()

        # 5. Logging (safe to use node_id here)
        emit_log_event(
            LogLevel.INFO,
            "NodeBridgeOrchestrator initialized successfully",
            {"node_id": self.node_id}  # ✅ Available from parent
        )
```

## Implementation Status

### ✅ NodeBridgeOrchestrator (`orchestrator/v1_0_0/node.py`)
- **Line 90**: `super().__init__(container)` - Sets `node_id` from `NodeOrchestrator` parent
- **Line 136**: `self.initialize_health_checks()`
- **Line 139**: `self.initialize_introspection()` - Uses `node_id` from parent
- **Line 145**: `self.node_id` safely used in logging
- **Status**: ✅ Correct pattern implemented

### ✅ NodeBridgeReducer (`reducer/v1_0_0/node.py`)
- **Line 105**: `super().__init__(container)` - Sets `node_id` from `NodeReducer` parent
- **Line 117**: `self.initialize_health_checks()`
- **Line 120**: `self.initialize_introspection()` - Uses `node_id` from parent
- **Line 417**: Uses defensive `getattr(self, 'node_id', 'unknown')` in startup logging
- **Status**: ✅ Correct pattern implemented

### ✅ NodeBridgeRegistry (`registry/v1_0_0/node.py`)
- **Line 104**: `super().__init__(container)` - Sets `node_id` from `NodeEffect` parent
- **Line 157**: `self.initialize_health_checks()`
- **Line 163**: `self.node_id` safely used in logging
- **Note**: Does NOT use `NodeIntrospectionMixin` (by design - no introspection needed)
- **Status**: ✅ Correct pattern implemented (no introspection required)

## How node_id is Set

### Parent Class Initialization

Each ONEX node type sets `node_id` in its parent constructor:

```python
# From omnibase_core.infrastructure (or stubs)

class NodeOrchestrator:
    def __init__(self, container: ModelONEXContainer):
        self.node_id = str(uuid4())  # or from container config
        self.container = container
        # ... rest of initialization

class NodeReducer:
    def __init__(self, container: ModelONEXContainer):
        self.node_id = str(uuid4())  # or from container config
        self.container = container
        # ... rest of initialization

class NodeEffect:
    def __init__(self, container: ModelONEXContainer):
        self.node_id = str(uuid4())  # or from container config
        self.container = container
        # ... rest of initialization
```

### Introspection Mixin Requirements

From `mixins/introspection_mixin.py` (line 55-63):

```python
"""
Requirements:
- Node must have `node_id` attribute (string)
- Node must have `container` attribute (ModelONEXContainer)
- Node must have `node_type` property or attribute (string: effect/compute/reducer/orchestrator)

Optional:
- `get_health_status()` method for health integration
- `workflow_fsm_states` dict for FSM state introspection
- `stamping_metrics` dict for performance metrics
"""
```

### Defensive Programming in Introspection Mixin

The introspection mixin uses defensive patterns throughout:

```python
# Line 225 in publish_introspection()
node_id = getattr(self, "node_id", str(uuid4()))

# Line 524 in _publish_heartbeat()
node_id = getattr(self, "node_id", str(uuid4()))

# Line 115, 123, 154, 164, 181, 192, etc.
node_id=getattr(self, "node_id", "unknown")
```

This means even if `node_id` is missing (which shouldn't happen), the mixin will:
1. Generate a fallback UUID for events
2. Use "unknown" for logging

## Best Practices

### ✅ DO: Correct Initialization Order

```python
def __init__(self, container: ModelONEXContainer) -> None:
    # 1. Parent first (sets node_id and container)
    super().__init__(container)

    # 2. Node-specific setup
    self.my_service_url = container.config.get("service_url")

    # 3. Optional: Initialize health checks
    self.initialize_health_checks()

    # 4. Initialize introspection (requires node_id from parent)
    self.initialize_introspection()

    # 5. Logging (node_id available)
    emit_log_event(LogLevel.INFO, "Initialized", {"node_id": self.node_id})
```

### ✅ DO: Use Defensive Pattern in Logging (Optional)

```python
# Recommended for robustness
logger.info(
    "Node starting up",
    node_id=getattr(self, "node_id", "unknown")
)
```

### ❌ DON'T: Initialize Introspection Before Parent

```python
def __init__(self, container: ModelONEXContainer) -> None:
    # ❌ WRONG: Introspection before parent init
    self.initialize_introspection()  # node_id doesn't exist yet!

    # Parent init (sets node_id)
    super().__init__(container)
```

### ❌ DON'T: Manually Set node_id (Unless Overriding)

```python
def __init__(self, container: ModelONEXContainer) -> None:
    super().__init__(container)  # Already sets node_id

    # ❌ WRONG: Overriding parent's node_id without good reason
    self.node_id = str(uuid4())
```

## Lifecycle Integration

### Startup Method

Both orchestrator and reducer nodes publish introspection during startup:

```python
async def startup(self) -> None:
    """Node startup lifecycle hook."""
    # Publish initial introspection broadcast
    await self.publish_introspection(reason="startup")

    # Start introspection background tasks (heartbeat, registry listener)
    await self.start_introspection_tasks(
        enable_heartbeat=True,
        heartbeat_interval_seconds=30,
        enable_registry_listener=True,
    )
```

### Shutdown Method

Clean up introspection tasks during shutdown:

```python
async def shutdown(self) -> None:
    """Node shutdown lifecycle hook."""
    # Stop introspection background tasks
    await self.stop_introspection_tasks()

    # Additional cleanup...
```

## Troubleshooting

### Issue: "node_id" attribute missing

**Symptom**: AttributeError when introspection tries to access `self.node_id`

**Root Cause**: `super().__init__(container)` not called or called after introspection

**Solution**: Ensure parent `__init__` is called FIRST in your node's `__init__` method

```python
def __init__(self, container: ModelONEXContainer) -> None:
    super().__init__(container)  # Must be first!
    self.initialize_introspection()  # Now safe
```

### Issue: node_id is "unknown" in logs

**Symptom**: Logs show `node_id="unknown"` instead of actual UUID

**Root Cause**: Either:
1. Parent init not called properly
2. Defensive `getattr(self, "node_id", "unknown")` pattern triggered

**Solution**: Check initialization order and ensure parent class is setting `node_id`

## Implementation Checklist

When creating a new ONEX bridge node with introspection:

- [ ] Inherit from appropriate base class (`NodeOrchestrator`, `NodeReducer`, `NodeEffect`, `NodeCompute`)
- [ ] Include `NodeIntrospectionMixin` in inheritance chain
- [ ] Call `super().__init__(container)` FIRST in `__init__` method
- [ ] Initialize health checks (if using `HealthCheckMixin`)
- [ ] Call `self.initialize_introspection()` AFTER parent init
- [ ] Implement `startup()` method with introspection broadcast
- [ ] Implement `shutdown()` method with introspection task cleanup
- [ ] Use `self.node_id` in logging (available after parent init)

## References

- **Orchestrator Implementation**: `src/omninode_bridge/nodes/orchestrator/v1_0_0/node.py`
- **Reducer Implementation**: `src/omninode_bridge/nodes/reducer/v1_0_0/node.py`
- **Registry Implementation**: `src/omninode_bridge/nodes/registry/v1_0_0/node.py` (no introspection)
- **Introspection Mixin**: `src/omninode_bridge/nodes/mixins/introspection_mixin.py`
- **Health Check Mixin**: `src/omninode_bridge/nodes/mixins/health_mixin.py`

## Related Documentation

- [Two-Way Registration Pattern](./architecture/TWO_WAY_REGISTRATION_PATTERN.md)
- [Node Introspection Integration](./NODE_INTROSPECTION_INTEGRATION.md)
- [Introspection Topics](./INTROSPECTION_TOPICS.md)
