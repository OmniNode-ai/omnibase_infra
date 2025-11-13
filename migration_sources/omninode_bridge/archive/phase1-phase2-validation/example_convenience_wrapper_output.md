# Convenience Wrapper Generation Examples

This document shows example outputs from the updated MixinInjector.

## Example 1: Orchestrator with Convenience Wrapper

### Input Contract
```python
{
    "name": "workflow_orchestrator",
    "node_type": "ORCHESTRATOR",
    "description": "Workflow orchestration with standard features",
    "mixins": []  # No mixins specified - use defaults
}
```

### Generated Import
```python
from omninode_bridge.utils.node_services import ModelServiceOrchestrator
```

### Generated Class
```python
class NodeWorkflowOrchestrator(ModelServiceOrchestrator):
    """
    Workflow orchestration with standard features

    ONEX v2.0 Compliant Orchestrator Node

    Base Class: ModelServiceOrchestrator (Pre-composed orchestrator with standard mixins)

    Pre-configured Capabilities:
        - MixinNodeService: Persistent service mode (MCP servers, tool invocation)
        - MixinHealthCheck: Health check implementation with async support
        - MixinEventBus: Event bus operations and publishing
        - MixinMetrics: Performance metrics collection
    """

    def __init__(self, container: ModelContainer):
        """Initialize node with container and mixins."""
        # Initialize base classes (Node + Mixins)
        super().__init__(container)

        # Initialize logger
        self.logger = logging.getLogger(self.__class__.__name__)

    async def initialize(self) -> None:
        """Initialize node resources and mixins."""
        # Initialize base ModelServiceOrchestrator
        await super().initialize()

        self.logger.info(f'Initializing {self.__class__.__name__}')

        self.logger.info(f'{self.__class__.__name__} initialized successfully')
```

### Key Points
- ✅ Single import from local convenience class
- ✅ Inherits from `ModelServiceOrchestrator` directly
- ✅ All standard mixins are pre-configured
- ✅ Simplified initialization (no mixin setup code)
- ✅ Docstring clearly indicates convenience wrapper usage

---

## Example 2: Reducer with Convenience Wrapper

### Input Contract
```python
{
    "name": "metrics_reducer",
    "node_type": "REDUCER",
    "description": "Metrics aggregation with caching",
    "mixins": []  # No mixins specified - use defaults
}
```

### Generated Import
```python
from omninode_bridge.utils.node_services import ModelServiceReducer
```

### Generated Class
```python
class NodeMetricsReducer(ModelServiceReducer):
    """
    Metrics aggregation with caching

    ONEX v2.0 Compliant Reducer Node

    Base Class: ModelServiceReducer (Pre-composed reducer with standard mixins)

    Pre-configured Capabilities:
        - MixinNodeService: Persistent service mode (MCP servers, tool invocation)
        - MixinHealthCheck: Health check implementation with async support
        - MixinCaching: Result caching for expensive operations
        - MixinMetrics: Performance metrics collection
    """

    def __init__(self, container: ModelContainer):
        """Initialize node with container and mixins."""
        # Initialize base classes (Node + Mixins)
        super().__init__(container)

        # Initialize logger
        self.logger = logging.getLogger(self.__class__.__name__)
```

### Key Points
- ✅ Single import from local convenience class
- ✅ Inherits from `ModelServiceReducer` directly
- ✅ Caching and metrics pre-configured
- ✅ Simplified class structure

---

## Example 3: Custom Composition (Fallback)

### Input Contract
```python
{
    "name": "database_effect",
    "node_type": "EFFECT",
    "description": "Database operations with health checks",
    "mixins": [
        {"name": "MixinHealthCheck", "enabled": True},
        {"name": "MixinMetrics", "enabled": True}
    ]
}
```

### Generated Imports
```python
from omnibase_core.mixins.mixin_health_check import MixinHealthCheck
from omnibase_core.mixins.mixin_metrics import MixinMetrics
from omnibase_core.nodes.node_effect import NodeEffect
```

### Generated Class
```python
class NodeDatabaseEffect(NodeEffect, MixinHealthCheck, MixinMetrics):
    """
    Database operations with health checks

    ONEX v2.0 Compliant Effect Node

    Capabilities:
      Built-in Features (NodeEffect):
        - Circuit breakers with failure threshold
        - Retry policies with exponential backoff
        - Transaction support with rollback
        - Concurrent execution control
        - Performance metrics tracking

      Enhanced Features (Mixins):
        - MixinHealthCheck: Health check implementation with async support
        - MixinMetrics: Performance metrics collection
    """

    def __init__(self, container: ModelContainer):
        """Initialize node with container and mixins."""
        # Initialize base classes (Node + Mixins)
        super().__init__(container)

        # Initialize logger
        self.logger = logging.getLogger(self.__class__.__name__)
```

### Key Points
- ✅ Multiple imports from omnibase_core mixins
- ✅ Traditional inheritance pattern (NodeEffect + Mixins)
- ✅ Detailed capability listing
- ✅ Used when no convenience wrapper available or custom mixins specified

---

## Example 4: Orchestrator with Custom Mixins (Fallback)

### Input Contract
```python
{
    "name": "custom_orchestrator",
    "node_type": "ORCHESTRATOR",
    "description": "Custom orchestration with caching",
    "mixins": [
        {"name": "MixinHealthCheck", "enabled": True},
        {"name": "MixinCaching", "enabled": True}  # Not standard for orchestrator
    ]
}
```

### Generated Imports
```python
from omnibase_core.mixins.mixin_health_check import MixinHealthCheck
from omnibase_core.mixins.mixin_caching import MixinCaching
from omnibase_core.nodes.node_orchestrator import NodeOrchestrator
```

### Generated Class
```python
class NodeCustomOrchestrator(NodeOrchestrator, MixinHealthCheck, MixinCaching):
    """
    Custom orchestration with caching

    ONEX v2.0 Compliant Orchestrator Node

    Capabilities:
      Built-in Features (NodeOrchestrator):
        - Circuit breakers with failure threshold
        - Retry policies with exponential backoff
        - Transaction support with rollback
        - Concurrent execution control
        - Performance metrics tracking

      Enhanced Features (Mixins):
        - MixinHealthCheck: Health check implementation with async support
        - MixinCaching: Result caching for expensive operations
    """
```

### Key Points
- ✅ Falls back to custom composition when non-standard mixins are used
- ✅ Even though node type has a convenience wrapper, custom mixins trigger fallback
- ✅ Maintains full flexibility for custom configurations

---

## Summary

### When Convenience Wrapper is Used
1. **Node type has wrapper available** (orchestrator, reducer)
2. **No mixins specified** → uses wrapper's defaults
3. **Standard mixins specified without custom config** → uses wrapper

### When Custom Composition is Used
1. **Node type has no wrapper** (effect, compute)
2. **Non-standard mixins specified** → custom composition needed
3. **Custom mixin configuration provided** → custom composition needed

### Benefits
- ✅ **Reduced boilerplate** for standard configurations
- ✅ **Simpler imports** (single import vs multiple)
- ✅ **Clearer intent** (convenience wrapper name indicates purpose)
- ✅ **Backward compatible** (existing contracts still work)
- ✅ **Flexible** (custom composition available when needed)
