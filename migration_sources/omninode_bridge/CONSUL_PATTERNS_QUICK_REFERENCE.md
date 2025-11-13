# Consul Integration Patterns - Quick Reference Guide

**Workstream 2**: Service Discovery Integration
**File**: `src/omninode_bridge/codegen/patterns/consul_integration.py`
**Status**: ✅ Complete (569 lines)

---

## Quick Start

```python
from omninode_bridge.codegen.patterns import (
    ConsulPatternGenerator,
    generate_consul_registration,
    generate_consul_discovery,
    generate_consul_deregistration
)
```

---

## 1. Generate All Patterns at Once

```python
generator = ConsulPatternGenerator()

patterns = generator.generate_all_patterns(
    node_type="effect",
    service_name="postgres_crud",
    port=8000,
    health_endpoint="/health",
    version="1.0.0",
    domain="bridge"
)

registration_code = patterns['registration']
discovery_code = patterns['discovery']
deregistration_code = patterns['deregistration']
```

---

## 2. Quick Convenience Functions

### Registration

```python
code = generate_consul_registration(
    node_type="orchestrator",
    service_name="workflow_orchestrator",
    port=8001,
    health_endpoint="/health",
    version="1.0.0",
    domain="default"
)
```

### Discovery

```python
code = generate_consul_discovery()
```

### Deregistration

```python
code = generate_consul_deregistration()
```

---

## 3. Integration with Node Generation

```python
from omninode_bridge.codegen.patterns import ConsulPatternGenerator

def add_consul_to_node(node_class_code: str, config: dict) -> str:
    """Add Consul integration to existing node class."""

    generator = ConsulPatternGenerator()
    patterns = generator.generate_all_patterns(
        node_type=config['node_type'],
        service_name=config['service_name'],
        port=config['port']
    )

    # Insert patterns into node class
    node_with_consul = node_class_code.replace(
        "# CONSUL_METHODS_PLACEHOLDER",
        f"{patterns['registration']}\n{patterns['discovery']}\n{patterns['deregistration']}"
    )

    return node_with_consul
```

---

## 4. Generated Method Signatures

### Registration Method

```python
async def _register_with_consul(self) -> None:
    """Register this node with Consul service discovery."""
    # Checks self.container.consul_client
    # Creates unique service_id
    # Registers with health check
    # Stores service_id for deregistration
```

### Discovery Method

```python
async def _discover_service(self, service_name: str) -> Optional[str]:
    """
    Discover service endpoint via Consul.

    Args:
        service_name: Service to discover (e.g., "postgres_crud")

    Returns:
        Endpoint URL (e.g., "http://192.168.1.100:8000") or None
    """
    # Checks cache first (5min TTL)
    # Queries Consul for healthy instances
    # Random load balancing
    # Caches result
```

### Deregistration Method

```python
async def _deregister_from_consul(self) -> None:
    """Deregister this node from Consul service discovery."""
    # Uses stored self._consul_service_id
    # Deregisters from Consul
    # Cleans up state
```

---

## 5. Required Imports for Generated Code

```python
from datetime import UTC, datetime
from typing import Optional
from omnibase_core.enums.enum_log_level import EnumLogLevel as LogLevel
from omnibase_core.logging.structured import emit_log_event_sync as emit_log_event
```

Get programmatically:

```python
generator = ConsulPatternGenerator()
imports = generator.get_required_imports()
```

---

## 6. Node Class Template with Consul

```python
from uuid import uuid4
from typing import Optional

class NodePostgresCRUDEffect:
    """Example node with Consul integration."""

    def __init__(self, container):
        self.container = container
        self.node_id = uuid4()
        self._consul_service_id: Optional[str] = None
        self._service_cache: dict = {}

    async def startup(self):
        """Startup with Consul registration."""
        await self._initialize_components()
        await self._register_with_consul()  # From generated code
        await self._start_health_checks()

    async def shutdown(self):
        """Shutdown with Consul deregistration."""
        self._is_shutting_down = True
        await self._deregister_from_consul()  # From generated code
        await self._cleanup_resources()

    # GENERATED CONSUL METHODS GO HERE
    async def _register_with_consul(self) -> None:
        # Generated registration code
        pass

    async def _discover_service(self, service_name: str) -> Optional[str]:
        # Generated discovery code
        pass

    async def _deregister_from_consul(self) -> None:
        # Generated deregistration code
        pass
```

---

## 7. Configuration Dataclass

```python
from dataclasses import dataclass

@dataclass
class ConsulRegistrationConfig:
    node_type: str          # "effect", "compute", "reducer", "orchestrator"
    service_name: str       # "postgres_crud"
    port: int              # 8000
    health_endpoint: str = "/health"
    version: str = "1.0.0"
    domain: str = "default"

# Usage
config = ConsulRegistrationConfig(
    node_type="effect",
    service_name="postgres_crud",
    port=8000
)
```

---

## 8. Pattern Features

### Graceful Degradation ✅

All patterns check for Consul availability:

```python
if not self.container.consul_client:
    emit_log_event(LogLevel.WARNING, "Consul not available, skipping", {})
    return  # Continue without Consul
```

### Structured Logging ✅

All operations log via `emit_log_event`:

```python
emit_log_event(
    LogLevel.INFO,
    "Successfully registered with Consul",
    {"node_id": str(self.node_id), "service_id": service_id}
)
```

### Health-Aware Discovery ✅

Discovery only returns healthy instances:

```python
_, services = await self.container.consul_client.health.service(
    service_name,
    passing=True  # Only healthy instances
)
```

### Caching ✅

Discovery caches endpoints for 5 minutes:

```python
if hasattr(self, '_service_cache'):
    cached = self._service_cache.get(cache_key)
    if cached and (datetime.now(UTC) - cached['timestamp']).seconds < 300:
        return cached['endpoint']
```

### Load Balancing ✅

Random selection from healthy instances:

```python
import random
service = random.choice(services)
```

---

## 9. Integration with Lifecycle Patterns

```python
from omninode_bridge.codegen.patterns import (
    ConsulPatternGenerator,
    LifecyclePatternGenerator
)

def generate_node_with_lifecycle_and_consul(config):
    """Generate node with both lifecycle and Consul integration."""

    # Generate lifecycle methods
    lifecycle_gen = LifecyclePatternGenerator()
    startup = lifecycle_gen.generate_startup_method(
        initialization_steps=[
            "await self._initialize_database()",
            "await self._register_with_consul()",  # Consul registration
            "await self._start_health_checks()"
        ]
    )

    shutdown = lifecycle_gen.generate_shutdown_method(
        cleanup_steps=[
            "self._is_shutting_down = True",
            "await self._deregister_from_consul()",  # Consul deregistration
            "await self._close_database()"
        ]
    )

    # Generate Consul methods
    consul_gen = ConsulPatternGenerator()
    consul_patterns = consul_gen.generate_all_patterns(
        node_type=config['node_type'],
        service_name=config['service_name'],
        port=config['port']
    )

    # Combine into complete node
    return {
        'startup': startup,
        'shutdown': shutdown,
        'consul_methods': consul_patterns
    }
```

---

## 10. Testing Generated Code

```python
def test_generated_consul_registration():
    """Test generated registration code."""
    generator = ConsulPatternGenerator()
    code = generator.generate_registration(
        node_type="effect",
        service_name="test_service",
        port=8000
    )

    # Verify required elements
    assert "_register_with_consul" in code
    assert "self.container.consul_client" in code
    assert "emit_log_event" in code
    assert "test_service" in code
    assert "8000" in code
    assert "LogLevel.WARNING" in code  # Graceful degradation

def test_generated_consul_discovery():
    """Test generated discovery code."""
    generator = ConsulPatternGenerator()
    code = generator.generate_discovery()

    # Verify required elements
    assert "_discover_service" in code
    assert "passing=True" in code  # Health-aware
    assert "_service_cache" in code  # Caching
    assert "random.choice" in code  # Load balancing
    assert "Optional[str]" in code  # Return type
```

---

## 11. Common Use Cases

### Use Case 1: Effect Node with Service Discovery

```python
# Effect node that discovers orchestrator service
patterns = generator.generate_all_patterns(
    node_type="effect",
    service_name="postgres_crud_effect",
    port=8000
)

# In the node's execute method:
# orchestrator_url = await self._discover_service("orchestrator")
# if orchestrator_url:
#     await self._call_orchestrator(orchestrator_url, data)
```

### Use Case 2: Orchestrator with Multiple Service Dependencies

```python
# Orchestrator that discovers effect and reducer services
patterns = generator.generate_all_patterns(
    node_type="orchestrator",
    service_name="workflow_orchestrator",
    port=8001
)

# In orchestration logic:
# effect_url = await self._discover_service("postgres_crud_effect")
# reducer_url = await self._discover_service("metrics_reducer")
```

### Use Case 3: Reducer with Registration Only

```python
# Reducer that registers but doesn't discover other services
code = generate_consul_registration(
    node_type="reducer",
    service_name="metrics_aggregator",
    port=8002
)

# Reducer just aggregates data, doesn't need to discover services
```

---

## 12. Debugging Tips

### Check if Consul client is available:

```python
if self.container.consul_client:
    print("✅ Consul client available")
else:
    print("⚠️  Consul client not configured")
```

### View registered services:

```python
# Query Consul API directly
services = await self.container.consul_client.agent.services()
print(f"Registered services: {services}")
```

### Check service cache:

```python
if hasattr(self, '_service_cache'):
    print(f"Cached endpoints: {self._service_cache}")
else:
    print("No cache initialized")
```

### Monitor registration status:

```python
if hasattr(self, '_consul_service_id'):
    print(f"✅ Registered as: {self._consul_service_id}")
else:
    print("⚠️  Not registered with Consul")
```

---

## Summary

**Workstream 2** provides production-ready Consul integration patterns:

✅ **3 Pattern Types**: Registration, Discovery, Deregistration
✅ **Graceful Degradation**: Nodes work without Consul
✅ **Health-Aware Discovery**: Only healthy instances returned
✅ **Caching**: 5-minute TTL for performance
✅ **Load Balancing**: Random selection from healthy instances
✅ **Structured Logging**: All operations logged
✅ **ONEX v2.0 Compliant**: Follows all conventions
✅ **omnibase_core Integration**: Uses ModelContainer pattern

**Quick Access**:
- Generator: `ConsulPatternGenerator()`
- Quick functions: `generate_consul_registration()`, `generate_consul_discovery()`, `generate_consul_deregistration()`
- Required imports: `generator.get_required_imports()`

**Integration Points**:
- Lifecycle: Call in `startup()` and `shutdown()`
- Health Checks: Referenced in registration config
- Events: Publish registration/discovery events
- Metrics: Track operation latency and success rate
