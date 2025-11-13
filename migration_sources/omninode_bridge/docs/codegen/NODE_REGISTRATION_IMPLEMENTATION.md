# Node Registration & Introspection Implementation

**Date**: 2025-10-31
**Status**: ✅ Complete
**Implementation**: Code Generation Pipeline Enhancement

## Overview

This document describes the implementation of automatic node registration and introspection events for generated ONEX nodes. When generated nodes come online, they now automatically fire introspection events that get picked up by NodeBridgeRegistry and Consul for service discovery.

## Critical Architectural Requirement

**Generated nodes MUST fire introspection events on initialization** to enable:

1. **Node Registration** - Automatic registration with NodeBridgeRegistry
2. **Service Discovery** - Registration with Consul for service discovery
3. **Health Monitoring** - Health check endpoints and monitoring
4. **Capability Broadcasting** - Publishing node capabilities and metadata
5. **Heartbeat Tracking** - Regular heartbeat signals for liveness detection

## Implementation Details

### 1. ContractIntrospector Utility ✅ (Already Existed)

**Location**: `src/omninode_bridge/codegen/contract_introspector.py`

**Status**: Already implemented and integrated into TemplateEngine

**Key Features**:
- Introspects Pydantic contract models to extract required fields
- Validates template context has all necessary data for contract generation
- Generates minimal valid contract instances for testing
- Prevents missing required field errors (e.g., `io_operations` for Effect nodes)

**Key Methods**:
```python
class ContractIntrospector:
    def get_required_fields(self, contract_class: Type[BaseModel]) -> list[str]
    def get_field_defaults(self, contract_class: Type[BaseModel]) -> dict[str, Any]
    def validate_contract_data(self, contract_class: Type[BaseModel], data: dict) -> tuple[bool, list[str]]
    def generate_minimal_contract(self, node_type: EnumNodeType) -> dict[str, Any]
    def validate_template_context_for_node_type(self, node_type: EnumNodeType, context: dict) -> tuple[bool, list[str]]
```

**Integration**:
- Template engine instantiates `ContractIntrospector` at initialization (line 146)
- Validation is called in `generate()` method (lines 435-444)
- Prevents generation of invalid contracts before files are written

### 2. Template Engine Validation ✅ (Already Existed)

**Location**: `src/omninode_bridge/codegen/template_engine.py` (lines 560-591)

**Functionality**:
- `validate_template_context()` method already validates context data
- Calls `ContractIntrospector.validate_template_context_for_node_type()`
- Logs warnings for missing data but allows generation to proceed
- Contract validation catches actual issues during YAML generation (lines 456-470)

### 3. Enhanced Node Templates ✅ (Newly Implemented)

**Location**: `src/omninode_bridge/codegen/template_engine.py` (inline templates)

**Updated Templates**:
- `_get_effect_template()` - Effect nodes (lines 999-1204)
- `_get_compute_template()` - Compute nodes (lines 1206-1299)
- `_get_reducer_template()` - Reducer nodes (lines 1301-1396)
- `_get_orchestrator_template()` - Orchestrator nodes (lines 1398-1491)

**Key Enhancements**:

#### Import Mixins with Graceful Fallback
```python
# Import mixins for registration and introspection
try:
    from omninode_bridge.nodes.mixins.introspection_mixin import IntrospectionMixin
    from omninode_bridge.nodes.mixins.health_mixin import HealthCheckMixin
    MIXINS_AVAILABLE = True
except ImportError:
    # Mixins not available in this environment
    MIXINS_AVAILABLE = False
    IntrospectionMixin = object  # type: ignore
    HealthCheckMixin = object  # type: ignore
```

**Why Graceful Fallback?**
- Allows generated nodes to work in environments where omninode_bridge isn't available
- Enables testing and development without full infrastructure
- Maintains backward compatibility with existing deployments

#### Mixin Inheritance
```python
class NodeXxxEffect(NodeEffect, HealthCheckMixin, IntrospectionMixin):
```

**Multiple Inheritance Order**:
1. **NodeEffect** (primary base class from omnibase_core)
2. **HealthCheckMixin** (health check capabilities)
3. **IntrospectionMixin** (introspection and registration)

#### Initialization with Registration
```python
def __init__(self, container: ModelContainer) -> None:
    """Initialize with registration and introspection."""
    super().__init__(container)

    self.config = container.value if isinstance(container.value, dict) else {}

    # Initialize health checks and introspection
    if MIXINS_AVAILABLE:
        self.initialize_health_checks()
        self._register_component_checks()  # Effect nodes only
        self.initialize_introspection()

    emit_log_event(
        LogLevel.INFO,
        "Node initialized with registration support",
        {
            "node_id": str(self.node_id),
            "mixins_available": MIXINS_AVAILABLE,
            "operations": operations_list,
            "features": features_list,
        }
    )
```

**Initialization Sequence**:
1. Call parent `__init__` (NodeEffect, NodeCompute, etc.)
2. Extract configuration from container
3. Initialize health checks (registers base health checks)
4. Register component checks (Effect nodes - allows custom health checks)
5. Initialize introspection (prepares introspection cache)
6. Log initialization with structured metadata

#### Lifecycle Methods

**Startup Hook**:
```python
async def startup(self) -> None:
    """
    Node startup lifecycle hook.

    Publishes introspection data to registry and starts background tasks.
    Should be called when node is ready to serve requests.
    """
    if not MIXINS_AVAILABLE:
        return

    # Publish introspection broadcast to registry
    await self.publish_introspection(reason="startup")

    # Start introspection background tasks (heartbeat, registry listener)
    await self.start_introspection_tasks(
        enable_heartbeat=True,
        heartbeat_interval_seconds=30,
        enable_registry_listener=True,
    )
```

**Shutdown Hook**:
```python
async def shutdown(self) -> None:
    """
    Node shutdown lifecycle hook.

    Stops background tasks and cleans up resources.
    Should be called when node is preparing to exit.
    """
    if not MIXINS_AVAILABLE:
        return

    await self.stop_introspection_tasks()
```

**Lifecycle Flow**:
1. **Initialization** → Prepare node, initialize mixins
2. **Startup** → Publish introspection events, start heartbeat
3. **Running** → Handle requests, send periodic heartbeats
4. **Shutdown** → Stop background tasks, cleanup resources

## Integration with Registry and Consul

### Introspection Event Flow

```
Generated Node (startup)
    │
    ├─→ publish_introspection(reason="startup")
    │       │
    │       ├─→ Kafka Topic: omninode.bridge.node_introspection
    │       │       └─→ ModelNodeIntrospectionEvent
    │       │           ├─ node_id: UUID
    │       │           ├─ node_type: "effect" | "compute" | "reducer" | "orchestrator"
    │       │           ├─ capabilities: operations, features, endpoints
    │       │           ├─ configuration: extracted from container
    │       │           └─ metadata: domain, version, generated_at
    │       │
    │       └─→ NodeBridgeRegistry (listens on Kafka)
    │               ├─→ Dual Registration:
    │               │   ├─ Consul (service discovery with health checks)
    │               │   └─ PostgreSQL (tool registry database)
    │               │
    │               └─→ Registry Response Event
    │                   └─→ Kafka: omninode.bridge.registry_response
    │
    └─→ start_introspection_tasks()
            ├─→ Heartbeat Task (every 30s)
            │   └─→ publish_introspection(reason="heartbeat")
            │
            └─→ Registry Listener Task (every 60s)
                └─→ Check for registry updates
```

### NodeBridgeRegistry Integration

**Registry Location**: `src/omninode_bridge/nodes/registry/v1_0_0/node.py`

**Key Features**:
- Listens for introspection events from generated nodes
- Performs dual registration:
  - **Consul**: Service discovery with health check URLs
  - **PostgreSQL**: Tool registry database for persistence
- Atomic registration with circuit breaker pattern
- TTL cache for offset tracking (memory leak prevention)
- Race condition prevention with cleanup locks

**Registration Process**:
1. Receive introspection event from Kafka
2. Extract node metadata (node_id, type, capabilities)
3. Register with Consul:
   - Service name: `omninode-{service_name}`
   - Health check: `http://{host}:{port}/health`
   - Tags: node_type, version, capabilities
4. Register with PostgreSQL:
   - Table: `node_registrations`
   - Fields: node_id, node_type, capabilities, config
5. Publish registry response event

### Consul Service Discovery

**Service Registration Format**:
```yaml
service_id: "omninode-{node_type}-{node_id}"
service_name: "omninode-{service_name}"
address: "{service_host}"
port: {service_port}
tags:
  - "onex"
  - "bridge"
  - "{node_type}"  # effect, compute, reducer, orchestrator
  - "version:{version}"
  - "node_type:{node_type}"
  - "namespace:{namespace}"
health_check:
  http: "http://{service_host}:{service_port}/health"
  interval: "30s"
  timeout: "5s"
```

**Service Discovery**:
- Nodes query Consul for available services by tag
- Load balancing across multiple instances
- Health check integration (failing health checks remove service)
- Dynamic service updates without code changes

## Testing Generated Nodes

### Verification Checklist

After generating a node, verify:

**1. Generated Files**:
- [ ] `node.py` includes mixin imports
- [ ] `node.py` class inherits from HealthCheckMixin and IntrospectionMixin
- [ ] `__init__` calls `initialize_health_checks()` and `initialize_introspection()`
- [ ] `startup()` and `shutdown()` methods exist
- [ ] `contract.yaml` includes all required fields (especially `io_operations` for Effect nodes)

**2. Registration Code**:
- [ ] `publish_introspection(reason="startup")` in startup()
- [ ] `start_introspection_tasks()` with heartbeat enabled
- [ ] `stop_introspection_tasks()` in shutdown()
- [ ] Graceful fallback when mixins not available

**3. Logging**:
- [ ] Initialization logs include `mixins_available` flag
- [ ] Startup logs confirm registration
- [ ] Structured logging with node_id and correlation_id

### Example Test

```python
from omnibase_core.models.core import ModelContainer
from generated_nodes.test_node.v1_0_0 import NodeTestEffect

async def test_registration():
    # Initialize node
    container = ModelContainer()
    node = NodeTestEffect(container)

    # Verify mixins initialized
    assert hasattr(node, 'initialize_introspection')
    assert hasattr(node, 'publish_introspection')

    # Start node (publishes introspection)
    await node.startup()

    # Node is now registered and sending heartbeats
    # Verify in Consul: curl http://localhost:8500/v1/catalog/service/omninode-test

    # Shutdown node
    await node.shutdown()
```

## Future Enhancements

### Short-term (Next Sprint)

1. **Jinja2 File Templates** (currently inline Python strings)
   - Create `node_templates/{node_type}/node.py.j2` files
   - Easier to maintain and customize
   - Better IDE support for template editing

2. **Custom Health Checks**
   - Generate `_register_component_checks()` with node-specific health checks
   - Database connection checks for Effect nodes
   - Dependency health checks based on requirements

3. **Consul Configuration**
   - Generate Consul-specific metadata in contract.yaml
   - Service mesh integration
   - Load balancing policies

### Medium-term (Next Month)

1. **Registry Query API**
   - Generated nodes query registry for dependent services
   - Dynamic service discovery in code
   - Circuit breaker integration

2. **Introspection API Endpoints**
   - Generate `/introspection` endpoint
   - Return full node capabilities
   - Enable runtime capability queries

3. **Test Generation Enhancement**
   - Generate tests that verify registration
   - Mock Kafka/Consul for isolated testing
   - Integration tests with real registry

### Long-term (Next Quarter)

1. **Multi-Instance Coordination**
   - Generated nodes coordinate with other instances
   - Distributed locking for singleton operations
   - Leader election for orchestrator nodes

2. **Dynamic Reconfiguration**
   - Listen for configuration updates from registry
   - Hot reload without restart
   - A/B testing support

3. **Observability Integration**
   - Automatic metrics export (Prometheus)
   - Distributed tracing (OpenTelemetry)
   - Log aggregation (ELK stack)

## References

### Key Files

- **ContractIntrospector**: `src/omninode_bridge/codegen/contract_introspector.py`
- **TemplateEngine**: `src/omninode_bridge/codegen/template_engine.py`
- **IntrospectionMixin**: `src/omninode_bridge/nodes/mixins/introspection_mixin.py`
- **HealthCheckMixin**: `src/omninode_bridge/nodes/mixins/health_mixin.py`
- **NodeBridgeRegistry**: `src/omninode_bridge/nodes/registry/v1_0_0/node.py`
- **NodeBridgeOrchestrator** (reference pattern): `src/omninode_bridge/nodes/orchestrator/v1_0_0/node.py`

### Documentation

- **Code Generation Guide**: `docs/guides/CODE_GENERATION_GUIDE.md`
- **Bridge Nodes Guide**: `docs/guides/BRIDGE_NODES_GUIDE.md`
- **Main README**: `CLAUDE.md`

### External References

- **ONEX v2.0 Specification**: omnibase_core contract models
- **Consul API**: https://www.consul.io/api-docs
- **Kafka Topics**: 13 topics defined in bridge infrastructure

## Implementation Summary

**What Changed**:
1. ✅ ContractIntrospector already existed and worked correctly
2. ✅ Template validation already existed and worked correctly
3. ✅ All 4 node templates enhanced with registration code (Effect, Compute, Reducer, Orchestrator)

**What's New**:
- Mixin inheritance in generated nodes
- `startup()` and `shutdown()` lifecycle methods
- Automatic introspection event publishing
- Health check initialization
- Graceful fallback when mixins unavailable

**Impact**:
- All newly generated nodes automatically register with registry and Consul
- Nodes send heartbeat signals every 30 seconds
- Health checks available at `/health` endpoint
- Service discovery via Consul
- No manual registration required

**Backward Compatibility**:
- Nodes work without mixins (graceful fallback)
- Existing generated nodes continue to function
- No breaking changes to contract format
- Optional lifecycle methods (can be ignored)

---

**Generated**: 2025-10-31
**Author**: Polymorphic Agent (Code Generation Pipeline)
**Version**: 1.0.0
