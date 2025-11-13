# Registry Node - Service Container Integration (TODO)

**Status**: In Progress
**Priority**: High
**Blocking**: Full deployment of registry node with dual registration

## Context

The registry node code is **functionally correct and tested** (57/65 tests passing), but requires proper integration with the ONEX service container pattern for full deployment.

## Current State

### ✅ What's Working
- **Core Logic**: Dual registration (Consul + PostgreSQL) code is correct
- **Tests**: 57 tests passing (88% coverage)
- **Code Quality**: Linting clean, type checking clean, EventBus compliant
- **Docker Image**: Builds successfully
- **Health Endpoint**: Responds in degraded mode

### ❌ What's Not Working
- **Service Initialization**: Node cannot access services (kafka_client, consul_client, postgres_client)
- **Error**: `'ModelContainer' object has no attribute 'get_service'`
- **Container Pattern**: Using wrong container type

## Root Cause

The registry node uses `container.get_service()` but `ModelContainer` doesn't have this method:

```python
# Current pattern in node.py (BROKEN):
self.kafka_client = container.get_service("kafka_client")  # ❌ Fails
self.consul_client = container.get_service("consul_client")  # ❌ Fails
self.postgres_client = container.get_service("postgres_client")  # ❌ Fails
```

## Solution

### Required Pattern (from omnibase_core)

The node should use **`ModelONEXContainer`** with proper service registration:

```python
from omnibase_core.models.container.model_onex_container import ModelONEXContainer

# Create container with service registration
container = ModelONEXContainer(
    enable_service_registry=True,
    # Other params...
)

# Register services
container.register_service("kafka_client", kafka_client_instance)
container.register_service("consul_client", consul_client_instance)
container.register_service("postgres_client", postgres_client_instance)

# Now node can access services
node = NodeBridgeRegistry(container)
```

### Files That Need Updates

1. **`src/omninode_bridge/nodes/registry/v1_0_0/main_standalone.py`**
   - Change from `ModelContainer` to `ModelONEXContainer`
   - Add service registration before node initialization
   - Initialize KafkaClient, ConsulClient, PostgresClient
   - Register services with container

2. **`src/omninode_bridge/nodes/registry/v1_0_0/node.py`**
   - Verify `container.get_service()` calls work with ModelONEXContainer
   - May need to update type hints: `ModelContainer` → `ModelONEXContainer`

3. **`tests/unit/nodes/registry/*.py`**
   - Update test fixtures to use ModelONEXContainer
   - Mock `get_service()` method properly
   - Ensure tests still pass

## Reference Implementations

### From omnibase_core

Check these files for the proper pattern:

```bash
/Volumes/PRO-G40/Code/omnibase_core/src/omnibase_core/models/container/model_onex_container.py
/Volumes/PRO-G40/Code/omnibase_core/src/omnibase_core/infrastructure/node_base.py
```

**Key Methods**:
- `ModelONEXContainer.__init__()`
- `ModelONEXContainer.get_service(protocol_type, service_name)`
- `ModelONEXContainer.get_service_optional()`

### Working Example Pattern

```python
# Example from NodeBase in omnibase_core
def __init__(
    self,
    contract_path: Path,
    node_id: UUID | None = None,
    event_bus: object | None = None,
    container: ModelONEXContainer | None = None,  # ← ModelONEXContainer!
    workflow_id: UUID | None = None,
):
    ...
```

## Implementation Steps

1. **Create Service Instances** (in main_standalone.py):
   ```python
   from omninode_bridge.clients.kafka_client import KafkaClient
   from omninode_bridge.clients.consul_client import ConsulClient
   from omninode_bridge.clients.postgres_client import PostgresClient

   # Initialize clients
   kafka_client = KafkaClient(config)
   consul_client = ConsulClient(config)
   postgres_client = PostgresClient(config)
   ```

2. **Create ModelONEXContainer**:
   ```python
   container = ModelONEXContainer(enable_service_registry=True)
   ```

3. **Register Services**:
   ```python
   container.register_service("kafka_client", kafka_client)
   container.register_service("consul_client", consul_client)
   container.register_service("postgres_client", postgres_client)
   ```

4. **Initialize Node**:
   ```python
   node = NodeBridgeRegistry(container)
   await node.on_startup()
   ```

5. **Test Thoroughly**:
   - Run all 57 registry tests
   - Test health endpoint
   - Test manual registration endpoint
   - Test introspection event consumption

## Testing Checklist

- [ ] All 57 registry tests still pass
- [ ] Docker image builds successfully
- [ ] Container starts without errors
- [ ] Health endpoint returns "healthy" status
- [ ] Node can connect to Kafka
- [ ] Node can connect to Consul
- [ ] Node can connect to PostgreSQL
- [ ] Dual registration works end-to-end
- [ ] Introspection events are consumed
- [ ] Registry request events are published

## Success Criteria

**The registry node should**:
1. Start successfully with all services connected
2. Pass health checks (not degraded mode)
3. Consume introspection events from Kafka
4. Register nodes in both Consul AND PostgreSQL
5. Respond to manual registration requests via API
6. Track metrics correctly
7. Handle failures gracefully (circuit breakers)

## Notes

### Similar Issue in deployment_receiver

The `deployment_receiver_effect` node has the **same pattern issue** - it also uses `container.get_service()` with `ModelContainer`. Consider fixing both nodes with the same pattern.

### EventBus Integration

Both nodes are already using `EventBusService` correctly - this work is purely about service container setup, not EventBus refactoring.

### Docker Deployment

Once this is fixed, the registry service in `deployment/docker-compose.yml` is ready to deploy. The Dockerfile and service configuration are correct.

## Related Files

- `src/omninode_bridge/nodes/registry/v1_0_0/node.py` (lines 662, 682, 702, 758)
- `src/omninode_bridge/nodes/registry/v1_0_0/main_standalone.py` (line 104+)
- `tests/unit/nodes/registry/test_*.py` (all fixtures)
- `deployment/docker-compose.yml` (registry service definition)

## Timeline Estimate

- **Research**: 1 hour (understand ModelONEXContainer service registration)
- **Implementation**: 2-3 hours (update main_standalone.py, node.py, tests)
- **Testing**: 1-2 hours (run tests, manual testing, Docker deployment)
- **Total**: 4-6 hours

## Contact

For questions about this work, refer to:
- omnibase_core documentation
- ModelONEXContainer class implementation
- NodeBase and NodeEffect base classes

---

**Created**: 2025-10-26
**Author**: Claude (Anthropic)
**Branch**: mvp_requirement_completion
**Commit**: b97cce4
