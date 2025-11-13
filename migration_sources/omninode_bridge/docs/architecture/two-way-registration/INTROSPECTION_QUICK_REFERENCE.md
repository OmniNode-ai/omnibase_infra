# NodeIntrospectionMixin - Quick Reference Card

## Import

```python
from omninode_bridge.nodes.mixins import NodeIntrospectionMixin
```

## Basic Integration (3 Steps)

```python
class NodeBridgeOrchestrator(NodeOrchestrator, HealthCheckMixin, NodeIntrospectionMixin):
    def __init__(self, container: ModelONEXContainer):
        super().__init__(container)
        self.initialize_introspection()  # 1. Initialize

    async def startup(self):
        await self.publish_introspection(reason="startup")  # 2. Broadcast
        await self.start_introspection_tasks()  # 3. Start background tasks

    async def shutdown(self):
        await self.stop_introspection_tasks()
```

## Key Methods

| Method | Purpose | When to Use |
|--------|---------|-------------|
| `initialize_introspection()` | Initialize system | In `__init__()` |
| `publish_introspection(reason)` | Broadcast capabilities | startup, changes, registry_request |
| `extract_capabilities()` | Get node capabilities | Override for custom data |
| `get_endpoints()` | Get node endpoints | Override for custom endpoints |
| `start_introspection_tasks()` | Start heartbeat/listener | In `startup()` |
| `stop_introspection_tasks()` | Stop background tasks | In `shutdown()` |

## Common Scenarios

### Startup Introspection
```python
await node.publish_introspection(reason="startup")
```

### Registry Request Response
```python
await node.publish_introspection(
    reason="registry_request",
    correlation_id=request_correlation_id
)
```

### After Major Changes
```python
await node.publish_introspection(
    reason="capability_change",
    force_refresh=True
)
```

### Custom Capabilities
```python
async def extract_capabilities(self) -> Dict[str, Any]:
    capabilities = await super().extract_capabilities()
    capabilities["custom_feature"] = self.custom_data
    return capabilities
```

## Event Types

1. **NODE_INTROSPECTION**: Full capability broadcast
   - Triggers: startup, registry_request, manual
   - Includes: node_type, capabilities, endpoints, metadata

2. **NODE_HEARTBEAT**: Periodic health status
   - Interval: 30s (configurable)
   - Includes: health_status, uptime, resource_usage

## Kafka Topics

- `dev.omninode_bridge.onex.evt.node-introspection.v1`
- `dev.omninode_bridge.onex.evt.node-heartbeat.v1`
- `dev.omninode_bridge.onex.evt.registry-request-introspection.v1`

(Replace `dev` with `staging` or `prod` based on environment)

## Configuration

```python
container = ModelONEXContainer(
    config={
        "api_port": 8053,
        "metrics_port": 9090,
        "environment": "production",
        "kafka_broker_url": "kafka:9092",
    }
)
```

## Node Attributes (Auto-Detected)

Required:
- `node_id: str`
- `node_type: str`
- `container: ModelONEXContainer`

Optional (for richer introspection):
- `workflow_fsm_states: dict` - FSM states
- `stamping_metrics: dict` - Performance metrics
- `metadata_stamping_service_url: str` - Service URLs

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No events in Kafka | Check Kafka producer in container |
| Heartbeat not working | Verify `start_introspection_tasks()` called |
| Missing capabilities | Check node attributes are set |
| Import errors | Verify OnexEnvelopeV1 wrappers available |

## Files

- Implementation: `src/omninode_bridge/nodes/mixins/introspection_mixin.py`
- Examples: `src/omninode_bridge/nodes/mixins/introspection_usage_examples.py`
- Tests: `tests/test_introspection_mixin.py`
- Full Docs: `docs/NODE_INTROSPECTION_INTEGRATION.md`

## Testing

```bash
# Run tests
pytest tests/test_introspection_mixin.py -v

# Test import
python -c "import sys; sys.path.insert(0, 'src'); from omninode_bridge.nodes.mixins import NodeIntrospectionMixin"
```

## Performance

- **Capability Caching**: Reduces extraction overhead
- **Async Operations**: Non-blocking broadcasts
- **Heartbeat Overhead**: <0.1% CPU per interval
- **Event Size**: ~1-2 KB per broadcast

## Wave 2 (Planned)

- Kafka consumer for registry requests
- Dynamic capability updates
- Enhanced health integration
- Full registry protocol

---

**Quick Reference Version**: 1.0.0
**Last Updated**: October 3, 2025
**See**: `docs/NODE_INTROSPECTION_INTEGRATION.md` for full documentation
