# Mixin-Enhanced Node Examples

This directory contains complete, runnable examples demonstrating the ONEX v2.0 mixin-enhanced code generation system. Each example showcases different mixin combinations and use cases.

## Overview

The mixin-enhanced code generator produces production-ready ONEX v2.0 compliant nodes with:
- **Type-safe inheritance** - Multiple mixins with proper MRO (Method Resolution Order)
- **Automatic method generation** - Required mixin methods with proper signatures
- **Contract-driven configuration** - All configuration via contract.yaml
- **Production patterns** - Health checks, metrics, logging, event handling

## Examples

### 1. Basic Effect with Health Check

**Directory**: `basic_effect_with_health_check/`

**Purpose**: Demonstrates the simplest mixin integration - a basic EFFECT node with health checking.

**Mixins Used**:
- `MixinHealthCheck` - Component health checks with Docker HEALTHCHECK compatibility

**Use Cases**:
- Simple data processing effects
- External service integrations requiring health monitoring
- Learning the basics of mixin integration

**Key Features**:
- Database health check implementation
- HTTP service health check
- Docker HEALTHCHECK exit codes
- Prometheus-compatible metrics

---

### 2. Advanced Orchestrator with Metrics

**Directory**: `advanced_orchestrator_with_metrics/`

**Purpose**: Demonstrates a production-grade orchestrator with comprehensive observability.

**Mixins Used**:
- `MixinHealthCheck` - Multi-component health monitoring
- `MixinMetrics` - Performance metrics collection
- `MixinLogData` - Structured logging with correlation tracking

**Use Cases**:
- Workflow orchestration requiring full observability
- Multi-step processing with performance tracking
- Production deployments requiring metrics and monitoring

**Key Features**:
- Multiple health check components (database, kafka, external service)
- Performance metrics (latency, throughput, error rates)
- Structured logging with correlation IDs
- Circuit breaker integration
- Retry policy configuration

---

### 3. Complete Node with All Mixins

**Directory**: `complete_node_all_mixins/`

**Purpose**: Comprehensive example showing the full power of the mixin system.

**Mixins Used**:
- `MixinHealthCheck` - Health monitoring
- `MixinMetrics` - Performance tracking
- `MixinLogData` - Structured logging
- `MixinEventBus` - Event publishing capabilities
- `MixinServiceRegistry` - Service discovery integration
- `MixinCaching` - Result caching for expensive operations

**Use Cases**:
- Complex microservices requiring all capabilities
- Event-driven architectures with service discovery
- High-performance systems requiring caching and metrics
- Learning the complete mixin ecosystem

**Key Features**:
- Full event-driven capabilities
- Service discovery and registration
- Result caching with TTL
- Complete observability stack
- Production-ready patterns

---

## Usage

### Generate a Node from Example Contract

```bash
# Using the code generator CLI
poetry run python -m omninode_bridge.cli.codegen.main generate \
  --contract examples/mixin_enhanced_nodes/basic_effect_with_health_check/contract.yaml \
  --output-dir ./output/basic_effect

# Using the Python API
from omninode_bridge.codegen.service import CodeGeneratorService

service = CodeGeneratorService()
result = service.generate_from_contract(
    contract_path="examples/mixin_enhanced_nodes/basic_effect_with_health_check/contract.yaml",
    output_dir="./output/basic_effect"
)
```

### Run an Example Node

```bash
# Copy the generated code to your project
cp -r output/basic_effect/* src/omninode_bridge/nodes/my_node/

# Import and use in your application
from omninode_bridge.nodes.my_node.node import NodeDataProcessingEffect
from omnibase_core.models.core.model_container import ModelContainer

# Initialize the node
container = ModelContainer()
node = NodeDataProcessingEffect(container)
await node.initialize()

# Perform health check
health_result = await node.check_health()
print(f"Node health: {health_result.overall_status}")

# Execute node operation
result = await node.execute(input_data)
```

### Customize an Example

1. **Copy the example directory**:
   ```bash
   cp -r examples/mixin_enhanced_nodes/basic_effect_with_health_check my_custom_node
   ```

2. **Edit contract.yaml**:
   - Change `name`, `description`, `version`
   - Add/remove mixins as needed
   - Configure mixin parameters
   - Update operations list

3. **Generate the node**:
   ```bash
   poetry run python -m omninode_bridge.cli.codegen.main generate \
     --contract my_custom_node/contract.yaml \
     --output-dir ./output/my_custom_node
   ```

4. **Implement business logic**:
   - Add custom methods to the generated node class
   - Implement operation handlers
   - Add additional health checks

---

## Mixin Reference

### Available Mixins

| Mixin | Purpose | Required Methods | Dependencies |
|-------|---------|-----------------|--------------|
| **MixinHealthCheck** | Health monitoring | `get_health_checks()` | None |
| **MixinMetrics** | Performance tracking | None | None |
| **MixinLogData** | Structured logging | None | None |
| **MixinEventBus** | Event publishing | `get_event_patterns()` | None |
| **MixinEventHandler** | Event handling | None | None |
| **MixinServiceRegistry** | Service discovery | None | None |
| **MixinCaching** | Result caching | None | None |
| **MixinHashComputation** | Hash utilities | None | None |
| **MixinCanonicalYAMLSerializer** | YAML serialization | None | None |

### Mixin Configuration

Mixins are configured in `contract.yaml`:

```yaml
mixins:
  - name: "MixinHealthCheck"
    enabled: true
    config:
      cache_ttl_seconds: 5.0
      critical_components:
        - database
        - kafka

  - name: "MixinMetrics"
    enabled: true
    config:
      collect_histograms: true
      histogram_buckets: [0.001, 0.01, 0.1, 1.0, 10.0]

  - name: "MixinCaching"
    enabled: true
    config:
      default_ttl_seconds: 300
      max_cache_size: 1000
```

---

## Best Practices

### 1. Start Simple

Begin with `basic_effect_with_health_check/` to understand the fundamentals, then progress to more complex examples.

### 2. Enable Mixins Incrementally

Add one mixin at a time and test thoroughly before adding more:

```yaml
# First: Just health checks
mixins:
  - name: "MixinHealthCheck"
    enabled: true

# Then: Add metrics
mixins:
  - name: "MixinHealthCheck"
    enabled: true
  - name: "MixinMetrics"
    enabled: true

# Finally: Add event capabilities
mixins:
  - name: "MixinHealthCheck"
    enabled: true
  - name: "MixinMetrics"
    enabled: true
  - name: "MixinEventBus"
    enabled: true
```

### 3. Implement Required Methods

Some mixins require you to implement specific methods. The generator creates stub implementations with TODO comments:

```python
def get_health_checks(self) -> list[tuple[str, Any]]:
    """Register health checks for this node."""
    return [
        ("self", self._check_self_health),
        ("database", self._check_database_health),  # TODO: Implement
        # Add more health checks as needed
    ]
```

### 4. Test Generated Code

Always test generated nodes thoroughly:

```bash
# Run unit tests
pytest tests/unit/nodes/my_node/

# Run integration tests
pytest tests/integration/nodes/my_node/

# Check health endpoint
curl http://localhost:8000/health
```

### 5. Use Type Hints

The generator produces fully type-hinted code. Maintain type safety when extending:

```python
async def custom_operation(
    self,
    input_data: ModelCustomInput
) -> ModelCustomOutput:
    """Process custom operation with proper types."""
    # Implementation
    return ModelCustomOutput(...)
```

---

## Troubleshooting

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'omnibase_core'`

**Solution**: Ensure omnibase_core is installed:
```bash
poetry add omnibase_core
# or
pip install omnibase_core
```

### Mixin Method Conflicts

**Problem**: Multiple mixins define the same method

**Solution**: Use explicit method resolution or override in your node class:
```python
class NodeMyEffect(NodeEffect, MixinHealthCheck, MixinMetrics):
    def initialize(self) -> None:
        # Explicitly call parent methods
        NodeEffect.initialize(self)
        MixinHealthCheck.initialize_health_checks(self)
        MixinMetrics.initialize_metrics(self)
```

### Missing Required Methods

**Problem**: Mixin requires methods not implemented

**Solution**: Check the generated TODO comments and implement required methods:
```python
# TODO: Implement this method for MixinHealthCheck
async def _check_database_health(self) -> tuple[HealthStatus, str, dict]:
    # Your implementation here
    pass
```

---

## Performance Considerations

### Health Check Caching

Health checks are cached by default (5 seconds TTL). Adjust in contract:

```yaml
mixins:
  - name: "MixinHealthCheck"
    config:
      cache_ttl_seconds: 10.0  # Cache for 10 seconds
```

### Metrics Collection Overhead

Metrics collection has minimal overhead (<1ms per operation), but can be disabled:

```yaml
mixins:
  - name: "MixinMetrics"
    enabled: false  # Disable metrics in development
```

### Caching Configuration

Configure cache size and TTL based on your workload:

```yaml
mixins:
  - name: "MixinCaching"
    config:
      default_ttl_seconds: 600  # 10 minutes
      max_cache_size: 5000      # 5000 entries
```

---

## Additional Resources

- **[Code Generator Guide](../../docs/codegen/CODE_GENERATOR_GUIDE.md)** - Complete code generation documentation
- **[Mixin Injector](../../src/omninode_bridge/codegen/mixin_injector.py)** - Mixin generation implementation
- **[Contract Schema](../../src/omninode_bridge/codegen/models_contract.py)** - Contract data models
- **[ONEX v2.0 Specification](../../docs/ONEX_V2_SPECIFICATION.md)** - Full ONEX v2.0 spec

---

## Contributing

To add new examples:

1. Create a new directory under `examples/mixin_enhanced_nodes/`
2. Add `contract.yaml` with complete configuration
3. Add `node.py` with the generated implementation
4. Update this README with example description
5. Add tests in `tests/integration/codegen/`

---

**Generated**: 2025-11-05
**ONEX Version**: v2.0
**Generator Version**: 1.0.0
