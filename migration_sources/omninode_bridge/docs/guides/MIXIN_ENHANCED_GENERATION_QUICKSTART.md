# Mixin-Enhanced Code Generation - Quick Start Guide

**🚀 Get started with mixin-enhanced node generation in 10 minutes**

---

## Prerequisites

✅ omninode_bridge repository cloned
✅ Python 3.12+ installed
✅ Poetry dependencies installed (`poetry install`)
✅ ZAI_API_KEY environment variable set (for LLM generation)

---

## Step 1: Choose Your Mixins (2 minutes)

Use the [Mixin Quick Reference](../reference/MIXIN_QUICK_REFERENCE.md) to select mixins for your node.

**Common Combinations**:

### Database Effect Node
```yaml
mixins:
  - MixinHealthCheck      # Monitor DB connection
  - MixinMetrics          # Track operation performance
  - MixinEventDrivenNode  # Consume Kafka events
  - MixinLogData          # Structured logging
  - MixinSensitiveFieldRedaction  # Redact secrets
```

### API Client Effect Node
```yaml
mixins:
  - MixinHealthCheck      # Monitor API health
  - MixinMetrics          # Track API latency
  - MixinLogData          # Request/response logging
```

### Orchestrator Node
```yaml
mixins:
  - MixinHealthCheck      # Monitor component health
  - MixinMetrics          # Track workflow metrics
  - MixinDagSupport       # Workflow DAGs
  - MixinLogData          # Structured logging
```

---

## Step 2: Create Enhanced Contract (3 minutes)

### Basic Contract (Without Mixins)
```yaml
# contract.yaml
name: my_effect
version:
  major: 1
  minor: 0
  patch: 0
node_type: effect

description: My custom effect node

input_model: ModelMyInput
output_model: ModelMyOutput

io_operations:
  - name: process_data
    description: Process input data
    input_model: ModelMyInput
    output_model: ModelMyOutput
```

### Enhanced Contract (With Mixins)
```yaml
# contract.yaml
name: my_effect
version:
  major: 1
  minor: 0
  patch: 0
node_type: effect

description: My custom effect node with production-ready features

input_model: ModelMyInput
output_model: ModelMyOutput

# ✨ NEW: Mixin declarations
mixins:
  # Health monitoring
  - name: MixinHealthCheck
    enabled: true
    config:
      check_interval_ms: 30000
      timeout_seconds: 5.0
      components:
        - name: "external_api"
          critical: true
          timeout_seconds: 10.0

  # Performance metrics
  - name: MixinMetrics
    enabled: true
    config:
      collect_latency: true
      collect_throughput: true
      percentiles: [50, 95, 99]

  # Structured logging
  - name: MixinLogData
    enabled: true
    config:
      log_level: "INFO"
      structured: true
      include_correlation_id: true

# ✨ NEW: Advanced features (NodeEffect built-ins)
advanced_features:
  # Circuit breaker for resilience
  circuit_breaker:
    enabled: true
    failure_threshold: 5
    recovery_timeout_ms: 60000

  # Retry policy with exponential backoff
  retry_policy:
    enabled: true
    max_attempts: 3
    initial_delay_ms: 1000
    backoff_multiplier: 2.0
    retryable_status_codes: [429, 500, 502, 503]

  # Security validation
  security_validation:
    enabled: true
    sanitize_inputs: true
    sanitize_logs: true

# IO operations (same as before)
io_operations:
  - name: process_data
    description: Process input data
    input_model: ModelMyInput
    output_model: ModelMyOutput
```

---

## Step 3: Validate Contract (1 minute)

```bash
# Validate contract before generation
omninode-generate \
  --contract path/to/contract.yaml \
  --validate-only

# Expected output:
# ✅ Contract validation passed
# Mixins found: 3
#   - MixinHealthCheck ✓
#   - MixinMetrics ✓
#   - MixinLogData ✓
# No errors found
```

---

## Step 4: Generate Node (2 minutes)

### Basic Generation (Mixins Only)
```bash
omninode-generate \
  --contract path/to/contract.yaml \
  --enable-mixins \
  --output src/omninode_bridge/nodes/my_effect/v1_0_0/
```

### Enhanced Generation (Mixins + LLM)
```bash
# Recommended: Use LLM for business logic
omninode-generate \
  --contract path/to/contract.yaml \
  --enable-mixins \
  --enable-llm \
  --llm-tier CLOUD_FAST \
  --validation-level strict \
  --output src/omninode_bridge/nodes/my_effect/v1_0_0/ \
  --verbose

# Generation will:
# 1. Parse contract with mixins (0.1s)
# 2. Inject mixins into node class (0.2s)
# 3. Generate business logic with LLM (2-5s)
# 4. Validate generated code (0.3s)
# 5. Generate tests (0.5s)
# Total: ~3-6 seconds
```

**Generation Output**:
```
✅ Contract parsed successfully
✅ Mixins validated: 3/3
✅ Node file generated: node.py (287 LOC)
✅ Models generated: 2 files
✅ Tests generated: 3 files
✅ Validation passed: 6/6 stages

Generated artifacts:
  src/omninode_bridge/nodes/my_effect/v1_0_0/
  ├── node.py                  # Main node implementation
  ├── contract.yaml            # Your contract (copied)
  ├── __init__.py              # Module init
  ├── models/
  │   ├── model_my_input.py
  │   └── model_my_output.py
  └── tests/
      ├── test_unit.py
      ├── test_integration.py
      └── conftest.py
```

---

## Step 5: Review Generated Code (2 minutes)

### Generated Node Structure

```python
# src/omninode_bridge/nodes/my_effect/v1_0_0/node.py

# ✅ Imports automatically generated
from omnibase_core.nodes.node_effect import NodeEffect
from omnibase_core.mixins.mixin_health_check import MixinHealthCheck
from omnibase_core.mixins.mixin_metrics import MixinMetrics
from omnibase_core.mixins.mixin_log_data import MixinLogData

# ✅ Class with mixin inheritance
class NodeMyEffect(NodeEffect, MixinHealthCheck, MixinMetrics, MixinLogData):
    """
    My custom effect node with production-ready features

    ONEX v2.0 Compliant Effect Node

    Capabilities:
      Built-in Features (NodeEffect):
        - Circuit breakers with failure threshold
        - Retry policies with exponential backoff
        - Transaction support with rollback
        - Concurrent execution control
        - Performance metrics tracking

      Enhanced Features (Mixins):
        - MixinHealthCheck
        - MixinMetrics
        - MixinLogData
    """

    # ✅ Mixin configuration in __init__
    def __init__(self, container: ModelContainer):
        """Initialize node with container and mixins."""
        super().__init__(container)

        # Configure MixinHealthCheck
        self.health_check_config = {
            "check_interval_ms": 30000,
            "timeout_seconds": 5.0,
        }

        # Configure MixinMetrics
        self.metrics_config = {
            "collect_latency": True,
            "collect_throughput": True,
            "percentiles": [50, 95, 99],
        }

    # ✅ Mixin setup in initialize()
    async def initialize(self) -> None:
        """Initialize node resources and mixins."""
        await super().initialize()

        # Setup health checks (MixinHealthCheck)
        self.register_health_check(
            "external_api",
            self._check_external_api_health,
            critical=True,
            timeout_seconds=10.0,
        )

        emit_log_event(
            LogLevel.INFO,
            "NodeMyEffect initialized",
            {"node_id": str(self.node_id)},
        )

    # ✅ Health check method generated
    async def _check_external_api_health(self) -> bool:
        """Check external_api health status."""
        try:
            # TODO: Implement actual health check for external_api
            return True
        except Exception as e:
            logger.error(f"external_api health check failed: {e}")
            return False

    # ✅ Business logic method generated (via LLM)
    async def process_data(
        self,
        input_data: ModelMyInput,
    ) -> ModelMyOutput:
        """
        Process input data

        Args:
            input_data: ModelMyInput instance

        Returns:
            ModelMyOutput with operation results

        Raises:
            ModelOnexError: On operation failure
        """
        # TODO: Implement process_data
        # (LLM would generate implementation here)
        raise NotImplementedError("process_data not yet implemented")
```

---

## Step 6: Run Tests (1 minute)

### Generated Tests

```bash
# Run unit tests
pytest src/omninode_bridge/nodes/my_effect/v1_0_0/tests/test_unit.py -v

# Expected output:
# test_node_initialization ... PASSED
# test_health_check_registered ... PASSED
# test_metrics_collection ... PASSED
# test_process_data ... PASSED  (or SKIPPED if not implemented)

# Run integration tests
pytest src/omninode_bridge/nodes/my_effect/v1_0_0/tests/test_integration.py -v
```

### Manual Testing

```python
# Quick manual test
import asyncio
import os
from omnibase_core.models.core import ModelContainer

os.environ["ZAI_API_KEY"] = "test_key"  # pragma: allowlist secret - For demo only - use real key from .env

# Import generated node
from omninode_bridge.nodes.my_effect.v1_0_0.node import NodeMyEffect

async def test_node():
    # Initialize
    container = ModelContainer(value={}, container_type="config")
    node = NodeMyEffect(container)
    await node.initialize()

    # Check health
    health = await node.health_check()
    print(f"Health: {health}")

    # Cleanup
    await node.shutdown()

asyncio.run(test_node())
```

---

## Step 7: Deploy (Optional)

### Add to Service

```python
# src/omninode_bridge/main.py
from omninode_bridge.nodes.my_effect.v1_0_0.node import NodeMyEffect

# Register node
app.register_node("my_effect", NodeMyEffect)
```

### Docker Deployment

```bash
# Build image
docker build -t omninode-bridge:latest .

# Run with environment variables
docker run -e ZAI_API_KEY=$ZAI_API_KEY \
           -e POSTGRES_HOST=192.168.86.200 \
           -e KAFKA_BOOTSTRAP_SERVERS=192.168.86.200:9092 \
           omninode-bridge:latest
```

---

## Common Patterns

### 1. Database Effect Node (Full Example)

```yaml
# contract.yaml
name: postgres_adapter
node_type: effect

mixins:
  - name: MixinHealthCheck
    config:
      components:
        - name: "database"
          critical: true
          timeout_seconds: 5.0
  - name: MixinMetrics
    config:
      collect_latency: true
  - name: MixinEventDrivenNode
    config:
      consumer_group: "postgres_adapter_consumers"
      topics: ["data-ingestion-requested"]
  - name: MixinLogData
  - name: MixinSensitiveFieldRedaction
    config:
      redact_fields: ["password", "api_key"]

advanced_features:
  circuit_breaker:
    enabled: true
    failure_threshold: 3
  dead_letter_queue:
    enabled: true
    max_retries: 3
```

### 2. API Client Effect Node

```yaml
# contract.yaml
name: external_api_client
node_type: effect

mixins:
  - name: MixinHealthCheck
    config:
      components:
        - name: "api"
          critical: true
          timeout_seconds: 10.0
  - name: MixinMetrics
  - name: MixinLogData

advanced_features:
  circuit_breaker:
    enabled: true
    failure_threshold: 5
  retry_policy:
    enabled: true
    max_attempts: 3
    retryable_status_codes: [429, 500, 502, 503, 504]
```

### 3. Orchestrator Node

```yaml
# contract.yaml
name: workflow_orchestrator
node_type: orchestrator

mixins:
  - name: MixinHealthCheck
    config:
      components:
        - name: "workflow_engine"
          critical: true
  - name: MixinMetrics
  - name: MixinDagSupport
  - name: MixinLogData
  - name: MixinServiceRegistry
    config:
      service_name: "workflow-orchestrator"
```

---

## Troubleshooting

### Issue: Mixin not found

**Error**: `Mixin 'MixinHealthCheck' not found in omnibase_core`

**Solution**: Check mixin name spelling. Use [Mixin Quick Reference](../reference/MIXIN_QUICK_REFERENCE.md).

```bash
# List available mixins
python -c "from omnibase_core.mixins import __all__; print(__all__)"
```

### Issue: Validation failed

**Error**: `Validation failed at stage: ONEX_COMPLIANCE`

**Solution**: Review validation errors, fix contract or code, regenerate.

```bash
# Validate only
omninode-generate --contract contract.yaml --validate-only

# View detailed errors
omninode-generate --contract contract.yaml --validate-only --verbose
```

### Issue: LLM generation slow

**Symptoms**: Generation takes > 10 seconds

**Solution**: Use faster model or disable LLM for simple nodes.

```bash
# Disable LLM for simple nodes
omninode-generate --contract contract.yaml --enable-mixins --no-llm

# Use caching for repeated generations
omninode-generate --contract contract.yaml --enable-mixins --enable-llm --cache-llm
```

### Issue: Generated code doesn't compile

**Error**: `SyntaxError` or `ImportError`

**Solution**: Check validation output, file bug report if generator issue.

```bash
# Check syntax
python -m py_compile src/omninode_bridge/nodes/my_effect/v1_0_0/node.py

# Check imports
python -c "from omninode_bridge.nodes.my_effect.v1_0_0.node import NodeMyEffect"
```

---

## Next Steps

1. **Explore Examples**: Check `examples/mixin_enhanced_nodes/`
2. **Read Full Docs**: [Mixin Catalog](../reference/OMNIBASE_CORE_MIXIN_CATALOG.md)
3. **Migrate Existing Node**: Use [Migration Checklist](../planning/NODE_MIGRATION_CHECKLIST_TEMPLATE.md)
4. **Customize Generation**: See [Advanced Generation Guide](./ADVANCED_CODE_GENERATION.md)

---

## Quick Reference Card

```bash
# Validate contract
omninode-generate --contract <path> --validate-only

# Generate with mixins only
omninode-generate --contract <path> --enable-mixins --output <dir>

# Generate with mixins + LLM
omninode-generate --contract <path> --enable-mixins --enable-llm --output <dir>

# Regenerate existing node
omninode-generate --contract <path> --enable-mixins --overwrite --backup

# Validate generated node
omninode-generate --validate-node <node.py> --contract <contract.yaml>
```

---

## Support

- **Documentation**: `docs/reference/MIXIN_QUICK_REFERENCE.md`
- **Examples**: `examples/mixin_enhanced_nodes/`
- **Issues**: GitHub Issues
- **Slack**: #omninode-codegen

**Happy generating! 🚀**
