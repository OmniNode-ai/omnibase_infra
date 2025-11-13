# Bridge Node Configuration

Comprehensive configuration system for omninode_bridge orchestrator and reducer nodes.

## Overview

The configuration system provides:

- **YAML-based configuration** - Human-readable configuration files
- **Hierarchical merging** - Base config + environment overrides
- **Environment variable support** - Override any setting via env vars
- **Type-safe validation** - Pydantic models ensure correctness
- **Multiple environments** - Development, production, staging, test

## Configuration Hierarchy

```
Base Configuration (orchestrator.yaml or reducer.yaml)
    ↓
Environment Configuration (development.yaml or production.yaml)
    ↓
Environment Variables (BRIDGE_* prefix)
    ↓
Validated Pydantic Models
```

## File Structure

```
config/
├── orchestrator.yaml       # Base orchestrator configuration
├── reducer.yaml            # Base reducer configuration
├── development.yaml        # Development environment overrides
├── production.yaml         # Production environment overrides
└── README.md               # This file

src/omninode_bridge/config/
├── __init__.py             # Exports all configuration components
├── settings.py             # Pydantic models for type-safe config
├── config_loader.py        # YAML loading and merging logic
└── environment_config.py   # Legacy environment configuration
```

## Quick Start

### Load Orchestrator Configuration

```python
from omninode_bridge.config import get_orchestrator_config

# Load development configuration (default)
config = get_orchestrator_config()
print(config.orchestrator.max_concurrent_workflows)  # 10

# Load production configuration
config = get_orchestrator_config(environment="production")
print(config.orchestrator.max_concurrent_workflows)  # 500
```

### Load Reducer Configuration

```python
from omninode_bridge.config import get_reducer_config

# Load development configuration
config = get_reducer_config()
print(config.reducer.aggregation_batch_size)  # 20

# Load production configuration
config = get_reducer_config(environment="production")
print(config.reducer.aggregation_batch_size)  # 500
```

### Access Configuration Values

```python
# Node identification
print(config.node.type)          # "orchestrator"
print(config.node.name)          # "bridge-orchestrator"
print(config.node.namespace)     # "omninode.bridge.orchestrator"

# Database settings
print(config.database.host)      # "localhost"
print(config.database.port)      # 5432
print(config.database.pool_max_size)  # 5 (dev) or 100 (prod)

# Kafka settings
print(config.kafka.bootstrap_servers)           # "localhost:9092"
print(config.kafka.topics.workflow_events)      # "dev.omninode.bridge.workflow.events.v1"
print(config.kafka.producer.compression_type)   # "none" (dev) or "zstd" (prod)

# Monitoring settings
print(config.monitoring.enable_prometheus)      # True
print(config.monitoring.prometheus_port)        # 9090
```

## Environment Variable Overrides

Override any configuration value using environment variables with the `BRIDGE_` prefix:

```bash
# Format: BRIDGE_<SECTION>_<KEY>=value
export BRIDGE_ORCHESTRATOR_MAX_CONCURRENT_WORKFLOWS=200
export BRIDGE_DATABASE_HOST=postgres-prod
export BRIDGE_KAFKA_BOOTSTRAP_SERVERS=kafka:9092

# Nested sections: BRIDGE_<SECTION>_<SUBSECTION>_<KEY>=value
export BRIDGE_KAFKA_PRODUCER_COMPRESSION_TYPE=zstd
export BRIDGE_KAFKA_TOPICS_WORKFLOW_EVENTS=prod.workflows.v1
```

### Environment Variable Type Conversion

Environment variables are automatically converted to appropriate types:

```bash
# Boolean values
export BRIDGE_ORCHESTRATOR_ENABLE_DEPENDENCY_TRACKING=true   # → True
export BRIDGE_CACHE_ENABLED=false                            # → False

# Numeric values
export BRIDGE_ORCHESTRATOR_MAX_CONCURRENT_WORKFLOWS=500      # → 500
export BRIDGE_DATABASE_POOL_MAX_SIZE=100                     # → 100

# List values (comma-separated)
export BRIDGE_ORCHESTRATOR_TASK_PRIORITY_LEVELS=critical,high,medium,low
# → ["critical", "high", "medium", "low"]

# String values
export BRIDGE_NODE_NAME=orchestrator-1                       # → "orchestrator-1"
```

## Configuration Sections

### Orchestrator Configuration

#### Node Settings
- `max_concurrent_workflows` - Maximum concurrent workflows
- `workflow_timeout_seconds` - Workflow timeout
- `task_batch_size` - Task processing batch size
- `worker_pool_size` - Worker pool size

#### Services
- `onextree` - OnexTree service endpoint
- `metadata_stamping` - Metadata stamping service endpoint

### Reducer Configuration

#### Aggregation Settings
- `aggregation_window_seconds` - Aggregation window
- `aggregation_batch_size` - Aggregation batch size
- `max_aggregation_buffer_size` - Maximum buffer size

#### State Management
- `state_persistence_interval_seconds` - State persistence interval
- `state_snapshot_interval_seconds` - State snapshot interval
- `enable_state_compression` - Enable state compression

#### Windowing
- `tumbling` - Tumbling window configuration
- `sliding` - Sliding window configuration
- `session` - Session window configuration

### Common Sections (Both Nodes)

#### Database
```yaml
database:
  host: localhost
  port: 5432
  database: omninode_bridge_dev
  pool_min_size: 2
  pool_max_size: 5
```

#### Kafka
```yaml
kafka:
  bootstrap_servers: localhost:9092
  producer:
    compression_type: snappy
    batch_size: 16384
  consumer:
    group_id: bridge-orchestrator-consumer
    auto_offset_reset: latest
  topics:
    workflow_events: dev.omninode.bridge.workflow.events.v1
```

#### Logging
```yaml
logging:
  level: DEBUG
  format: pretty
  enable_structured_logging: false
  outputs:
    - type: console
      enabled: true
```

#### Monitoring
```yaml
monitoring:
  enable_prometheus: true
  prometheus_port: 9090
  metrics_interval_seconds: 15
```

#### Circuit Breaker
```yaml
circuit_breaker:
  enabled: true
  failure_threshold: 5
  recovery_timeout_seconds: 60
```

#### Cache
```yaml
cache:
  enabled: true
  workflow_state_ttl_seconds: 3600
  max_cache_size_mb: 256
```

## Environment-Specific Overrides

### Development Environment

Optimized for local development and debugging:

- **Lower concurrency** - Easier to debug (10 workflows vs 100)
- **Longer timeouts** - More time for debugging (600s vs 300s)
- **Verbose logging** - DEBUG level, pretty format
- **No compression** - Easier to inspect Kafka messages
- **Smaller pools** - Reduced resource usage

### Production Environment

Optimized for high performance and reliability:

- **High concurrency** - Handle production load (500 workflows)
- **Strict timeouts** - Fast failure detection (180s)
- **Optimized logging** - INFO level, JSON format
- **Compression enabled** - zstd compression for Kafka
- **Large pools** - Maximum performance (100 connections)
- **Enhanced monitoring** - Comprehensive metrics and alerts

## Advanced Usage

### Reload Configuration

```python
from omninode_bridge.config import reload_config, get_orchestrator_config
import os

# Update environment variable
os.environ["BRIDGE_ORCHESTRATOR_MAX_CONCURRENT_WORKFLOWS"] = "300"

# Force reload (clears cache)
reload_config()

# Get fresh configuration
config = get_orchestrator_config()
print(config.orchestrator.max_concurrent_workflows)  # 300
```

### Validate Configuration Files

```python
from omninode_bridge.config import validate_config_files

# Validate all configuration files
results = validate_config_files()
print(results)
# {
#     "orchestrator.yaml": True,
#     "reducer.yaml": True,
#     "development.yaml": True,
#     "production.yaml": True
# }

# Check for invalid files
invalid_files = [name for name, valid in results.items() if not valid]
if invalid_files:
    print(f"Invalid configuration files: {invalid_files}")
```

### Get Configuration Info

```python
from omninode_bridge.config import get_config_info

# Get metadata about configuration
info = get_config_info("orchestrator", "development")
print(info)
# {
#     "node_type": "orchestrator",
#     "environment": "development",
#     "config_dir": "/path/to/omninode_bridge/config",
#     "config_files": ["orchestrator.yaml", "development.yaml"],
#     "files_exist": {
#         "orchestrator.yaml": True,
#         "development.yaml": True
#     },
#     "env_overrides_active": True
# }
```

### Custom Configuration Directory

```python
from omninode_bridge.config import load_node_config
from pathlib import Path

# Load from custom directory
config = load_node_config(
    node_type="orchestrator",
    environment="production",
    config_dir=Path("/etc/omninode/config")
)
```

## Docker/Kubernetes Usage

### Docker Compose

```yaml
services:
  orchestrator:
    image: omninode-bridge-orchestrator:latest
    environment:
      - ENVIRONMENT=production
      - BRIDGE_DATABASE_HOST=postgres
      - BRIDGE_KAFKA_BOOTSTRAP_SERVERS=kafka:9092
      - BRIDGE_ORCHESTRATOR_MAX_CONCURRENT_WORKFLOWS=500
    volumes:
      - ./config:/app/config:ro
```

### Kubernetes ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: orchestrator-config
data:
  ENVIRONMENT: "production"
  BRIDGE_DATABASE_HOST: "postgres-primary"
  BRIDGE_KAFKA_BOOTSTRAP_SERVERS: "kafka-0.kafka-headless:9092,kafka-1.kafka-headless:9092"
  BRIDGE_ORCHESTRATOR_MAX_CONCURRENT_WORKFLOWS: "500"
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: orchestrator
spec:
  template:
    spec:
      containers:
      - name: orchestrator
        image: omninode-bridge-orchestrator:latest
        envFrom:
        - configMapRef:
            name: orchestrator-config
        env:
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-credentials
              key: password
```

## Testing

### Unit Tests

```python
import pytest
from omninode_bridge.config import load_node_config, ConfigurationError

def test_load_orchestrator_config():
    """Test loading orchestrator configuration."""
    config = load_node_config("orchestrator", environment="development")

    assert config.node.type == "orchestrator"
    assert config.orchestrator.max_concurrent_workflows > 0
    assert config.database.pool_max_size > 0

def test_invalid_environment():
    """Test loading with invalid environment."""
    with pytest.raises(ConfigurationError):
        load_node_config("orchestrator", environment="invalid")
```

### Integration Tests

```python
import os
from omninode_bridge.config import get_orchestrator_config, reload_config

def test_environment_override():
    """Test environment variable override."""
    # Set environment override
    os.environ["BRIDGE_ORCHESTRATOR_MAX_CONCURRENT_WORKFLOWS"] = "999"
    reload_config()

    # Load configuration
    config = get_orchestrator_config()

    # Verify override
    assert config.orchestrator.max_concurrent_workflows == 999

    # Cleanup
    del os.environ["BRIDGE_ORCHESTRATOR_MAX_CONCURRENT_WORKFLOWS"]
    reload_config()
```

## Troubleshooting

### Configuration Not Loading

**Problem**: Configuration file not found

```python
ConfigurationError: Configuration file not found: /path/to/config/orchestrator.yaml
```

**Solution**: Verify config directory exists and contains YAML files

```bash
ls -la /path/to/omninode_bridge/config/
# Should show: orchestrator.yaml, reducer.yaml, development.yaml, production.yaml
```

### Validation Errors

**Problem**: Configuration validation fails

```python
ConfigurationError: Configuration validation failed for orchestrator node
```

**Solution**: Check YAML syntax and required fields

```bash
# Validate YAML syntax
yamllint config/orchestrator.yaml

# Check configuration structure
python -c "from omninode_bridge.config import validate_config_files; print(validate_config_files())"
```

### Environment Variables Not Applied

**Problem**: Environment variable overrides not working

**Solution**: Verify environment variable format and reload cache

```python
import os
from omninode_bridge.config import reload_config, get_orchestrator_config

# Check environment variable format
os.environ["BRIDGE_ORCHESTRATOR_MAX_CONCURRENT_WORKFLOWS"] = "200"

# Force cache reload
reload_config()

# Load fresh config
config = get_orchestrator_config()
print(config.orchestrator.max_concurrent_workflows)  # Should be 200
```

## Best Practices

1. **Use environment-specific configs** - Keep dev/prod separate
2. **Don't commit secrets** - Use environment variables for sensitive data
3. **Validate before deployment** - Run `validate_config_files()` in CI/CD
4. **Document custom settings** - Add comments to YAML files
5. **Test configuration changes** - Use unit tests for config validation
6. **Monitor configuration** - Log active configuration on startup
7. **Use type hints** - Leverage Pydantic models for type safety

## Migration from Legacy Config

If migrating from the legacy `EnvironmentConfig`:

```python
# Old way (legacy)
from omninode_bridge.config import get_config
config = get_config()
print(config.database.host)

# New way (bridge nodes)
from omninode_bridge.config import get_orchestrator_config
config = get_orchestrator_config()
print(config.database.host)
```

Both configuration systems can coexist during migration.

## See Also

- [Bridge Node Implementation Plan](../BRIDGE_NODE_IMPLEMENTATION_PLAN.md)
- [ONEX Node Implementation Plan](../ONEX_NODE_IMPLEMENTATION_PLAN.md)
- [Pydantic Settings Documentation](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
