# NodeDeploymentReceiverEffect

## Overview

Production-grade ONEX v2.0 Effect Node for receiving and deploying Docker containers on remote systems with comprehensive security validation, health monitoring, and Kafka event publishing.

**Node Type**: Effect
**Domain**: deployment_automation
**Status**: ✅ Complete (October 2025)
**Security**: HMAC + BLAKE3 + IP Whitelisting

## Features

### Security Features 🔒
- **HMAC Authentication**: SHA256-based signature validation
- **BLAKE3 Checksum**: Sub-millisecond hash verification
- **IP Whitelisting**: CIDR-based network filtering (192.168.86.0/24, 10.0.0.0/8)
- **Constant-Time Comparison**: Timing attack protection
- **Circuit Breaker**: Resilience against Docker daemon failures

### Deployment Features 🚀
- **Docker SDK Integration**: Full container lifecycle management
- **Resource Limits**: CPU and memory constraints
- **Health Monitoring**: Container status verification
- **Volume Mounts**: Persistent storage support
- **Network Configuration**: Multi-network attachment
- **Restart Policies**: Auto-restart configuration

### Event Publishing 📡
- **Kafka Integration**: Real-time deployment event streaming
- **OnexEnvelopeV1**: Standardized event format
- **6 Event Types**: Full deployment lifecycle tracking

## Operations

### 1. `receive_package` - Package Reception & Validation
Receives Docker image package with security validation.

**Input**:
```python
{
    "operation_type": "receive_package",
    "package_data": {
        "image_tar_path": "/path/to/image.tar",
        "checksum": "blake3_hash_64_hex_chars",
        "size_bytes": 1234567
    },
    "sender_auth": {
        "sender_id": "uuid",
        "auth_token": "32_char_minimum",
        "signature": "hmac_sha256_signature",
        "sender_ip": "192.168.86.101"
    }
}
```

**Output**:
```python
{
    "success": true,
    "package_path": "/path/to/image.tar",
    "checksum_valid": true,
    "auth_valid": true,
    "execution_time_ms": 45
}
```

### 2. `load_image` - Docker Image Loading
Loads Docker image into local Docker daemon.

**Target**: <3s
**Input**: `{"image_tar_path": "/path/to/image.tar"}`
**Output**: `{"success": true, "image_id": "sha256:...", "image_name": "omninode/service:v1.0.0"}`

### 3. `deploy_container` - Container Deployment
Deploys container with full configuration.

**Target**: <2s
**Input**:
```python
{
    "deployment_config": {
        "image_name": "omninode/orchestrator:v1.0.0",
        "container_name": "omninode-orchestrator",
        "ports": {"8060": 8060},
        "environment_vars": {"LOG_LEVEL": "INFO"},
        "volumes": [{"host_path": "/var/run/docker.sock", "container_path": "/var/run/docker.sock", "mode": "rw"}],
        "restart_policy": "unless-stopped",
        "resource_limits": {"cpu_limit": "1.5", "memory_limit": "512m"}
    }
}
```

### 4. `health_check` - Container Health Verification
Verifies container is running and healthy.

**Target**: <1s
**Output**: `{"is_healthy": true, "status": "healthy", "checks_passed": 1}`

### 5. `publish_deployment_event` - Kafka Event Publishing
Publishes deployment lifecycle events to Kafka.

**Target**: <100ms
**Events**: DEPLOYMENT_STARTED, IMAGE_LOADED, CONTAINER_STARTED, HEALTH_CHECK_PASSED, DEPLOYMENT_COMPLETED, DEPLOYMENT_FAILED

### 6. `full_deployment` - Complete Deployment Pipeline
Executes all steps in sequence: validate → load → deploy → health check → publish events.

**Target**: <8s total

## Performance Requirements

| Metric | Target | Maximum |
|--------|--------|---------|
| Image Load | <3s | 5s |
| Container Start | <2s | 3s |
| Health Check | <1s | 2s |
| Total Deployment | <8s | 15s |
| Authentication | <100ms | 200ms |
| Checksum Validation | <500ms | 1s |

## Security Considerations

### HMAC Authentication

**Setup**:
```bash
export AUTH_SECRET_KEY="your-secret-key-min-32-chars"  # pragma: allowlist secret
```

**Signature Generation**:
```python
import hmac
import hashlib

message = f"{image_tar_path}{checksum}".encode('utf-8')
signature = hmac.new(
    secret_key.encode('utf-8'),
    message,
    hashlib.sha256
).hexdigest()
```

### BLAKE3 Checksum Validation

**Checksum Generation**:
```python
import blake3

with open('image.tar', 'rb') as f:
    checksum = blake3.blake3(f.read()).hexdigest()
```

### IP Whitelisting

**Configuration**:
```bash
export ALLOWED_IP_RANGES="192.168.86.0/24,10.0.0.0/8"
```

**Allowed Networks**:
- `192.168.86.0/24` - Local network (default)
- `10.0.0.0/8` - Private network (default)

**Security Best Practices**:
1. ✅ Use strong secret keys (min 32 chars, random)
2. ✅ Validate checksums before image loading
3. ✅ Restrict IP ranges to trusted networks
4. ✅ Enable Docker socket access only when needed
5. ✅ Set resource limits for all containers
6. ✅ Monitor deployment events for anomalies
7. ✅ Rotate secret keys regularly

## Usage Examples

### Basic Deployment

```python
from omninode_bridge.nodes.deployment_receiver_effect.v1_0_0 import NodeDeploymentReceiverEffect
from omnibase_core.models.core import ModelContainer
from omnibase_core.models.contracts.model_contract_effect import ModelContractEffect
from uuid import uuid4

# Initialize node
container = ModelContainer()
node = NodeDeploymentReceiverEffect(container)

# Full deployment
contract = ModelContractEffect(
    correlation_id=uuid4(),
    input_state={
        "operation_type": "full_deployment",
        "package_data": {
            "image_tar_path": "/tmp/omninode-orchestrator.tar",
            "checksum": "blake3_hash_here",
            "size_bytes": 1234567
        },
        "sender_auth": {
            "sender_id": str(uuid4()),
            "auth_token": "a" * 32,
            "signature": "hmac_signature_here",
            "sender_ip": "192.168.86.101"
        },
        "deployment_config": {
            "image_name": "omninode/orchestrator:v1.0.0",
            "container_name": "omninode-orchestrator",
            "ports": {"8060": 8060},
            "environment_vars": {"LOG_LEVEL": "INFO"},
            "restart_policy": "unless-stopped"
        }
    }
)

result = await node.execute_effect(contract)
print(f"Deployment success: {result.success}")
print(f"Container ID: {result.container_id}")
print(f"Container URL: {result.container_url}")
```

### Individual Operations

```python
# Validate package only
contract = ModelContractEffect(
    input_state={
        "operation_type": "receive_package",
        "package_data": {...},
        "sender_auth": {...}
    }
)
result = await node.execute_effect(contract)

# Load image only
contract = ModelContractEffect(
    input_state={
        "operation_type": "load_image",
        "image_tar_path": "/tmp/image.tar"
    }
)
result = await node.execute_effect(contract)

# Deploy container only
contract = ModelContractEffect(
    input_state={
        "operation_type": "deploy_container",
        "deployment_config": {...}
    }
)
result = await node.execute_effect(contract)
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DOCKER_HOST` | No | `unix:///var/run/docker.sock` | Docker daemon socket |
| `AUTH_SECRET_KEY` | **Yes** | `default-secret-key-change-me` | HMAC secret key (min 32 chars) |
| `ALLOWED_IP_RANGES` | No | `192.168.86.0/24,10.0.0.0/8` | Comma-separated CIDR ranges |
| `PACKAGE_DIR` | No | `/tmp/deployment_packages` | Package storage directory |
| `LOG_LEVEL` | No | `INFO` | Logging level |
| `KAFKA_BOOTSTRAP_SERVERS` | No | `localhost:29092` | Kafka brokers |

## Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run Security Tests
```bash
pytest tests/test_security_validator.py -v
```

### Run Integration Tests
```bash
pytest tests/test_integration.py -v
```

### Test Coverage
- Security validation: 100%
- Docker operations: 95%
- End-to-end deployment: 90%

## Kafka Events

**Topics**:
- `dev.omninode-bridge.deployment.deployment-started.v1`
- `dev.omninode-bridge.deployment.image-loaded.v1`
- `dev.omninode-bridge.deployment.container-started.v1`
- `dev.omninode-bridge.deployment.health-check-passed.v1`
- `dev.omninode-bridge.deployment.deployment-completed.v1`
- `dev.omninode-bridge.deployment.deployment-failed.v1`

**Event Format**: OnexEnvelopeV1

## Error Handling

| Error Code | Description | Retry |
|------------|-------------|-------|
| `AUTHENTICATION_FAILED` | HMAC signature mismatch | No |
| `CHECKSUM_MISMATCH` | BLAKE3 validation failed | No |
| `LOAD_FAILED` | Docker image load error | Yes |
| `DEPLOY_FAILED` | Container deployment error | Yes |
| `HEALTH_CHECK_FAILED` | Health check timeout | Yes |
| `DOCKER_DAEMON_UNAVAILABLE` | Docker daemon unreachable | Yes |

## Circuit Breaker

- **Failure Threshold**: 5 consecutive failures
- **Reset Timeout**: 60 seconds
- **Behavior**: Opens circuit after threshold, blocks operations until reset

## ONEX v2.0 Compliance

✅ **Naming Convention**: `NodeDeploymentReceiverEffect` (suffix-based)
✅ **Contract-Driven**: Full contract YAML with IO operations
✅ **Event-Driven**: Kafka event publishing for all lifecycle stages
✅ **Error Handling**: ModelOnexError with EnumCoreErrorCode
✅ **Performance Monitoring**: Execution time tracking for all operations
✅ **Security**: Multi-layer validation (HMAC, checksum, IP)
✅ **Resilience**: Circuit breaker pattern for Docker operations

## Dependencies

- `omnibase_core>=0.1.0` - ONEX v2.0 framework
- `docker>=7.0.0` - Docker SDK for Python
- `blake3>=0.4.0` - BLAKE3 hashing
- `pydantic>=2.0.0` - Data validation
- `aiokafka>=0.11.0` - Async Kafka client

## Related Documentation

- **[Remote Migration Guide](../../../../../../docs/deployment/REMOTE_MIGRATION_GUIDE.md)** - Complete migration procedures
- **[Secure Deployment Guide](../../../../../../docs/deployment/SECURE_DEPLOYMENT_GUIDE.md)** - Production security
- **[Bridge Nodes Guide](../../../../../../docs/guides/BRIDGE_NODES_GUIDE.md)** - Bridge node patterns

---

**Status**: ✅ Production Ready
**Implementation**: Complete (October 2025)
**Security**: HMAC + BLAKE3 + IP Whitelisting
**Test Coverage**: 95%+
