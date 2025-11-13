# Integration Testing Guide - OmniNode Bridge

✅ **Status**: Active
**Last Updated**: October 2025

This guide covers integration testing strategies for OmniNode Bridge, including local Docker Compose testing and remote distributed system testing.

---

## Table of Contents

1. [Overview](#overview)
2. [Test Modes](#test-modes)
3. [Local Testing (Docker Compose)](#local-testing)
4. [Remote Testing (Distributed System)](#remote-testing)
5. [Test Configuration](#test-configuration)
6. [Running Tests](#running-tests)
7. [Writing Integration Tests](#writing-integration-tests)
8. [Troubleshooting](#troubleshooting)

---

## Overview

OmniNode Bridge integration tests validate the complete system behavior including:

- **Kafka Event Publishing**: Real Kafka broker integration with event validation
- **PostgreSQL Persistence**: Database operations with transaction support
- **Service Integration**: Bridge nodes (Orchestrator, Reducer) with dependencies
- **Consul Service Discovery**: Service registration and health checks
- **End-to-End Workflows**: Complete workflows from request to aggregation

### Test Infrastructure

- **Kafka/RedPanda**: Event streaming and message queues
- **PostgreSQL 16**: State persistence and transaction management
- **Consul**: Service discovery and configuration
- **Docker Compose**: Local containerized environment
- **Remote System**: Distributed deployment on 192.168.86.200

---

## Test Modes

Integration tests support two execution modes:

### Local Mode (Default)

- **Environment**: Docker Compose containers on localhost
- **Use Case**: Development and CI/CD testing
- **Services**: All services run in Docker containers
- **Configuration**: `.env` file or environment defaults

### Remote Mode

- **Environment**: Distributed system on remote host
- **Use Case**: Production-like validation and distributed system testing
- **Services**: Services deployed on remote infrastructure (192.168.86.200)
- **Configuration**: `remote.env` file with remote system addresses

---

## Local Testing

### Prerequisites

1. **Docker Compose**: Ensure Docker and Docker Compose are installed
2. **Environment File**: Copy `.env.example` to `.env` and configure
3. **Services Running**: Start all services with Docker Compose

### Starting Local Services

```bash
# Start all services
docker compose -f deployment/docker-compose.yml up -d

# Verify services are running
docker compose -f deployment/docker-compose.yml ps

# Check service logs
docker compose -f deployment/docker-compose.yml logs -f
```

### Running Local Tests

```bash
# Run all integration tests (local mode)
poetry run pytest tests/integration/ -v

# Run specific test file
poetry run pytest tests/integration/test_kafka_event_publishing.py -v

# Run with markers
poetry run pytest tests/integration/ -m integration -v

# Run with coverage
poetry run pytest tests/integration/ --cov=src/omninode_bridge --cov-report=html
```

### Local Configuration

By default, tests use localhost addresses:

- **Kafka**: `localhost:29092`
- **PostgreSQL**: `localhost:5436`
- **Consul**: `localhost:28500`

These are automatically configured when `TEST_MODE` is not set or is set to `local`.

---

## Remote Testing

### Prerequisites

1. **Remote System**: Services deployed on 192.168.86.200
2. **Network Access**: Ensure network connectivity to remote host
3. **Environment File**: Configure `remote.env` with remote addresses
4. **Service Verification**: Verify all services are accessible

### Remote System Configuration

The remote system must have the following services running and accessible:

| Service | Port | URL |
|---------|------|-----|
| Kafka/RedPanda | 29102 | 192.168.86.200:29102 |
| PostgreSQL | 5436 | 192.168.86.200:5436 |
| Consul | 28500 | http://192.168.86.200:28500 |
| Metadata Stamping | 8057 | http://192.168.86.200:8057 |
| OnexTree | 8058 | http://192.168.86.200:8058 |
| Orchestrator | 8060 | http://192.168.86.200:8060 |
| Reducer | 8061 | http://192.168.86.200:8061 |

### Running Remote Tests

#### Using the Test Runner Script (Recommended)

```bash
# Run all integration tests against remote system
./run-integration-tests-remote.sh

# Run with verbose output
./run-integration-tests-remote.sh -v

# Run specific test pattern
./run-integration-tests-remote.sh -k test_kafka

# Run with coverage report
./run-integration-tests-remote.sh --coverage

# Skip service verification
./run-integration-tests-remote.sh --no-verify

# Use different remote host
./run-integration-tests-remote.sh --remote-host 192.168.86.201
```

#### Manual Execution

```bash
# Set test mode to remote
export TEST_MODE=remote

# Configure remote services
export KAFKA_BOOTSTRAP_SERVERS=192.168.86.200:29102
export POSTGRES_HOST=192.168.86.200
export POSTGRES_PORT=5436
export CONSUL_HOST=192.168.86.200
export CONSUL_PORT=28500

# Run tests
poetry run pytest tests/integration/ -v -m integration
```

### Remote Configuration File

Create or update `remote.env` with remote system configuration:

```env
# Remote Environment Configuration
ENVIRONMENT=production
TEST_MODE=remote

# Kafka Configuration
KAFKA_ADVERTISED_HOST=192.168.86.200
KAFKA_ADVERTISED_PORT=29102
KAFKA_BOOTSTRAP_SERVERS=192.168.86.200:29102

# PostgreSQL Configuration
POSTGRES_HOST=192.168.86.200
POSTGRES_PORT=5436
POSTGRES_DATABASE=omninode_bridge
POSTGRES_USER=postgres
POSTGRES_PASSWORD=omninode_remote_2024_secure

# Consul Configuration
CONSUL_HOST=192.168.86.200
CONSUL_PORT=28500

# Service URLs
METADATA_STAMPING_URL=http://192.168.86.200:8057
ONEXTREE_URL=http://192.168.86.200:8058
ORCHESTRATOR_URL=http://192.168.86.200:8060
REDUCER_URL=http://192.168.86.200:8061
```

---

## Test Configuration

### Centralized Configuration Module

Tests use `tests/integration/remote_config.py` for centralized configuration:

```python
from tests.integration.remote_config import get_test_config

# Get test configuration (auto-detects local vs remote)
config = get_test_config()

# Use in tests
kafka_servers = config.kafka_bootstrap_servers
postgres_url = config.postgres_connection_url
```

### Configuration Loading

The configuration module:

1. **Detects Test Mode**: Checks `TEST_MODE` environment variable
2. **Loads Environment**: Loads `.env` (local) or `remote.env` (remote)
3. **Resolves Addresses**: Maps Docker network names to actual addresses
4. **Provides Utilities**: Helper functions for common configurations

### Using Configuration in Tests

```python
import pytest
from tests.integration.remote_config import get_test_config

@pytest.fixture
async def kafka_consumer():
    """Create Kafka consumer using test configuration."""
    config = get_test_config()

    consumer = AIOKafkaConsumer(
        bootstrap_servers=config.kafka_bootstrap_servers,
        group_id=f"test_consumer_{uuid4().hex[:8]}",
        auto_offset_reset="earliest",
    )

    await consumer.start()
    yield consumer
    await consumer.stop()
```

---

## Running Tests

### Quick Reference

```bash
# Local testing (default)
poetry run pytest tests/integration/ -v

# Remote testing (automated)
./run-integration-tests-remote.sh

# Remote testing (manual)
export TEST_MODE=remote
poetry run pytest tests/integration/ -v

# Specific test file
poetry run pytest tests/integration/test_kafka_event_publishing.py -v

# Test pattern matching
poetry run pytest tests/integration/ -k "kafka" -v

# With markers
poetry run pytest tests/integration/ -m "integration and not slow" -v

# With coverage
poetry run pytest tests/integration/ --cov=src --cov-report=html
```

### Test Markers

Integration tests support these markers:

- `@pytest.mark.integration` - Mark as integration test
- `@pytest.mark.requires_infrastructure` - Requires Kafka/PostgreSQL
- `@pytest.mark.performance` - Performance validation test
- `@pytest.mark.slow` - Slow-running test (>5 seconds)

### Performance Considerations

Remote tests may have higher latency:

- **Network Latency**: ~1-5ms for local network
- **Kafka Operations**: May be slower due to network round trips
- **PostgreSQL Queries**: Network overhead for queries
- **Timeout Configuration**: Consider increasing timeouts for remote tests

---

## Writing Integration Tests

### Test Structure

```python
#!/usr/bin/env python3
"""
Integration test for [Component Name].

Tests real integration with:
- Kafka event publishing
- PostgreSQL persistence
- Service coordination
"""

import pytest
from tests.integration.remote_config import get_test_config

# Load configuration
_test_config = get_test_config()
KAFKA_BROKER = _test_config.kafka_bootstrap_servers
TEST_NAMESPACE = _test_config.test_namespaces[0]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_kafka_integration(kafka_consumer):
    """Test Kafka event publishing and consumption."""
    # Test implementation
    pass
```

### Best Practices

1. **Use Configuration Module**: Always use `get_test_config()` for addresses
2. **Support Both Modes**: Tests should work in local and remote modes
3. **Clean Up Resources**: Use fixtures for proper cleanup
4. **Verify Services**: Check service health before running tests
5. **Handle Timeouts**: Use appropriate timeouts for network operations
6. **Log Context**: Include correlation IDs and context in test output

### Example: Kafka Integration Test

```python
from tests.integration.remote_config import get_test_config
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

@pytest.mark.integration
@pytest.mark.asyncio
async def test_kafka_event_publishing():
    """Test event publishing to Kafka."""
    config = get_test_config()

    # Create producer
    producer = AIOKafkaProducer(
        bootstrap_servers=config.kafka_bootstrap_servers
    )
    await producer.start()

    try:
        # Publish event
        await producer.send(
            "test.events.v1",
            value={"test": "data"}.encode()
        )

        # Verify event published
        assert True  # Add validation logic

    finally:
        await producer.stop()
```

---

## Troubleshooting

### Common Issues

#### 1. Service Not Accessible

**Symptom**: Connection refused or timeout errors

**Solutions**:
- Verify service is running: `docker compose ps`
- Check network connectivity: `ping 192.168.86.200`
- Verify port accessibility: `nc -zv 192.168.86.200 29102`
- Check firewall rules on remote host
- Ensure services bind to 0.0.0.0 (not 127.0.0.1)

#### 2. Kafka Connection Failures

**Symptom**: Kafka connection timeout or metadata errors

**Solutions**:
- Verify Kafka advertised listeners: Check `docker-compose.remote.yml`
- Test Kafka connectivity: `kafkacat -b 192.168.86.200:29102 -L`
- Check Kafka broker logs: `docker logs omninode-bridge-redpanda`
- Ensure KAFKA_ADVERTISED_HOST is set correctly

#### 3. PostgreSQL Authentication Failures

**Symptom**: Authentication failed for user postgres

**Solutions**:
- Verify password in `remote.env` matches remote system
- Check PostgreSQL logs: `docker logs omninode-bridge-postgres`
- Test connection manually: `psql -h 192.168.86.200 -p 5436 -U postgres`
- Verify pg_hba.conf allows remote connections

#### 4. Tests Pass Locally But Fail Remotely

**Symptom**: Tests work in local mode but fail in remote mode

**Solutions**:
- Check network latency: Increase timeouts for remote tests
- Verify remote services are same version as local
- Check for hardcoded localhost references
- Review test logs for specific error messages
- Ensure TEST_MODE=remote is set

#### 5. Configuration Not Loading

**Symptom**: Tests still use localhost instead of remote addresses

**Solutions**:
- Verify TEST_MODE environment variable: `echo $TEST_MODE`
- Check remote.env exists in project root
- Ensure environment is loaded: Add debug logging
- Verify no conflicting environment variables

### Debug Mode

Enable debug logging for test configuration:

```bash
# Run with debug output
export LOG_LEVEL=DEBUG
export TEST_MODE=remote
poetry run pytest tests/integration/ -v -s
```

### Service Health Checks

Verify services before running tests:

```bash
# Check Kafka
echo | nc -w 1 192.168.86.200 29102 && echo "Kafka OK" || echo "Kafka FAIL"

# Check PostgreSQL
pg_isready -h 192.168.86.200 -p 5436 && echo "PostgreSQL OK" || echo "PostgreSQL FAIL"

# Check Consul
curl -s http://192.168.86.200:28500/v1/status/leader && echo "Consul OK" || echo "Consul FAIL"

# Check all services (automated)
./run-integration-tests-remote.sh --no-verify
```

### Log Analysis

Check service logs for errors:

```bash
# Remote system logs (SSH to remote host)
ssh user@192.168.86.200
docker logs omninode-bridge-orchestrator --tail 100
docker logs omninode-bridge-reducer --tail 100
docker logs omninode-bridge-redpanda --tail 100

# Local test logs
poetry run pytest tests/integration/ -v -s --log-cli-level=DEBUG
```

---

## Performance Benchmarks

### Expected Performance

| Metric | Local | Remote | Notes |
|--------|-------|--------|-------|
| Kafka Publish Latency | <5ms | <10ms | Network overhead |
| PostgreSQL Query | <10ms | <20ms | Network latency |
| Full Workflow | <300ms | <400ms | End-to-end time |
| Test Suite Duration | ~30s | ~45s | All integration tests |

### Performance Testing

```bash
# Run performance-marked tests
poetry run pytest tests/integration/ -m performance -v

# Generate performance report
poetry run pytest tests/integration/ --benchmark-only
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Integration Tests

on: [push, pull_request]

jobs:
  integration-tests:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Start services
        run: docker compose up -d

      - name: Run integration tests
        run: poetry run pytest tests/integration/ -v

      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### Remote Testing in CI

```yaml
jobs:
  remote-integration-tests:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Configure remote testing
        env:
          REMOTE_HOST: ${{ secrets.REMOTE_TEST_HOST }}
        run: ./run-integration-tests-remote.sh
```

---

## Additional Resources

- **[Testing Guide](./TESTING_GUIDE.md)** - General testing overview
- **[Setup Guide](../SETUP.md)** - Development environment setup
- **[Remote Migration Guide](../deployment/REMOTE_MIGRATION_GUIDE.md)** - Remote deployment
- **[Bridge Nodes Guide](../guides/BRIDGE_NODES_GUIDE.md)** - Bridge node testing patterns

---

**Document Status**: Active
**Last Updated**: October 2025
**Maintained By**: omninode_bridge team

[← Back to Documentation Index](../INDEX.md)
