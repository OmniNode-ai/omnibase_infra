# Handoff: OMN-892 - 2-Way Registration E2E Integration Tests

**Date**: 2025-12-26
**Status**: Partially Complete - Architectural Gap Identified
**Branch**: `jonah/omn-892-infra-mvp-2-way-registration-e2e-integration-test`
**Linear Issue**: [OMN-892](https://linear.app/omninode/issue/OMN-892)

---

## Executive Summary

Implemented comprehensive E2E integration tests for the 2-way registration pattern. During implementation, identified an **architectural gap**: the tests validate components in isolation but don't test the actual production runtime (`onex-runtime` / `kernel.py`).

**Test Results**: 42 passed, 3 skipped
**Key Gap**: Tests should use the real `RuntimeHostProcess` / `kernel.main()` instead of custom test harnesses.

---

## What Was Accomplished

### 1. Component Integration Tests (Complete)

Created `/tests/integration/registration/e2e/` with:

| File | Lines | Purpose |
|------|-------|---------|
| `conftest.py` | ~850 | E2E fixtures with container wiring, .env loading |
| `verification_helpers.py` | ~850 | Consul/PostgreSQL/Kafka verification utilities |
| `performance_utils.py` | ~700 | Performance timing and threshold utilities |
| `test_two_way_registration_e2e.py` | ~2,800 | 36 component integration tests |
| `test_full_orchestrator_flow.py` | ~950 | 9 pipeline tests (uses OrchestratorPipeline harness) |

### 2. Test Suites Implemented

| Suite | Tests | What It Tests |
|-------|-------|---------------|
| Suite 1: Node Startup Introspection | 5 | Node broadcasts introspection to Kafka |
| Suite 2: Registry Dual Registration | 7 | Consul + PostgreSQL registration |
| Suite 3: Re-Introspection Protocol | 3 | Registry requests re-introspection |
| Suite 4: Heartbeat Publishing | 4 | Periodic heartbeat events |
| Suite 5: Registry Recovery | 3 | Recovery after restart |
| Suite 6: Multiple Nodes | 3 | Concurrent registration |
| Suite 7: Graceful Degradation | 5 | Partial failure handling |
| Suite 8: Self-Registration | 5 | Registry registers itself |
| Full Pipeline | 9 | Handler → Reducer → Effect chain |

### 3. Key Fixes Applied

1. **Model field alignment** - Fixed `ModelRegistrationProjection` usage
2. **Schema initialization** - Postgres fixture auto-creates `registration_projections` table
3. **Performance thresholds** - Calibrated for remote infrastructure (200ms introspection, 1000ms dual registration)
4. **Method signature fix** - Changed `upsert_projection()` to `persist()` with `ModelSequenceInfo`
5. **Automatic .env loading** - Tests load environment from project root `.env`

---

## Architectural Gap Identified

### The Problem

The tests call components directly instead of testing the actual production runtime:

```python
# CURRENT (Component Testing)
handler = HandlerNodeIntrospected(projection_reader)
result = await handler.handle(event, now, correlation_id)  # Direct call

# SHOULD BE (Runtime Testing)
# Start actual runtime that consumes from Kafka
await kernel.main()  # or docker compose up
# Publish event to Kafka
await event_bus.publish(topic, event)
# Verify runtime processed it
```

### Why This Matters

The production runtime uses:
- `kernel.py` → Entry point (`onex-runtime` CLI command)
- `RuntimeHostProcess` → Wires handlers, subscribes to Kafka
- `HealthServer` → Exposes `/health` and `/ready` endpoints
- `container_wiring.py` → DI container setup

**None of this is tested by the current E2E tests.**

### Production Runtime Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    onex-runtime (kernel.py)                  │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              RuntimeHostProcess                      │    │
│  │                                                      │    │
│  │  ┌──────────────┐    ┌──────────────────────────┐   │    │
│  │  │ EventBus     │───▶│ ProtocolBindingRegistry  │   │    │
│  │  │ (Kafka)      │    │ - HandlerNodeIntrospected│   │    │
│  │  │              │    │ - HandlerRuntimeTick     │   │    │
│  │  │ subscribe()  │    │ - HandlerNodeRegAcked    │   │    │
│  │  └──────────────┘    └──────────────────────────┘   │    │
│  │                                                      │    │
│  │  ┌──────────────┐    ┌──────────────────────────┐   │    │
│  │  │ HealthServer │    │ ModelONEXContainer (DI)  │   │    │
│  │  │ :8085        │    │ - Projector              │   │    │
│  │  │ /health      │    │ - ProjectionReader       │   │    │
│  │  │ /ready       │    │ - ConsulHandler          │   │    │
│  │  └──────────────┘    └──────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼               ▼               ▼
      ┌───────────┐   ┌───────────┐   ┌───────────┐
      │ PostgreSQL│   │  Kafka    │   │  Consul   │
      │ :5436     │   │  :29092   │   │  :28500   │
      └───────────┘   └───────────┘   └───────────┘
                   192.168.86.200
```

---

## Existing Infrastructure (Already Implemented)

### Docker Compose Files

| File | Purpose |
|------|---------|
| `docker/docker-compose.runtime.yml` | Production runtime with 3 service profiles |
| `docker/Dockerfile.runtime` | Multi-stage production Dockerfile |
| `docker/.env.example` | Environment variable template |
| `docker/README.md` | Deployment documentation |

### Runtime Entry Points

```toml
# pyproject.toml
[tool.poetry.scripts]
omni-infra = "omnibase_infra.cli.commands:cli"
onex-runtime = "omnibase_infra.runtime.kernel:main"  # <-- Production runtime
```

### Service Profiles

```bash
# Start core runtime
docker compose -f docker/docker-compose.runtime.yml --profile main up

# Start effects processor
docker compose -f docker/docker-compose.runtime.yml --profile effects up

# Start workers
docker compose -f docker/docker-compose.runtime.yml --profile workers up

# Start everything
docker compose -f docker/docker-compose.runtime.yml --profile all up
```

---

## Next Steps (Recommended)

### Option 1: True Runtime E2E Tests (Recommended)

Create tests that start the actual runtime:

```python
# tests/integration/registration/e2e/test_runtime_e2e.py

import asyncio
import subprocess
import pytest

@pytest.fixture
async def running_runtime():
    """Start the actual onex-runtime in background."""
    proc = subprocess.Popen(
        ["poetry", "run", "onex-runtime"],
        env={
            "KAFKA_BOOTSTRAP_SERVERS": "192.168.86.200:29092",
            "POSTGRES_HOST": "192.168.86.200",
            "POSTGRES_PORT": "5436",
            # ... other env vars
        }
    )
    # Wait for health check
    await wait_for_health("http://localhost:8085/health")
    yield proc
    proc.terminate()
    proc.wait()

@pytest.mark.e2e
async def test_full_registration_via_runtime(running_runtime, real_kafka_event_bus):
    """Test the ACTUAL runtime processes registration."""
    node_id = uuid4()

    # Publish introspection to Kafka
    await real_kafka_event_bus.publish(
        topic="dev.onex.evt.node-introspection.v1",
        event=create_introspection_event(node_id),
    )

    # Wait for runtime to process
    await asyncio.sleep(5)

    # Verify dual registration happened
    consul_result = await verify_consul_registration(node_id)
    postgres_result = await verify_postgres_registration(node_id)

    assert consul_result is not None, "Runtime should register in Consul"
    assert postgres_result is not None, "Runtime should register in PostgreSQL"
```

### Option 2: Docker Compose Integration Tests

```python
@pytest.fixture(scope="session")
async def runtime_containers():
    """Start runtime via docker compose."""
    subprocess.run([
        "docker", "compose",
        "-f", "docker/docker-compose.runtime.yml",
        "--profile", "main",
        "up", "-d"
    ], check=True)

    await wait_for_health("http://localhost:8085/health")

    yield

    subprocess.run([
        "docker", "compose",
        "-f", "docker/docker-compose.runtime.yml",
        "down"
    ])
```

### Option 3: Refactor OrchestratorPipeline

Replace the custom `OrchestratorPipeline` test harness with `RuntimeHostProcess`:

```python
@pytest.fixture
async def runtime_host_process(wired_container, real_kafka_event_bus):
    """Use the ACTUAL RuntimeHostProcess."""
    from omnibase_infra.runtime import RuntimeHostProcess

    registry = await wired_container.resolve(ProtocolBindingRegistry)

    runtime = RuntimeHostProcess(
        event_bus=real_kafka_event_bus,
        binding_registry=registry,
        input_topic="test.node.introspection",
        output_topic="test.registration.responses",
    )

    await runtime.start()
    yield runtime
    await runtime.stop()
```

---

## Key Files Reference

### Test Files (Created)

```
tests/integration/registration/e2e/
├── __init__.py
├── conftest.py                          # E2E fixtures
├── verification_helpers.py              # Verification utilities
├── performance_utils.py                 # Timing utilities
├── test_two_way_registration_e2e.py     # Component tests (36 tests)
└── test_full_orchestrator_flow.py       # Pipeline tests (9 tests)
```

### Runtime Files (Existing)

```
src/omnibase_infra/runtime/
├── kernel.py                            # Entry point (onex-runtime)
├── runtime_host_process.py              # Core runtime
├── health_server.py                     # Health endpoints
├── container_wiring.py                  # DI wiring
└── wiring.py                            # Handler registration

docker/
├── docker-compose.runtime.yml           # Production compose
├── Dockerfile.runtime                   # Production Dockerfile
├── .env.example                         # Environment template
└── README.md                            # Deployment docs
```

### Orchestrator Files

```
src/omnibase_infra/nodes/node_registration_orchestrator/
├── node.py                              # NodeRegistrationOrchestrator
├── contract.yaml                        # Workflow definition
├── handlers/
│   ├── handler_node_introspected.py     # Introspection handler
│   ├── handler_runtime_tick.py          # Timeout handler
│   └── handler_node_registration_acked.py
└── models/
```

---

## Environment Variables Required

```bash
# Infrastructure (remote at 192.168.86.200)
KAFKA_BOOTSTRAP_SERVERS=192.168.86.200:29092
POSTGRES_HOST=192.168.86.200
POSTGRES_PORT=5436
POSTGRES_DATABASE=omninode_bridge
POSTGRES_USER=postgres
POSTGRES_PASSWORD=<from .env>
CONSUL_HOST=192.168.86.200
CONSUL_PORT=28500

# Runtime configuration
ONEX_ENVIRONMENT=test
ONEX_LOG_LEVEL=INFO
ONEX_INPUT_TOPIC=test.node.introspection
ONEX_OUTPUT_TOPIC=test.registration.responses
```

---

## Commits

| Commit | Description |
|--------|-------------|
| `c54f5b5` | `test(e2e): implement 2-way registration E2E integration tests [OMN-892]` |

---

## Open Questions

1. **Should the E2E tests start the runtime via subprocess or docker compose?**
   - Subprocess is faster but less production-like
   - Docker compose matches production but adds complexity

2. **Should we create a separate test profile in docker-compose?**
   - Could have `--profile test` with test-specific configuration

3. **How do we handle test isolation with a running runtime?**
   - Unique topic names per test?
   - Unique node IDs with cleanup?

---

## Contact

- **Linear Issue**: OMN-892
- **Branch**: `jonah/omn-892-infra-mvp-2-way-registration-e2e-integration-test`
- **PR**: https://github.com/OmniNode-ai/omnibase_infra/pull/101
