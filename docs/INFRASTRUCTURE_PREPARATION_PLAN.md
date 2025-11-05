# Infrastructure Preparation Plan for Code Generation

**Status:** Planning Phase
**Goal:** Prepare omnibase_infra to receive generated infrastructure nodes
**Strategy:** Build stable foundation utilities, then generate all node implementations

---

## 🎯 Executive Summary

### The Three-Phase Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                    INFRASTRUCTURE STRATEGY                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ✅ Phase 1: omninode_bridge (COMPLETE)                     │
│  └─ Quick MVP for immediate infrastructure needs            │
│                                                              │
│  🔄 Phase 2: Generation Pipeline (IN PROGRESS)              │
│  └─ Perfect code generation for all ONEX node types         │
│                                                              │
│  📋 Phase 3: omnibase_infra Regeneration (THIS PLAN)        │
│  └─ Receive and deploy properly generated nodes             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Key Insight

**Don't hand-write what you're going to generate.**

Instead, prepare:
- ✅ Foundation utilities (connection pools, circuit breakers)
- ✅ Shared models and protocols
- ✅ Testing infrastructure
- ✅ CI/CD validation pipelines
- ✅ Documentation frameworks

Then let the generation pipeline create:
- 🤖 All adapter nodes (postgres, kafka, vault, consul)
- 🤖 All orchestrator and reducer nodes
- 🤖 All node-specific handlers and processors
- 🤖 All node-specific registries

---

## 📊 Current State Assessment

### What We Have (Completed)

✅ **Clean repository structure** - Temp files archived
✅ **Production dependencies** - Aligned with omninode_bridge (v0.1.0 tags)
✅ **Observability stack** - OpenTelemetry, Prometheus, structlog
✅ **Resilience libraries** - Circuit breaker, retry, rate limiting
✅ **Bridge analysis** - BRIDGE_IMPLEMENTATION_FINDINGS.md

### What We Need (This Plan)

🔨 **Foundation utilities** - Connection pools, health monitors
🔨 **Shared models** - Contract-driven model definitions
🔨 **Testing infrastructure** - Fixtures for generated nodes
🔨 **Generation landing zone** - Directory structure ready
🔨 **CI/CD validation** - Automated quality checks

---

## 🏗️ Phase 1: Foundation Utilities (Hand-Written)

### Objective
Build the **stable infrastructure layer** that generated nodes will depend on.

### 1.1 PostgreSQL Infrastructure

**Location:** `src/omnibase_infra/infrastructure/postgres/`

**Components to Extract/Build:**

#### `connection_pool.py` (~200-300 lines)
```python
"""
PostgreSQL connection pool manager with health monitoring.
Based on omninode_bridge postgres_connection_manager.py patterns.
"""

Key Features:
- Connection pool (10-50 connections, configurable)
- Async/await support (asyncpg)
- Pool health monitoring
- Connection lifecycle management
- Metrics collection (pool size, active connections, wait times)

NOT included (belongs in generated nodes):
- SQL query execution (that's node-specific)
- Transaction management (that's node-specific)
- CRUD operations (that's node-specific)
```

#### `health_monitor.py` (~150-200 lines)
```python
"""
PostgreSQL health monitoring and diagnostics.
"""

Key Features:
- Connection health checks
- Pool status monitoring
- Performance metrics
- Latency tracking
- Error rate monitoring
```

#### `query_metrics.py` (~100-150 lines)
```python
"""
PostgreSQL query performance metrics collection.
"""

Key Features:
- Query duration tracking
- Slow query detection
- Query pattern analysis
- Prometheus metrics export
```

**Rationale:**
- These are **infrastructure utilities**, not nodes
- Used BY multiple generated adapter nodes
- Stable APIs that won't change frequently
- Cross-cutting concerns

### 1.2 Kafka Infrastructure

**Location:** `src/omnibase_infra/infrastructure/kafka/`

**Components to Build:**

#### `producer_pool.py` (~200-300 lines)
```python
"""
Kafka producer pool manager with connection pooling.
Supports both aiokafka (async) and confluent-kafka (performance).
"""

Key Features:
- Producer pool management
- Connection pooling
- Health monitoring
- Metrics collection
- Error handling and retry
- Topic validation
```

#### `consumer_factory.py` (~150-200 lines)
```python
"""
Kafka consumer factory for creating configured consumers.
"""

Key Features:
- Consumer group management
- Offset management
- Partition assignment
- Health monitoring
- Graceful shutdown
```

#### `topic_registry.py` (~100-150 lines)
```python
"""
Kafka topic registry and naming conventions.
"""

Key Features:
- Topic naming validation
- Environment-specific prefixes
- Topic metadata tracking
- Schema registry integration (future)
```

### 1.3 Resilience Infrastructure

**Location:** `src/omnibase_infra/infrastructure/resilience/`

**Components to Build:**

#### `circuit_breaker_factory.py` (~150-200 lines)
```python
"""
Circuit breaker factory for infrastructure components.
Based on circuitbreaker library with custom configurations.
"""

Key Features:
- Pre-configured circuit breakers
- Database circuit breakers
- Kafka circuit breakers
- External service circuit breakers
- Metrics and monitoring
```

#### `retry_policy.py` (~100-150 lines)
```python
"""
Retry policy configurations using tenacity.
"""

Key Features:
- Exponential backoff policies
- Jitter configurations
- Max attempt limits
- Conditional retry logic
- Per-service policies
```

#### `rate_limiter.py` (~100-150 lines)
```python
"""
Rate limiting configurations using slowapi.
"""

Key Features:
- Per-endpoint rate limits
- Per-client throttling
- Burst handling
- Metrics collection
```

### 1.4 Observability Infrastructure

**Location:** `src/omnibase_infra/infrastructure/observability/`

**Components to Build:**

#### `structured_logger.py` (~150-200 lines)
```python
"""
Structured logging setup using structlog.
"""

Key Features:
- JSON log formatting
- Correlation ID injection
- Context propagation
- Log level management
- Environment-specific configs
```

#### `tracer_factory.py` (~150-200 lines)
```python
"""
OpenTelemetry tracer factory and configuration.
"""

Key Features:
- Tracer initialization
- OTLP exporter setup
- Instrumentation configuration
- Sampling strategies
- Context propagation
```

#### `metrics_registry.py` (~150-200 lines)
```python
"""
Prometheus metrics registry for infrastructure.
"""

Key Features:
- Metrics collection
- Custom metric registration
- Common infrastructure metrics
- Export endpoints
- Aggregation strategies
```

---

## 🎨 Phase 2: Shared Models Strategy

### Objective
Define the **contract-driven model layer** that nodes will use for communication.

### 2.1 Model Categories

#### Category A: Infrastructure-Wide Models
**Location:** `src/omnibase_infra/models/infrastructure/`

```python
# Models used across ALL infrastructure
model_health_check.py              # Health check responses
model_service_status.py            # Service status tracking
model_error_response.py            # Standardized errors
model_metrics_snapshot.py          # Metrics snapshots
```

**Decision:**
- ✅ Hand-write these (stable, foundational)
- ✅ Update when ONEX protocols change
- ✅ Not frequently regenerated

#### Category B: Service-Specific Shared Models
**Location:** `src/omnibase_infra/models/{service}/`

```python
# PostgreSQL shared models
models/postgres/
├── model_postgres_query_request.py        # Query request envelope
├── model_postgres_query_response.py       # Query response envelope
├── model_postgres_transaction_request.py  # Transaction operations
└── model_postgres_health_response.py      # Health check response

# Kafka shared models
models/kafka/
├── model_kafka_message.py                 # Message envelope
├── model_kafka_event_envelope.py          # Event structure
└── model_kafka_producer_config.py         # Producer configuration

# Consul shared models
models/consul/
├── model_consul_kv_request.py             # KV operations
├── model_consul_kv_response.py            # KV responses
└── model_consul_service_registration.py   # Service registration

# Vault shared models
models/vault/
├── model_vault_secret_request.py          # Secret operations
└── model_vault_secret_response.py         # Secret responses
```

**Decision:**
- 🤖 **GENERATE from contracts** (recommended)
- ✅ Referenced as dependencies in node contracts
- ✅ Automatically updated when contracts change
- ✅ No manual sync required

**Rationale:**
- Contract-driven consistency
- Automatic updates
- No drift between contract and implementation
- Regeneration is fast and safe

### 2.2 Contract Dependencies Pattern

**Example Node Contract:**
```yaml
# nodes/postgres_adapter/v1_0_0/contract.yaml

dependencies:
  # Protocol dependencies (existing pattern)
  - name: "protocol_event_bus"
    type: "protocol"
    class_name: "ProtocolEventBus"
    module: "omnibase_core.protocol.protocol_event_bus"

  # Shared model dependencies (NEW pattern)
  - name: "model_postgres_query_request"
    type: "model"
    class_name: "ModelPostgresQueryRequest"
    module: "omnibase_infra.models.postgres.model_postgres_query_request"

  - name: "model_postgres_query_response"
    type: "model"
    class_name: "ModelPostgresQueryResponse"
    module: "omnibase_infra.models.postgres.model_postgres_query_response"
```

**Generated node references shared models:**
```python
# GENERATED: nodes/postgres_adapter/v1_0_0/node.py

from omnibase_infra.models.postgres.model_postgres_query_request import (
    ModelPostgresQueryRequest
)
from omnibase_infra.models.postgres.model_postgres_query_response import (
    ModelPostgresQueryResponse
)

class NodePostgresAdapterEffect(NodeEffectService):
    async def process(
        self,
        request: ModelPostgresQueryRequest
    ) -> ModelPostgresQueryResponse:
        # Generated implementation uses shared models
        ...
```

---

## 🧪 Phase 3: Testing Infrastructure

### Objective
Build the **test foundation** for validating generated nodes.

### 3.1 Test Fixtures

**Location:** `tests/infrastructure/fixtures/`

#### `postgres_fixtures.py`
```python
"""
PostgreSQL test fixtures using testcontainers.
"""

Fixtures:
- postgres_container: Testcontainer PostgreSQL instance
- postgres_connection_pool: Test connection pool
- postgres_test_db: Pre-populated test database
- postgres_migration_applied: Database with migrations
```

#### `kafka_fixtures.py`
```python
"""
Kafka test fixtures using testcontainers.
"""

Fixtures:
- kafka_container: Testcontainer Kafka/Redpanda instance
- kafka_producer_pool: Test producer pool
- kafka_consumer_factory: Test consumer factory
- kafka_topics_created: Pre-created test topics
```

#### `container_fixtures.py`
```python
"""
Dependency injection container fixtures.
"""

Fixtures:
- test_container: ONEXContainer with mocked dependencies
- infrastructure_services: Registered infrastructure services
- protocol_mocks: Mocked protocol implementations
```

### 3.2 Test Utilities

**Location:** `tests/infrastructure/utils/`

#### `test_helpers.py`
```python
"""
Common test utilities for infrastructure tests.
"""

Utilities:
- assert_health_check_valid()
- assert_metrics_collected()
- assert_event_published()
- wait_for_condition()
- mock_external_service()
```

#### `mock_factories.py`
```python
"""
Mock object factories for testing.
"""

Factories:
- create_mock_postgres_pool()
- create_mock_kafka_producer()
- create_mock_circuit_breaker()
- create_mock_tracer()
```

### 3.3 Test Structure for Generated Nodes

```
tests/
├── infrastructure/           # Infrastructure utility tests (hand-written)
│   ├── postgres/
│   │   ├── test_connection_pool.py
│   │   ├── test_health_monitor.py
│   │   └── test_query_metrics.py
│   ├── kafka/
│   │   ├── test_producer_pool.py
│   │   ├── test_consumer_factory.py
│   │   └── test_topic_registry.py
│   └── resilience/
│       ├── test_circuit_breaker.py
│       ├── test_retry_policy.py
│       └── test_rate_limiter.py
│
├── nodes/                    # Generated node tests (GENERATED)
│   ├── postgres_adapter/
│   │   └── v1_0_0/
│   │       ├── test_node.py              # GENERATED
│   │       ├── test_handlers.py          # GENERATED
│   │       └── test_integration.py       # GENERATED
│   ├── kafka_adapter/
│   └── vault_adapter/
│
└── integration/              # End-to-end integration tests
    ├── test_postgres_kafka_flow.py
    ├── test_event_bus_integration.py
    └── test_service_discovery.py
```

---

## 📁 Phase 4: Generation Landing Zone

### Objective
Prepare the **directory structure** to receive generated nodes cleanly.

### 4.1 Directory Structure

```
src/omnibase_infra/
├── infrastructure/           # Hand-written foundation (Phase 1)
│   ├── __init__.py
│   ├── postgres/
│   │   ├── __init__.py
│   │   ├── connection_pool.py
│   │   ├── health_monitor.py
│   │   └── query_metrics.py
│   ├── kafka/
│   │   ├── __init__.py
│   │   ├── producer_pool.py
│   │   ├── consumer_factory.py
│   │   └── topic_registry.py
│   ├── resilience/
│   │   ├── __init__.py
│   │   ├── circuit_breaker_factory.py
│   │   ├── retry_policy.py
│   │   └── rate_limiter.py
│   └── observability/
│       ├── __init__.py
│       ├── structured_logger.py
│       ├── tracer_factory.py
│       └── metrics_registry.py
│
├── models/                   # Shared models (Phase 2)
│   ├── __init__.py
│   ├── infrastructure/       # Hand-written
│   │   ├── model_health_check.py
│   │   └── model_service_status.py
│   ├── postgres/             # GENERATED or hand-written (TBD)
│   │   ├── model_postgres_query_request.py
│   │   └── model_postgres_query_response.py
│   ├── kafka/                # GENERATED or hand-written (TBD)
│   └── consul/               # GENERATED or hand-written (TBD)
│
├── nodes/                    # ALL GENERATED (Phase 3+)
│   ├── __init__.py
│   ├── README.md             # "⚠️ ALL NODES IN THIS DIRECTORY ARE GENERATED"
│   ├── .gitignore            # Ignore generation artifacts
│   │
│   ├── postgres_adapter/     # GENERATED
│   │   └── v1_0_0/
│   │       ├── contract.yaml
│   │       ├── node.py
│   │       ├── handlers/
│   │       ├── processors/
│   │       ├── models/       # Node-specific models
│   │       └── registry/
│   │
│   ├── kafka_adapter/        # GENERATED
│   │   └── v1_0_0/
│   │
│   ├── consul_adapter/       # GENERATED
│   │   └── v1_0_0/
│   │
│   ├── vault_adapter/        # GENERATED
│   │   └── v1_0_0/
│   │
│   ├── infrastructure_reducer/     # GENERATED
│   │   └── v1_0_0/
│   │
│   └── infrastructure_orchestrator/  # GENERATED
│       └── v1_0_0/
│
└── registry/                 # Container and DI setup
    ├── __init__.py
    ├── container.py          # ONEXContainer configuration
    ├── protocol_registry.py  # Protocol resolution
    └── service_registry.py   # Service injection
```

### 4.2 Generation Marker Files

#### `nodes/README.md`
```markdown
# Generated Infrastructure Nodes

⚠️ **WARNING: ALL NODES IN THIS DIRECTORY ARE GENERATED**

Do not edit generated files directly. Changes will be overwritten.

## How to Modify Nodes

1. Update the contract file: `{node_name}/v1_0_0/contract.yaml`
2. Run the generation pipeline
3. Review and test generated code
4. Commit both contract and generated code

## Generation Metadata

- **Generator Version:** Will be tracked in each node
- **Generation Date:** Embedded in generated files
- **Contract Version:** Contract version determines generation
- **Dependencies:** Listed in contract.yaml

## Node Directory Structure

Each generated node follows this structure:

\`\`\`
{node_name}/v1_0_0/
├── contract.yaml           # Source of truth (manually edited)
├── node.py                 # GENERATED - Main node implementation
├── handlers/               # GENERATED - Request handlers
├── processors/             # GENERATED - Business logic
├── models/                 # GENERATED - Node-specific models
└── registry/               # GENERATED - DI setup
\`\`\`

## Supported Node Types

- **EFFECT Nodes:** External system adapters (postgres, kafka, vault, consul)
- **COMPUTE Nodes:** Data transformation and processing
- **REDUCER Nodes:** State aggregation and consolidation
- **ORCHESTRATOR Nodes:** Workflow coordination
```

#### `nodes/.gitignore`
```gitignore
# Generation artifacts (during development)
*.gen.tmp
*.backup
.generation_cache/

# Keep contract files and generated code in version control
!contract.yaml
!*.py
!*/
```

### 4.3 Contract Templates

Create **starter contract templates** for each node type:

#### `templates/contracts/effect_node_template.yaml`
```yaml
# EFFECT Node Contract Template
# Copy and customize for new effect nodes

name: "Node{ServiceName}AdapterEffect"
version:
  major: 1
  minor: 0
  patch: 0
node_type: "effect"

description: >
  {Service} adapter node for external system integration.
  Bridges ONEX event bus with {Service} infrastructure.

capabilities:
  - name: "resource_management"
    description: "Manage {Service} resources"
  - name: "health_monitoring"
    description: "Monitor {Service} health"

dependencies:
  services:
    - name: "{service}_connection_manager"
      required: true
      description: "{Service} connection pool manager"

# ... rest of template
```

---

## 🔄 Phase 5: CI/CD for Generated Code

### Objective
Automated **validation and quality checks** for generated nodes.

### 5.1 Generation Pipeline Validation

#### `.github/workflows/validate-generated-nodes.yml`
```yaml
name: Validate Generated Nodes

on:
  pull_request:
    paths:
      - 'src/omnibase_infra/nodes/**'
      - 'src/omnibase_infra/models/**'

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - name: Check for manual edits
        run: |
          # Verify no manual edits to generated files
          # Check generation metadata

      - name: Validate contracts
        run: |
          # YAML schema validation
          # Contract compliance checks

      - name: Type checking
        run: poetry run mypy src/omnibase_infra/nodes/

      - name: Linting
        run: poetry run ruff check src/omnibase_infra/nodes/

      - name: Unit tests
        run: poetry run pytest tests/nodes/ -v

      - name: Integration tests
        run: poetry run pytest tests/integration/ -v

      - name: Contract-code consistency
        run: |
          # Verify generated code matches contracts
          # Detect drift
```

### 5.2 Quality Gates

**Automated Checks:**
- ✅ No manual edits to generated files
- ✅ Contract YAML schema validation
- ✅ Type checking passes (mypy)
- ✅ Linting passes (ruff)
- ✅ Unit tests pass (pytest)
- ✅ Integration tests pass
- ✅ Generated code matches contract

**Performance Benchmarks:**
- ✅ Node startup time < 100ms
- ✅ Request processing < 50ms p95
- ✅ Memory usage < 512MB per node
- ✅ No connection leaks

**Security Checks:**
- ✅ SQL injection prevention
- ✅ Input validation
- ✅ No hardcoded credentials
- ✅ Secure configuration

---

## 📚 Phase 6: Documentation Framework

### Objective
Establish **documentation standards** for generated infrastructure.

### 6.1 Documentation Structure

```
docs/
├── INFRASTRUCTURE_PREPARATION_PLAN.md    # This document
├── BRIDGE_IMPLEMENTATION_FINDINGS.md     # Bridge analysis (exists)
│
├── architecture/
│   ├── INFRASTRUCTURE_ARCHITECTURE.md    # Overall architecture
│   ├── SHARED_MODELS_STRATEGY.md         # Model sharing patterns
│   └── GENERATION_WORKFLOW.md            # How generation works
│
├── guides/
│   ├── ADDING_NEW_ADAPTER_NODE.md        # How to add adapters
│   ├── INFRASTRUCTURE_UTILITIES.md       # Using foundation utilities
│   └── TESTING_GENERATED_NODES.md        # Testing guidelines
│
└── nodes/                                 # Node-specific docs
    ├── postgres_adapter/
    │   └── README.md                      # GENERATED - Usage guide
    ├── kafka_adapter/
    │   └── README.md                      # GENERATED - Usage guide
    └── vault_adapter/
        └── README.md                      # GENERATED - Usage guide
```

### 6.2 Generated Documentation

Each generated node should include:

#### `nodes/{node_name}/v1_0_0/README.md` (GENERATED)
```markdown
# {NodeName} - v1.0.0

**Type:** {NODE_TYPE}
**Generated:** {TIMESTAMP}
**Generator Version:** {VERSION}

## Overview

{Description from contract}

## Capabilities

{List capabilities from contract}

## Dependencies

{List service dependencies}

## Configuration

{Configuration options from contract}

## Usage Examples

{Generated usage examples}

## API Reference

{Generated API documentation}

## Health & Monitoring

{Health check endpoints and metrics}

## Troubleshooting

{Common issues and solutions}
```

---

## 🗺️ Implementation Roadmap

### Week 1-2: Foundation Utilities

**Focus:** PostgreSQL and Kafka infrastructure

- [ ] Extract postgres connection pool from existing code
- [ ] Build postgres health monitor
- [ ] Build postgres query metrics
- [ ] Create kafka producer pool
- [ ] Create kafka consumer factory
- [ ] Create kafka topic registry
- [ ] Write comprehensive unit tests
- [ ] Document APIs and usage

**Deliverables:**
- `infrastructure/postgres/` package (3 modules)
- `infrastructure/kafka/` package (3 modules)
- Unit tests (>90% coverage)
- API documentation

### Week 3: Resilience & Observability

**Focus:** Cross-cutting infrastructure concerns

- [ ] Build circuit breaker factory
- [ ] Create retry policy configurations
- [ ] Build rate limiter setup
- [ ] Configure structured logging
- [ ] Set up OpenTelemetry tracing
- [ ] Create Prometheus metrics registry
- [ ] Integration tests
- [ ] Documentation

**Deliverables:**
- `infrastructure/resilience/` package (3 modules)
- `infrastructure/observability/` package (3 modules)
- Integration tests
- Configuration guides

### Week 4: Shared Models & Contracts

**Focus:** Model layer preparation

**Option A: Hand-Written Shared Models**
- [ ] Create infrastructure-wide models
- [ ] Create postgres shared models
- [ ] Create kafka shared models
- [ ] Create consul shared models
- [ ] Create vault shared models
- [ ] Write model tests
- [ ] Document model usage

**Option B: Generated Shared Models** (RECOMMENDED)
- [ ] Define model contracts
- [ ] Set up model generation pipeline
- [ ] Generate initial shared models
- [ ] Validate generated models
- [ ] Document contract patterns

**Deliverables:**
- `models/` package structure
- Shared models (generated or hand-written)
- Model contracts (if generated)
- Documentation

### Week 5: Testing Infrastructure

**Focus:** Test foundation for generated nodes

- [ ] Create postgres test fixtures
- [ ] Create kafka test fixtures
- [ ] Create container fixtures
- [ ] Build test helpers
- [ ] Build mock factories
- [ ] Create test templates
- [ ] Document testing patterns

**Deliverables:**
- `tests/infrastructure/fixtures/` package
- `tests/infrastructure/utils/` package
- Test templates for nodes
- Testing documentation

### Week 6: Generation Landing Zone

**Focus:** Directory structure and validation

- [ ] Set up nodes/ directory structure
- [ ] Create README.md markers
- [ ] Create .gitignore rules
- [ ] Build contract templates
- [ ] Set up CI/CD validation
- [ ] Create quality gates
- [ ] Document generation workflow

**Deliverables:**
- Clean `nodes/` directory structure
- Contract templates
- CI/CD pipeline
- Generation documentation

### Week 7-8: Documentation & Polish

**Focus:** Comprehensive documentation

- [ ] Architecture documentation
- [ ] Infrastructure utilities guide
- [ ] Shared models guide
- [ ] Testing guide
- [ ] Generation workflow guide
- [ ] Troubleshooting guide
- [ ] API reference

**Deliverables:**
- Complete documentation suite
- Architecture diagrams
- Usage examples
- Best practices guide

---

## ✅ Success Criteria

### Phase 1 Complete When:
- ✅ All foundation utilities implemented and tested
- ✅ >90% test coverage on infrastructure code
- ✅ APIs documented and stable
- ✅ Performance benchmarks met

### Phase 2 Complete When:
- ✅ Shared model strategy decided and implemented
- ✅ Model contracts defined (if generated)
- ✅ Models tested and validated
- ✅ Documentation complete

### Phase 3 Complete When:
- ✅ Test fixtures cover all infrastructure components
- ✅ Test utilities available for node testing
- ✅ Testing documentation complete
- ✅ CI/CD integration working

### Phase 4 Complete When:
- ✅ Directory structure ready for generation
- ✅ Contract templates available
- ✅ Generation markers in place
- ✅ CI/CD validation configured

### Ready for Node Generation When:
- ✅ All phases 1-4 complete
- ✅ Quality gates passing
- ✅ Documentation complete
- ✅ Team trained on workflow

---

## 🚀 Post-Generation Workflow

### When Generation Pipeline is Ready:

1. **Generate Adapter Nodes**
   ```bash
   # Generate postgres adapter from contract
   omninode-generate --type effect \
                     --contract contracts/postgres_adapter.yaml \
                     --output src/omnibase_infra/nodes/postgres_adapter/v1_0_0/

   # Repeat for kafka, vault, consul adapters
   ```

2. **Validate Generated Code**
   ```bash
   # Run CI/CD validation
   poetry run pytest tests/nodes/postgres_adapter/
   poetry run mypy src/omnibase_infra/nodes/postgres_adapter/
   poetry run ruff check src/omnibase_infra/nodes/postgres_adapter/
   ```

3. **Integration Testing**
   ```bash
   # Test with real infrastructure
   poetry run pytest tests/integration/ -v
   ```

4. **Performance Benchmarking**
   ```bash
   # Measure performance
   poetry run pytest tests/performance/ -v
   ```

5. **Deploy to Dev/Staging**
   ```bash
   # Deploy generated infrastructure
   docker compose -f deployment/docker-compose.dev.yml up -d
   ```

6. **Monitor & Iterate**
   - Review observability data
   - Identify issues
   - Update contracts
   - Regenerate
   - Repeat

---

## 🎯 Key Decisions Required

### Decision 1: Shared Models Strategy

**Question:** Should shared models be hand-written or generated?

**Options:**
- **A:** Hand-written stable models (predictable, more work)
- **B:** Generated from model contracts (DRY, automatic updates)
- **C:** Hybrid (core hand-written, extensions generated)

**Recommendation:** Option B (Generated)
- Contract-driven consistency
- Automatic updates
- No manual sync

**Action Required:** Choose and document decision

### Decision 2: Infrastructure Utility Scope

**Question:** What should be hand-written vs generated?

**Hand-Written (Recommended):**
- Connection pools
- Health monitors
- Circuit breakers
- Logging/tracing setup
- Metrics collection

**Generated:**
- All node implementations
- Node-specific handlers
- Node-specific models
- DI registries

**Action Required:** Finalize scope and document

### Decision 3: Testing Strategy

**Question:** How much test coverage before generation?

**Recommendation:**
- Infrastructure utilities: >90% coverage
- Test fixtures: Comprehensive (all services)
- Generated nodes: Tests also generated
- Integration tests: Hand-written

**Action Required:** Set coverage requirements

### Decision 4: CI/CD Validation

**Question:** What quality gates for generated code?

**Recommendation:**
- Type checking: Required
- Linting: Required
- Unit tests: Required (>80% coverage)
- Integration tests: Required
- Security scans: Required
- Performance benchmarks: Optional (initially)

**Action Required:** Configure CI/CD pipeline

---

## 📊 Risk Assessment

### Low Risk ✅

- Foundation utilities implementation
- Test infrastructure setup
- Documentation framework
- Directory structure

### Medium Risk ⚠️

- Shared model generation strategy
- CI/CD pipeline complexity
- Performance tuning
- Migration timing

### High Risk 🚨

- Contract-code consistency drift
- Breaking changes during regeneration
- Test coverage gaps
- Production deployment timing

**Mitigation Strategies:**
- Start simple, iterate
- Comprehensive testing
- Incremental rollout
- Rollback procedures

---

## 🎬 Next Steps

### Immediate (This Week)

1. **Commit current archiving work** ✅
2. **Review this plan with team**
3. **Make key decisions** (models strategy, scope)
4. **Start Phase 1: PostgreSQL utilities**

### Short Term (Next 2 Weeks)

1. Complete foundation utilities
2. Build testing infrastructure
3. Set up generation landing zone
4. Create contract templates

### Medium Term (4-6 Weeks)

1. Complete all preparation phases
2. Validate with generation pipeline
3. Generate first adapter node
4. Integration testing

### Long Term (2-3 Months)

1. Generate all infrastructure nodes
2. Comprehensive testing
3. Production deployment
4. Monitoring and optimization

---

## 📝 Conclusion

This plan provides a **clear, structured approach** to preparing omnibase_infra for code generation:

✅ **Build stable foundation** - Hand-written utilities
✅ **Define model strategy** - Contract-driven generation
✅ **Create test infrastructure** - Comprehensive validation
✅ **Prepare landing zone** - Clean directory structure
✅ **Establish quality gates** - CI/CD automation
✅ **Document everything** - Clear guidance

**The Result:**
A production-ready infrastructure repository that receives generated nodes cleanly, validates them automatically, and deploys them confidently.

**Key Principle:**
> "Don't hand-write what you're going to generate. Build the foundation, then generate the house."

---

**Status:** Ready for review and team decision-making
**Next Action:** Commit this plan and begin Phase 1 implementation
