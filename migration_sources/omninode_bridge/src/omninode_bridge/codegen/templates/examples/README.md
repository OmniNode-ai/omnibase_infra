# Test Contract Examples - ONEX Node Types

This directory contains comprehensive test contract examples for all ONEX node types, demonstrating best practices for testing each node type.

## Available Examples

### 1. Effect Node Test Contract
**File**: `test_contract_effect_example.yaml`
**Node**: `NodePostgresCrudEffect`
**Purpose**: Database CRUD operations with connection pooling

**Key Testing Patterns**:
- Unit tests for database operations (mocked connections)
- Integration tests for connection pool behavior
- Performance tests for query latency
- Security tests for SQL injection prevention
- Contract tests for ONEX Effect compliance (no side effects in pure compute)

**Mock Strategy**:
- Mock database connections using AsyncMock
- Mock SQL queries with patch and call tracking
- Simulate connection pool exhaustion
- Simulate database failures and recovery

**Performance Targets**:
- Create/Update operations: <50ms per operation
- Read operations: <20ms per operation
- Throughput: >100 operations/second

---

### 2. Compute Node Test Contract
**File**: `test_contract_compute_example.yaml`
**Node**: `NodeDataTransformerCompute`
**Purpose**: Pure data transformation without side effects

**Key Testing Patterns**:
- Unit tests for pure functions (no mocks needed)
- Contract tests for deterministic behavior
- Performance tests for throughput
- Property-based testing for edge cases
- Contract tests for ONEX Compute compliance (zero side effects)

**Mock Strategy**:
- Minimal mocking (compute nodes should be pure)
- Mock input data generators only
- No database, HTTP, or external service mocks

**Performance Targets**:
- Data transformation: >1000 operations/second
- Memory efficiency: <100MB for 10K records
- Deterministic results: 100% consistency

---

### 3. Reducer Node Test Contract
**File**: `test_contract_reducer_example.yaml`
**Node**: `NodeMetricsAggregatorReducer`
**Purpose**: Streaming aggregation with state management

**Key Testing Patterns**:
- Unit tests for aggregation logic
- Integration tests for state persistence
- Performance tests for streaming throughput
- Contract tests for ONEX Reducer compliance (state isolation, aggregation semantics)
- Stress tests for high-volume event streams

**Mock Strategy**:
- Mock event stream generators
- Mock state storage (Redis/PostgreSQL)
- Simulate event ordering and out-of-order delivery
- Simulate state corruption and recovery

**Performance Targets**:
- Event processing: >500 events/second
- State persistence: <100ms per checkpoint
- Memory efficiency: <200MB for 100K events

---

### 4. Orchestrator Node Test Contract
**File**: `test_contract_orchestrator_example.yaml`
**Node**: `NodeWorkflowCoordinatorOrchestrator`
**Purpose**: Multi-stage workflow coordination

**Key Testing Patterns**:
- Unit tests for stage execution
- Integration tests for workflow coordination
- Contract tests for event publishing and FSM transitions
- Performance tests for parallel stage execution
- Contract tests for ONEX Orchestrator compliance (workflow semantics, event ordering)

**Mock Strategy**:
- Mock stage handlers and executors
- Mock event bus (Kafka/RabbitMQ)
- Mock dependent services
- Simulate stage failures and retries

**Performance Targets**:
- 3-stage workflow: <2 seconds end-to-end
- 5-stage workflow: <5 seconds end-to-end
- Parallel stage execution: 2-3x speedup vs sequential

---

## Using the Examples

### 1. Copy Example to Your Node

```bash
# Copy the appropriate example for your node type
cp examples/test_contract_effect_example.yaml my_node_test_contract.yaml

# Edit to match your node
vim my_node_test_contract.yaml
```

### 2. Customize for Your Node

Update these sections:
- `name`: Your node's test contract name
- `target_node`: Your node class name
- `test_targets`: Methods/functions to test
- `mock_requirements`: Dependencies specific to your node
- `test_configuration.environment_variables`: Environment setup

### 3. Validate Your Contract

```bash
# Validate contract against ModelContractTest schema
poetry run python src/omninode_bridge/codegen/validate_test_contract.py \
  my_node_test_contract.yaml
```

### 4. Generate Tests (Future)

```bash
# Once test generator is implemented
poetry run python src/omninode_bridge/codegen/generate_tests.py \
  --contract my_node_test_contract.yaml \
  --output tests/nodes/my_node/
```

---

## Node Type Comparison

| Aspect | Effect | Compute | Reducer | Orchestrator |
|--------|--------|---------|---------|--------------|
| **Primary Purpose** | External I/O operations | Pure data transformation | Streaming aggregation | Workflow coordination |
| **Side Effects** | Yes (external systems) | No (pure functions) | Limited (state updates) | Yes (event publishing) |
| **State Management** | Transactional | Stateless | Stateful (aggregation) | Stateful (workflow) |
| **Mock Complexity** | High (DB, HTTP, etc.) | Low (input generators) | Medium (streams, state) | High (stages, events) |
| **Performance Focus** | Latency (<100ms) | Throughput (>1000/s) | Streaming (>500/s) | Workflow (<5s) |
| **Key Tests** | Integration, Security | Unit, Property-based | Integration, Stress | Integration, Contract |

---

## Best Practices

### 1. Test Naming Conventions

```python
# Good test names (descriptive and specific)
test_execute_effect_creates_record_successfully()
test_execute_effect_handles_connection_pool_exhaustion()
test_execute_effect_prevents_sql_injection_attacks()

# Bad test names (vague and generic)
test_execute()
test_database()
test_error()
```

### 2. Mock Requirements

**Effect Nodes**:
- Always mock external dependencies (DB, HTTP, filesystem)
- Use async mocks for async operations
- Simulate both success and failure scenarios

**Compute Nodes**:
- Minimal mocking (only for complex input generation)
- Focus on property-based testing
- Test deterministic behavior

**Reducer Nodes**:
- Mock event streams with realistic data
- Mock state storage with persistence semantics
- Test ordering and out-of-order scenarios

**Orchestrator Nodes**:
- Mock individual stages/steps
- Mock event bus for integration tests
- Test FSM state transitions exhaustively

### 3. Coverage Targets

**Minimum Coverage**: 85% (enforced by CI/CD)
**Target Coverage**: 95% (best practice)

**Critical Paths**: 100% coverage required
- Error handling code
- Security validation
- State transitions
- Event publishing

### 4. Performance Benchmarks

Include performance benchmarks for:
- **Effect**: Database query latency, HTTP request latency
- **Compute**: Data transformation throughput
- **Reducer**: Event processing rate, state checkpoint latency
- **Orchestrator**: Workflow execution time, stage parallelism

### 5. Test Isolation

**Enforce**:
- No shared state between tests
- Each test creates its own fixtures
- Tests can run in any order
- Tests can run in parallel

**Avoid**:
- Global variables
- Shared database state without cleanup
- File system pollution
- Network dependencies

### 6. Deterministic Tests

**Ensure**:
- Fixed random seeds for reproducibility
- Mocked time sources
- Controlled concurrency
- Predictable ordering

**Avoid**:
- Real time dependencies
- Uncontrolled randomness
- Race conditions
- External service dependencies

---

## Validation Checklist

Before submitting your test contract:

- [ ] Contract validates against ModelContractTest schema
- [ ] All test targets have comprehensive test scenarios
- [ ] Mock requirements are complete and specific
- [ ] Performance benchmarks are realistic and measurable
- [ ] Coverage targets are >= 85%
- [ ] Test names follow naming conventions
- [ ] Quality gates are enforced (isolation, determinism)
- [ ] Documentation is complete and accurate

---

## Common Issues and Solutions

### Issue: Contract validation fails with "High coverage targets require explicit test_targets"

**Solution**: Add specific test_targets for coverage >90%:

```yaml
test_targets:
  - target_name: execute_effect
    target_type: method
    test_scenarios:
      - "successful operation"
      - "error handling"
```

### Issue: Mock requirements validation warning

**Solution**: For integration/e2e tests, specify mock dependencies:

```yaml
mock_requirements:
  mock_dependencies:
    - "module.path.to.DependencyClass"
  mock_external_services:
    - "PostgreSQL database"
    - "Kafka broker"
```

### Issue: Performance benchmarks are unrealistic

**Solution**: Base benchmarks on real measurements:

```bash
# Run actual benchmarks first
pytest tests/ -m performance --benchmark-only

# Use P95 latencies as targets
```

### Issue: Tests are flaky (pass/fail randomly)

**Solution**: Enforce determinism:

```yaml
enforce_deterministic_tests: true

# In test setup
random.seed(42)
time.freeze("2024-01-01T00:00:00Z")
```

---

## Reference Documentation

- **[ModelContractTest Schema](../models/model_contract_test.py)** - Contract model definition
- **[Test Generation Guide](../../../../docs/guides/TEST_GENERATION_GUIDE.md)** - Complete test generation guide (future)
- **[ONEX Architecture Patterns](../../../../docs/architecture/ONEX_ARCHITECTURE_PATTERNS.md)** - ONEX design patterns
- **[Testing Best Practices](../../../../docs/CONTRIBUTING.md#testing)** - Project testing standards

---

## Contributing

When adding new examples:

1. Follow the existing structure and naming conventions
2. Validate against ModelContractTest schema
3. Include comprehensive test scenarios
4. Document node-specific patterns
5. Update this README with the new example

---

**Last Updated**: October 2025
**Version**: 1.0.0
**Status**: Production-ready examples
