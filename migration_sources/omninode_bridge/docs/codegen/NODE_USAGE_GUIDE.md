# Node-Based Code Generator - Usage Guide

**Complete**: All 5 nodes implemented with comprehensive testing

This guide shows how to use the node-based code generator architecture with all implemented nodes.

---

## Overview

The node-based code generator provides a modular, testable architecture for code generation workflows:

### Core Nodes (All Implemented ✅)

1. **NodeCodegenStubExtractorEffect** - Extract method stubs from generated files
2. **NodeCodegenCodeValidatorEffect** - Validate code for compliance and quality
3. **NodeCodegenCodeInjectorEffect** - Inject validated code back into files
4. **NodeCodegenStoreEffect** - Persist artifacts to file system
5. **NodeCodegenMetricsReducer** - Pure metrics aggregation with intent publishing

---

## Quick Start

### 1. Extract Stubs from Generated Node

```python
from omnibase_core.models.core import ModelContainer
from omnibase_core.models.contracts.model_contract_effect import ModelContractEffect
from omninode_bridge.nodes.codegen_stub_extractor_effect.v1_0_0.node import NodeCodegenStubExtractorEffect

# Initialize node with container
container = ModelContainer(...)
extractor = NodeCodegenStubExtractorEffect(container)

# Create contract
contract = ModelContractEffect(
    correlation_id=uuid4(),
    input_state={
        "node_file_content": node_source_code,
        "extraction_patterns": ["# IMPLEMENTATION REQUIRED", "pass  # Stub"],
    }
)

# Execute extraction
result = await extractor.execute_effect(contract)

print(f"Found {result.total_stubs_found} stubs")
for stub in result.stubs:
    print(f"  - {stub.name} at line {stub.line_number}")
```

**Performance**: <100ms for typical node file

---

### 2. Validate Generated Code

```python
from omninode_bridge.nodes.codegen_code_validator_effect.v1_0_0.node import NodeCodegenCodeValidatorEffect
from omninode_bridge.nodes.codegen_code_validator_effect.v1_0_0.models.enum_validation_rule import EnumValidationRule

# Initialize validator
validator = NodeCodegenCodeValidatorEffect(container)

# Create contract
contract = ModelContractEffect(
    correlation_id=uuid4(),
    input_state={
        "generated_code": implementation_code,
        "validation_rules": [
            EnumValidationRule.SYNTAX,
            EnumValidationRule.ONEX_COMPLIANCE,
            EnumValidationRule.SECURITY,
        ],
    }
)

# Execute validation
result = await validator.execute_effect(contract)

if result.is_valid:
    print(f"✅ Code passed all validations (score: {result.quality_score})")
else:
    print(f"❌ Validation failed:")
    for error in result.validation_errors:
        print(f"  - {error.error_type}: {error.message}")
```

**Performance**: <500ms for typical validation

---

### 3. Inject Code into Node File

```python
from omninode_bridge.nodes.codegen_code_injector_effect.v1_0_0.node import NodeCodegenCodeInjectorEffect

# Initialize injector
injector = NodeCodegenCodeInjectorEffect(container)

# Create injection requests
injection_requests = [
    {
        "method_name": "execute_effect",
        "line_number": 42,
        "generated_code": "    result = await self._process(contract)\n    return result",
        "preserve_signature": True,
        "preserve_docstring": True,
    }
]

# Create contract
contract = ModelContractEffect(
    correlation_id=uuid4(),
    input_state={
        "source_code": original_source,
        "injection_requests": injection_requests,
    }
)

# Execute injection
result = await injector.execute_effect(contract)

print(f"Injected code into {result.injections_performed} methods")
print(f"Modified source:\n{result.modified_source}")
```

**Performance**: <200ms for typical injection

---

### 4. Store Generated Artifacts

```python
from omninode_bridge.nodes.codegen_store_effect.v1_0_0.node import NodeCodegenStoreEffect

# Initialize store
store = NodeCodegenStoreEffect(container)

# Create storage requests
storage_requests = [
    {
        "file_path": "nodes/my_node/v1_0_0/node.py",
        "content": modified_source,
        "artifact_type": "node_file",
        "create_directories": True,
        "file_permissions": "0644",
    }
]

# Create contract
contract = ModelContractEffect(
    correlation_id=uuid4(),
    input_state={
        "storage_requests": storage_requests,
        "base_directory": "./generated",
    }
)

# Execute storage
result = await store.execute_effect(contract)

print(f"Stored {result.artifacts_stored} artifacts")
print(f"Total bytes written: {result.total_bytes_written}")
for file_path in result.stored_files:
    print(f"  ✓ {file_path}")
```

**Performance**: <1s for typical storage operation

---

### 5. Aggregate Metrics (Pure Reducer)

```python
from omninode_bridge.nodes.codegen_metrics_reducer.v1_0_0.node import NodeCodegenMetricsReducer
from omnibase_core.models.contracts.model_contract_reducer import ModelContractReducer

# Initialize reducer (pure, with MixinIntentPublisher)
reducer = NodeCodegenMetricsReducer(container)

# Create event stream
async def event_stream():
    for event in codegen_events:
        yield event

# Create contract
contract = ModelContractReducer(
    correlation_id=uuid4(),
    input_stream=event_stream(),
    input_state={"window_type": "hourly"}
)

# Execute reduction (pure aggregation + intent publishing)
metrics_state = await reducer.execute_reduction(contract)

print(f"Aggregated {metrics_state.total_events_processed} events")
print(f"Throughput: {metrics_state.items_per_second:.0f} events/sec")
print(f"Duration: {metrics_state.aggregation_duration_ms:.2f}ms")
```

**Performance**: >1000 events/sec, <100ms for 1000 items

---

## Complete Workflow Example

See `tests/integration/test_node_based_codegen_workflow.py` for complete end-to-end workflow:

```python
# 1. Extract stubs from generated file
extraction_result = await extractor.execute_effect(extract_contract)

# 2. Validate generated implementations
for stub in extraction_result.stubs:
    validation_result = await validator.execute_effect(validate_contract)

# 3. Inject validated code
injection_result = await injector.execute_effect(inject_contract)

# 4. Store modified artifacts
storage_result = await store.execute_effect(storage_contract)

# 5. Aggregate workflow metrics
metrics_state = await reducer.execute_reduction(metrics_contract)
```

---

## Orchestrator Integration

The orchestrator now supports **intent execution** for routing intents from reducer:

```python
from omninode_bridge.nodes.codegen_orchestrator.v1_0_0.node import NodeCodegenOrchestrator

# Initialize orchestrator
orchestrator = NodeCodegenOrchestrator(container)

# Reducer returns intents (not direct I/O)
intents = [
    {
        "intent_type": "STORE_ARTIFACT",
        "target": "store_effect",
        "payload": {
            "file_path": "/path/to/artifact.py",
            "content": "...",
        },
        "priority": 1
    }
]

# Orchestrator executes intents
results = await orchestrator.execute_intents(intents)

print(f"Executed {results['executed_intents']} intents")
print(f"Failed {results['failed_intents']} intents")
```

---

## Contract-Based Testing

All nodes support contract-based testing - same contract works with mock or real services:

```python
# Mock container for unit tests
mock_container = Mock(spec=ModelContainer)
mock_container.config.get = Mock(side_effect=lambda k, default: default)
node = NodeCodegenStubExtractorEffect(mock_container)

# Real container for integration tests
real_container = ModelContainer(
    config={
        "stub_extraction_patterns": ["# TODO", "# FIXME"],
        "kafka_broker_url": "localhost:9092",
    }
)
node = NodeCodegenStubExtractorEffect(real_container)

# Same contract works with both!
contract = ModelContractEffect(...)
result = await node.execute_effect(contract)
```

---

## Performance Targets (All Met ✅)

| Node | Target | Achieved |
|------|--------|----------|
| StubExtractor | <100ms | ✅ <100ms |
| CodeValidator | <500ms | ✅ <500ms |
| CodeInjector | <200ms | ✅ <200ms |
| StoreEffect | <1s | ✅ <1s |
| MetricsReducer | >1000 events/sec | ✅ >1000 events/sec |

---

## Architecture Patterns

### Pure Reducer Pattern (MetricsReducer)

```python
class NodeCodegenMetricsReducer(NodeReducer, MixinIntentPublisher):

    async def execute_reduction(self, contract):
        # 1. Pure aggregation (NO I/O)
        metrics = self.aggregator.aggregate_events(events)

        # 2. Build event (pure)
        event = ModelMetricsRecordedEvent(metrics=metrics)

        # 3. Publish intent (coordination I/O via mixin)
        await self.publish_event_intent(
            target_topic=TOPIC_CODEGEN_METRICS,
            target_key=str(metrics.aggregation_id),
            event=event
        )

        # 4. Return state (NO direct I/O)
        return metrics
```

### Effect Node Pattern

```python
class NodeMyEffect(NodeEffect):

    async def execute_effect(self, contract: ModelContractEffect):
        # 1. Extract input from contract
        input_data = contract.input_state

        # 2. Execute effect logic
        result = await self._process(input_data)

        # 3. Return typed result
        return ModelMyEffectResult(...)
```

---

## Testing

### Run All Integration Tests

```bash
# All node-based codegen tests
pytest tests/integration/test_node_based_codegen_workflow.py -v

# Performance tests
pytest tests/integration/test_node_based_codegen_workflow.py -v -m performance
```

### Run Individual Node Tests

```bash
# Stub Extractor tests
pytest src/omninode_bridge/nodes/codegen_stub_extractor_effect/v1_0_0/tests/ -v

# Validator tests
pytest src/omninode_bridge/nodes/codegen_code_validator_effect/v1_0_0/tests/ -v

# Injector tests
pytest src/omninode_bridge/nodes/codegen_code_injector_effect/v1_0_0/tests/ -v

# Store Effect tests
pytest src/omninode_bridge/nodes/codegen_store_effect/v1_0_0/tests/ -v

# Metrics Reducer tests
pytest src/omninode_bridge/nodes/codegen_metrics_reducer/v1_0_0/tests/ -v
```

---

## Next Steps

1. **Integration into existing workflow**: Add refinement steps to `CodeGenerationWorkflow`
2. **Enhanced validation rules**: Extend `EnumValidationRule` for domain-specific checks
3. **Custom stub patterns**: Configure extraction patterns per node type
4. **Metrics visualization**: Build dashboard from aggregated metrics
5. **Horizontal scaling**: Deploy multiple reducer instances for throughput

---

## Reference Documentation

- **[Implementation Status](./IMPLEMENTATION_STATUS.md)** - Current progress and metrics
- **[Implementation Guide](./NODE_BASED_CODEGEN_IMPLEMENTATION_GUIDE.md)** - Complete patterns and examples
- **[Integration Tests](../../tests/integration/test_node_based_codegen_workflow.py)** - Working examples

---

**Status**: ✅ Production Ready - All 5 nodes complete with comprehensive tests
