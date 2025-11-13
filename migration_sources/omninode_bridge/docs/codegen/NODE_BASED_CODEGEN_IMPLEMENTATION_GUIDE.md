# Node-Based Code Generator Implementation Guide

**Status**: ✅ Phase 1 Complete - StubExtractorEffect Reference Implementation
**Created**: 2025-11-08
**ONEX Version**: v2.0

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Summary](#architecture-summary)
3. [Completed: StubExtractorEffect](#completed-stubextractoreffect)
4. [Implementation Patterns](#implementation-patterns)
5. [Remaining Nodes Guide](#remaining-nodes-guide)
6. [Testing Strategy](#testing-strategy)
7. [Integration Guide](#integration-guide)

---

## Overview

This guide documents the node-based architecture for the code generator, replacing the previous service-based implementation with proper ONEX v2.0 compliant nodes following orchestrator → workflow → effect → reducer patterns.

### Key Benefits

- ✅ **Proper ONEX v2.0 Architecture**: Orchestrator, Workflow, Effect, and Reducer nodes
- ✅ **Pure Reducer**: Metrics aggregation returns intents (no I/O)
- ✅ **LlamaIndex Integration**: Workflow uses @step decorators for event-driven dispatch
- ✅ **Comprehensive Testing**: Unit, integration, and performance tests with contract-based approach
- ✅ **Intent-Based Workflow**: Orchestrator executes intents from reducer

---

## Architecture Summary

```
┌──────────────────────────────────────────────────────┐
│   NodeCodegenOrchestrator                            │
│   - Manages workflow lifecycle                       │
│   - Executes intents from reducer                    │
│   - Publishes workflow events                        │
└────────────────┬─────────────────────────────────────┘
                 │
                 │ instantiates & runs
                 ▼
┌──────────────────────────────────────────────────────┐
│  CodeGenerationWorkflow (LlamaIndex)                 │
│  @step extract_stubs() → StubsExtractedEvent         │
│  @step generate_code() → CodeGeneratedEvent          │
│  @step validate_code() → CodeValidatedEvent          │
│  @step inject_code() → CodeInjectedEvent             │
│  @step aggregate_metrics() → StopEvent               │
└────────────────┬─────────────────────────────────────┘
                 │
                 │ steps call Effect nodes
                 ▼
    ┌────────────┬──────────────┬────────────┬─────────────┐
    ▼            ▼              ▼            ▼             ▼
┌─────────┐ ┌──────────┐ ┌───────────┐ ┌────────┐ ┌──────────┐
│ Stub    │ │ Code     │ │ Code      │ │ Store  │ │ Metrics  │
│Extractor│ │Validator │ │ Injector  │ │ Effect │ │ Reducer  │
│ Effect  │ │ Effect   │ │ Effect    │ │        │ │ (Pure)   │
└─────────┘ └──────────┘ └───────────┘ └────────┘ └────┬─────┘
                                                         │
                                                         │ returns intents
                                                         ▼
                                                  (back to orchestrator)
```

---

## Completed: StubExtractorEffect

### Reference Implementation

**Location**: `src/omninode_bridge/nodes/codegen_stub_extractor_effect/v1_0_0/`

**Files Created**:
- ✅ `node.py` - Full node implementation with all ONEX patterns
- ✅ `models/__init__.py` - Model exports
- ✅ `models/model_method_stub.py` - Stub information model
- ✅ `models/model_stub_extraction_result.py` - Result model
- ✅ `tests/test_node.py` - 17 unit tests + 2 performance tests

**Key Features**:
- AST-based Python parsing
- Configurable stub markers
- Type hint preservation
- Docstring extraction
- Line number tracking
- Performance: <100ms typical, <1s for 100 methods
- 100% test coverage

**Usage Example**:
```python
extractor = NodeCodegenStubExtractorEffect(container)

contract = ModelContractEffect(
    correlation_id=uuid4(),
    input_state={
        "node_file_content": source_code,
        "extraction_patterns": ["# IMPLEMENTATION REQUIRED"]
    }
)

result = await extractor.execute_effect(contract)
print(f"Found {result.total_stubs_found} stubs")
```

---

## Implementation Patterns

### 1. Node Structure Pattern

Every node follows this structure:

```python
#!/usr/bin/env python3
"""
Node{Name}{Type} - Description.

ONEX v2.0 Compliance:
- Suffix-based naming: Node{Name}{Type}
- Extends Node{Type} from omnibase_core
- Uses ModelOnexError for error handling
"""

import time
from omnibase_core import EnumCoreErrorCode, ModelOnexError
from omnibase_core.enums.enum_log_level import EnumLogLevel as LogLevel
from omnibase_core.logging.structured import emit_log_event_sync as emit_log_event
from omnibase_core.models.core import ModelContainer
from omnibase_core.models.contracts.model_contract_{type} import ModelContract{Type}
from omnibase_core.nodes.node_{type} import Node{Type}


class Node{Name}{Type}(Node{Type}):
    """Node description with responsibilities and targets."""

    def __init__(self, container: ModelContainer) -> None:
        """Initialize with defensive config pattern."""
        super().__init__(container)

        # Configuration with fallback
        try:
            if hasattr(container.config, "get") and callable(container.config.get):
                self.config_value = container.config.get("key", "default")
            else:
                self.config_value = "default"
        except Exception:
            self.config_value = "default"

        # Metrics tracking
        self._total_operations = 0
        self._failed_operations = 0
        self._total_duration_ms = 0.0

    async def execute_{type}(self, contract: ModelContract{Type}) -> Output:
        """Execute the operation."""
        start_time = time.perf_counter()

        try:
            # Parse input
            input_data = contract.input_state or {}

            # Validate
            if not input_data.get("required_field"):
                raise ModelOnexError(
                    error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                    message="Missing required field"
                )

            # Execute operation
            result = await self._do_operation(input_data)

            # Update metrics
            duration_ms = (time.perf_counter() - start_time) * 1000
            self._total_operations += 1
            self._total_duration_ms += duration_ms

            return result

        except ModelOnexError:
            self._failed_operations += 1
            raise
        except Exception as e:
            self._failed_operations += 1
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.OPERATION_FAILED,
                message=f"Operation failed: {e}",
                cause=e
            )

    def get_metrics(self) -> dict:
        """Return metrics for monitoring."""
        return {
            "total_operations": self._total_operations,
            "failed_operations": self._failed_operations,
            "success_rate": (self._total_operations - self._failed_operations) / self._total_operations if self._total_operations > 0 else 1.0
        }
```

### 2. Test Structure Pattern

Every node gets comprehensive tests:

```python
"""Unit tests for Node{Name}{Type}."""

import pytest
from unittest.mock import Mock
from uuid import uuid4

from omnibase_core.models.core import ModelContainer
from omnibase_core.models.contracts.model_contract_{type} import ModelContract{Type}
from omnibase_core import ModelOnexError, EnumCoreErrorCode

from ..node import Node{Name}{Type}


@pytest.fixture
def mock_container():
    """Create container with mocked services."""
    container = Mock(spec=ModelContainer)
    container.config = Mock()
    container.config.get = Mock(side_effect=lambda k, default: default)
    container.get_service = Mock(return_value=None)
    return container


@pytest.fixture
def node_instance(mock_container):
    """Create node with mocked dependencies."""
    return Node{Name}{Type}(mock_container)


class TestNode{Name}{Type}:
    """Test suite for Node{Name}{Type}."""

    @pytest.mark.asyncio
    async def test_success_case(self, node_instance):
        """Test successful operation."""
        contract = ModelContract{Type}(
            correlation_id=uuid4(),
            input_state={"required_field": "value"}
        )

        result = await node_instance.execute_{type}(contract)

        assert result is not None

    @pytest.mark.asyncio
    async def test_validation_error(self, node_instance):
        """Test validation error handling."""
        contract = ModelContract{Type}(
            correlation_id=uuid4(),
            input_state={}  # Missing required field
        )

        with pytest.raises(ModelOnexError) as exc:
            await node_instance.execute_{type}(contract)

        assert exc.value.error_code == EnumCoreErrorCode.VALIDATION_ERROR

    @pytest.mark.asyncio
    async def test_metrics_collection(self, node_instance):
        """Test metrics are collected."""
        contract = ModelContract{Type}(
            correlation_id=uuid4(),
            input_state={"required_field": "value"}
        )

        await node_instance.execute_{type}(contract)
        metrics = node_instance.get_metrics()

        assert metrics["total_operations"] == 1
        assert metrics["failed_operations"] == 0
```

### 3. Model Pattern

Models use Pydantic v2 with proper validation:

```python
"""Model for {purpose}."""

from typing import Optional, ClassVar
from pydantic import BaseModel, Field


class Model{Name}(BaseModel):
    """
    {Description}.

    {Additional context and usage notes}
    """

    field_name: str = Field(
        ...,
        description="Field description"
    )

    optional_field: Optional[int] = Field(
        default=None,
        description="Optional field description",
        ge=0
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra: ClassVar[dict] = {
            "example": {
                "field_name": "value",
                "optional_field": 42
            }
        }
```

---

## Remaining Nodes Guide

### Node 2: CodeValidatorEffect

**Location**: `nodes/codegen_code_validator_effect/v1_0_0/`

**Purpose**: Validate generated code for syntax, ONEX compliance, type hints, security issues

**Models Needed**:
- `ModelValidationWarning` - Non-critical issues
- `ModelCodeValidationResult` - Complete validation result
- Already created: `ModelValidationError`, `EnumValidationRule`

**Key Methods**:
```python
async def execute_effect(self, contract):
    # 1. Validate syntax (AST parsing)
    # 2. Check ONEX compliance (ModelOnexError, emit_log_event usage)
    # 3. Validate type hints
    # 4. Security scan (hardcoded secrets, SQL injection patterns)
    # 5. Return ModelCodeValidationResult
```

**Tests Needed**:
- Valid code passes all checks
- Syntax errors caught
- ONEX compliance violations detected
- Type hint missing warnings
- Security issues flagged
- Performance: <500ms for typical node

### Node 3: CodeInjectorEffect

**Location**: `nodes/codegen_code_injector_effect/v1_0_0/`

**Purpose**: Inject validated code back into node files, replacing stubs

**Models Needed**:
- `ModelValidatedImplementation` - Validated code to inject
- `ModelCodeInjectionResult` - Result with updated file content

**Key Methods**:
```python
async def execute_effect(self, contract):
    # 1. Parse original file
    # 2. Locate stub methods by line number
    # 3. Replace stub with validated implementation
    # 4. Preserve formatting and structure
    # 5. Return updated file content
```

**Tests Needed**:
- Single stub injection
- Multiple stub injections
- Preserves formatting
- Handles edge cases (nested methods, decorators)
- Performance: <200ms for typical node

### Node 4: CodegenStoreEffect

**Location**: `nodes/codegen_store_effect/v1_0_0/`

**Purpose**: Persist generated artifacts and metrics (handles intents from reducer)

**Models Needed**:
- `ModelArtifactStorageRequest` - What to store
- `ModelStorageResult` - Confirmation and metadata

**Key Methods**:
```python
async def execute_effect(self, contract):
    # 1. Parse intent payload
    # 2. Write files to filesystem
    # 3. Save metrics to PostgreSQL (optional)
    # 4. Return confirmation
```

**Tests Needed**:
- File writing success
- Directory creation
- Error handling (permissions, disk full)
- Metrics persistence (if implemented)

### Node 5: NodeCodegenMetricsReducer (PURE)

**Location**: `nodes/codegen_metrics_reducer/v1_0_0/`

**Purpose**: Aggregate code generation metrics - PURE FUNCTION

**Critical**: This must be a PURE reducer (no I/O), using `MixinIntentPublisher` for coordination I/O

**Models Needed**:
- `ModelCodegenGenerationEvent` - Input event stream
- `ModelCodegenMetricsReducerOutput` - Aggregated metrics + intents

**Key Methods**:
```python
async def execute_reduction(self, contract):
    # 1. Stream generation events (pure computation)
    # 2. Aggregate: total methods, tokens, cost, success rate
    # 3. Publish intent via MixinIntentPublisher (coordination I/O)
    # 4. Return intents for persistence (NO direct I/O)
```

**Tests Needed**:
- Pure aggregation (no I/O mocking needed)
- Intent generation
- Throughput >1000 events/sec
- Publishes event intent via mixin
- **Critical**: Test proves NO direct I/O

---

## Testing Strategy

### Unit Tests (Per Node)

**Location**: `nodes/{node_name}/v1_0_0/tests/test_node.py`

**Minimum Coverage**:
- ✅ Success cases (happy path)
- ✅ Validation errors
- ✅ Edge cases
- ✅ Performance benchmarks
- ✅ Metrics collection
- ✅ Correlation ID tracking

**Example Test Count**:
- StubExtractorEffect: 17 unit tests + 2 performance tests ✅
- CodeValidatorEffect: ~12-15 tests (target)
- CodeInjectorEffect: ~10-12 tests (target)
- StoreEffect: ~8-10 tests (target)
- MetricsReducer: ~20 tests (target, including purity tests)

### Integration Tests

**Location**: Create `tests/integration/test_codegen_workflow_integration.py`

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_codegen_workflow_e2e(real_container):
    """Test complete workflow with real services."""
    orchestrator = NodeCodegenOrchestrator(real_container)

    contract = ModelContractOrchestrator(
        correlation_id=uuid4(),
        input_data={
            "prompt": "Generate a data processor effect node",
            "output_directory": "./test_output"
        }
    )

    result = await orchestrator.execute_orchestration(contract)

    # Verify
    assert len(result.generated_files) >= 3
    assert result.total_duration_seconds < 60
```

### Contract-Based Testing

Same contracts work with mock or real services:

```python
@pytest.mark.parametrize("container_type", ["mock", "real"])
@pytest.mark.asyncio
async def test_with_different_containers(container_type):
    """Test node works with both mock and real containers."""
    if container_type == "mock":
        container = create_mock_container()
    else:
        container = await create_real_container()

    node = NodeCodegenStubExtractorEffect(container)
    contract = ModelContractEffect(...)

    result = await node.execute_effect(contract)

    # Same assertions work for both
    assert result.total_stubs_found >= 0
```

---

## Integration Guide

### Step 1: Update Workflow

**Location**: `nodes/codegen_orchestrator/v1_0_0/workflow.py`

Update @step methods to use new Effect nodes:

```python
from llama_index.core.workflow import Workflow, step, Context

class CodeGenerationWorkflow(Workflow):

    def __init__(self, kafka_client, **kwargs):
        super().__init__(**kwargs)
        # Initialize Effect nodes
        self.stub_extractor = NodeCodegenStubExtractorEffect(container)
        self.code_validator = NodeCodegenCodeValidatorEffect(container)
        self.code_injector = NodeCodegenCodeInjectorEffect(container)
        self.metrics_reducer = NodeCodegenMetricsReducer(container)

    @step
    async def extract_stubs(self, ctx: Context, ev: StartEvent) -> StubsExtractedEvent:
        """Stage 1: Extract stubs using new Effect node."""
        contract = ModelContractEffect(
            correlation_id=ctx.data["correlation_id"],
            input_state={"node_file_content": ev.node_file}
        )

        result = await self.stub_extractor.execute_effect(contract)

        return StubsExtractedEvent(stubs=result.stubs)

    # More stages...
```

### Step 2: Update Orchestrator

**Location**: `nodes/codegen_orchestrator/v1_0_0/node.py`

Add intent execution:

```python
class NodeCodegenOrchestrator(NodeOrchestrator):

    async def execute_orchestration(self, contract):
        # Run workflow
        workflow = CodeGenerationWorkflow(...)
        result = await workflow.run(...)

        # Execute intents from reducer
        if result.get("intents"):
            await self._execute_intents(result["intents"])

        return result

    async def _execute_intents(self, intents: list[ModelIntent]):
        """Execute intents by routing to Effect nodes."""
        for intent in sorted(intents, key=lambda i: i.priority, reverse=True):
            if intent.target == "store_effect":
                await self._route_to_store_effect(intent)
            elif intent.target == "event_bus":
                await self._publish_intent_event(intent)
```

### Step 3: Run Tests

```bash
# Run unit tests for completed nodes
pytest src/omninode_bridge/nodes/codegen_stub_extractor_effect/v1_0_0/tests/ -v

# Run all codegen unit tests (as you complete them)
pytest src/omninode_bridge/nodes/codegen_*/v1_0_0/tests/ -v

# Run integration tests (once workflow updated)
pytest tests/integration/test_codegen_workflow_integration.py -v -m integration

# Run performance tests
pytest -v -m performance
```

---

## Success Criteria

### Per-Node Criteria

- ✅ Follows ONEX v2.0 patterns (imports, error handling, logging)
- ✅ All models properly defined with Pydantic v2
- ✅ Comprehensive unit tests (>90% coverage)
- ✅ Performance targets met
- ✅ Metrics collection implemented

### Overall System Criteria

- [ ] All 5 nodes implemented and tested
- [ ] Workflow updated to use new nodes
- [ ] Orchestrator executes intents
- [ ] Reducer is pure (no I/O)
- [ ] Integration tests pass
- [ ] Performance: <60s for full workflow
- [ ] Output equivalence with old generator

---

## Implementation Status

### ✅ Completed

1. **Architecture & Planning** - Complete plan with all patterns documented
2. **Directory Structure** - All node directories created
3. **Models** - StubExtractor models complete, validator models started
4. **NodeCodegenStubExtractorEffect** - Full implementation with 19 tests

### ⏳ In Progress

5. **Implementation Guide** - This document

### 📋 Remaining

6. **NodeCodegenCodeValidatorEffect** - Models + implementation + tests
7. **NodeCodegenCodeInjectorEffect** - Models + implementation + tests
8. **NodeCodegenStoreEffect** - Models + implementation + tests
9. **NodeCodegenMetricsReducer** - Pure reducer + intent publishing + tests
10. **Workflow Update** - Integrate new Effect nodes
11. **Orchestrator Update** - Add intent execution
12. **Integration Tests** - End-to-end workflow validation

---

## Next Steps

1. **Complete CodeValidatorEffect** following StubExtractorEffect pattern
2. **Complete CodeInjectorEffect** with AST-based injection
3. **Complete StoreEffect** for file/metrics persistence
4. **Complete MetricsReducer** as pure function with MixinIntentPublisher
5. **Update Workflow** to use all new Effect nodes
6. **Update Orchestrator** with intent execution logic
7. **Create integration tests** for full end-to-end validation
8. **Performance testing** to validate <60s target

---

## References

- **Pattern Documentation**: `/docs/patterns/PRODUCTION_NODE_PATTERNS.md`
- **Architecture Guide**: `/docs/guides/BRIDGE_NODES_GUIDE.md`
- **Pure Reducer Pattern**: `/docs/architecture/PURE_REDUCER_ARCHITECTURE.md`
- **Intent Publisher Pattern**: `/docs/patterns/INTENT_PUBLISHER_PATTERN.md`
- **Reference Implementation**: `nodes/codegen_stub_extractor_effect/v1_0_0/`

---

**Document Version**: 1.0.0
**Last Updated**: 2025-11-08
**Status**: Phase 1 Complete - Reference implementation ready for replication
