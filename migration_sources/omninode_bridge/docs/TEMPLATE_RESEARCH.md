# ONEX Node Template Structure - Research Report

## Executive Summary

The OmniNode Bridge codebase contains a **complete, production-grade template system** for ONEX v2.0 node generation. A `TemplateEngine` class already exists and uses **Jinja2 for template rendering** with comprehensive inline fallback support. The template structure is organized by node type (Effect, Compute, Reducer, Orchestrator) with specific stub patterns for marking incomplete code.

---

## 1. Template Directories and Structure

### Location 1: Main Template Directories
```
/Volumes/PRO-G40/Code/omninode_bridge/templates/node_templates/
├── compute/          (empty - reserved for Jinja2 templates)
├── effect/           (empty - reserved for Jinja2 templates)
├── orchestrator/     (empty - reserved for Jinja2 templates)
└── reducer/          (empty - reserved for Jinja2 templates)
```

**Status**: Currently empty - designed to hold Jinja2 template files
- Each subdirectory would contain `node.py.j2` and `contract.yaml.j2` files
- The TemplateEngine looks for templates here first

### Location 2: Code Generation Templates
```
/Volumes/PRO-G40/Code/omninode_bridge/src/omninode_bridge/codegen/templates/
├── test_templates/
│   ├── __init__.py
│   ├── conftest.py.j2              # Pytest configuration & fixtures
│   ├── test_unit.py.j2              # Unit test template
│   ├── test_integration.py.j2       # Integration test template
│   ├── test_performance.py.j2       # Performance test template
│   └── test_contract.py.j2          # Contract compliance tests
├── examples/
│   ├── test_contract_effect_example.yaml
│   ├── test_contract_compute_example.yaml
│   ├── test_contract_reducer_example.yaml
│   └── test_contract_orchestrator_example.yaml
├── test_contract_example.yaml
└── business_logic/
    ├── system_prompt.txt            # LLM system prompt
    └── user_prompt.txt              # LLM user prompt template
```

### Location 3: CI/CD Templates
```
/Volumes/PRO-G40/Code/omninode_bridge/src/omninode_bridge/ci/templates/
├── __init__.py
└── templates.py                      # CI/CD template code
```

---

## 2. Template Engine System

### TemplateEngine Class
**Location**: `/Volumes/PRO-G40/Code/omninode_bridge/src/omninode_bridge/codegen/template_engine.py`

**Key Features**:
1. **Jinja2-based** with environment configuration
2. **Dual rendering**: Filesystem-based OR inline fallback
3. **Security validation** for user inputs (service_name, domain)
4. **Contract validation** against omnibase_core schemas
5. **Comprehensive artifact generation**:
   - Main node file (`node.py`)
   - Contract file (`contract.yaml`)
   - Init file (`__init__.py`)
   - Data models (request/response)
   - Test files (unit, integration, performance, contract)
   - Documentation (README.md)

### Generated Artifacts Model
```python
class ModelGeneratedArtifacts(BaseModel):
    node_file: str                          # Main node implementation
    contract_file: str                      # ONEX contract
    init_file: str                          # Module init
    models: dict[str, str]                  # Data models
    tests: dict[str, str]                   # Test files
    documentation: dict[str, str]           # Documentation
    node_type: str                          # Generated node type
    node_name: str                          # Node class name
    service_name: str                       # Service name
    test_results: Optional[ModelTestResults] # If run_tests=True
    failure_analysis: Optional[ModelFailureAnalysis]
```

---

## 3. Node Template Structure (Inline Fallback)

The TemplateEngine provides **inline templates** for each node type when Jinja2 files aren't available.

### Effect Node Template Structure
```python
class NodeXxxEffect(NodeEffect, HealthCheckMixin, IntrospectionMixin):
    def __init__(self, container: ModelContainer) -> None:
        # Initialization with config extraction
        # Health checks registration
        # Introspection system setup
        pass

    def _register_component_checks(self) -> None:
        # Override for custom health checks
        pass

    async def execute_effect(self, contract: ModelContractEffect) -> Any:
        # IMPLEMENTATION REQUIRED: Add effect-specific logic here
        pass

    async def startup(self) -> None:
        # Node startup with introspection registration
        pass

    async def shutdown(self) -> None:
        # Node shutdown with cleanup
        pass
```

### Compute Node Template Structure
```python
class NodeXxxCompute(NodeCompute, HealthCheckMixin, IntrospectionMixin):
    def __init__(self, container: ModelContainer) -> None:
        pass

    async def execute_compute(self, contract: ModelContractCompute) -> Any:
        # IMPLEMENTATION REQUIRED: Add pure computation logic here
        pass

    async def startup(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass
```

### Reducer Node Template Structure
```python
class NodeXxxReducer(NodeReducer, HealthCheckMixin, IntrospectionMixin):
    def __init__(self, container: ModelContainer) -> None:
        self.accumulated_state: dict[str, Any] = {}
        pass

    async def execute_reduction(self, contract: ModelContractReducer) -> Any:
        # IMPLEMENTATION REQUIRED: Add aggregation/reduction logic here
        pass

    async def startup(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass
```

### Orchestrator Node Template Structure
```python
class NodeXxxOrchestrator(NodeOrchestrator, HealthCheckMixin, IntrospectionMixin):
    def __init__(self, container: ModelContainer) -> None:
        pass

    async def execute_orchestration(self, contract: ModelContractOrchestrator) -> Any:
        # IMPLEMENTATION REQUIRED: Add workflow orchestration logic here
        pass

    async def startup(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass
```

---

## 4. Contract Template Structure (YAML)

**Format**: ONEX v2.0 Compliant

### Required Fields (11 Total):
1. **name**: Service name (string)
2. **version**: Version dict with major/minor/patch
3. **description**: Business description
4. **node_type**: UPPERCASE enum (EFFECT, COMPUTE, REDUCER, ORCHESTRATOR)
5. **input_model**: Input model class name (string)
6. **output_model**: Output model class name (string)
7. **tool_specification**: Tool configuration dict
8. **node_name**: Legacy compatibility (string)
9. **node_version**: Legacy compatibility (version dict)
10. **contract_name**: Legacy compatibility
11. **author**: Legacy compatibility

### Optional Fields:
- **io_operations**: For Effect nodes only (array of ModelIOOperationConfig)
- **metadata**: Domain, tags, generated timestamp
- **operations**: List of supported operations
- **performance_requirements**: Target/max latency and memory
- **testing**: Unit/integration test requirements

### Example Effect Contract:
```yaml
name: "postgres_crud"
version:
  major: 1
  minor: 0
  patch: 0
description: "PostgreSQL CRUD operations with connection pooling"
node_type: "EFFECT"
input_model: "ModelPostgresCrudRequest"
output_model: "ModelPostgresCrudResponse"
tool_specification:
  tool_name: "postgres_crud"
  main_tool_class: "omninode_bridge.nodes.postgres_crud.v1_0_0.node.NodePostgresCrudEffect"

io_operations:
  - operation_type: "database_query"
    atomic: true
    timeout_seconds: 30
    validation_enabled: true

metadata:
  domain: "data_processing"
  generated: true
  generated_at: "2024-10-31T12:00:00Z"
  tags:
    - "database"
    - "postgres"
    - "crud"

operations:
  - "create"
  - "read"
  - "update"
  - "delete"

performance_requirements:
  execution_time:
    target_ms: 100
    max_ms: 1000
  memory_usage:
    target_mb: 128
    max_mb: 512
```

---

## 5. Stub Patterns Used in Templates

### Stub Indicators (from `BusinessLogicConfig`)
```python
STUB_INDICATORS = (
    "# IMPLEMENTATION REQUIRED",    # Primary marker for methods
    "# TODO:",                        # Legacy TODO comments
    "pass  # Stub",                   # Simple pass statements
    "raise NotImplementedError",      # Error-based stub
)
```

### Pattern Examples in Node Templates:

#### Pattern 1: IMPLEMENTATION REQUIRED (Primary)
```python
async def execute_effect(self, contract: ModelContractEffect) -> Any:
    """Execute effect operation."""
    try:
        # IMPLEMENTATION REQUIRED: Add effect-specific logic here
        # This is a generated stub - replace with actual external I/O operations
        # Examples: database queries, API calls, file operations, message publishing

        result = {"status": "success", "message": "Effect executed"}
        return result
    except Exception as e:
        # Error handling...
        raise ModelOnexError(...)
```

#### Pattern 2: TODO Comments (Legacy)
```python
def method_name(self):
    """Method documentation."""
    # TODO: Implement business logic
    pass
```

#### Pattern 3: Simple Pass
```python
def method_name(self):
    """Method documentation."""
    pass
```

#### Pattern 4: NotImplementedError
```python
def method_name(self):
    """Method documentation."""
    raise NotImplementedError("TODO: Implement")
```

### Formatting Conventions
- **Indentation**: 4 spaces (Python standard)
- **Location**: Inside method body, after docstring
- **Comment style**: Inline comments with explanation
- **Context**: Includes examples of expected operations

---

## 6. Test Template Structure

### Test Templates (Jinja2-based)

#### 6.1 conftest.py.j2
**Purpose**: Pytest configuration and shared fixtures

**Key Sections**:
- Pytest configuration (markers, plugins)
- Logging setup
- Reusable fixtures:
  - `container`: ModelContainer with mocked dependencies
  - `node`: Test node instance
  - `sample_contract`: Test contract
  - Database fixtures (if Effect node)
  - Mock fixtures for external services

**Variables Available**:
```jinja2
{{ node_name }}              # Node class name
{{ node_type }}              # Node type (effect, compute, etc.)
{{ test_contract }}          # Test contract configuration
{{ module_path }}            # Module import path
```

#### 6.2 test_unit.py.j2
**Purpose**: Unit tests with mocked dependencies

**Key Sections**:
- Fixtures (container, node, sample_contract)
- Initialization tests
- Mock testing for:
  - Database operations (Effect)
  - HTTP clients
  - Kafka producers
- Edge cases and error handling
- Performance assertions (if applicable)

**Coverage Target**: {{ test_contract.coverage_target }}% (default 85%)

#### 6.3 test_integration.py.j2
**Purpose**: Integration tests with real dependencies

**Key Sections**:
- Database setup/teardown (if Effect)
- End-to-end workflow tests
- Event publishing verification (Kafka)
- Transaction management tests
- Real dependency testing

**Markers**:
- `@pytest.mark.integration`
- `@pytest.mark.asyncio`
- `@pytest.mark.slow` (optional)

#### 6.4 test_contract.py.j2
**Purpose**: Contract compliance tests

**Validates**:
- Contract structure against ModelContract{Type} schema
- Required fields present
- Version format correct
- Node type valid
- I/O operations configured (Effect only)

#### 6.5 test_performance.py.j2
**Purpose**: Performance and load tests

**Benchmarks** (with pytest-benchmark):
- Method execution time (target: <100ms for most operations)
- Throughput (operations/second)
- Memory usage
- Connection pool efficiency (Effect nodes)
- Concurrent request handling

---

## 7. Test Contract Example Structure

**Location**: `src/omninode_bridge/codegen/templates/examples/test_contract_effect_example.yaml`

### Core Sections:
```yaml
# 1. Test Target Identification
name: postgres_crud_effect_tests
version: { major: 1, minor: 0, patch: 0 }
target_node: NodePostgresCrudEffect
target_node_type: effect

# 2. Coverage Requirements
coverage_minimum: 90
coverage_target: 95
coverage_by_file: true

# 3. Test Type Specifications
test_types:
  - unit
  - integration
  - contract
  - performance
  - security

# 4. Test Targets and Scenarios
test_targets:
  - target_name: execute_effect
    test_scenarios:
      - "successful record creation"
      - "creation with duplicate key violation"
    expected_behaviors:
      - "creates record in database"
      - "returns result with success=True"
    edge_cases:
      - "empty data dictionary"
      - "data with extremely long strings"
    error_conditions:
      - "database connection failure"
      - "deadlock detection"

# 5. Mock Requirements
mock_requirements:
  mock_database: true
  mock_external_services:
    - "PostgreSQL database"
  mock_exceptions:
    - "asyncpg.exceptions.ConnectionDoesNotExistError"

# 6. Assertion Types
assertion_types:
  - equality
  - type
  - exception
  - database_state
  - mock_call_verification

# 7. Test Configuration
test_configuration:
  pytest_markers: [unit, integration, asyncio, security]
  parallel_execution: true
  parallel_workers: 4
  timeout_seconds: 300
  coverage_threshold: 90

# 8. Environment Variables
environment_variables:
  TESTING: "true"
  LOG_LEVEL: "DEBUG"
  POSTGRES_HOST: "localhost"
```

---

## 8. Existing Template Systems

### System 1: TemplateEngine (template_engine.py)
- **Status**: Fully implemented and operational
- **Purpose**: Main code generation orchestrator
- **Input**: Requirements (ModelPRDRequirements) + Classification (ModelClassificationResult)
- **Output**: ModelGeneratedArtifacts with all files and metadata

### System 2: BusinessLogicGenerator (business_logic/generator.py)
- **Status**: Fully implemented
- **Purpose**: LLM-based business logic generation
- **Input**: Node stub code + Context
- **Output**: Intelligent implementation for marked stubs

### System 3: CodeInjector (business_logic/injector.py)
- **Status**: Fully implemented
- **Purpose**: Injects generated code into template stubs
- **Features**:
  - Detects 4 stub pattern types
  - Preserves indentation and formatting
  - Maintains method signatures
  - Type-safe validation

### System 4: TestExecutor (test_executor.py)
- **Status**: Fully implemented
- **Purpose**: Runs generated tests and reports results
- **Features**:
  - Pytest integration
  - Coverage reporting
  - Performance metrics
  - Failure analysis

---

## 9. Naming Conventions

### Service Name Format
- **Input**: user-provided (e.g., "postgres_crud")
- **Validation**: lowercase letters, numbers, underscores only, 2-64 characters
- **Transformation**: snake_case → PascalCase

### Node Class Naming
**Pattern**: `Node<PascalName><Type>`

**Examples**:
- `NodePostgresCrudEffect` (from "postgres_crud" + "effect")
- `NodeDataProcessorCompute` (from "data_processor" + "compute")
- `NodeEventAggregatorReducer` (from "event_aggregator" + "reducer")
- `NodeWorkflowOrchestratorOrchestrator` (from "workflow_orchestrator" + "orchestrator")

### Model Naming
**Input Model**: `Model<PascalName>Request`
**Output Model**: `Model<PascalName>Response`

**Examples**:
- `ModelPostgresCrudRequest`
- `ModelPostgresCrudResponse`

### Contract Naming
**Format**: `{service_name}_contract`
**Example**: `postgres_crud_contract`

### Package Path
**Format**: `omninode_bridge.nodes.{service_name}.v1_0_0.node.{NodeClassName}`
**Example**: `omninode_bridge.nodes.postgres_crud.v1_0_0.node.NodePostgresCrudEffect`

---

## 10. ONEX v2.0 Compliance Features

### Built-in Compliance Elements:

1. **Suffix-based Naming**
   - Effect → `NodeXxxEffect`
   - Compute → `NodeXxxCompute`
   - Reducer → `NodeXxxReducer`
   - Orchestrator → `NodeXxxOrchestrator`

2. **Error Handling**
   - Uses `ModelOnexError` with `EnumCoreErrorCode`
   - Structured error details
   - Proper exception chaining

3. **Logging & Observability**
   - Uses `emit_log_event` from omnibase_core
   - Structured log events with context
   - Correlation ID tracking

4. **Mixins for Auto-Registration**
   - `IntrospectionMixin`: Automatic node registration
   - `HealthCheckMixin`: Health check endpoints
   - Consul service discovery integration

5. **Contract-Driven Architecture**
   - ModelContract{Type} validation
   - Required field enforcement
   - Version tracking

6. **Event-Driven Patterns**
   - Kafka event publishing
   - OnexEnvelopeV1 format
   - Workflow tracking events

---

## 11. Security Validation in Templates

### Input Validation
```python
# Service name validation
- Pattern: ^[a-z][a-z0-9_]*$
- Length: 2-64 characters
- Prevents: code injection, filesystem traversal

# Domain validation
- Pattern: ^[a-z][a-z0-9_]*$
- Length: max 64 characters

# List validation
- Type checking for all items
- No control characters (\x00, etc.)
- Max string length per item: 500
```

### Generated Code Security Checks
- SQL injection prevention (parameterized queries)
- Hardcoded secret detection
- API key patterns
- Token patterns

---

## 12. Template Rendering Mechanism

### Rendering Flow:
```
1. Template Context Building
   ├── Validate inputs (service_name, domain, operations, features)
   ├── Calculate derived names (PascalCase, node_class_name)
   ├── Build version dict and package path
   └── Merge with requirements and classification

2. File Generation
   ├── Node file (from filesystem or inline template)
   ├── Contract file (YAML validation)
   ├── Init file (module initialization)
   ├── Models (request/response)
   ├── Tests (unit, integration, contract, performance)
   └── Documentation (README.md)

3. Validation
   ├── Contract YAML parsing
   ├── Schema validation against omnibase_core
   ├── Required fields verification
   ├── Type checking for all fields

4. Optional Test Execution
   ├── Write files to disk
   ├── Run pytest
   ├── Collect metrics
   ├── Analyze failures (if any)
```

---

## 13. Key Files and Locations

| Component | Location | Status |
|-----------|----------|--------|
| **TemplateEngine** | `src/omninode_bridge/codegen/template_engine.py` | Complete |
| **Node Templates (Jinja2)** | `templates/node_templates/{type}/` | Empty (reserved) |
| **Test Templates (Jinja2)** | `src/omninode_bridge/codegen/templates/test_templates/` | Complete (5 files) |
| **Test Examples** | `src/omninode_bridge/codegen/templates/examples/` | Complete (4 files) |
| **Business Logic Generator** | `src/omninode_bridge/codegen/business_logic/generator.py` | Complete |
| **Code Injector** | `src/omninode_bridge/codegen/business_logic/injector.py` | Complete |
| **Config** | `src/omninode_bridge/codegen/business_logic/config.py` | Complete |
| **Test Executor** | `src/omninode_bridge/codegen/test_executor.py` | Complete |
| **Contract Introspector** | `src/omninode_bridge/codegen/contract_introspector.py` | Complete |

---

## 14. Recommended Next Steps for TemplateEngine Enhancement

### Phase 1: Populate Jinja2 Templates
```bash
# Create filesystem Jinja2 templates in:
templates/node_templates/effect/
  ├── node.py.j2
  └── contract.yaml.j2

templates/node_templates/compute/
  ├── node.py.j2
  └── contract.yaml.j2

templates/node_templates/reducer/
  ├── node.py.j2
  └── contract.yaml.j2

templates/node_templates/orchestrator/
  ├── node.py.j2
  └── contract.yaml.j2
```

### Phase 2: Enhanced Template Features
- Template variants (e.g., "database", "api", "ml")
- Template inheritance chains
- Snippet composition
- Plugin-based template extensions

### Phase 3: Advanced Capabilities
- Template validation framework
- Template versioning
- Template marketplace integration
- Custom template directives

---

## Conclusion

The OmniNode Bridge codebase already has a **comprehensive, production-grade template system** with:

✅ **TemplateEngine** class for orchestration
✅ **Jinja2 support** with inline fallback
✅ **4 node types** (Effect, Compute, Reducer, Orchestrator)
✅ **5 test template types** (Unit, Integration, Contract, Performance, Conftest)
✅ **ONEX v2.0 compliance** built-in
✅ **Security validation** throughout
✅ **Contract validation** against schemas
✅ **Business logic injection** system
✅ **Test execution** framework
✅ **Code generation** pipeline

The template system is **ready for production use** and can be enhanced with additional features as needed.
