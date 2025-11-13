# Code Generator Architecture & Contract Processing Flow

**Purpose**: Comprehensive architecture documentation for the OmniNode Bridge code generation system, including component relationships, data flows, and mixin injection points.

**Status**: Current Architecture (October 2025)
**Target**: Mixin Enhancement Implementation (Phase 3)

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
3. [Generation Strategies](#generation-strategies)
4. [Contract Processing Flow](#contract-processing-flow)
5. [Template Processing](#template-processing)
6. [LLM Enhancement Pipeline](#llm-enhancement-pipeline)
7. [Mixin Injection Points](#mixin-injection-points)
8. [Data Flow Diagrams](#data-flow-diagrams)
9. [Sequence Diagrams](#sequence-diagrams)

---

## Architecture Overview

The code generation system uses a **Strategy Pattern** to support multiple generation approaches, unified through a common facade.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CodeGenerationService (Facade)                    │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Entry Point: generate_node(requirements, strategy, ...)       │ │
│  │  - Node classification (NodeClassifier)                        │ │
│  │  - Strategy selection (StrategyRegistry)                       │ │
│  │  - Validation & error handling                                 │ │
│  │  - Performance monitoring                                       │ │
│  │  - File distribution                                            │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│                         StrategyRegistry                             │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Strategy Selection Logic:                                      │ │
│  │  1. Preferred strategy (if specified)                          │ │
│  │  2. LLM-powered (if enable_llm=True)                           │ │
│  │  3. Jinja2 (template-based)                                    │ │
│  │  4. Default fallback                                            │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                    │                    │                    │
        ┌───────────┴──────────┬─────────┴──────────┬────────┴────────┐
        │                      │                    │                  │
        ↓                      ↓                    ↓                  ↓
┌──────────────┐      ┌──────────────┐    ┌──────────────┐   ┌──────────────┐
│   Jinja2     │      │  Template    │    │   Hybrid     │   │     Auto     │
│   Strategy   │      │   Loading    │    │   Strategy   │   │   Strategy   │
└──────────────┘      └──────────────┘    └──────────────┘   └──────────────┘
     │                      │                     │
     │                      │                     │
     ↓                      ↓                     ↓
┌──────────────┐      ┌──────────────┐    ┌──────────────┐
│  Template    │      │  Template    │    │   Combined   │
│  Engine      │      │  Engine      │    │   Approach   │
│  (Jinja2)    │      │  Loader +    │    │              │
└──────────────┘      │  Business    │    └──────────────┘
                      │  Logic Gen   │
                      └──────────────┘

```

**Key Characteristics**:
- **Strategy Pattern**: Pluggable generation approaches
- **Unified API**: Single entry point via `CodeGenerationService`
- **Async/Await**: Non-blocking operations throughout
- **Type Safety**: Pydantic models for all data structures
- **Performance Monitoring**: Metrics collection at all stages
- **Validation**: Multi-level validation (requirements, templates, generated code)

---

## Core Components

### 1. CodeGenerationService (Facade)

**Location**: `src/omninode_bridge/codegen/service.py`

**Purpose**: Unified entry point for all code generation requests.

**Responsibilities**:
- Accept generation requests with PRD requirements
- Classify node type if not provided (via `NodeClassifier`)
- Select optimal strategy based on requirements
- Coordinate generation pipeline
- Distribute generated files to target directories
- Collect and log performance metrics

**Key Methods**:
```python
async def generate_node(
    requirements: ModelPRDRequirements,
    classification: Optional[ModelClassificationResult] = None,
    output_directory: Optional[Path] = None,
    strategy: str = "auto",
    enable_llm: bool = True,
    validation_level: str = "standard",
    run_tests: bool = False,
    strict_mode: bool = False,
    correlation_id: Optional[UUID] = None,
) -> ModelGenerationResult
```

**Strategy Selection Logic**:
1. If `strategy != "auto"`: Use specified strategy
2. If `enable_llm=True`: Prefer TemplateLoadingStrategy (LLM-powered)
3. Fallback: Jinja2Strategy (template-based, zero cost)
4. Default: First registered strategy

**File Distribution**:
Writes generated artifacts to 3 locations:
- Base: `generated_nodes/{service_name}/`
- Final: `generated_nodes/{service_name}_final/`
- LLM: `generated_nodes/{service_name}_llm/`

---

### 2. StrategyRegistry

**Location**: `src/omninode_bridge/codegen/service.py` (inner class)

**Purpose**: Manage strategy registration and selection.

**Responsibilities**:
- Register strategies at initialization
- Maintain default strategy
- Select optimal strategy based on:
  - Node type compatibility
  - LLM availability
  - User preference

**Registered Strategies**:
1. **Jinja2Strategy** (default) - Template-based, fast, zero cost
2. **TemplateLoadStrategy** - Hand-written templates + LLM enhancement
3. **HybridStrategy** - Combined Jinja2 + LLM + validation

**Strategy Selection Algorithm**:
```python
def select_strategy(
    node_type: EnumNodeType,
    enable_llm: bool,
    prefer_strategy: Optional[EnumStrategyType] = None,
) -> BaseGenerationStrategy:
    # 1. Try preferred strategy (if specified)
    if prefer_strategy and supports_node_type(node_type):
        return preferred_strategy

    # 2. Try LLM-powered strategy (if enabled)
    if enable_llm and TemplateLoadingStrategy.supports_node_type(node_type):
        return template_loading_strategy

    # 3. Try Jinja2 strategy
    if Jinja2Strategy.supports_node_type(node_type):
        return jinja2_strategy

    # 4. Fall back to default
    return default_strategy
```

---

### 3. NodeClassifier

**Location**: `src/omninode_bridge/codegen/node_classifier.py`

**Purpose**: Classify node type from PRD requirements using keyword analysis.

**Classification Logic**:
- Analyzes `business_description` and `operations`
- Scores each node type based on keyword matches
- Returns classification with confidence score

**Node Type Keywords**:
```python
EFFECT_KEYWORDS = ["api", "database", "http", "fetch", "store", "io", "external"]
COMPUTE_KEYWORDS = ["calculate", "transform", "process", "algorithm", "compute"]
REDUCER_KEYWORDS = ["aggregate", "reduce", "accumulate", "summarize", "collect"]
ORCHESTRATOR_KEYWORDS = ["orchestrate", "coordinate", "workflow", "pipeline"]
```

**Example**:
```python
classifier = NodeClassifier()
classification = classifier.classify(requirements)
# ModelClassificationResult(
#     node_type=EnumNodeType.EFFECT,
#     confidence=0.85,
#     reasoning="Strong indicators: api, database, fetch"
# )
```

---

### 4. ContractIntrospector

**Location**: `src/omninode_bridge/codegen/contract_introspector.py`

**Purpose**: Introspect omnibase_core contract models to extract field requirements.

**Key Capabilities**:
- **Field Analysis**: Extract required vs optional fields from Pydantic models
- **Default Values**: Get default values for optional fields
- **Validation**: Validate data dict contains all required fields
- **Minimal Contract Generation**: Generate minimal valid contract for testing

**Contract Class Mapping**:
```python
CONTRACT_CLASS_MAP = {
    EnumNodeType.EFFECT: ModelContractEffect,
    EnumNodeType.COMPUTE: ModelContractCompute,
    EnumNodeType.REDUCER: ModelContractReducer,
    EnumNodeType.ORCHESTRATOR: ModelContractOrchestrator,
}
```

**Key Methods**:
```python
# Extract required fields from contract class
def get_required_fields(contract_class: type[BaseModel]) -> list[str]

# Get default values for optional fields
def get_field_defaults(contract_class: type[BaseModel]) -> dict[str, Any]

# Validate contract data has all required fields
def validate_contract_data(
    contract_class: type[BaseModel],
    data: dict[str, Any]
) -> tuple[bool, list[str]]

# Generate minimal valid contract for node type
def generate_minimal_contract(node_type: EnumNodeType) -> dict[str, Any]
```

**🎯 MIXIN INJECTION POINT**: This component will be extended to:
- Parse mixin declarations from contracts
- Validate mixin references
- Extract mixin metadata for template context

---

### 5. TemplateEngine (Jinja2)

**Location**: `src/omninode_bridge/codegen/template_engine.py`

**Purpose**: Generate ONEX-compliant nodes from Jinja2 templates.

**Generated Artifacts**:
- `node.py` - Main node implementation
- `contract.yaml` - ONEX v2.0 contract
- `__init__.py` - Module initialization
- `models/*.py` - Pydantic data models (optional)
- `tests/*.py` - Unit and integration tests (optional)
- `README.md` - Documentation (optional)

**Template Context Building**:
```python
context = {
    # Node identification
    "service_name": "postgres_crud",
    "node_class_name": "NodePostgresCRUDEffect",
    "node_type": "effect",
    "node_type_upper": "EFFECT",

    # Contract fields
    "business_description": "PostgreSQL CRUD operations",
    "version_dict": {"major": 1, "minor": 0, "patch": 0},
    "input_model": "ModelRequest",
    "output_model": "ModelResponse",

    # Operations and features
    "operations": ["create", "read", "update", "delete"],
    "features": ["Connection pooling", "Async operations"],

    # Metadata
    "generated_at": datetime.now(UTC),
    "domain": "database",
}
```

**🎯 MIXIN INJECTION POINT**: Template context will be extended to include:
- `mixins`: List of mixin declarations
- `mixin_imports`: Required imports for mixins
- `mixin_inheritance`: Mixin classes to inherit from
- `mixin_methods`: Methods to include/override

**Validation**:
1. **Input Validation**: Service name, domain, string lists
2. **Contract Validation**: Generated YAML validates against omnibase_core schema
3. **Security**: Prevents code injection via template variables

---

### 6. TemplateEngineLoader

**Location**: `src/omninode_bridge/codegen/template_engine_loader/engine.py`

**Purpose**: Load hand-written Python templates from filesystem (alternative to Jinja2).

**Key Features**:
- **Template Discovery**: Scan directories for templates by node type/version
- **Stub Detection**: Identify methods needing implementation
- **Metadata Extraction**: Extract docstrings and configuration
- **Integration**: Prepares artifacts for BusinessLogicGenerator

**Directory Structure**:
```
templates/node_templates/
├── effect/
│   └── v1_0_0/
│       └── node.py          # Hand-written template with stubs
├── compute/
│   └── v1_0_0/
│       └── node.py
├── reducer/
│   └── v1_0_0/
│       └── node.py
└── orchestrator/
    └── v1_0_0/
        └── node.py
```

**Stub Indicators**:
Templates mark methods needing implementation with:
```python
async def execute_effect(self, contract: ModelContractEffect):
    """Execute effect operation."""
    # IMPLEMENTATION REQUIRED
    pass
```

**🎯 MIXIN INJECTION POINT**: Template loading will support:
- Mixin-enhanced templates
- Mixin composition markers
- Dynamic method injection

---

### 7. BusinessLogicGenerator

**Location**: `src/omninode_bridge/codegen/business_logic/generator.py`

**Purpose**: Generate intelligent business logic implementations using LLMs.

**Pipeline**:
1. **Extract Stubs**: Parse node file to find methods with stub indicators
2. **Build Context**: Gather requirements, patterns, best practices
3. **Generate Implementation**: Call NodeLLMEffect with structured prompt
4. **Validate**: Check syntax, ONEX compliance, security
5. **Inject**: Replace stub with generated implementation
6. **Track Metrics**: Tokens, cost, latency, success rate

**Stub Detection Markers**:
```python
STUB_INDICATORS = [
    "# IMPLEMENTATION REQUIRED",
    "# TODO: Implement",
    "pass  # Stub",
    "raise NotImplementedError",
]
```

**Generation Context**:
```python
ModelBusinessLogicContext(
    node_type="effect",
    service_name="postgres_crud",
    business_description="PostgreSQL CRUD operations",
    operations=["create", "read", "update", "delete"],
    features=["Async operations", "Connection pooling"],
    method_name="execute_effect",
    method_signature="async def execute_effect(self, contract: ModelContractEffect)",
    method_docstring="Execute database operation...",
    similar_patterns=["Use asyncpg", "Pool connections"],
    best_practices=["Handle errors", "Log queries"],
    performance_requirements={"max_latency_ms": 100},
)
```

**Validation Checks**:
- **Syntax**: AST parsing
- **ONEX Compliance**: ModelOnexError, emit_log_event patterns
- **Type Hints**: Check for type annotations
- **Security**: Detect hardcoded secrets, SQL injection risks

**🎯 MIXIN INJECTION POINT**: Generation prompts will include:
- Mixin method requirements
- Mixin-specific patterns
- Mixin interaction guidelines

---

## Generation Strategies

### Strategy Pattern Implementation

All strategies implement `BaseGenerationStrategy`:

```python
class BaseGenerationStrategy(ABC):
    @abstractmethod
    async def generate(
        self,
        request: ModelGenerationRequest,
    ) -> ModelGenerationResult:
        """Generate node code using this strategy."""
        pass

    @abstractmethod
    def supports_node_type(self, node_type: EnumNodeType) -> bool:
        """Check if this strategy supports the given node type."""
        pass

    @abstractmethod
    def get_strategy_info(self) -> dict[str, Any]:
        """Get strategy information and capabilities."""
        pass
```

---

### 1. Jinja2Strategy

**Location**: `src/omninode_bridge/codegen/strategies/jinja2_strategy.py`

**Characteristics**:
- **Speed**: <1s generation time
- **Cost**: $0 (no LLM calls)
- **Deterministic**: Same input → same output
- **Offline**: No network calls

**Best For**:
- Standard CRUD operations
- Well-defined patterns
- Rapid prototyping
- Cost-sensitive scenarios

**Limitations**:
- No custom business logic generation
- Limited to predefined templates
- Requires template updates for new patterns

**Performance Profile**:
```python
{
    "avg_generation_time_ms": 500,
    "max_generation_time_ms": 2000,
    "memory_usage_mb": 50,
    "cost_per_generation_usd": 0.0,
}
```

---

### 2. TemplateLoadStrategy

**Location**: `src/omninode_bridge/codegen/strategies/template_load_strategy.py`

**Characteristics**:
- **Template Loading**: <100ms
- **LLM Enhancement**: Variable (200-5000ms)
- **Cost**: Variable (depends on stubs count)
- **Intelligent**: Context-aware business logic

**Pipeline**:
1. Load hand-written template from filesystem
2. Detect stubs in template code
3. Convert to BusinessLogicGenerator format
4. Optionally enhance stubs with LLM
5. Validate enhanced code
6. Return complete implementation

**Best For**:
- Custom nodes with hand-written templates
- Iterative development
- Nodes requiring specific patterns not in Jinja2
- Prototyping with real template files

**Limitations**:
- Requires hand-written templates in filesystem
- LLM enhancement requires ZAI_API_KEY
- Template must exist for specified node_type/version

**Performance Profile**:
```python
{
    "template_loading_ms": "<100",
    "llm_enhancement_ms": "200-5000 (variable)",
    "total_generation_ms": "200-5000",
}
```

---

### 3. HybridStrategy

**Location**: `src/omninode_bridge/codegen/strategies/hybrid_strategy.py`

**Characteristics**:
- Combines Jinja2 templates with LLM enhancement
- Strict validation
- Best of both worlds: Speed + intelligence

**Pipeline**:
1. Generate initial code with Jinja2
2. Detect enhancement opportunities
3. Apply LLM enhancement to complex methods
4. Validate with strict rules
5. Return enhanced, validated code

**Best For**:
- Production-grade nodes
- Maximum quality requirements
- Balanced speed/intelligence

---

### 4. AutoStrategy

**Purpose**: Automatically selects optimal strategy based on:
- Node type
- Complexity indicators
- LLM availability
- Performance requirements

**Selection Logic**:
```python
if complexity_score > 0.7 and llm_available:
    return TemplateLoadStrategy
elif standard_pattern_detected:
    return Jinja2Strategy
else:
    return HybridStrategy
```

---

## Contract Processing Flow

### Current Contract Generation

```
ModelPRDRequirements
         │
         ↓
┌────────────────────────┐
│  NodeClassifier         │  ← Analyzes requirements
│  - Extract keywords     │
│  - Score node types     │
│  - Return classification│
└────────────────────────┘
         │
         ↓
┌────────────────────────┐
│  ContractIntrospector   │  ← Get contract schema
│  - Load contract class  │
│  - Extract required     │
│    fields               │
│  - Get field defaults   │
└────────────────────────┘
         │
         ↓
┌────────────────────────┐
│  TemplateEngine         │  ← Build template context
│  - Build context dict   │
│  - Validate inputs      │
│  - Render template      │
└────────────────────────┘
         │
         ↓
┌────────────────────────┐
│  Generated contract.yaml│
│  ├─ name               │
│  ├─ version            │
│  ├─ description        │
│  ├─ node_type          │
│  ├─ input_model        │
│  ├─ output_model       │
│  └─ node_type_specific │
│     ├─ io_operations   │  (Effect)
│     ├─ algorithm       │  (Compute)
│     ├─ aggregation     │  (Reducer)
│     └─ workflow        │  (Orchestrator)
└────────────────────────┘
```

### Contract Parsing Details

**Step 1: Requirements Analysis**
```python
requirements = ModelPRDRequirements(
    node_type="effect",
    service_name="postgres_crud",
    business_description="PostgreSQL CRUD operations",
    operations=["create", "read", "update", "delete"],
    domain="database",
    features=["Connection pooling", "Async operations"],
    dependencies={"asyncpg": "^0.27.0"},
    performance_requirements={"max_latency_ms": 100},
)
```

**Step 2: Node Type Classification**
```python
classifier = NodeClassifier()
classification = classifier.classify(requirements)
# Returns: ModelClassificationResult(
#     node_type=EnumNodeType.EFFECT,
#     confidence=0.85,
#     reasoning="Keywords: api, database, io operations"
# )
```

**Step 3: Contract Schema Extraction**
```python
introspector = ContractIntrospector()
required_fields = introspector.get_required_fields(ModelContractEffect)
# Returns: [
#     'name', 'version', 'description', 'node_type',
#     'input_model', 'output_model', 'io_operations'
# ]

field_defaults = introspector.get_field_defaults(ModelContractEffect)
# Returns: {
#     'idempotent_operations': True,
#     'audit_trail_enabled': True,
# }
```

**Step 4: Template Context Building**
```python
context = {
    "service_name": requirements.service_name,
    "node_class_name": f"Node{pascal_case(service_name)}{node_type.capitalize()}",
    "node_type": classification.node_type.value,
    "node_type_upper": classification.node_type.value.upper(),
    "business_description": requirements.business_description,
    "version_dict": {"major": 1, "minor": 0, "patch": 0},
    "input_model": f"Model{pascal_case(service_name)}Request",
    "output_model": f"Model{pascal_case(service_name)}Response",
    "operations": requirements.operations,
    "features": requirements.features,
    "dependencies": requirements.dependencies,
    "performance_requirements": requirements.performance_requirements,
    "generated_at": datetime.now(UTC),
    "domain": requirements.domain,
}
```

**Step 5: Contract YAML Generation**
```yaml
name: postgres_crud_effect
version:
  major: 1
  minor: 0
  patch: 0
description: PostgreSQL CRUD operations
node_type: EFFECT
input_model: ModelPostgresCRUDRequest
output_model: ModelPostgresCRUDResponse

# Effect-specific configuration
io_operations:
  - operation_type: database_query
    atomic: true
    timeout_seconds: 30
    validation_enabled: true

# Optional fields (from defaults)
idempotent_operations: true
audit_trail_enabled: true
```

**Step 6: Contract Validation**
```python
# Parse YAML to dict
contract_data = yaml.safe_load(contract_yaml)

# Convert to omnibase_core model
contract = ModelContractEffect(**contract_data)

# Validation happens automatically in Pydantic model
# Raises ValidationError if any required fields missing
```

---

## Template Processing

### Jinja2 Template Flow

```
Template Context (dict)
         │
         ↓
┌────────────────────────┐
│  Jinja2 Environment     │
│  - Load template file   │
│  - Parse Jinja2 syntax  │
│  - Apply filters        │
└────────────────────────┘
         │
         ↓
┌────────────────────────┐
│  Template Rendering     │
│  - Substitute variables │
│  - Execute conditionals │
│  - Process loops        │
└────────────────────────┘
         │
         ↓
┌────────────────────────┐
│  Generated Python Code  │
│  - node.py             │
│  - contract.yaml       │
│  - models/*.py         │
│  - tests/*.py          │
└────────────────────────┘
```

### Hand-written Template Flow

```
Template File (Python)
         │
         ↓
┌────────────────────────┐
│  TemplateEngineLoader   │
│  - Read template file   │
│  - Parse Python AST     │
│  - Extract metadata     │
└────────────────────────┘
         │
         ↓
┌────────────────────────┐
│  Stub Detection         │
│  - Find methods with    │
│    stub markers         │
│  - Extract signatures   │
│  - Extract docstrings   │
└────────────────────────┘
         │
         ↓
┌────────────────────────┐
│  ModelTemplateArtifacts │
│  - template_code       │
│  - stubs: [...]        │
│  - metadata            │
└────────────────────────┘
         │
         ↓
┌────────────────────────┐
│  ArtifactConverter      │
│  - Convert to           │
│    ModelGeneratedArtifacts│
│  - Prepare for LLM      │
└────────────────────────┘
```

---

## LLM Enhancement Pipeline

### Complete Pipeline Flow

```
ModelGeneratedArtifacts (from template)
         │
         ↓
┌────────────────────────┐
│  BusinessLogicGenerator │
│  Step 1: Extract Stubs  │
│  - Parse node.py AST    │
│  - Find stub markers    │
│  - Extract method info  │
└────────────────────────┘
         │
         ↓
┌────────────────────────┐
│  Step 2: Build Context  │
│  - Gather requirements  │
│  - Add similar patterns │
│  - Add best practices   │
└────────────────────────┘
         │
         ↓
┌────────────────────────┐
│  Step 3: Generate       │
│  - Build prompt         │
│  - Call NodeLLMEffect   │
│  - Strip code fences    │
└────────────────────────┘
         │
         ↓
┌────────────────────────┐
│  Step 4: Validate       │
│  - AST syntax check     │
│  - ONEX compliance      │
│  - Security scan        │
└────────────────────────┘
         │
         ↓
┌────────────────────────┐
│  Step 5: Inject         │
│  - Replace stub marker  │
│  - Maintain indentation │
│  - Preserve docstring   │
└────────────────────────┘
         │
         ↓
ModelEnhancedArtifacts
  ├─ original_artifacts
  ├─ enhanced_node_file
  ├─ methods_generated: [...]
  ├─ total_tokens_used
  ├─ total_cost_usd
  ├─ total_latency_ms
  └─ generation_success_rate
```

### LLM Prompt Structure

```
# Task: Implement {method_name} for {service_name}

## Context
Node Type: {node_type}
Purpose: {business_description}
Operations: {operations}
Features: {features}

## Method Signature
```python
{method_signature}
```

## Docstring
{method_docstring}

## Similar Patterns (for reference)
- {pattern_1}
- {pattern_2}
- {pattern_3}

## ONEX Best Practices
- {practice_1}
- {practice_2}
- {practice_3}

## Requirements
1. Return ONLY the method body (indented, starting with try/except)
2. Include proper error handling with ModelOnexError
3. Add emit_log_event calls for INFO and ERROR
4. Use type hints for all variables
5. Follow ONEX patterns (correlation tracking, structured logging)
6. No hardcoded secrets or sensitive data

## IMPORTANT: Imports Already Available
The following imports are ALREADY available in the node file:
- Standard library: os
- Typing: Any, Dict, List, Optional
- External dependencies: Check if hvac, requests, etc. are needed

DO NOT add import statements in your implementation.

Generate the implementation:
```

### Code Injection Mechanism

**Pattern Matching**:
```python
stub_pattern = (
    rf"(    async def {method_name}\([^)]+\)[^:]*:.*?\n"  # Signature (4 spaces)
    r"(?:        \"\"\"[\s\S]*?\"\"\"\n)?)"  # Optional docstring (8 spaces)
    r"        # IMPLEMENTATION REQUIRED\n"  # Stub marker (8 spaces)
    r"        pass"  # Stub body (8 spaces)
)
```

**Replacement**:
```python
# Indent all lines by 8 spaces for method body
indented_implementation = "\n".join(
    "        " + line if line.strip() else line
    for line in implementation.splitlines()
)

# Replace with signature/docstring + indented implementation
replacement = rf"\1{indented_implementation}"
enhanced = re.sub(stub_pattern, replacement, node_file, flags=re.DOTALL)
```

---

## Mixin Injection Points

### 1. Contract Parsing (ContractIntrospector)

**Current State**: Parses contract YAML to extract required fields

**Mixin Enhancement**: Parse mixin declarations from contract

```yaml
# Current contract.yaml (no mixins)
name: postgres_crud_effect
version: {major: 1, minor: 0, patch: 0}
description: PostgreSQL CRUD operations
node_type: EFFECT
input_model: ModelRequest
output_model: ModelResponse
io_operations: [...]

# Enhanced contract.yaml (with mixins)
name: postgres_crud_effect
version: {major: 1, minor: 0, patch: 0}
description: PostgreSQL CRUD operations
node_type: EFFECT
input_model: ModelRequest
output_model: ModelResponse
io_operations: [...]

# NEW: Mixin declarations
mixins:
  - name: AsyncConnectionPool
    version: v1_0_0
    config:
      min_connections: 5
      max_connections: 20
      timeout_seconds: 30

  - name: RetryMechanism
    version: v1_0_0
    config:
      max_retries: 3
      backoff_strategy: exponential
      retry_on: [TimeoutError, ConnectionError]
```

**Implementation**:
```python
class ContractIntrospector:
    def parse_mixins(self, contract_data: dict) -> list[ModelMixinDeclaration]:
        """
        Parse mixin declarations from contract.

        Returns list of ModelMixinDeclaration with:
        - name: Mixin name
        - version: Mixin version
        - config: Mixin-specific configuration
        """
        mixins = contract_data.get("mixins", [])
        return [
            ModelMixinDeclaration(
                name=m["name"],
                version=m["version"],
                config=m.get("config", {}),
            )
            for m in mixins
        ]

    def validate_mixin_compatibility(
        self,
        node_type: EnumNodeType,
        mixins: list[ModelMixinDeclaration],
    ) -> tuple[bool, list[str]]:
        """
        Validate that mixins are compatible with node type.

        Returns (is_valid, error_messages).
        """
        errors = []

        for mixin in mixins:
            # Check if mixin exists
            if not mixin_exists(mixin.name, mixin.version):
                errors.append(f"Mixin not found: {mixin.name} {mixin.version}")

            # Check if mixin supports this node type
            if not mixin_supports_node_type(mixin.name, node_type):
                errors.append(
                    f"Mixin {mixin.name} does not support {node_type.value} nodes"
                )

        return len(errors) == 0, errors
```

---

### 2. Template Context Building (TemplateEngine)

**Current State**: Builds context dict with node metadata

**Mixin Enhancement**: Add mixin-specific context

```python
def _build_template_context(
    self,
    requirements: ModelPRDRequirements,
    classification: ModelClassificationResult,
    mixins: list[ModelMixinDeclaration],  # NEW
) -> dict:
    """Build template context with mixin support."""

    context = {
        # ... existing context fields ...

        # NEW: Mixin-specific context
        "mixins": [
            {
                "name": mixin.name,
                "version": mixin.version,
                "class_name": f"Mixin{pascal_case(mixin.name)}",
                "config": mixin.config,
                "imports": self._get_mixin_imports(mixin),
                "methods": self._get_mixin_methods(mixin),
                "init_params": self._get_mixin_init_params(mixin),
            }
            for mixin in mixins
        ],

        # Aggregated mixin data
        "mixin_imports": self._aggregate_mixin_imports(mixins),
        "mixin_inheritance": [
            f"Mixin{pascal_case(m.name)}" for m in mixins
        ],
        "mixin_init_code": self._generate_mixin_init_code(mixins),
        "has_mixins": len(mixins) > 0,
    }

    return context
```

**Helper Methods**:
```python
def _get_mixin_imports(self, mixin: ModelMixinDeclaration) -> list[str]:
    """Get required imports for mixin."""
    mixin_path = self._locate_mixin(mixin.name, mixin.version)
    mixin_code = mixin_path.read_text()

    # Extract imports from mixin code
    imports = self._extract_imports(mixin_code)
    return imports

def _get_mixin_methods(self, mixin: ModelMixinDeclaration) -> list[dict]:
    """Get method signatures from mixin."""
    mixin_code = self._load_mixin_code(mixin.name, mixin.version)

    # Parse AST to extract method signatures
    methods = self._extract_method_signatures(mixin_code)
    return methods

def _generate_mixin_init_code(self, mixins: list[ModelMixinDeclaration]) -> str:
    """Generate initialization code for mixins."""
    init_lines = []

    for mixin in mixins:
        init_lines.append(
            f"self.{snake_case(mixin.name)} = "
            f"Mixin{pascal_case(mixin.name)}(**{mixin.config})"
        )

    return "\n        ".join(init_lines)
```

---

### 3. Mixin Code Injection (New Component)

**New Component**: `MixinInjector`

**Location**: `src/omninode_bridge/codegen/mixin_injector.py`

**Purpose**: Inject mixin code into generated nodes

```python
class MixinInjector:
    """
    Inject mixin code into generated node implementations.

    Handles:
    - Import injection
    - Class inheritance modification
    - Method injection/override
    - Initialization code injection
    - Configuration merging
    """

    def inject_mixins(
        self,
        node_code: str,
        mixins: list[ModelMixinDeclaration],
    ) -> str:
        """
        Inject mixin code into node implementation.

        Steps:
        1. Parse node code AST
        2. Add mixin imports
        3. Modify class inheritance
        4. Inject mixin initialization
        5. Add/override methods
        6. Return modified code
        """
        # Parse node code
        tree = ast.parse(node_code)

        # Step 1: Add imports
        node_code = self._inject_imports(node_code, mixins)

        # Step 2: Modify class inheritance
        node_code = self._modify_inheritance(node_code, mixins)

        # Step 3: Inject initialization
        node_code = self._inject_initialization(node_code, mixins)

        # Step 4: Inject/override methods
        node_code = self._inject_methods(node_code, mixins)

        return node_code

    def _inject_imports(
        self,
        node_code: str,
        mixins: list[ModelMixinDeclaration],
    ) -> str:
        """Add mixin imports to node code."""
        imports = []

        for mixin in mixins:
            imports.append(
                f"from omninode_bridge.mixins.{snake_case(mixin.name)}.{mixin.version} "
                f"import Mixin{pascal_case(mixin.name)}"
            )

        # Find last import line
        lines = node_code.splitlines()
        last_import_idx = self._find_last_import(lines)

        # Insert mixin imports after last import
        lines.insert(last_import_idx + 1, "\n".join(imports))

        return "\n".join(lines)

    def _modify_inheritance(
        self,
        node_code: str,
        mixins: list[ModelMixinDeclaration],
    ) -> str:
        """Modify class inheritance to include mixins."""
        tree = ast.parse(node_code)

        # Find node class definition
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name.startswith("Node"):
                # Add mixin base classes
                for mixin in mixins:
                    mixin_class = f"Mixin{pascal_case(mixin.name)}"
                    node.bases.append(ast.Name(id=mixin_class, ctx=ast.Load()))

        # Convert back to code
        return ast.unparse(tree)

    def _inject_initialization(
        self,
        node_code: str,
        mixins: list[ModelMixinDeclaration],
    ) -> str:
        """Inject mixin initialization code into __init__."""
        # Find __init__ method
        init_pattern = r"(    def __init__\(self[^)]*\):[^\n]*\n(?:        [^\n]+\n)*)"

        # Build initialization code
        init_code = "\n".join(
            f"        self.{snake_case(m.name)} = "
            f"Mixin{pascal_case(m.name)}(**{m.config})"
            for m in mixins
        )

        # Inject after super().__init__()
        def inject_after_super(match):
            existing_init = match.group(1)
            return f"{existing_init}\n{init_code}"

        return re.sub(init_pattern, inject_after_super, node_code)

    def _inject_methods(
        self,
        node_code: str,
        mixins: list[ModelMixinDeclaration],
    ) -> str:
        """Inject/override methods from mixins."""
        for mixin in mixins:
            # Load mixin methods
            methods = self._load_mixin_methods(mixin)

            for method in methods:
                # Check if method should be overridden
                if self._should_override(method.name, node_code):
                    node_code = self._override_method(
                        node_code, method.name, method.implementation
                    )
                else:
                    # Add new method
                    node_code = self._add_method(
                        node_code, method.name, method.implementation
                    )

        return node_code
```

---

### 4. Template Selection (New Logic)

**Enhancement**: Select templates based on mixin requirements

```python
class TemplateSelector:
    """
    Select optimal template based on requirements and mixins.

    Considerations:
    - Node type
    - Mixin requirements
    - Template availability
    - Complexity score
    """

    def select_template(
        self,
        node_type: EnumNodeType,
        mixins: list[ModelMixinDeclaration],
        requirements: ModelPRDRequirements,
    ) -> Path:
        """
        Select template that best matches requirements and mixins.

        Selection logic:
        1. Check for mixin-specific templates first
        2. Fall back to base templates
        3. Consider template complexity score
        """
        # Try mixin-specific template
        if mixins:
            mixin_template = self._find_mixin_template(node_type, mixins)
            if mixin_template:
                return mixin_template

        # Fall back to base template
        return self._get_base_template(node_type)

    def _find_mixin_template(
        self,
        node_type: EnumNodeType,
        mixins: list[ModelMixinDeclaration],
    ) -> Optional[Path]:
        """Find template optimized for specific mixin combination."""
        # Check for template matching mixin set
        mixin_names = sorted(m.name for m in mixins)
        template_key = f"{node_type.value}_{'_'.join(mixin_names)}"

        template_path = (
            self.templates_dir / "mixin_optimized" / f"{template_key}.py.j2"
        )

        if template_path.exists():
            logger.info(f"Found mixin-optimized template: {template_path}")
            return template_path

        return None
```

---

### 5. Validation (New Checks)

**Enhancement**: Validate mixin integration

```python
class MixinValidator:
    """
    Validate mixin integration in generated code.

    Checks:
    - Mixin imports present
    - Mixin inheritance correct
    - Mixin methods callable
    - Mixin configuration valid
    """

    def validate_mixin_integration(
        self,
        node_code: str,
        mixins: list[ModelMixinDeclaration],
    ) -> tuple[bool, list[str]]:
        """
        Validate that mixins are properly integrated.

        Returns (is_valid, error_messages).
        """
        errors = []

        # Parse code
        try:
            tree = ast.parse(node_code)
        except SyntaxError as e:
            return False, [f"Syntax error: {e}"]

        # Check 1: Imports present
        imports_valid, import_errors = self._validate_imports(tree, mixins)
        errors.extend(import_errors)

        # Check 2: Inheritance correct
        inheritance_valid, inheritance_errors = self._validate_inheritance(tree, mixins)
        errors.extend(inheritance_errors)

        # Check 3: Methods accessible
        methods_valid, method_errors = self._validate_methods(tree, mixins)
        errors.extend(method_errors)

        # Check 4: Configuration valid
        config_valid, config_errors = self._validate_configuration(mixins)
        errors.extend(config_errors)

        is_valid = len(errors) == 0
        return is_valid, errors

    def _validate_imports(
        self,
        tree: ast.AST,
        mixins: list[ModelMixinDeclaration],
    ) -> tuple[bool, list[str]]:
        """Validate mixin imports are present."""
        errors = []

        # Extract all imports from AST
        imports = self._extract_imports_from_ast(tree)

        # Check each mixin has corresponding import
        for mixin in mixins:
            expected_import = f"Mixin{pascal_case(mixin.name)}"

            if expected_import not in imports:
                errors.append(f"Missing import for mixin: {mixin.name}")

        return len(errors) == 0, errors

    def _validate_inheritance(
        self,
        tree: ast.AST,
        mixins: list[ModelMixinDeclaration],
    ) -> tuple[bool, list[str]]:
        """Validate mixin classes in inheritance."""
        errors = []

        # Find node class
        node_class = self._find_node_class(tree)
        if not node_class:
            return False, ["Node class not found"]

        # Extract base classes
        base_classes = [
            base.id for base in node_class.bases
            if isinstance(base, ast.Name)
        ]

        # Check each mixin is in base classes
        for mixin in mixins:
            expected_base = f"Mixin{pascal_case(mixin.name)}"

            if expected_base not in base_classes:
                errors.append(
                    f"Mixin {mixin.name} not in class inheritance"
                )

        return len(errors) == 0, errors

    def _validate_methods(
        self,
        tree: ast.AST,
        mixins: list[ModelMixinDeclaration],
    ) -> tuple[bool, list[str]]:
        """Validate mixin methods are accessible/implemented."""
        errors = []

        # Find node class methods
        node_methods = self._extract_method_names(tree)

        # Check each mixin's required methods
        for mixin in mixins:
            required_methods = self._get_mixin_required_methods(mixin)

            for method in required_methods:
                if method not in node_methods:
                    errors.append(
                        f"Mixin {mixin.name} requires method: {method}"
                    )

        return len(errors) == 0, errors

    def _validate_configuration(
        self,
        mixins: list[ModelMixinDeclaration],
    ) -> tuple[bool, list[str]]:
        """Validate mixin configurations are valid."""
        errors = []

        for mixin in mixins:
            # Load mixin configuration schema
            config_schema = self._load_mixin_config_schema(mixin.name, mixin.version)

            # Validate mixin.config against schema
            try:
                validate(mixin.config, config_schema)
            except ValidationError as e:
                errors.append(f"Invalid config for {mixin.name}: {e.message}")

        return len(errors) == 0, errors
```

---

## Data Flow Diagrams

### High-Level Generation Flow

```
User Request
    │
    ├─ requirements: ModelPRDRequirements
    ├─ strategy: "auto" | "jinja2" | "template_loading" | "hybrid"
    ├─ enable_llm: bool
    └─ validation_level: "none" | "basic" | "standard" | "strict"
    │
    ↓
┌───────────────────────────────────────────────────────────────┐
│  CodeGenerationService.generate_node()                        │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  1. Classify node type (if not provided)                │ │
│  │  2. Validate requirements                               │ │
│  │  3. Select strategy                                     │ │
│  │  4. Execute strategy.generate()                         │ │
│  │  5. Distribute files to target directories              │ │
│  │  6. Collect metrics                                     │ │
│  └─────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────┘
    │
    ↓
Strategy.generate()
    │
    ├─ Jinja2Strategy → TemplateEngine (Jinja2) → Generated Code
    │
    ├─ TemplateLoadStrategy → TemplateEngineLoader → BusinessLogicGenerator → Enhanced Code
    │
    └─ HybridStrategy → Both approaches + validation
    │
    ↓
ModelGenerationResult
    │
    ├─ artifacts: ModelGeneratedArtifacts
    ├─ strategy_used: EnumStrategyType
    ├─ generation_time_ms: float
    ├─ validation_passed: bool
    ├─ llm_used: bool
    └─ intelligence_sources: list[str]
```

### Jinja2 Strategy Flow

```
ModelGenerationRequest
    │
    ↓
┌────────────────────────────┐
│  Jinja2Strategy.generate() │
└────────────────────────────┘
    │
    ├─ Validate requirements
    │
    ↓
┌────────────────────────────┐
│  TemplateEngine.generate() │
│  ┌──────────────────────┐  │
│  │  1. Validate inputs  │  │
│  │  2. Build context    │  │
│  │  3. Render templates │  │
│  │  4. Validate output  │  │
│  └──────────────────────┘  │
└────────────────────────────┘
    │
    ├─ node.py (rendered from template)
    ├─ contract.yaml (rendered from template)
    ├─ __init__.py (rendered from template)
    ├─ models/*.py (rendered from templates)
    └─ tests/*.py (rendered from templates)
    │
    ↓
ModelGeneratedArtifacts
    │
    ↓
ModelGenerationResult
```

### TemplateLoad Strategy Flow

```
ModelGenerationRequest
    │
    ↓
┌────────────────────────────────┐
│  TemplateLoadStrategy.generate()│
└────────────────────────────────┘
    │
    ├─ Validate requirements
    │
    ↓
┌────────────────────────────────┐
│  TemplateEngine.load_template() │
│  ┌──────────────────────────┐   │
│  │  1. Locate template file │   │
│  │  2. Read Python code     │   │
│  │  3. Detect stubs (AST)   │   │
│  │  4. Extract metadata     │   │
│  └──────────────────────────┘   │
└────────────────────────────────┘
    │
    ↓
ModelTemplateArtifacts
    ├─ template_code: str
    ├─ stubs: list[ModelStubInfo]
    └─ metadata: ModelTemplateMetadata
    │
    ↓
┌────────────────────────────────┐
│  ArtifactConverter              │
│  .template_to_generated()      │
│  ┌──────────────────────────┐  │
│  │  Convert to standard     │  │
│  │  ModelGeneratedArtifacts │  │
│  └──────────────────────────┘  │
└────────────────────────────────┘
    │
    ↓
ModelGeneratedArtifacts (with stubs)
    │
    ↓
┌────────────────────────────────┐
│  BusinessLogicGenerator         │
│  .enhance_artifacts()           │
│  ┌──────────────────────────┐  │
│  │  1. Extract stubs        │  │
│  │  2. Build context        │  │
│  │  3. Generate via LLM     │  │
│  │  4. Validate generated   │  │
│  │  5. Inject into code     │  │
│  │  6. Collect metrics      │  │
│  └──────────────────────────┘  │
└────────────────────────────────┘
    │
    ↓
ModelEnhancedArtifacts
    ├─ original_artifacts
    ├─ enhanced_node_file: str
    ├─ methods_generated: list[ModelGeneratedMethod]
    ├─ total_tokens_used: int
    ├─ total_cost_usd: float
    └─ generation_success_rate: float
    │
    ↓
ModelGenerationResult
```

### Mixin-Enhanced Flow (Proposed)

```
ModelGenerationRequest + Mixins
    │
    ↓
┌────────────────────────────────┐
│  CodeGenerationService          │
│  .generate_node()               │
│  ┌──────────────────────────┐  │
│  │  1. Parse mixins from    │  │
│  │     contract/requirements│  │
│  │  2. Validate mixins      │  │
│  │  3. Select strategy      │  │
│  └──────────────────────────┘  │
└────────────────────────────────┘
    │
    ↓
┌────────────────────────────────┐
│  ContractIntrospector           │
│  .parse_mixins()                │
│  .validate_mixin_compatibility()│
└────────────────────────────────┘
    │
    ↓
list[ModelMixinDeclaration]
    │
    ↓
┌────────────────────────────────┐
│  Strategy.generate()            │
│  (with mixin support)           │
│  ┌──────────────────────────┐  │
│  │  1. Generate base code   │  │
│  │  2. Inject mixins        │  │
│  │  3. Validate integration │  │
│  └──────────────────────────┘  │
└────────────────────────────────┘
    │
    ↓
┌────────────────────────────────┐
│  MixinInjector                  │
│  .inject_mixins()               │
│  ┌──────────────────────────┐  │
│  │  1. Add imports          │  │
│  │  2. Modify inheritance   │  │
│  │  3. Inject init code     │  │
│  │  4. Add/override methods │  │
│  └──────────────────────────┘  │
└────────────────────────────────┘
    │
    ↓
Enhanced Node Code (with mixins)
    │
    ↓
┌────────────────────────────────┐
│  MixinValidator                 │
│  .validate_mixin_integration()  │
│  ┌──────────────────────────┐  │
│  │  1. Check imports        │  │
│  │  2. Check inheritance    │  │
│  │  3. Check methods        │  │
│  │  4. Check configuration  │  │
│  └──────────────────────────┘  │
└────────────────────────────────┘
    │
    ↓
ModelGenerationResult
    ├─ artifacts (with mixins)
    ├─ mixins_applied: list[str]
    └─ mixin_validation_passed: bool
```

---

## Sequence Diagrams

### Current Generation Flow

```
User                CodeGenerationService    StrategyRegistry    Strategy    TemplateEngine
 │                           │                      │               │              │
 │ generate_node()           │                      │               │              │
 ├──────────────────────────>│                      │               │              │
 │                           │                      │               │              │
 │                           │ select_strategy()    │               │              │
 │                           ├─────────────────────>│               │              │
 │                           │                      │               │              │
 │                           │       Strategy       │               │              │
 │                           │<─────────────────────┤               │              │
 │                           │                      │               │              │
 │                           │ generate()           │               │              │
 │                           ├──────────────────────┼──────────────>│              │
 │                           │                      │               │              │
 │                           │                      │               │ generate()   │
 │                           │                      │               ├─────────────>│
 │                           │                      │               │              │
 │                           │                      │               │ Artifacts    │
 │                           │                      │               │<─────────────┤
 │                           │                      │               │              │
 │                           │       Result         │               │              │
 │                           │<─────────────────────┼───────────────┤              │
 │                           │                      │               │              │
 │       Result              │                      │               │              │
 │<──────────────────────────┤                      │               │              │
 │                           │                      │               │              │
```

### Proposed Mixin-Enhanced Flow

```
User    CodeGen   Contract     Mixin      Strategy   Mixin      Mixin      Template
        Service   Introspector Selector              Injector   Validator  Engine
 │         │          │           │           │          │          │         │
 │ generate_node     │           │           │          │          │         │
 │  (with mixins)    │           │           │          │          │         │
 ├──────────────────>│           │           │          │          │         │
 │         │          │           │           │          │          │         │
 │         │ parse_mixins()      │           │          │          │         │
 │         ├─────────────────────>│           │          │          │         │
 │         │          │           │           │          │          │         │
 │         │    list[Mixins]     │           │          │          │         │
 │         │<─────────────────────┤           │          │          │         │
 │         │          │           │           │          │          │         │
 │         │ validate_mixin_compatibility()  │          │          │         │
 │         ├─────────────────────>│           │          │          │         │
 │         │          │           │           │          │          │         │
 │         │      (valid, [])    │           │          │          │         │
 │         │<─────────────────────┤           │          │          │         │
 │         │          │           │           │          │          │         │
 │         │ select_template()   │           │          │          │         │
 │         ├─────────────────────────────────>│          │          │         │
 │         │          │           │           │          │          │         │
 │         │     Template Path   │           │          │          │         │
 │         │<─────────────────────────────────┤          │          │         │
 │         │          │           │           │          │          │         │
 │         │ generate() (with mixins)        │          │          │         │
 │         ├─────────────────────────────────────────────>│          │         │
 │         │          │           │           │          │          │         │
 │         │          │           │           │      generate()     │         │
 │         │          │           │           │          ├───────────────────>│
 │         │          │           │           │          │          │         │
 │         │          │           │           │     Base Code       │         │
 │         │          │           │           │          │<──────────────────┤
 │         │          │           │           │          │          │         │
 │         │          │           │           │  inject_mixins()    │         │
 │         │          │           │           │<─────────┤          │         │
 │         │          │           │           │          │          │         │
 │         │          │           │           │  Enhanced Code      │         │
 │         │          │           │           │          ├─────────>│         │
 │         │          │           │           │          │          │         │
 │         │          │           │           │       validate()    │         │
 │         │          │           │           │          │<─────────┤         │
 │         │          │           │           │          │          │         │
 │         │          │           │           │      (valid, [])    │         │
 │         │          │           │           │          ├─────────>│         │
 │         │          │           │           │          │          │         │
 │         │                     Result (with mixins)   │          │         │
 │         │<───────────────────────────────────────────────────────┤         │
 │         │          │           │           │          │          │         │
 │  Result │          │           │           │          │          │         │
 │<────────┤          │           │           │          │          │         │
 │         │          │           │           │          │          │         │
```

---

## Summary

### Current Architecture Strengths

1. **Strategy Pattern**: Clean separation of concerns, easy to extend
2. **Type Safety**: Pydantic models throughout
3. **Async/Await**: Non-blocking operations
4. **Validation**: Multi-level validation at each stage
5. **Performance Monitoring**: Comprehensive metrics collection
6. **LLM Integration**: Intelligent business logic generation

### Mixin Integration Requirements

1. **Contract Parsing** (ContractIntrospector):
   - Parse mixin declarations from YAML
   - Validate mixin compatibility
   - Extract mixin metadata

2. **Template Context** (TemplateEngine):
   - Add mixin-specific context
   - Include mixin imports
   - Generate mixin initialization

3. **Code Injection** (MixinInjector - NEW):
   - Inject imports
   - Modify class inheritance
   - Add/override methods
   - Inject initialization code

4. **Template Selection** (TemplateSelector - NEW):
   - Select templates based on mixins
   - Support mixin-optimized templates
   - Fall back to base templates

5. **Validation** (MixinValidator - NEW):
   - Validate imports present
   - Validate inheritance correct
   - Validate methods accessible
   - Validate configuration valid

### Next Steps

1. **Phase 1**: Implement mixin parsing in ContractIntrospector
2. **Phase 2**: Create MixinInjector component
3. **Phase 3**: Extend template context building
4. **Phase 4**: Create MixinValidator component
5. **Phase 5**: Integrate with generation strategies
6. **Phase 6**: Add mixin-optimized templates
7. **Phase 7**: Test and validate mixin system

---

## References

- **Code Generation Service**: `src/omninode_bridge/codegen/service.py`
- **Strategies**: `src/omninode_bridge/codegen/strategies/`
- **Template Engine**: `src/omninode_bridge/codegen/template_engine.py`
- **Template Loader**: `src/omninode_bridge/codegen/template_engine_loader/engine.py`
- **Business Logic Generator**: `src/omninode_bridge/codegen/business_logic/generator.py`
- **Contract Introspector**: `src/omninode_bridge/codegen/contract_introspector.py`
- **Node Classifier**: `src/omninode_bridge/codegen/node_classifier.py`

---

**Document Status**: ✅ Complete
**Last Updated**: 2025-11-04
**Author**: Claude Code (Polymorphic Agent)
**Purpose**: Mixin Enhancement Implementation Planning
