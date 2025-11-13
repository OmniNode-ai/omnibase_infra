# OmniNode Code Generation Guide

**Status**: ✅ Production Ready (Phase 2 Complete)
**Last Updated**: 2025-10-30
**Version**: 2.0.0

## Overview

The OmniNode Code Generation System provides intelligent, automated generation of ONEX v2.0-compliant nodes from natural language descriptions. It combines PRD analysis, intelligent classification, template-based generation, and comprehensive quality validation through a 9-stage event-driven pipeline.

**Key Features**:
- **9-Stage Pipeline**: LlamaIndex Workflows-based orchestration (38.5 second target)
- **CLI Command**: `omninode-generate` for command-line node generation
- **Event-Driven**: 13 Kafka topics for real-time progress tracking and observability
- **Remote Infrastructure**: Deployed on 192.168.86.200 (Redpanda, PostgreSQL, Consul)
- **Proven Working**: Successfully generated production nodes (e.g., postgres_crud_effect, Oct 27, 2025)
- **ONEX v2.0 Compliant**: 100% compliance with ONEX architectural patterns

## Architecture

### High-Level Overview

The code generation system operates at two levels:

1. **NodeCodegenOrchestrator**: High-level 9-stage event-driven pipeline coordinator
2. **Code Generation Modules**: Low-level building blocks (PRDAnalyzer, NodeClassifier, TemplateEngine, QualityValidator)

```
┌──────────────────────────────────────────────────────────────────────┐
│                    NodeCodegenOrchestrator                            │
│                    (9-Stage LlamaIndex Workflow)                      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  CLI Command (omninode-generate)                                      │
│         ↓                                                              │
│  Kafka Event Bus (13 topics on 192.168.86.200:9092)                  │
│         ↓                                                              │
│  ┌──────────────────────────────────────────────────────────┐        │
│  │ Stage 1: Prompt Parsing              (5s)  ──→ Event     │        │
│  │ Stage 2: Intelligence Gathering      (3s)  ──→ Event     │        │
│  │ Stage 3: Contract Building           (2s)  ──→ Event     │        │
│  │ Stage 4: Code Generation          (10-15s) ──→ Event     │        │
│  │ Stage 5: Event Bus Integration       (2s)  ──→ Event     │        │
│  │ Stage 6: Validation                  (5s)  ──→ Event     │        │
│  │ Stage 7: Refinement                  (3s)  ──→ Event     │        │
│  │ Stage 8: File Writing                (3s)  ──→ Event     │        │
│  │ Stage 9: Test Generation             (3s)  ──→ Event     │        │
│  └──────────────────────────────────────────────────────────┘        │
│         ↓                                                              │
│  Generated ONEX Node (contract.yaml, node.py, models/, tests/)       │
│                                                                        │
└──────────────────────────────────────────────────────────────────────┘

Underlying Modules (used by Stage 1, 4, 6):
  • PRDAnalyzer      → Requirements extraction
  • NodeClassifier   → Node type classification
  • TemplateEngine   → Code generation from templates
  • QualityValidator → ONEX compliance validation
```

### 9-Stage Pipeline (NodeCodegenOrchestrator)

The **NodeCodegenOrchestrator** coordinates the complete code generation workflow using **LlamaIndex Workflows** for event-driven stage management.

#### Stage 1: Prompt Parsing (5s target)

**Purpose**: Extract structured requirements from natural language prompt

**Input**: Natural language prompt (e.g., "Create PostgreSQL CRUD Effect node")

**Processing**:
- Uses `PRDAnalyzer` for requirement extraction
- Keyword-based domain detection (database, api, ml, messaging, etc.)
- Operation identification (CRUD, transform, aggregate, orchestrate)
- Feature extraction (connection pooling, retry logic, caching, etc.)

**Output**: `ModelPRDRequirements` with node type, domain, operations, features

**Event Published**: `omninode_codegen_stage_completed` (stage=1)

#### Stage 2: Intelligence Gathering (3s target)

**Purpose**: Query RAG system for relevant patterns and best practices (optional)

**Input**: Parsed requirements from Stage 1

**Processing**:
- Optional RAG query to Archon MCP intelligence service
- Pattern matching for similar existing nodes
- Best practices and architectural recommendations
- **Graceful degradation**: Continues without intelligence if service unavailable
- **Circuit breaker**: Protects against intelligence service failures

**Output**: Intelligence data (patterns, recommendations) or empty dict

**Event Published**: `omninode_codegen_stage_completed` (stage=2)

**Note**: Intelligence gathering is **optional**. Disable with `--disable-intelligence` CLI flag.

#### Stage 3: Contract Building (2s target)

**Purpose**: Generate ONEX v2.0 contract YAML

**Input**: Requirements + intelligence data

**Processing**:
- Contract metadata (name, version, description)
- Node type and base class specification
- Method signatures and parameters
- Error handling patterns
- Subcontract references (FSM, caching, etc.)

**Output**: `contract.yaml` content

**Event Published**: `omninode_codegen_stage_completed` (stage=3)

#### Stage 4: Code Generation (10-15s target)

**Purpose**: Generate complete ONEX node implementation

**Input**: Requirements + classification + contract

**Processing**:
- Uses `NodeClassifier` for node type classification (Effect/Compute/Reducer/Orchestrator)
- Uses `TemplateEngine` for Jinja2-based code generation
- Generates all artifacts:
  - `node.py`: Main node implementation
  - `__init__.py`: Module exports
  - `models/*.py`: Pydantic data models (3+ files)
  - `tests/*.py`: Test scaffolding (3+ files)
  - `README.md`: Documentation

**Output**: `ModelGeneratedArtifacts` with all file contents

**Event Published**: `omninode_codegen_stage_completed` (stage=4)

#### Stage 5: Event Bus Integration (2s target)

**Purpose**: Generate Kafka event schemas for the new node

**Input**: Generated node artifacts

**Processing**:
- Event schema generation (request/response/status)
- Topic naming conventions (ONEX-compliant)
- Producer/consumer code generation
- Event envelope integration (OnexEnvelopeV1)

**Output**: Updated artifacts with event bus integration code

**Event Published**: `omninode_codegen_stage_completed` (stage=5)

#### Stage 6: Validation (5s target)

**Purpose**: Validate ONEX compliance and code quality

**Input**: Generated artifacts

**Processing**:
- Uses `QualityValidator` for comprehensive validation
- **ONEX Compliance** (30% weight): Naming, base class, method signatures, error handling
- **Type Safety** (25% weight): Type hints, return annotations, Pydantic models
- **Code Quality** (20% weight): Complexity, imports, formatting
- **Documentation** (15% weight): Docstrings, README, contract docs
- **Test Coverage** (10% weight): Test file presence, assertion coverage

**Output**: `ModelValidationResult` with quality scores and issues

**Event Published**: `omninode_codegen_stage_completed` (stage=6)

**Quality Gate**: Overall score must be ≥ 0.8 to pass (configurable)

#### Stage 7: Refinement (3s target)

**Purpose**: Apply fixes and improvements based on validation results

**Input**: Artifacts + validation results

**Processing**:
- Fix critical errors (ONEX compliance violations)
- Apply suggestions (type hints, docstrings)
- Optional: AI Quorum validation for architectural decisions
- Re-validate after refinements

**Output**: Refined artifacts

**Event Published**: `omninode_codegen_stage_completed` (stage=7)

#### Stage 8: File Writing (3s target)

**Purpose**: Write generated files to disk

**Input**: Refined artifacts

**Processing**:
- Create output directory structure
- Write all files (node, contract, models, tests, docs)
- Set proper file permissions
- Generate summary report

**Output**: File paths of written files

**Event Published**: `omninode_codegen_stage_completed` (stage=8)

#### Stage 9: Test Generation (3s target)

**Purpose**: Generate comprehensive test files using NodeTestGeneratorEffect

**Input**: Generated artifacts from Stages 4-8

**Processing**:
- Build test contract YAML from generation context
- Initialize NodeTestGeneratorEffect
- Generate test files using Jinja2 templates:
  - `tests/test_unit.py`: Unit tests with mocks
  - `tests/test_integration.py`: Integration tests
  - `tests/test_contract.py`: ONEX compliance tests
  - `tests/test_performance.py`: Performance benchmarks
  - `tests/conftest.py`: Pytest fixtures
- Write test files to output directory
- Validate Python syntax (AST-based)
- Calculate test quality metrics

**Output**: Test files with comprehensive coverage

**Event Published**: `omninode_codegen_completed` (success) or `omninode_codegen_failed` (failure)

**Quality Metrics**:
- Test file count
- Total lines of test code
- Quality score (AST validation)
- Coverage target compliance

**Test Types by Node Type**:
- **Effect**: Unit, Contract, Integration, Performance
- **Compute**: Unit, Contract, Performance
- **Reducer**: Unit, Contract, Integration, Performance
- **Orchestrator**: Unit, Contract, Integration, Performance

**Configuration**:
```yaml
test_configuration:
  pytest_plugins:
    - pytest-asyncio
    - pytest-cov
    - pytest-mock
  coverage_target: 90
  enable_fixtures: true
  enable_parametrized_tests: true
```

**Total Pipeline Time**: ~38.5 seconds (target) | ~35-40 seconds (typical actual)
  - Stages 1-8: Code generation and validation (35.5s)
  - Stage 9: Test generation (3s)

### Event-Driven Architecture

The code generation system uses **13 Kafka topics** for real-time progress tracking and observability:

#### Request Topics (4)

| Topic | Purpose | Schema |
|-------|---------|--------|
| `omninode_codegen_request_analyze_v1` | PRD analysis requests | `CodegenAnalysisRequest` |
| `omninode_codegen_request_validate_v1` | Code validation requests | `CodegenValidationRequest` |
| `omninode_codegen_request_pattern_v1` | Pattern matching requests | `CodegenPatternRequest` |
| `omninode_codegen_request_mixin_v1` | Mixin generation requests | `CodegenMixinRequest` |

#### Response Topics (4)

| Topic | Purpose | Schema |
|-------|---------|--------|
| `omninode_codegen_response_analyze_v1` | Analysis results | `CodegenAnalysisResponse` |
| `omninode_codegen_response_validate_v1` | Validation results | `CodegenValidationResponse` |
| `omninode_codegen_response_pattern_v1` | Pattern matching results | `CodegenPatternResponse` |
| `omninode_codegen_response_mixin_v1` | Mixin generation results | `CodegenMixinResponse` |

#### Status Topics (4)

| Topic | Purpose | Schema | Partitions |
|-------|---------|--------|-----------|
| `omninode_codegen_status_session_v1` | Workflow status and progress | `CodegenSessionStatus` | 6 |
| `omninode_test_generation_started_v1` | Test generation started | `TestGenerationStarted` | 3 |
| `omninode_test_generation_completed_v1` | Test generation completed | `TestGenerationCompleted` | 3 |
| `omninode_test_generation_failed_v1` | Test generation failed | `TestGenerationFailed` | 3 |

#### Dead Letter Queue (DLQ) Topics (4)

| Topic | Purpose |
|-------|---------|
| `omninode_codegen_request_analyze_v1_dlq` | Failed analysis requests |
| `omninode_codegen_request_validate_v1_dlq` | Failed validation requests |
| `omninode_codegen_request_pattern_v1_dlq` | Failed pattern requests |
| `omninode_codegen_request_mixin_v1_dlq` | Failed mixin requests |

**Event Flow**:

```
CLI (omninode-generate)
    ↓ publishes to
Kafka: omninode_codegen_request_analyze_v1
    ↓ consumed by
NodeCodegenOrchestrator (Stage 1-8)
    ↓ publishes to
Kafka: omninode_codegen_status_session_v1 (8 stage events)
    ↓ consumed by
CLI Progress Display (real-time UI updates)
    ↓
Completion Event (success/failure)
```

**Kafka Infrastructure**: Redpanda cluster on **192.168.86.200:9092**

### Remote Infrastructure

**Production Deployment** on `192.168.86.200`:

| Service | Port | Purpose |
|---------|------|---------|
| **Redpanda** (Kafka) | 9092 | Event bus for code generation events |
| **PostgreSQL** | 5432 | Persistence for workflow state and metrics |
| **Consul** | 8500 | Service discovery and configuration |
| **Hook Receiver** | 8001 | Webhook endpoint for external triggers |
| **Orchestrator** | 8060 | NodeCodegenOrchestrator HTTP API |
| **Reducer** | 8061 | Metrics aggregation service |
| **Metadata Stamping** | 8057 | Metadata stamping service |
| **OnexTree** | 8058 | Intelligence tree service |

**Network Configuration**:
```bash
# Required for Kafka connectivity (one-time setup)
echo "192.168.86.200 omninode-bridge-redpanda" | sudo tee -a /etc/hosts
```

**Health Check**:
```bash
# Verify all services are healthy
curl http://192.168.86.200:8060/health  # Orchestrator
curl http://192.168.86.200:8061/health  # Reducer
curl http://192.168.86.200:8001/health  # Hook Receiver
```

### Components (Low-Level Modules)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Code Generation Pipeline                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. PRD Analysis       → Extract requirements from prompt        │
│     ├─ Keyword extraction                                        │
│     ├─ Domain detection                                          │
│     ├─ Operation identification                                  │
│     └─ Archon MCP intelligence (optional)                       │
│                                                                   │
│  2. Node Classification → Determine node type & template         │
│     ├─ Multi-factor scoring (domain, ops, keywords, features)   │
│     ├─ Confidence calculation                                    │
│     └─ Template selection                                        │
│                                                                   │
│  3. Code Generation    → Generate ONEX-compliant code           │
│     ├─ Jinja2 template rendering                                │
│     ├─ ONEX naming conventions                                   │
│     ├─ Contract generation                                       │
│     ├─ Model generation                                          │
│     └─ Test scaffolding                                          │
│                                                                   │
│  4. Quality Validation → Ensure compliance & quality            │
│     ├─ ONEX v2.0 compliance                                     │
│     ├─ Type safety (AST-based)                                  │
│     ├─ Code quality checks                                       │
│     └─ Documentation completeness                                │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Module Hierarchy

```
src/omninode_bridge/codegen/
├── __init__.py                 # Main exports
├── prd_analyzer.py             # PRD analysis & requirement extraction
├── node_classifier.py          # Node type classification
├── template_engine.py          # Code generation from templates
└── quality_validator.py        # Quality & compliance validation
```

## Quick Start

### Prerequisites

Before using the code generation system, ensure:

1. **Kafka Infrastructure**: Redpanda running on 192.168.86.200:9092
2. **Hostname Resolution**: Add Kafka hostname to `/etc/hosts`
   ```bash
   echo "192.168.86.200 omninode-bridge-redpanda" | sudo tee -a /etc/hosts
   ```
3. **Poetry Installation**: Install dependencies
   ```bash
   cd /path/to/omninode_bridge
   poetry install
   ```

### Installation

```bash
# Install omninode_bridge with all dependencies
cd /path/to/omninode_bridge
poetry install

# Verify installation
python -c "from omninode_bridge.codegen import PRDAnalyzer; print('✅ Codegen installed')"

# Verify CLI command is available
omninode-generate --help
```

## CLI Usage (`omninode-generate`)

The **omninode-generate** command provides a simple interface for generating ONEX nodes via the event-driven orchestrator.

### Basic Command Syntax

```bash
omninode-generate "PROMPT" [OPTIONS]
```

### Common Usage Examples

**1. Generate PostgreSQL CRUD Effect Node**:
```bash
omninode-generate "Create PostgreSQL CRUD Effect node with connection pooling and retry logic"
```

**2. Generate with Custom Output Directory**:
```bash
omninode-generate "Create API Gateway Orchestrator" --output-dir ./my_nodes/api_gateway
```

**3. Generate with Node Type Hint**:
```bash
omninode-generate "Create data transformation node" --node-type compute
```

**4. Interactive Mode (checkpoints at each stage)**:
```bash
omninode-generate "Create metrics Reducer" --interactive
```

**5. Disable Intelligence Gathering (faster, no RAG)**:
```bash
omninode-generate "Create simple Effect node" --disable-intelligence
```

**6. Enable AI Quorum Validation (multi-model consensus)**:
```bash
omninode-generate "Create critical Orchestrator" --enable-quorum
```

**7. Custom Timeout**:
```bash
omninode-generate "Create complex node" --timeout 600  # 10 minutes
```

**8. Remote Kafka Server**:
```bash
omninode-generate "Create Effect node" --kafka-servers 192.168.86.200:9092
```

### CLI Options Reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `PROMPT` | string | *required* | Natural language description of the node |
| `--output-dir` | string | `./generated_nodes` | Output directory for generated files |
| `--node-type` | choice | auto-detect | Node type hint: `effect`, `orchestrator`, `reducer`, `compute` |
| `--interactive` | flag | disabled | Enable interactive checkpoints at each stage |
| `--enable-intelligence` | flag | enabled | Use RAG intelligence gathering (default) |
| `--disable-intelligence` | flag | - | Disable RAG intelligence gathering |
| `--enable-quorum` | flag | disabled | Use AI quorum validation (multi-model consensus) |
| `--timeout` | int | 300 | Timeout in seconds (1-3600) |
| `--kafka-servers` | string | `localhost:29092` | Kafka bootstrap servers |

### CLI Output Example

```bash
$ omninode-generate "Create PostgreSQL CRUD Effect"

🚀 Generating ONEX node...
   Correlation ID: 3f2a1b5c-8d4e-4f1a-9c3b-7e6d5a4c3b2a
   Prompt: Create PostgreSQL CRUD Effect
   Output: ./generated_nodes

┌──────────────────────────────────────────────────────────┐
│ Stage 1/9: Prompt Parsing                    [✓] 2.3s   │
│ Stage 2/9: Intelligence Gathering            [✓] 3.1s   │
│ Stage 3/9: Contract Building                 [✓] 1.8s   │
│ Stage 4/9: Code Generation                   [✓] 12.4s  │
│ Stage 5/9: Event Bus Integration             [✓] 2.1s   │
│ Stage 6/9: Validation                        [✓] 4.7s   │
│ Stage 7/9: Refinement                        [✓] 2.9s   │
│ Stage 8/9: File Writing                      [✓] 1.5s   │
│ Stage 9/9: Test Generation                   [✓] 2.8s   │
└──────────────────────────────────────────────────────────┘

✅ Generation complete!
   Duration: 33.6s
   Quality Score: 0.94
   Files Generated: 15 (10 code + 5 tests)

   Generated files:
   - generated_nodes/postgres_crud_effect_3f2a1b5c/node.py
   - generated_nodes/postgres_crud_effect_3f2a1b5c/contract.yaml
   - generated_nodes/postgres_crud_effect_3f2a1b5c/__init__.py
   - generated_nodes/postgres_crud_effect_3f2a1b5c/models/model_request.py
   - generated_nodes/postgres_crud_effect_3f2a1b5c/models/model_response.py
   - generated_nodes/postgres_crud_effect_3f2a1b5c/models/model_config.py
   - generated_nodes/postgres_crud_effect_3f2a1b5c/tests/test_node.py
   - generated_nodes/postgres_crud_effect_3f2a1b5c/tests/test_integration.py
   - generated_nodes/postgres_crud_effect_3f2a1b5c/tests/test_unit.py
   - generated_nodes/postgres_crud_effect_3f2a1b5c/README.md
```

### Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success - node generated successfully |
| `1` | Error - generation failed (check error message) |
| `130` | Cancelled - user interrupted with Ctrl+C |

### Environment Variables

Configure default behavior with environment variables:

```bash
# Kafka configuration
export OMNINODE_KAFKA_BOOTSTRAP_SERVERS="192.168.86.200:9092"

# Default output directory
export OMNINODE_DEFAULT_OUTPUT_DIR="./generated_nodes"

# Default timeout
export OMNINODE_DEFAULT_TIMEOUT_SECONDS=300

# Run generation
omninode-generate "Create Effect node"  # Uses env vars
```

### Programmatic Usage (Python API)

For advanced use cases, use the Python API directly:

```python
import asyncio
from omninode_bridge.cli.codegen.commands.generate import generate_node_async
from omninode_bridge.cli.codegen.client import CLIKafkaClient
from omninode_bridge.cli.codegen.ui import ProgressDisplay
from uuid import uuid4

async def generate_custom_node():
    """Generate node with full control over parameters."""

    # Initialize clients
    kafka_client = CLIKafkaClient(bootstrap_servers="192.168.86.200:9092")
    await kafka_client.connect()

    try:
        correlation_id = uuid4()
        progress_display = ProgressDisplay(correlation_id=correlation_id)

        result = await generate_node_async(
            prompt="Create PostgreSQL CRUD Effect node",
            output_dir="./my_custom_output",
            kafka_client=kafka_client,
            progress_display=progress_display,
            node_type="effect",
            interactive=False,
            enable_intelligence=True,
            enable_quorum=False,
            timeout_seconds=300,
        )

        if result.success:
            print(f"✅ Generated {len(result.files)} files")
            print(f"   Quality: {result.quality_score:.2f}")
            print(f"   Duration: {result.duration_seconds:.1f}s")
        else:
            print(f"❌ Failed: {result.error}")

    finally:
        await kafka_client.disconnect()

# Run
asyncio.run(generate_custom_node())
```

## Programmatic Usage (Low-Level Modules)

For fine-grained control over the generation process, use the low-level modules directly.

```python
from pathlib import Path
from omninode_bridge.codegen import (
    PRDAnalyzer,
    NodeClassifier,
    TemplateEngine,
    QualityValidator,
)

async def generate_node(prompt: str, output_dir: Path):
    """Generate ONEX node from natural language prompt."""

    # Step 1: Analyze prompt
    analyzer = PRDAnalyzer()
    requirements = await analyzer.analyze_prompt(prompt)

    print(f"📋 Extracted requirements:")
    print(f"   Node Type: {requirements.node_type}")
    print(f"   Domain: {requirements.domain}")
    print(f"   Operations: {requirements.operations}")

    # Step 2: Classify node type
    classifier = NodeClassifier()
    classification = classifier.classify(requirements)

    print(f"\n🎯 Classification:")
    print(f"   Type: {classification.node_type}")
    print(f"   Confidence: {classification.confidence:.1%}")
    print(f"   Template: {classification.template_name}")

    # Step 3: Generate code
    engine = TemplateEngine()
    artifacts = await engine.generate(
        requirements=requirements,
        classification=classification,
        output_directory=output_dir,
    )

    print(f"\n📦 Generated artifacts:")
    print(f"   Node: {artifacts.node_name}")
    print(f"   Files: {len(artifacts.get_all_files())}")

    # Step 4: Validate quality
    validator = QualityValidator()
    validation = await validator.validate(artifacts)

    print(f"\n✅ Quality validation:")
    print(f"   Overall Score: {validation.quality_score:.1%}")
    print(f"   ONEX Compliance: {validation.onex_compliance_score:.1%}")
    print(f"   Passed: {'✅' if validation.passed else '❌'}")

    if validation.warnings:
        print(f"\n⚠️  Warnings:")
        for warning in validation.warnings:
            print(f"   - {warning}")

    return artifacts, validation

# Usage
prompt = "Create PostgreSQL CRUD Effect node with connection pooling"
output_dir = Path("./generated_nodes/postgres_crud")
artifacts, validation = await generate_node(prompt, output_dir)
```

## Component Details

### 1. PRDAnalyzer

**Purpose**: Extracts structured requirements from natural language prompts.

**Features**:
- Keyword-based extraction (regex patterns)
- Domain detection (database, api, ml, messaging, storage, cache, monitoring)
- Operation identification (CRUD, transform, aggregate, orchestrate)
- Feature extraction (connection pooling, caching, retry logic, etc.)
- Archon MCP intelligence integration (optional)
- Graceful degradation when intelligence unavailable

**API**:

```python
analyzer = PRDAnalyzer(
    archon_mcp_url="http://localhost:8060",  # Optional
    enable_intelligence=True,                 # Optional
    timeout_seconds=30,                       # Optional
)

requirements = await analyzer.analyze_prompt(
    prompt="Create database node...",
    correlation_id=uuid4(),                   # Optional
    node_type_hint="effect",                  # Optional
)

# Access requirements
print(requirements.node_type)           # "effect"
print(requirements.domain)              # "database"
print(requirements.operations)          # ["create", "read", "update", "delete"]
print(requirements.features)            # ["connection_pooling", "retry_logic"]
print(requirements.extraction_confidence)  # 0.7-1.0
```

**Output**: `ModelPRDRequirements` with:
- `node_type`: Inferred node type (effect|compute|reducer|orchestrator)
- `service_name`: Service name in snake_case
- `domain`: Domain classification
- `operations`: List of operations to implement
- `business_description`: Human-readable purpose
- `features`: Key features to implement
- `dependencies`: External service dependencies
- `performance_requirements`: Performance targets
- `best_practices`: Best practices from intelligence service
- `extraction_confidence`: Confidence score (0.0-1.0)

### 2. NodeClassifier

**Purpose**: Classifies requirements into ONEX node types and selects templates.

**Classification Strategy**:
- **Multi-factor analysis** (4 factors, weighted):
  - Domain analysis (30% weight)
  - Operation analysis (30% weight)
  - Keyword analysis (25% weight)
  - Feature analysis (15% weight)
- **Confidence scoring** (0.0-1.0)
- **Template selection** based on node type and domain
- **Alternative suggestions** for ambiguous cases

**Node Type Characteristics**:

| Node Type | Indicators | Examples |
|-----------|-----------|----------|
| **Effect** | I/O operations, external services, side effects | Database CRUD, HTTP requests, Kafka publishing |
| **Compute** | Pure transformations, calculations, algorithms | Data parsing, hashing, encryption, formatting |
| **Reducer** | Aggregation, accumulation, state management | Metrics collection, data summarization, grouping |
| **Orchestrator** | Coordination, workflow, multi-step processes | Pipeline management, service routing, parallel execution |

**API**:

```python
classifier = NodeClassifier()

classification = classifier.classify(requirements)

# Access classification
print(classification.node_type)          # EnumNodeType.EFFECT
print(classification.confidence)         # 0.92
print(classification.template_name)      # "effect_database"
print(classification.template_variant)   # "pooled"
print(classification.primary_indicators) # ["Domain match: database", ...]
print(classification.alternatives)       # [{"node_type": "compute", "confidence": 0.4}]
```

**Output**: `ModelClassificationResult` with:
- `node_type`: Classified type (Effect, Compute, Reducer, Orchestrator)
- `confidence`: Classification confidence (0.0-1.0)
- `template_name`: Selected template
- `template_variant`: Template variant (optional)
- `primary_indicators`: Reasons for classification
- `alternatives`: Alternative classifications with confidence scores

### 3. TemplateEngine

**Purpose**: Generates ONEX-compliant code from templates.

**Features**:
- Jinja2 template rendering (filesystem-based)
- Inline template fallback (no external templates required)
- ONEX v2.0 naming conventions (Node<Name><Type>)
- Complete artifact generation:
  - `node.py`: Main node implementation
  - `contract.yaml`: ONEX v2.0 contract
  - `__init__.py`: Module exports
  - `models/*.py`: Pydantic data models
  - `tests/*.py`: Test scaffolding
  - `README.md`: Documentation

**Templates Available** (inline):
- `effect_generic`: General Effect nodes
- `effect_database`: Database-specific Effect nodes
- `compute_generic`: General Compute nodes
- `reducer_generic`: General Reducer nodes
- `orchestrator_workflow`: Workflow Orchestrator nodes

**API**:

```python
engine = TemplateEngine(
    templates_directory=Path("./templates"),  # Optional
    enable_inline_templates=True,             # Optional
)

artifacts = await engine.generate(
    requirements=requirements,
    classification=classification,
    output_directory=Path("./output"),
)

# Access artifacts
print(artifacts.node_name)           # "NodePostgresCrudEffect"
print(artifacts.service_name)        # "postgres_crud"
print(artifacts.node_type)           # "effect"
print(len(artifacts.node_file))      # 1500+ chars
print(artifacts.get_all_files())     # Dict of all files

# Write to disk
for filename, content in artifacts.get_all_files().items():
    file_path = artifacts.output_directory / filename
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content)
```

**Output**: `ModelGeneratedArtifacts` with:
- `node_file`: Main node implementation
- `contract_file`: ONEX contract YAML
- `init_file`: Module __init__.py
- `models`: Dict of model files
- `tests`: Dict of test files
- `documentation`: Dict of documentation files
- `node_name`: Generated node class name
- `service_name`: Service name

### 4. QualityValidator

**Purpose**: Validates generated code quality and ONEX compliance.

**Validation Checks**:

| Check | Weight | Details |
|-------|--------|---------|
| **ONEX Compliance** | 30% | Naming, base class, method signatures, error handling |
| **Type Safety** | 25% | Type hints, return annotations, Pydantic models |
| **Code Quality** | 20% | Complexity, formatting, imports (optional: ruff) |
| **Documentation** | 15% | Docstrings, README, contract documentation |
| **Test Coverage** | 10% | Test file presence, test types, assertions |

**Quality Gates**:
- **Pass**: Overall score ≥ 0.8 (configurable)
- **Warnings**: Component scores < 0.8
- **Errors**: Component scores < 0.5

**API**:

```python
validator = QualityValidator(
    enable_mypy=False,              # Optional: Enable mypy type checking
    enable_ruff=False,              # Optional: Enable ruff linting
    min_quality_threshold=0.8,      # Optional: Minimum passing score
)

validation = await validator.validate(artifacts)

# Access validation results
print(f"Overall: {validation.quality_score:.1%}")
print(f"ONEX Compliance: {validation.onex_compliance_score:.1%}")
print(f"Type Safety: {validation.type_safety_score:.1%}")
print(f"Code Quality: {validation.code_quality_score:.1%}")
print(f"Documentation: {validation.documentation_score:.1%}")
print(f"Test Coverage: {validation.test_coverage_score:.1%}")

print(f"Passed: {validation.passed}")

# Review issues
for error in validation.errors:
    print(f"❌ {error}")
for warning in validation.warnings:
    print(f"⚠️  {warning}")
for suggestion in validation.suggestions:
    print(f"💡 {suggestion}")
```

**Output**: `ModelValidationResult` with:
- `quality_score`: Overall quality (0.0-1.0)
- Component scores (onex_compliance, type_safety, code_quality, documentation, test_coverage)
- `passed`: Boolean validation status
- `errors`: Critical issues (score < 0.5)
- `warnings`: Non-critical issues (score < 0.8)
- `suggestions`: Improvement recommendations
- Detailed issue lists per component

### 5. Test Execution & Failure Analysis (NEW!)

**Status**: ✅ Production Ready (Phase 2.5 - October 2025)

**Purpose**: Execute generated tests and analyze failures to validate that generated code actually works.

**Problem Solved**: Previously, code generation created tests but never ran them. This meant generated code was never validated to work correctly, creating a critical production readiness gap.

**Key Features**:
- **Automatic Test Execution**: Run pytest on generated tests via subprocess
- **Failure Classification**: Identify root causes (import errors, type errors, runtime errors, etc.)
- **Actionable Recommendations**: Specific fix suggestions for each failure type
- **Coverage Measurement**: Track test coverage percentage
- **Strict/Permissive Modes**: Configurable error handling
- **Detailed Reporting**: Comprehensive test execution and failure analysis reports

#### 5.1. TestExecutor

**Purpose**: Execute generated tests using pytest in clean subprocess isolation.

**Features**:
- Subprocess-based pytest execution (clean isolation)
- JSON output parsing for detailed results
- Coverage measurement integration
- Timeout handling (default: 300s)
- Test type filtering (unit, integration, performance)
- Graceful fallback to text parsing

**API**:

```python
from omninode_bridge.codegen import TestExecutor, TestExecutionConfig

# Configure test execution
config = TestExecutionConfig(
    test_types=["unit", "integration"],  # Skip performance tests
    timeout_seconds=60,                  # 1 minute timeout
    verbose=True,                        # Verbose output
    enable_coverage=True,                # Measure coverage
    coverage_threshold=80.0,             # Minimum coverage
)

# Create executor
executor = TestExecutor(config)

# Run tests
results = await executor.run_tests(
    output_directory=Path("./generated_nodes/postgres_crud"),
    test_types=["unit"],  # Override config
    timeout_seconds=30,   # Override timeout
)

# Access results
print(results.get_summary())
print(f"Success Rate: {results.success_rate:.1%}")
print(f"Coverage: {results.coverage_percent:.1f}%")
print(f"Duration: {results.duration_seconds:.2f}s")

# Check failures
if not results.is_passing:
    print(f"Failed tests ({results.failed}):")
    for test in results.failed_tests:
        print(f"  - {test['name']}: {test['error']}")
```

**Output**: `ModelTestResults` with:
- `passed`, `failed`, `skipped`, `total`: Test counts
- `duration_seconds`: Total execution time
- `coverage_percent`: Code coverage (if enabled)
- `success_rate`: Pass rate (0.0-1.0)
- `is_passing`: Boolean (all tests passed)
- `failed_tests`: Detailed failure information (name, error, traceback, file)
- `exit_code`: Pytest exit code
- `stdout`, `stderr`: Raw output
- `pytest_version`, `python_version`: Environment info

**Test Execution Workflow**:

```
1. Validate test directory exists
2. Build pytest command with args
   - JSON report generation (if pytest-json-report installed)
   - Coverage measurement (--cov)
   - Test type filtering (-m unit/integration)
   - Verbosity and output capture
3. Execute pytest in subprocess
4. Parse JSON output (or fallback to text parsing)
5. Extract test counts, failures, coverage
6. Return ModelTestResults
```

#### 5.2. FailureAnalyzer

**Purpose**: Analyze test failures and provide actionable recommendations.

**Features**:
- Root cause classification (7 categories)
- Severity assessment (critical, high, medium, low)
- Auto-fix feasibility detection
- Estimated fix time calculation
- Affected file tracking
- Pattern-based error detection

**Failure Categories**:

| Category | Indicators | Example | Auto-Fixable |
|----------|-----------|---------|--------------|
| **Import Error** | `ModuleNotFoundError`, `ImportError` | Missing dependencies (hvac, pytest-asyncio) | ✅ Yes |
| **Type Error** | `TypeError`, `ValidationError` | Incorrect type annotations | ❌ No |
| **Runtime Error** | `AttributeError`, `KeyError`, `ValueError` | Logic errors in implementation | ❌ No |
| **Assertion Error** | `AssertionError`, failed assertions | Test expectations don't match behavior | ❌ No |
| **Configuration Error** | Missing config, invalid settings | Environment variables not set | ⚠️ Maybe |
| **Infrastructure Error** | Connection failures, timeouts | External services unavailable | ⚠️ Maybe |
| **Timeout Error** | Test execution timeout | Performance issues | ❌ No |

**API**:

```python
from omninode_bridge.codegen import FailureAnalyzer

# Create analyzer
analyzer = FailureAnalyzer()

# Analyze test results
analysis = analyzer.analyze(test_results)

# Access analysis
print(analysis.get_report())  # Markdown report
print(f"Severity: {analysis.severity.value}")
print(f"Auto-fixable: {analysis.auto_fixable}")
print(f"Fix time: {analysis.estimated_fix_time_minutes} minutes")

# Review root causes
for cause in analysis.root_causes:
    print(f"Category: {cause.category.value}")
    print(f"Description: {cause.description}")
    print(f"Affected files: {', '.join(cause.affected_files)}")

# Apply recommendations
for fix in analysis.recommended_fixes:
    print(f"Fix: {fix}")
```

**Output**: `ModelFailureAnalysis` with:
- `root_causes`: List of identified causes with categories and descriptions
- `affected_files`: All files affected by failures
- `affected_tests`: Names of failed tests
- `recommended_fixes`: Actionable fix recommendations
- `severity`: Overall severity (critical, high, medium, low, info)
- `auto_fixable`: Whether failures can be automatically fixed
- `estimated_fix_time_minutes`: Manual fix time estimate
- `failure_count`: Total failures
- `unique_error_types`: Number of distinct error types
- `summary`: Executive summary

**Severity Assessment**:
- **CRITICAL**: All tests failed (0% success rate)
- **HIGH**: >50% failure rate OR import errors present
- **MEDIUM**: 20-50% failure rate
- **LOW**: <20% failure rate

**Fix Time Estimation**:
- Import errors: 5 minutes (add to pyproject.toml)
- Type errors: 15 minutes (fix annotations)
- Assertion errors: 30 minutes (fix logic)
- Runtime errors: 20 minutes (debug)
- Configuration errors: 10 minutes (update config)
- Infrastructure errors: 30 minutes (fix external deps)

#### 5.3. Integrated Test Execution in TemplateEngine

**Purpose**: Automatically execute tests during code generation for immediate validation.

**Usage**:

```python
from omninode_bridge.codegen import TemplateEngine

engine = TemplateEngine(enable_inline_templates=True)

# Generate code WITH test execution
artifacts = await engine.generate(
    requirements=requirements,
    classification=classification,
    output_directory=output_dir,
    run_tests=True,        # NEW: Execute tests automatically
    strict_mode=False,     # NEW: Don't raise exception on failures
)

# Check test results
if artifacts.test_results:
    print(f"Tests: {artifacts.test_results.passed}/{artifacts.test_results.total}")
    print(f"Coverage: {artifacts.test_results.coverage_percent:.1f}%")

    if not artifacts.test_results.is_passing:
        # Review failure analysis
        if artifacts.failure_analysis:
            print(artifacts.failure_analysis.get_report())
            for fix in artifacts.failure_analysis.recommended_fixes:
                print(f"Fix: {fix}")
```

**Modes**:

1. **Strict Mode** (`strict_mode=True`):
   - Raises `ValueError` if tests fail
   - Stops generation process
   - Use for CI/CD pipelines
   - Ensures only passing code is generated

2. **Permissive Mode** (`strict_mode=False`, default):
   - Attaches test results and failure analysis to artifacts
   - Continues generation even if tests fail
   - Logs warnings for failures
   - Use for exploratory development

**Workflow**:

```
1. Generate code artifacts (node, contract, models, tests, docs)
2. If run_tests=True:
   a. Write artifacts to disk
   b. Execute TestExecutor.run_tests()
   c. Attach results to artifacts.test_results
   d. If tests failed:
      - Run FailureAnalyzer.analyze()
      - Attach analysis to artifacts.failure_analysis
      - Log failure summary and recommendations
      - If strict_mode=True: raise ValueError
   e. If tests passed:
      - Log success message
3. Return artifacts (with test results and analysis)
```

**Benefits**:
- ✅ Validates generated code actually works
- ✅ Catches failures immediately (not in CI/CD)
- ✅ Provides actionable fix recommendations
- ✅ Auto-classifies failure root causes
- ✅ Estimates fix time and severity
- ✅ Closes production readiness gap

**Example Output**:

```
✅ Code generated: NodeVaultSecretsEffect
   Files: 12
   Tests executed: Yes (run_tests=True)

Test Execution Results:
❌ FAILED
  Tests: 8/10 passed (80.0% success rate)
  Failed: 2
  Skipped: 0
  Duration: 4.23s
  Coverage: 76.5%

Failure Analysis:
  Severity: MEDIUM
  Auto-fixable: Yes
  Est. Fix Time: 10 minutes

Root Causes:
1. Import Error
   Missing Python dependencies: hvac, pytest-asyncio
   Affected Files: pyproject.toml

Recommended Fixes:
1. Add missing dependencies to pyproject.toml: hvac, pytest-asyncio
2. Run `poetry install` to install dependencies
3. Re-run tests: pytest ./generated_nodes/vault_secrets/tests/
```

#### 5.4. Configuration

**Test Execution Config**:

```python
TEST_EXECUTION_CONFIG = {
    "run_tests_by_default": False,  # Enable globally
    "strict_mode": False,           # Raise on failure
    "test_timeout_seconds": 300,    # 5 minute timeout
    "required_coverage_percent": 80,# Minimum coverage
    "test_types": ["unit", "integration"],  # Skip performance
    "enable_coverage": True,        # Measure coverage
    "verbose": True,                # Verbose output
    "retry_on_failure": False,      # Retry failed tests
}
```

**pytest Dependencies**:
- `pytest` (required)
- `pytest-asyncio` (required for async tests)
- `pytest-cov` (optional, for coverage)
- `pytest-json-report` (optional, for JSON output)
- `pytest-mock` (optional, for mocking)

**Installation**:

```bash
# Core dependencies (already in omninode_bridge)
poetry add pytest pytest-asyncio

# Optional (for enhanced features)
poetry add pytest-cov pytest-json-report pytest-mock
```

#### 5.5. Complete Example

See `examples/codegen_with_test_execution.py` for a complete working example demonstrating:
1. Code generation without test execution
2. Code generation with test execution
3. Manual test execution using TestExecutor
4. Failure analysis using FailureAnalyzer
5. Quality validation integration

**Run Example**:

```bash
poetry run python examples/codegen_with_test_execution.py
```

### 6. NodeTestGeneratorEffect

**Purpose**: Generates comprehensive test files from ModelContractTest contracts.

**Features**:
- Jinja2 template-based test generation
- Multiple test types (unit, integration, contract, performance)
- AST-based Python syntax validation
- Quality metrics calculation
- Fixture and mock generation
- Parametrized test support

**API**:

```python
from omninode_bridge.nodes.test_generator_effect.v1_0_0.node import NodeTestGeneratorEffect

# Initialize
test_generator = NodeTestGeneratorEffect(container)

# Create contract
contract = ModelContractEffect(
    name="generate_tests",
    version={"major": 1, "minor": 0, "patch": 0},
    node_type=EnumNodeType.EFFECT,
    io_operations=[{"operation_type": "file_write"}],
    input_state={
        "test_contract_yaml": yaml_string,
        "output_directory": "./tests",
        "node_name": "NodePostgresCrudEffect",
    }
)

# Execute
response = await test_generator.execute_effect(contract)
print(f"Generated {response.file_count} test files")
```

**Output**: `ModelTestGeneratorResponse` with:
- `generated_files`: Dict of test files
- `file_count`: Number of files generated
- `total_lines`: Total lines of test code
- `quality_score`: AST validation score
- `duration_ms`: Generation time

## Integration Architecture

### How NodeCodegenOrchestrator Uses Low-Level Modules

The **NodeCodegenOrchestrator** coordinates the 9-stage pipeline by orchestrating the low-level code generation modules:

```python
# In workflow.py (NodeCodegenOrchestrator implementation)

class CodeGenerationWorkflow(Workflow):
    """9-stage LlamaIndex Workflow for ONEX node code generation."""

    # Stage 1: Prompt Parsing
    @step
    async def parse_prompt(self, ctx: Context, ev: StartEvent) -> PromptParsedEvent:
        """Parse natural language prompt and extract requirements."""
        from omninode_bridge.codegen import PRDAnalyzer

        analyzer = PRDAnalyzer(
            enable_intelligence=self.enable_intelligence
        )
        requirements = await analyzer.analyze_prompt(
            prompt=ev.prompt,
            correlation_id=ev.correlation_id,
        )

        # Publish stage completed event to Kafka
        await self._publish_stage_completed(stage=1, ...)

        return PromptParsedEvent(requirements=requirements)

    # Stage 2: Intelligence Gathering (Optional)
    @step
    async def gather_intelligence(
        self, ctx: Context, ev: PromptParsedEvent
    ) -> IntelligenceGatheredEvent:
        """Query RAG system for patterns and best practices."""
        # Optional: Query Archon MCP intelligence service
        # Graceful degradation if service unavailable
        intelligence_data = await self._query_intelligence_with_protection(...)

        await self._publish_stage_completed(stage=2, ...)

        return IntelligenceGatheredEvent(intelligence_data=intelligence_data)

    # Stage 3: Contract Building
    @step
    async def build_contract(
        self, ctx: Context, ev: IntelligenceGatheredEvent
    ) -> ContractBuiltEvent:
        """Generate ONEX v2.0 contract YAML."""
        contract_yaml = self._build_contract_yaml(...)

        await self._publish_stage_completed(stage=3, ...)

        return ContractBuiltEvent(contract_yaml=contract_yaml)

    # Stage 4: Code Generation
    @step
    async def generate_code(
        self, ctx: Context, ev: ContractBuiltEvent
    ) -> CodeGeneratedEvent:
        """Generate complete ONEX node implementation."""
        from omninode_bridge.codegen import NodeClassifier, TemplateEngine

        # Classify node type
        classifier = NodeClassifier()
        classification = classifier.classify(requirements)

        # Generate code from templates
        engine = TemplateEngine(enable_inline_templates=True)
        artifacts = await engine.generate(
            requirements=requirements,
            classification=classification,
            output_directory=output_dir,
        )

        await self._publish_stage_completed(stage=4, ...)

        return CodeGeneratedEvent(generated_files=artifacts.get_all_files())

    # Stage 5: Event Bus Integration
    @step
    async def integrate_event_bus(
        self, ctx: Context, ev: CodeGeneratedEvent
    ) -> EventBusIntegratedEvent:
        """Generate Kafka event schemas for the new node."""
        # Generate event schemas and topic configurations
        integration_details = self._generate_event_schemas(...)

        await self._publish_stage_completed(stage=5, ...)

        return EventBusIntegratedEvent(integration_details=integration_details)

    # Stage 6: Validation
    @step
    async def validate_code(
        self, ctx: Context, ev: EventBusIntegratedEvent
    ) -> ValidationCompleteEvent:
        """Validate ONEX compliance and code quality."""
        from omninode_bridge.codegen import QualityValidator

        validator = QualityValidator(
            enable_mypy=False,
            enable_ruff=False,
            min_quality_threshold=0.8,
        )
        validation = await validator.validate(artifacts)

        await self._publish_stage_completed(stage=6, ...)

        return ValidationCompleteEvent(validation_results=validation.model_dump())

    # Stage 7: Refinement
    @step
    async def refine_code(
        self, ctx: Context, ev: ValidationCompleteEvent
    ) -> RefinementCompleteEvent:
        """Apply fixes and improvements based on validation results."""
        refined_files = await self._apply_refinements(...)

        await self._publish_stage_completed(stage=7, ...)

        return RefinementCompleteEvent(refined_files=refined_files)

    # Stage 8: File Writing
    @step
    async def write_files(
        self, ctx: Context, ev: RefinementCompleteEvent
    ) -> FileWritingCompleteEvent:
        """Write generated files to disk."""
        file_paths = await self._write_files_to_disk(...)

        await self._publish_stage_completed(stage=8, ...)

        return FileWritingCompleteEvent(file_paths=file_paths)

    # Stage 9: Test Generation
    @step
    async def generate_tests(
        self, ctx: Context, ev: FileWritingCompleteEvent
    ) -> StopEvent:
        """Generate comprehensive test files."""
        from omninode_bridge.nodes.test_generator_effect.v1_0_0.node import NodeTestGeneratorEffect

        # Build test contract
        test_contract_yaml = self._build_test_contract_yaml(...)

        # Initialize test generator
        test_generator = NodeTestGeneratorEffect(self.container)

        # Execute test generation
        response = await test_generator.execute_effect(contract)

        await self._publish_stage_completed(stage=9, ...)
        await self._publish_completed_event(success=True, ...)

        return StopEvent(result={
            "success": True,
            "generated_files": all_files,
            "test_files": response.generated_files,
            "quality_score": validation.quality_score,
        })
```

### Event Publishing at Each Stage

Every stage publishes events to Kafka for real-time progress tracking:

```python
async def _publish_stage_completed(
    self,
    stage: EnumPipelineStage,
    duration: float,
    success: bool,
) -> None:
    """Publish stage completion event to Kafka."""
    if self.kafka_client:
        event = ModelEventCodegenStageCompleted(
            correlation_id=gen_ctx.correlation_id,
            workflow_id=gen_ctx.workflow_id,
            stage=stage,
            stage_number=stage.stage_number,
            duration_seconds=duration,
            success=success,
        )

        await self.kafka_client.publish_event(
            topic=TOPIC_CODEGEN_STAGE_COMPLETED,
            event=event,
        )
```

### CLI Integration

The CLI command (`omninode-generate`) consumes events from Kafka and displays progress:

```python
# In cli/codegen/commands/generate.py

async def generate_node_async(...):
    """Generate ONEX node via event-driven orchestration."""

    # Start consuming events in background
    consumer_task = asyncio.create_task(
        kafka_client.consume_progress_events(
            correlation_id=correlation_id,
            callback=progress_display.on_event,
        )
    )

    # Publish generation request event
    request_event = ModelEventNodeGenerationRequested(
        correlation_id=correlation_id,
        prompt=prompt,
        output_directory=output_dir,
        ...
    )
    await kafka_client.publish_request(request_event)

    # Wait for completion with real-time progress updates
    result = await progress_display.wait_for_completion(
        timeout_seconds=timeout_seconds
    )

    return GenerationResult(success=True, ...)
```

## Testing

### Unit Tests

Test individual components:

```bash
# Test PRD analyzer
pytest tests/unit/codegen/test_prd_analyzer.py -v

# Test node classifier
pytest tests/unit/codegen/test_node_classifier.py -v

# Test template engine
pytest tests/unit/codegen/test_template_engine.py -v

# Test quality validator
pytest tests/unit/codegen/test_quality_validator.py -v
```

### Integration Tests

Test complete workflow:

```bash
# Test all node types
pytest tests/integration/test_codegen_workflow_complete.py -v

# Test specific node type
pytest tests/integration/test_codegen_workflow_complete.py::test_effect_node_generation_workflow -v

# Test with markers
pytest tests/integration/ -m integration -v
```

### Performance Benchmarks

Target performance (38.5 seconds total):

| Stage | Target | Component |
|-------|--------|-----------|
| PRD Analysis | 5s | PRDAnalyzer + Archon MCP |
| Classification | <1s | NodeClassifier |
| Code Generation | 10-15s | TemplateEngine |
| Validation | 5s | QualityValidator |
| Test Generation | 3s | NodeTestGeneratorEffect |
| Total | ~30s | (without LLM calls) |

## Examples

### Example 1: Database Effect Node

```python
prompt = """
Create a PostgreSQL CRUD Effect node with:
- Connection pooling (10-50 connections)
- Automatic retry logic with exponential backoff
- Circuit breaker pattern for resilience
- Structured logging and metrics
- Full CRUD operations (Create, Read, Update, Delete, List)
"""

analyzer = PRDAnalyzer()
requirements = await analyzer.analyze_prompt(prompt)

# Result:
# - node_type: "effect"
# - domain: "database"
# - operations: ["create", "read", "update", "delete"]
# - features: ["connection_pooling", "retry_logic", "circuit_breaker", "logging", "metrics"]
```

### Example 2: Data Transformation Compute Node

```python
prompt = """
Create a data transformation Compute node that:
- Parses JSON input
- Validates against schema
- Transforms to XML format
- Handles nested structures
- Pure function (no side effects)
"""

requirements = await analyzer.analyze_prompt(prompt)
# - node_type: "compute"
# - domain: "general"
# - operations: ["transform", "parse", "validate"]
# - features: ["validation"]
```

### Example 3: Metrics Reducer Node

```python
prompt = """
Create a metrics aggregation Reducer node that:
- Collects performance metrics from multiple sources
- Groups by time windows (1min, 5min, 1hour)
- Calculates statistics (avg, p50, p95, p99)
- Maintains rolling state
- Publishes to monitoring dashboard
"""

requirements = await analyzer.analyze_prompt(prompt)
# - node_type: "reducer"
# - domain: "monitoring"
# - operations: ["aggregate", "collect", "group"]
# - features: ["metrics"]
```

### Example 4: Workflow Orchestrator Node

```python
prompt = """
Create a workflow Orchestrator node that:
- Coordinates multi-stage data processing pipeline
- Manages 5 sequential stages with parallel sub-tasks
- Handles stage failures with retry and rollback
- Tracks workflow state and progress
- Publishes events to Kafka for monitoring
"""

requirements = await analyzer.analyze_prompt(prompt)
# - node_type: "orchestrator"
# - domain: "general"
# - operations: ["orchestrate", "coordinate", "execute"]
# - features: ["retry_logic", "circuit_breaker"]
```

## Real-World Generated Nodes

### Example: PostgreSQL CRUD Effect (Oct 27, 2025)

**Production Node**: `postgres_crud_effect`

This node was successfully generated and validated in production, demonstrating the full capabilities of the code generation system.

**Generation Command**:
```bash
omninode-generate "Create PostgreSQL CRUD Effect node with connection pooling, retry logic, and circuit breaker"
```

**Generated Artifacts** (15 files):
```
generated_nodes/postgres_crud_effect_a3f7b2c1/
├── node.py                    # 3,521 bytes - Main Effect implementation
├── contract.yaml              # 1,095 bytes - ONEX v2.0 contract
├── __init__.py                # 500 bytes - Module exports
├── models/
│   ├── model_request.py       # 842 bytes - Request data model
│   ├── model_response.py      # 759 bytes - Response data model
│   └── model_config.py        # 931 bytes - Configuration model
├── tests/
│   ├── test_unit.py           # 1,456 bytes - Unit tests with mocks
│   ├── test_integration.py    # 1,678 bytes - Integration tests
│   ├── test_contract.py       # 1,234 bytes - ONEX compliance tests
│   ├── test_performance.py    # 1,123 bytes - Performance benchmarks
│   └── conftest.py            # 987 bytes - Pytest fixtures
└── README.md                  # 1,240 bytes - Documentation
```

**Quality Metrics**:
- **Overall Quality**: 100.0%
- **ONEX Compliance**: 100.0%
- **Type Safety**: 100.0%
- **Code Quality**: 100.0%
- **Documentation**: 100.0%
- **Test Coverage**: 100.0%

**Total Pipeline Time**: ~35.2s → ~33.6s (with Stage 9)

**Generation Stages**:
- Stages 1-8: Code generation (30.8s)
- Stage 9: Test generation (2.8s)

**Key Features Implemented**:
- ✅ Connection pooling (10-50 connections)
- ✅ Automatic retry logic with exponential backoff
- ✅ Circuit breaker pattern for resilience
- ✅ Full CRUD operations (Create, Read, Update, Delete, List)
- ✅ Structured logging with correlation tracking
- ✅ Metrics collection for monitoring
- ✅ Async/await support throughout
- ✅ Proper error handling with ModelOnexError
- ✅ Type hints on all methods
- ✅ Comprehensive docstrings
- ✅ Test scaffolding with mocks

**node.py Highlights**:
```python
class NodePostgresCrudEffect(NodeEffect):
    """
    PostgreSQL CRUD Effect Node.

    Provides full CRUD operations with connection pooling, retry logic,
    and circuit breaker pattern for resilience.

    ONEX v2.0 Compliance:
    - Extends NodeEffect from omnibase_core
    - Implements execute_effect method signature
    - Uses ModelOnexError for error handling
    - Structured logging with correlation tracking
    """

    async def execute_effect(
        self,
        contract: ModelContractEffect
    ) -> ModelPostgresCrudResponse:
        """
        Execute PostgreSQL CRUD operation.

        Args:
            contract: Effect contract with operation parameters

        Returns:
            ModelPostgresCrudResponse with operation results

        Raises:
            ModelOnexError: On database errors or validation failures
        """
        # Implementation generated with connection pooling,
        # retry logic, and circuit breaker...
```

**contract.yaml Highlights**:
```yaml
name: postgres_crud_effect
version: 1.0.0
node_type: effect
base_class: NodeEffect

description: |
  PostgreSQL CRUD Effect node with connection pooling and resilience patterns.

subcontracts:
  - type: circuit_breaker
    enabled: true
    threshold: 5
    timeout_seconds: 60

  - type: retry
    enabled: true
    max_attempts: 3
    backoff_multiplier: 2

performance:
  target_latency_ms: 100
  max_connections: 50
  min_connections: 10
```

**Usage Example**:
```python
from omnibase_core import ModelContainer
from generated_nodes.postgres_crud_effect import NodePostgresCrudEffect

# Initialize node
container = ModelContainer(config={
    "postgres_host": "localhost",
    "postgres_port": 5432,
    "postgres_database": "mydb",
    "postgres_user": "<your-username>",
    "postgres_password": "<your-password>",
})

node = NodePostgresCrudEffect(container)

# Execute CRUD operation
contract = ModelContractEffect(
    operation="create",
    data={"name": "John Doe", "email": "john@example.com"},
)

response = await node.execute_effect(contract)
print(f"Created record with ID: {response.record_id}")
```

**Lessons Learned**:
1. **Generation Time**: Actual time (33.6s) was 13% faster than target (38.5s) with test generation
2. **Quality Scores**: 100% across all metrics demonstrates template quality
3. **ONEX Compliance**: Zero manual fixes required for compliance
4. **Immediate Usability**: Generated code is production-ready with TODO markers for business logic
5. **Event Tracking**: Full observability via 13 Kafka topics enabled debugging
6. **Test Generation**: Stage 9 adds comprehensive test scaffolding with minimal overhead (2.8s)

## Best Practices

### 1. Writing Good Prompts

**✅ Good Prompts**:
- Clear node purpose: "Create a PostgreSQL CRUD Effect node..."
- Specific operations: "...with create, read, update, delete operations"
- Explicit features: "...using connection pooling and retry logic"
- Performance requirements: "...supporting 1000+ QPS"
- Domain context: "...for user management system"

**❌ Poor Prompts**:
- Vague: "Create a node"
- Ambiguous: "Create something for data"
- Missing context: "Handle requests"

### 2. Reviewing Generated Code

Always review generated code for:
- [ ] Business logic correctness (templates are stubs)
- [ ] Security considerations (authentication, validation)
- [ ] Performance optimizations (caching, pooling)
- [ ] Error handling completeness
- [ ] Test coverage adequacy

### 3. Extending Templates

To create custom templates:

1. Create template directory:
```bash
mkdir -p templates/node_templates/effect_custom
```

2. Create Jinja2 templates:
```bash
templates/node_templates/effect_custom/
├── node.py.j2
├── contract.yaml.j2
└── models/
    └── model_request.py.j2
```

3. Use TemplateEngine with custom templates:
```python
engine = TemplateEngine(
    templates_directory=Path("./templates"),
    enable_inline_templates=False,  # Force custom templates
)
```

### 4. Quality Thresholds

Recommended thresholds by environment:

| Environment | Min Quality | ONEX Compliance | Type Safety |
|-------------|-------------|-----------------|-------------|
| Development | 0.6 | 0.5 | 0.5 |
| Staging | 0.7 | 0.7 | 0.7 |
| Production | 0.8 | 0.9 | 0.8 |

## Troubleshooting

### Issue: Generation Hangs or Times Out

**Symptoms**: CLI shows "Generating..." but no progress updates, or timeout after 5 minutes

**Root Causes**:
1. Kafka connection failure (cannot reach 192.168.86.200:9092)
2. NodeCodegenOrchestrator service not running
3. Missing hostname resolution for Kafka

**Solutions**:

```bash
# 1. Verify Kafka hostname resolution
ping omninode-bridge-redpanda
# If fails: echo "192.168.86.200 omninode-bridge-redpanda" | sudo tee -a /etc/hosts

# 2. Test Kafka connectivity
telnet 192.168.86.200 9092
# Should connect successfully

# 3. Verify Orchestrator health
curl http://192.168.86.200:8060/health
# Expected: {"status": "healthy"}

# 4. Check Kafka topics exist
docker exec -it omninode-bridge-redpanda rpk topic list | grep codegen
# Expected: 13 codegen topics

# 5. Increase timeout if processing is slow
omninode-generate "your prompt" --timeout 600  # 10 minutes
```

### Issue: Event Bus Connection Errors

**Symptoms**: `KafkaConnectionError`, `BrokerNotAvailableError`

**Root Causes**:
1. Incorrect Kafka bootstrap servers
2. Firewall blocking port 9092
3. Kafka service not running

**Solutions**:

```bash
# 1. Verify Kafka is running
docker ps | grep redpanda
# Expected: omninode-bridge-redpanda container running

# 2. Check firewall rules (if applicable)
sudo ufw status | grep 9092
# If blocked: sudo ufw allow 9092/tcp

# 3. Test with explicit Kafka servers
omninode-generate "prompt" --kafka-servers 192.168.86.200:9092

# 4. Check Kafka logs for errors
docker logs omninode-bridge-redpanda --tail 50
```

### Issue: No Progress Updates During Generation

**Symptoms**: CLI shows initial message but no stage updates

**Root Causes**:
1. Stage completion events not being published
2. Consumer not subscribed to correct topics
3. Correlation ID mismatch

**Debugging Steps**:

```bash
# 1. Monitor Kafka topics in real-time (separate terminal)
docker exec -it omninode-bridge-redpanda rpk topic consume omninode_codegen_status_session_v1

# 2. Check if events are being published
docker exec -it omninode-bridge-redpanda rpk topic list --detailed | grep codegen

# 3. Verify correlation ID in logs
docker logs omninode-bridge-orchestrator --tail 100 | grep "correlation_id"

# 4. Enable debug logging
export LOG_LEVEL=DEBUG
omninode-generate "prompt"
```

### Issue: Low ONEX Compliance Score

**Symptoms**: `onex_compliance_score < 0.7`

**Solutions**:
1. Check node naming: Must follow `Node<Name><Type>` pattern
2. Verify base class: Must extend correct ONEX base class
3. Check method signature: Must have correct `execute_*` method
4. Review error handling: Must use `ModelOnexError`

### Issue: Low Classification Confidence

**Symptoms**: `classification.confidence < 0.7`

**Solutions**:
1. Add more specific keywords to prompt
2. Explicitly specify node type hint: `--node-type effect`
3. Add domain context: "Create **database** Effect node..."
4. List specific operations: "with **CRUD operations**..."

### Issue: Intelligence Gathering Fails

**Symptoms**: Stage 2 fails with "Intelligence service unavailable"

**Solutions**:

```bash
# 1. Disable intelligence gathering (faster, no RAG)
omninode-generate "prompt" --disable-intelligence

# 2. Verify intelligence service health
curl http://192.168.86.200:8058/health  # OnexTree service

# 3. Check circuit breaker status (if enabled)
# Circuit breaker automatically disables intelligence after 5 consecutive failures

# 4. Review intelligence service logs
docker logs omninode-bridge-onextree --tail 50
```

**Note**: Intelligence gathering is **optional**. The system gracefully degrades and continues generation without intelligence data.

### Issue: Generation Fails at Validation Stage

**Symptoms**: Stage 6 (Validation) fails with quality score errors

**Root Causes**:
1. Generated code has ONEX compliance violations
2. Type hints missing or incorrect
3. Docstrings incomplete

**Solutions**:

```bash
# 1. Review validation errors in output
# CLI shows detailed error messages with line numbers

# 2. Lower quality threshold (for development)
# Edit generated node and re-validate manually:
from omninode_bridge.codegen import QualityValidator
validator = QualityValidator(min_quality_threshold=0.6)  # Lower threshold

# 3. Enable AI Quorum for better validation
omninode-generate "prompt" --enable-quorum

# 4. Manually inspect generated code
ls -la generated_nodes/*/node.py
cat generated_nodes/*/node.py | grep "class Node"
```

### Issue: Files Not Written to Disk

**Symptoms**: Generation completes but no files in output directory

**Root Causes**:
1. Permission denied on output directory
2. Output directory path traversal blocked
3. Disk space full

**Solutions**:

```bash
# 1. Check directory permissions
ls -ld ./generated_nodes
# Fix: chmod 755 ./generated_nodes

# 2. Verify disk space
df -h .
# Ensure sufficient space (at least 10MB per node)

# 3. Use absolute path for output directory
omninode-generate "prompt" --output-dir /full/path/to/output

# 4. Check file system errors
dmesg | tail -20
```

### Debugging Tips

**Enable Verbose Logging**:
```bash
export LOG_LEVEL=DEBUG
export OMNINODE_DEBUG=1
omninode-generate "prompt"
```

**Monitor All Kafka Topics**:
```bash
# In separate terminal, monitor all events
watch -n 1 'docker exec omninode-bridge-redpanda rpk topic list --detailed | grep codegen'
```

**Check Service Health**:
```bash
# Verify all services are healthy
for port in 8001 8057 8058 8060 8061; do
    echo "Checking port $port..."
    curl -s http://192.168.86.200:$port/health | jq .
done
```

**Replay Failed Generation**:
```bash
# If generation fails, check DLQ topics for failed events
docker exec -it omninode-bridge-redpanda rpk topic consume omninode_codegen_request_analyze_v1_dlq --offset start
```

## Phase 3: Intelligent Code Generation ✅ (Complete)

**Status**: ✅ Complete (November 2025)
**Version**: 3.0.0

Phase 3 adds intelligent, pattern-driven code generation with LLM enhancement capabilities:

### Implemented Features

#### 1. Template Variant Selection

**Component**: `TemplateSelector`
**Purpose**: Intelligent template selection based on requirements analysis

**Features**:
- Analyzes requirements to select optimal template variant
- Confidence scoring (0.0-1.0) for selection decisions
- Pattern recommendations for selected variant
- Performance target: <5ms per selection, >95% accuracy

**Supported Variants**:
- **Standard**: Basic ONEX node structure (default)
- **Database-Heavy**: Optimized for database operations
- **API-Heavy**: Optimized for external API integration
- **Kafka-Heavy**: Optimized for event streaming
- **ML-Inference**: Optimized for machine learning operations
- **Analytics**: Optimized for data analytics

**Usage Example**:
```python
from omninode_bridge.codegen.template_selector import TemplateSelector

selector = TemplateSelector()
result = selector.select_template(
    requirements=contract,
    node_type="effect",
    target_environment="production"
)

print(f"Selected: {result.variant.value}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Rationale: {result.rationale}")
```

#### 2. Production Pattern Library

**Component**: `ProductionPatternLibrary`
**Purpose**: Pattern matching and code generation from production patterns

**Available Patterns**:
- **Lifecycle**: Startup/shutdown patterns with proper resource management
- **Health Checks**: Multi-level health check implementation
- **Metrics**: Prometheus-compatible metrics collection
- **Event Publishing**: Kafka event publishing with OnexEnvelopeV1 format
- **Consul Integration**: Service discovery and configuration

**Features**:
- Pattern discovery and categorization
- Similarity-based pattern matching (>90% relevance)
- Code generation from matched patterns
- Usage tracking for adaptive learning
- Performance target: <10ms per pattern search

**Usage Example**:
```python
from omninode_bridge.codegen.pattern_library import ProductionPatternLibrary

library = ProductionPatternLibrary()

# Find matching patterns
matches = library.find_patterns(
    requirements=contract,
    node_type="effect",
    max_results=5
)

for match in matches:
    print(f"{match.pattern_info.name}: {match.relevance_score:.2f}")
    code = library.get_pattern_code(match.pattern_info.name)
```

#### 3. Intelligent Mixin Recommendation

**Component**: `MixinRecommender`
**Purpose**: Smart mixin selection with conflict detection

**Features**:
- Requirements analysis (8 category scores: database, API, Kafka, security, observability, resilience, caching, performance)
- Intelligent mixin scoring based on requirements
- Conflict detection and resolution
- Priority-based recommendation ordering
- Performance target: <20ms per recommendation set, >90% useful recommendations

**Available Mixins**:
- Database connection pooling
- API client with retry logic
- Kafka producer/consumer
- Metrics collection
- Health check endpoints
- Consul service registration
- Caching layer
- Circuit breaker pattern

**Usage Example**:
```python
from omninode_bridge.codegen.mixins.mixin_recommender import MixinRecommender

recommender = MixinRecommender()

recommendations = recommender.recommend_mixins(
    requirements=contract,
    node_type="effect",
    max_recommendations=5,
    min_score_threshold=0.6
)

for rec in recommendations:
    print(f"{rec.mixin_name}: {rec.relevance_score:.2f}")
    print(f"  Rationale: {rec.rationale}")
```

#### 4. Enhanced Context Builder

**Component**: `EnhancedContextBuilder`
**Purpose**: Build comprehensive LLM context for code generation

**Features**:
- Aggregates data from all Phase 3 components
- Token count estimation and management (target: <8K tokens)
- ONEX best practices integration
- Similar code examples
- Performance target: <50ms per context build

**Context Includes**:
- Operation name and description
- Selected template variant
- Matched production patterns with code examples
- Recommended mixins with usage instructions
- Requirements and constraints
- Best practices and examples

**Usage Example**:
```python
from omninode_bridge.codegen.context_builder import EnhancedContextBuilder

builder = EnhancedContextBuilder(max_tokens=8000)

context = builder.build_context(
    operation_name="fetch_user_data",
    operation_description="Fetch user data from PostgreSQL with caching",
    node_type="effect",
    template_selection=template_result,
    pattern_matches=pattern_results,
    mixin_recommendations=mixin_results,
    requirements={"database": "postgresql", "cache": True}
)

print(f"Context: {context.estimated_tokens} tokens")
```

#### 5. Subcontract Processing

**Component**: `TemplateEngine.process_subcontracts()`
**Purpose**: Generate ONEX v2.0 subcontract YAML files

**Supported Subcontract Types**:
- **API**: External API integration specifications
- **Compute**: Computational operation specifications
- **Database**: Database operation specifications
- **Event**: Event publishing/consuming specifications
- **State**: State management specifications
- **Workflow**: Workflow coordination specifications

**Usage Example**:
```python
from omninode_bridge.codegen.template_engine import TemplateEngine

engine = TemplateEngine()

subcontracts = engine.process_subcontracts(
    contract={
        "subcontracts": {
            "database": {"type": "postgresql", "operations": ["read", "write"]},
            "events": {"topics": ["user.created", "user.updated"]}
        }
    },
    node_type="effect"
)

# subcontracts = {"database_subcontract.yaml": "...", "event_subcontract.yaml": "..."}
```

### Phase 3 Performance Metrics

| Component | Target | Achieved |
|-----------|--------|----------|
| Template Selection | <5ms, >95% accuracy | ✅ |
| Pattern Search | <10ms, >90% relevance | ✅ |
| Mixin Recommendation | <20ms, >90% useful | ✅ |
| Context Building | <50ms, <8K tokens | ✅ |

### Phase 3 Integration with Pipeline

Phase 3 components integrate seamlessly with the existing 9-stage pipeline:

**Stage 1: Prompt Parsing**
- Uses enhanced requirement extraction
- Feeds into Phase 3 components

**Stage 2: Intelligence Gathering**
- Pattern library queries for similar nodes
- Historical usage data for adaptive learning

**Stage 3: Contract Building**
- Subcontract processor generates YAML files
- Template selector chooses optimal variant

**Stage 4: Code Generation**
- Enhanced context builder aggregates all data
- Template engine uses Phase 3 context for generation
- Mixin injector applies recommended mixins
- Pattern library injects production patterns

**Stage 6: Validation**
- Validates Phase 3 enhancements (patterns, mixins, subcontracts)

### End-to-End Phase 3 Example

```python
from omninode_bridge.codegen.template_selector import TemplateSelector
from omninode_bridge.codegen.pattern_library import ProductionPatternLibrary
from omninode_bridge.codegen.mixins.mixin_recommender import MixinRecommender
from omninode_bridge.codegen.context_builder import EnhancedContextBuilder
from omninode_bridge.codegen.template_engine import TemplateEngine

# Initialize Phase 3 components
template_selector = TemplateSelector()
pattern_library = ProductionPatternLibrary()
mixin_recommender = MixinRecommender()
context_builder = EnhancedContextBuilder()
template_engine = TemplateEngine()

# Step 1: Select template variant (5ms)
template_result = template_selector.select_template(
    requirements=contract,
    node_type="effect"
)
print(f"✅ Template: {template_result.variant.value}")

# Step 2: Find matching patterns (10ms)
pattern_results = pattern_library.find_patterns(
    requirements=contract,
    node_type="effect"
)
print(f"✅ Patterns: {len(pattern_results)} matches")

# Step 3: Recommend mixins (20ms)
mixin_results = mixin_recommender.recommend_mixins(
    requirements=contract,
    node_type="effect"
)
print(f"✅ Mixins: {len(mixin_results)} recommendations")

# Step 4: Build LLM context (50ms)
llm_context = context_builder.build_context(
    operation_name="fetch_user_data",
    operation_description="Fetch user with caching",
    node_type="effect",
    template_selection=template_result,
    pattern_matches=pattern_results,
    mixin_recommendations=mixin_results,
    requirements=contract
)
print(f"✅ Context: {llm_context.estimated_tokens} tokens")

# Step 5: Generate code with Phase 3 enhancements
artifacts = template_engine.generate(
    contract=contract,
    context=llm_context
)
print(f"✅ Generated: {len(artifacts)} files")

# Total Phase 3 overhead: ~85ms (5+10+20+50)
```

### Documentation

- **[API Reference](../api/API_REFERENCE.md#phase-3-intelligent-code-generation-api)** - Complete Phase 3 API documentation
- **[Phase 3 Task Breakdown](../planning/PHASE_3_TASK_BREAKDOWN.md)** - Implementation planning
- **[Phase 3 Architecture Design](../planning/PHASE_3_ARCHITECTURE_DESIGN.md)** - Detailed architecture

---

## Future Enhancements

### Phase 4 (Future)

- [ ] Visual workflow designer
- [ ] AI-powered code review with GPT-4
- [ ] Automatic optimization suggestions
- [ ] Integration test generation
- [ ] Performance profiling integration
- [ ] Deployment manifest generation
- [ ] Multi-language support (TypeScript, Go, Rust)
- [ ] IDE plugin for VSCode/JetBrains

## References

**Core Documentation**:
- [ONEX Architecture Patterns](../../Archon/docs/ONEX_ARCHITECTURE_PATTERNS_COMPLETE.md) - Complete ONEX v2.0 patterns
- [Bridge Nodes Guide](./BRIDGE_NODES_GUIDE.md) - Bridge node implementation guide
- [LlamaIndex Workflows Guide](../LLAMAINDEX_WORKFLOWS_GUIDE.md) - Event-driven workflow integration
- [API Reference](../api/API_REFERENCE.md) - Complete API documentation

**Source Code**:
- [NodeCodegenOrchestrator](../../../src/omninode_bridge/nodes/codegen_orchestrator/v1_0_0/node.py) - 9-stage orchestrator implementation
- [CodeGenerationWorkflow](../../../src/omninode_bridge/nodes/codegen_orchestrator/v1_0_0/workflow.py) - LlamaIndex workflow
- [NodeTestGeneratorEffect](../../../src/omninode_bridge/nodes/test_generator_effect/v1_0_0/node.py) - Test generation Effect node
- [CLI Command](../../../src/omninode_bridge/cli/codegen/commands/generate.py) - `omninode-generate` command
- [Event Schemas](../../../src/omninode_bridge/events/codegen_schemas.py) - Kafka event definitions

**Templates & Examples**:
- [Test Templates](../../../src/omninode_bridge/codegen/templates/test_templates/) - Jinja2 test templates
- [Test Contract Examples](../../../src/omninode_bridge/codegen/templates/examples/) - Example test contracts

**Examples**:
- [Simple Example](../../../examples/codegen_simple_example.py) - Complete generation workflow demo
- [Generated Node](../../../generated_nodes/) - Real generated nodes (postgres_crud_effect, etc.)

**Related Guides**:
- [Remote Migration Guide](../deployment/REMOTE_MIGRATION_GUIDE.md) - Deploy to remote infrastructure
- [Database Migrations](../deployment/DATABASE_MIGRATIONS.md) - PostgreSQL setup for codegen persistence
- [Deployment System](../deployment/DEPLOYMENT_SYSTEM_MIGRATION.md) - Automated deployment workflows

**External Resources**:
- [LlamaIndex Official Docs](https://docs.llamaindex.ai/en/stable/understanding/workflows/) - LlamaIndex Workflows documentation
- [Redpanda Documentation](https://docs.redpanda.com/) - Kafka-compatible event streaming
- [Pydantic v2 Documentation](https://docs.pydantic.dev/latest/) - Data validation and models

---

**Document Information**:
- **Last Updated**: 2025-10-30
- **Authors**: OmniNode Team
- **Status**: ✅ Production Ready (Phase 2 Complete)
- **Version**: 2.0.0 (Stage 9 integration)
- **Repository**: omninode_bridge
- **Remote Infrastructure**: 192.168.86.200 (Redpanda, PostgreSQL, Consul)

**Quick Links**:
- 🚀 **Quick Start**: See [CLI Usage](#cli-usage-omninode-generate)
- 📚 **API Reference**: See [Component Details](#component-details)
- 🔧 **Troubleshooting**: See [Troubleshooting](#troubleshooting)
- 💡 **Examples**: See [Real-World Generated Nodes](#real-world-generated-nodes)
