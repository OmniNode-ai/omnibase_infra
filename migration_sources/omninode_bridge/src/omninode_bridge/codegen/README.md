# OmniNode Code Generation System

**Purpose**: Automated generation of ONEX v2.0 compliant node implementations from PRD requirements.

## Overview

The code generation system creates production-ready ONEX nodes with complete scaffolding including:
- Node implementations (Effect, Compute, Reducer, Orchestrator)
- ONEX v2.0 contract definitions
- Pydantic data models
- Comprehensive test suites
- Documentation

## Architecture

```
CodeGenerationService (Facade)
    ↓
Strategy Registry (Strategy Selection)
    ↓
BaseGenerationStrategy (Abstract Interface)
    ↓
[Jinja2Strategy | TemplateLoadingStrategy | HybridStrategy | AutoStrategy]
    ↓
Generated Artifacts
```

## Generated Files

The code generation system creates the following files:

```
nodes/<service_name>/v1_0_0/
├── node.py              # Main node implementation (REQUIRED)
├── contract.yaml        # ONEX v2.0 contract definition (REQUIRED)
├── __init__.py          # Module initialization (REQUIRED)
├── models/              # Data models directory
│   ├── __init__.py
│   ├── model_request.py
│   └── model_response.py
├── tests/               # Test suite
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_node.py
│   └── test_integration.py (for Effect nodes)
└── README.md           # Node documentation
```

## ⚠️ IMPORTANT: main_standalone.py is NOT Generated

**ONEX v2.0 Standard**: Nodes are invoked via orchestration contracts, NOT as standalone REST APIs.

### Why main_standalone.py is NOT Generated

1. **ONEX v2.0 Architecture**: Nodes are pure business logic components invoked through contracts
2. **Separation of Concerns**: REST API layer belongs in API gateway/orchestrator, not in nodes
3. **Scalability**: Contract-based invocation enables better orchestration and workflow management
4. **Consistency**: All nodes follow the same invocation pattern

### Legacy main_standalone.py Files

Some existing nodes have `main_standalone.py` files for **backward compatibility only**:
- These were hand-written, NOT generated
- They are marked as LEGACY in Dockerfiles with TODO comments
- New nodes should NOT include main_standalone.py

### If You Need REST API Access

**DO NOT add main_standalone.py to your node.**

Instead:
1. **Use NodeBridgeOrchestrator** - Provides REST API wrapper for workflow orchestration
2. **Use API Gateway** - Dedicated service for HTTP → contract translation
3. **Invoke via Contract** - Direct contract-based invocation from orchestrator

Example (correct approach):
```python
# In orchestrator or API gateway
from omninode_bridge.nodes.your_service.v1_0_0 import NodeYourServiceEffect
from omnibase_core.models.contracts.model_contract_effect import ModelContractEffect
from omnibase_core.models.core import ModelContainer

# Create contract from REST request
contract = ModelContractEffect(
    name="your_operation",
    version={"major": 1, "minor": 0, "patch": 0},
    description="Execute your operation",
    node_type=EnumNodeType.EFFECT,
    input_model="ModelYourServiceInput",
    output_model="ModelYourServiceOutput",
    correlation_id=uuid4(),
    execution_id=uuid4()
)

# Invoke node via contract (ONEX standard)
node = NodeYourServiceEffect(container)
result = await node.execute_effect(contract)
```

## Usage

### Basic Code Generation

```python
from omninode_bridge.codegen.service import CodeGenerationService
from omninode_bridge.codegen.prd_analyzer import ModelPRDRequirements

# Create service
service = CodeGenerationService()

# Define requirements
requirements = ModelPRDRequirements(
    node_type="effect",
    service_name="postgres_crud",
    domain="database",
    business_description="PostgreSQL CRUD operations",
    operations=["create", "read", "update", "delete"],
)

# Generate node
result = await service.generate_node(
    requirements=requirements,
    strategy="auto",
    enable_llm=True,
    validation_level="strict",
)

print(f"Generated: {result.artifacts.node_name}")
# Output: Generated: NodePostgresCrudEffect
```

### Available Strategies

1. **Jinja2Strategy** (Default) - Template-based generation with inline templates
2. **TemplateLoadStrategy** - LLM-enhanced generation with business logic
3. **HybridStrategy** - Combines templates with LLM enhancement
4. **AutoStrategy** - Automatically selects optimal strategy

## Validation

Generated code is validated against:
1. **ONEX v2.0 compliance** - Naming conventions, contract structure
2. **omnibase_core schemas** - Contract models, data models
3. **Test execution** - Optional test runs with failure analysis

## Files in This Directory

- `service.py` - Main code generation service (facade)
- `template_engine.py` - Jinja2 template rendering engine
- `prd_analyzer.py` - PRD requirements extraction
- `node_classifier.py` - Node type classification
- `contract_introspector.py` - Contract validation
- `test_executor.py` - Generated test execution
- `failure_analyzer.py` - Test failure analysis
- `strategies/` - Generation strategy implementations
- `business_logic/` - LLM-powered business logic generation
- `models/` - Pydantic models for codegen system
- `templates/` - Jinja2 templates

## Contributing

When modifying the code generation system:

1. **DO NOT** add main_standalone.py generation
2. **DO** ensure ONEX v2.0 compliance
3. **DO** validate generated contracts against omnibase_core
4. **DO** add tests for new strategies
5. **DO** update this README with changes

## References

- **[ONEX v2.0 Specification](../../docs/architecture/ONEX_V2_COMPLIANCE.md)** - Complete ONEX v2.0 standard
- **[Contract Reference](../../docs/guides/BRIDGE_NODES_GUIDE.md)** - Contract patterns and examples
- **[Template Examples](./templates/examples/)** - Template usage examples
- **[Strategy Guide](./strategies/README.md)** - Generation strategy documentation

---

**Last Updated**: 2025-11-03
**Maintainer**: OmniNode Code Generation Team
