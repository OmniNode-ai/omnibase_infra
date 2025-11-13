# Code Generation Service Usage Guide

**Version**: 1.0.0
**Last Updated**: 2025-11-01
**Status**: Production Ready

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Quick Start (5 Minutes)](#quick-start-5-minutes)
3. [Strategy Selection Guide](#strategy-selection-guide)
4. [API Reference](#api-reference)
5. [Code Examples](#code-examples)
6. [Advanced Features](#advanced-features)
7. [Performance Tuning](#performance-tuning)
8. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Installation

The code generation service is included in the `omninode_bridge` package:

```bash
# Install via Poetry (recommended)
poetry add omninode_bridge

# Or pip
pip install omninode_bridge
```

### Prerequisites

```bash
# Required
- Python 3.11+
- Poetry or pip

# Optional (for LLM-powered generation)
- Archon MCP service running (for intelligence)
- Valid API key for LLM provider
```

### Basic Imports

```python
from omninode_bridge.codegen import (
    CodeGenerationService,      # Main service
    ModelPRDRequirements,        # Requirements model
    EnumStrategyType,            # Strategy selection
    EnumValidationLevel,         # Validation levels
)
```

---

## Quick Start (5 Minutes)

### 1. Simple Node Generation

```python
import asyncio
from pathlib import Path
from omninode_bridge.codegen import (
    CodeGenerationService,
    ModelPRDRequirements,
)

async def generate_simple_node():
    """Generate a simple CRUD node in 5 minutes."""

    # Step 1: Initialize service
    service = CodeGenerationService()

    # Step 2: Define requirements
    requirements = ModelPRDRequirements(
        node_type="effect",
        service_name="user_crud",
        domain="database",
        business_description="User CRUD operations with PostgreSQL",
        operations=["create", "read", "update", "delete", "list"],
        features=["connection pooling", "automatic retry", "metrics"],
    )

    # Step 3: Generate code
    result = await service.generate_node(
        requirements=requirements,
        output_directory=Path("./generated/user_crud"),
        strategy="auto",  # Automatic strategy selection
        validation_level="standard",  # Standard validation
    )

    # Step 4: Check results
    print(f"✅ Generated: {result.artifacts.node_name}")
    print(f"⏱️  Time: {result.generation_time_ms:.0f}ms")
    print(f"🎯 Strategy: {result.strategy_used.value}")
    print(f"✨ Validation: {'PASSED' if result.validation_passed else 'FAILED'}")

    # Step 5: Access generated files
    all_files = result.artifacts.get_all_files()
    print(f"📦 Files: {len(all_files)}")
    for filename in sorted(all_files.keys()):
        print(f"   - {filename}")

    return result

# Run
if __name__ == "__main__":
    asyncio.run(generate_simple_node())
```

**Output**:
```
✅ Generated: NodeUserCrudEffect
⏱️  Time: 1245ms
🎯 Strategy: jinja2
✨ Validation: PASSED
📦 Files: 7
   - __init__.py
   - contract.yaml
   - models.py
   - node.py
   - tests/__init__.py
   - tests/test_node.py
   - tests/test_contract.py
```

### 2. From Natural Language Prompt

```python
from omninode_bridge.codegen import PRDAnalyzer, CodeGenerationService

async def generate_from_prompt():
    """Generate node from natural language prompt."""

    # Step 1: Analyze prompt
    analyzer = PRDAnalyzer()
    prompt = """
    Create a Redis caching node with the following features:
    - Connection pooling with 10-50 connections
    - TTL support for cache expiration
    - Automatic serialization/deserialization
    - Circuit breaker pattern
    - Metrics collection
    """

    requirements = await analyzer.analyze_prompt(prompt)

    # Step 2: Generate with service
    service = CodeGenerationService()
    result = await service.generate_node(
        requirements=requirements,
        strategy="auto",
    )

    return result

# Run
asyncio.run(generate_from_prompt())
```

---

## Strategy Selection Guide

### Overview

The service supports multiple generation strategies:

| Strategy | Description | Best For | Speed | Quality |
|----------|-------------|----------|-------|---------|
| **auto** | Automatic selection | General use | - | - |
| **jinja2** | Template-based | Simple CRUD, high-speed | ⚡⚡⚡ | ⭐⭐⭐ |
| **template_loading** | LLM-powered | Complex logic, high-quality | ⚡ | ⭐⭐⭐⭐⭐ |
| **hybrid** | Best of both | Critical features | ⚡⚡ | ⭐⭐⭐⭐ |

### When to Use Each Strategy

#### Use **auto** (Automatic) When:
✅ You're not sure which strategy to use
✅ Requirements are clear and well-defined
✅ You want the service to optimize for you

```python
result = await service.generate_node(
    requirements=requirements,
    strategy="auto",  # Service decides
    enable_llm=True,  # Allow LLM if beneficial
)
```

#### Use **jinja2** (Template-Based) When:
✅ Simple CRUD operations
✅ Speed is critical (<500ms generation time)
✅ Well-defined patterns (Effect, Compute, Reducer)
✅ No complex business logic

```python
result = await service.generate_node(
    requirements=requirements,
    strategy="jinja2",  # Fast template-based
)
```

**Example Use Cases**:
- Database CRUD operations
- API client wrappers
- Simple data transformations
- Cache integrations

#### Use **template_loading** (LLM-Powered) When:
✅ Complex business logic required
✅ High-quality code is critical
✅ Novel or unique requirements
✅ Willing to wait for LLM processing (2-5s)

```python
result = await service.generate_node(
    requirements=requirements,
    strategy="template_loading",  # LLM-powered
    enable_llm=True,
)
```

**Example Use Cases**:
- Custom algorithms
- Complex state machines
- Integration with multiple services
- Business rule engines

#### Use **hybrid** When:
✅ Production-critical features
✅ Need both speed and quality
✅ Template foundation + LLM enhancement

```python
result = await service.generate_node(
    requirements=requirements,
    strategy="hybrid",  # Best of both worlds
    enable_llm=True,
)
```

**Example Use Cases**:
- Payment processing
- Authentication/authorization
- Compliance-critical features
- High-visibility services

---

## API Reference

### CodeGenerationService

#### Constructor

```python
def __init__(
    self,
    templates_directory: Optional[Path] = None,
    archon_mcp_url: Optional[str] = None,
    enable_intelligence: bool = True,
)
```

**Parameters**:
- `templates_directory`: Path to Jinja2 templates (optional, uses built-in if not provided)
- `archon_mcp_url`: Archon MCP endpoint URL (e.g., `http://archon:8060`)
- `enable_intelligence`: Enable RAG intelligence gathering from Archon MCP

**Example**:
```python
# Default configuration
service = CodeGenerationService()

# Custom configuration
service = CodeGenerationService(
    templates_directory=Path("./custom_templates"),
    archon_mcp_url="http://localhost:8060",
    enable_intelligence=True,
)
```

#### generate_node()

```python
async def generate_node(
    self,
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

**Parameters**:
- `requirements`: Extracted PRD requirements (required)
- `classification`: Node type classification (optional, will classify if not provided)
- `output_directory`: Target directory (optional, uses temp dir if not provided)
- `strategy`: Generation strategy (`"auto"`, `"jinja2"`, `"template_loading"`, `"hybrid"`)
- `enable_llm`: Enable LLM-powered business logic generation
- `validation_level`: Validation strictness (`"none"`, `"basic"`, `"standard"`, `"strict"`)
- `run_tests`: Execute generated tests after code generation
- `strict_mode`: Raise exception if tests fail (vs. attach to artifacts)
- `correlation_id`: Correlation ID for tracing (optional)

**Returns**: `ModelGenerationResult`

**Raises**:
- `ValueError`: If requirements invalid or validation fails
- `RuntimeError`: If no suitable strategy found or generation fails

**Example**:
```python
result = await service.generate_node(
    requirements=requirements,
    output_directory=Path("./output"),
    strategy="auto",
    enable_llm=True,
    validation_level="strict",
    run_tests=True,
    strict_mode=False,
)
```

#### list_strategies()

```python
def list_strategies(self) -> list[dict[str, str]]
```

**Returns**: List of registered strategies with metadata

**Example**:
```python
strategies = service.list_strategies()
for strategy in strategies:
    print(f"{strategy['name']}: {strategy['type']} (default: {strategy['is_default']})")
```

#### get_strategy_info()

```python
def get_strategy_info(self, strategy_type: str) -> dict
```

**Parameters**:
- `strategy_type`: Strategy type to query

**Returns**: Detailed strategy information

**Example**:
```python
info = service.get_strategy_info("jinja2")
print(f"Supported node types: {info['supported_node_types']}")
```

### ModelGenerationResult

**Attributes**:
```python
class ModelGenerationResult:
    artifacts: ModelGeneratedArtifacts  # Generated code artifacts
    strategy_used: EnumStrategyType     # Strategy that generated the code
    generation_time_ms: float           # Generation time in milliseconds
    validation_passed: bool             # Whether validation passed
    validation_errors: list[str]        # Validation errors (if any)
    llm_used: bool                      # Whether LLM was used
    intelligence_sources: list[str]     # Intelligence sources used
    correlation_id: UUID                # Correlation ID for tracing
    generated_at: datetime              # Generation timestamp
```

**Example**:
```python
result = await service.generate_node(...)

# Access generated artifacts
artifacts = result.artifacts
node_code = artifacts.node_file
contract = artifacts.contract_file

# Check metadata
print(f"Strategy: {result.strategy_used.value}")
print(f"Time: {result.generation_time_ms}ms")
print(f"Validation: {result.validation_passed}")
print(f"LLM used: {result.llm_used}")
```

### ModelPRDRequirements

**Attributes**:
```python
class ModelPRDRequirements:
    node_type: str                      # "effect", "compute", "reducer", "orchestrator"
    service_name: str                   # Service name (e.g., "postgres_crud")
    domain: str                         # Domain (e.g., "database", "api", "cache")
    business_description: str           # Business description
    operations: list[str]               # Operations (e.g., ["create", "read"])
    features: list[str]                 # Features (e.g., ["retry", "metrics"])
    extraction_confidence: float        # Extraction confidence (0.0-1.0)
```

**Example**:
```python
requirements = ModelPRDRequirements(
    node_type="effect",
    service_name="postgres_crud",
    domain="database",
    business_description="PostgreSQL CRUD operations",
    operations=["create", "read", "update", "delete"],
    features=["connection pooling", "retry logic", "metrics"],
    extraction_confidence=0.95,
)
```

---

## Code Examples

### Example 1: Generate Database Effect Node

```python
from omninode_bridge.codegen import CodeGenerationService, ModelPRDRequirements
from pathlib import Path

async def generate_postgres_node():
    """Generate PostgreSQL CRUD Effect node."""

    service = CodeGenerationService()

    requirements = ModelPRDRequirements(
        node_type="effect",
        service_name="postgres_crud",
        domain="database",
        business_description="PostgreSQL CRUD operations with connection pooling",
        operations=["create", "read", "update", "delete", "list"],
        features=[
            "connection pooling (10-50 connections)",
            "automatic retry with exponential backoff",
            "circuit breaker pattern",
            "structured logging",
            "metrics collection",
        ],
    )

    result = await service.generate_node(
        requirements=requirements,
        output_directory=Path("./generated/postgres_crud_effect"),
        strategy="jinja2",  # Fast template-based
        validation_level="strict",
    )

    # Write files to disk
    output_dir = Path("./generated/postgres_crud_effect")
    for filename, content in result.artifacts.get_all_files().items():
        file_path = output_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)

    print(f"✅ Generated {result.artifacts.node_name}")
    print(f"📁 Output: {output_dir}")

    return result
```

### Example 2: Generate API Client Effect Node

```python
async def generate_api_client():
    """Generate HTTP API client Effect node."""

    service = CodeGenerationService()

    requirements = ModelPRDRequirements(
        node_type="effect",
        service_name="github_api_client",
        domain="api",
        business_description="GitHub API client with rate limiting and retry",
        operations=["get_user", "get_repos", "create_issue", "update_issue"],
        features=[
            "rate limiting (5000 requests/hour)",
            "automatic retry with backoff",
            "response caching",
            "authentication via token",
            "pagination support",
        ],
    )

    result = await service.generate_node(
        requirements=requirements,
        strategy="auto",
        enable_llm=True,  # Use LLM for API client logic
        validation_level="standard",
    )

    return result
```

### Example 3: Generate Compute Node with Complex Logic

```python
async def generate_ml_compute():
    """Generate ML prediction Compute node with LLM."""

    service = CodeGenerationService()

    requirements = ModelPRDRequirements(
        node_type="compute",
        service_name="ml_prediction",
        domain="machine_learning",
        business_description="ML model prediction with preprocessing and postprocessing",
        operations=["preprocess", "predict", "postprocess", "batch_predict"],
        features=[
            "input validation and normalization",
            "model loading and caching",
            "batch processing support",
            "confidence scoring",
            "error handling for invalid inputs",
        ],
    )

    result = await service.generate_node(
        requirements=requirements,
        strategy="template_loading",  # Use LLM for complex logic
        enable_llm=True,
        validation_level="strict",
    )

    return result
```

### Example 4: Generate Reducer Node

```python
async def generate_metrics_reducer():
    """Generate metrics aggregation Reducer node."""

    service = CodeGenerationService()

    requirements = ModelPRDRequirements(
        node_type="reducer",
        service_name="metrics_aggregator",
        domain="observability",
        business_description="Real-time metrics aggregation with windowing",
        operations=["aggregate", "window", "reduce", "emit"],
        features=[
            "time-based windowing (1min, 5min, 1hour)",
            "count, sum, avg, min, max aggregations",
            "late data handling",
            "state persistence to Redis",
            "periodic emission",
        ],
    )

    result = await service.generate_node(
        requirements=requirements,
        strategy="auto",
        validation_level="standard",
    )

    return result
```

### Example 5: Generate Orchestrator Node

```python
async def generate_workflow_orchestrator():
    """Generate workflow orchestration Orchestrator node."""

    service = CodeGenerationService()

    requirements = ModelPRDRequirements(
        node_type="orchestrator",
        service_name="data_pipeline",
        domain="workflow",
        business_description="Data pipeline orchestration with error handling",
        operations=["start", "execute_step", "handle_error", "complete"],
        features=[
            "multi-step workflow coordination",
            "parallel step execution",
            "error recovery strategies",
            "workflow state persistence",
            "event publishing to Kafka",
        ],
    )

    result = await service.generate_node(
        requirements=requirements,
        strategy="template_loading",  # Complex orchestration logic
        enable_llm=True,
        validation_level="strict",
    )

    return result
```

### Example 6: Batch Generation

```python
async def generate_multiple_nodes():
    """Generate multiple related nodes in parallel."""

    service = CodeGenerationService()

    # Define requirements for multiple nodes
    node_specs = [
        {
            "node_type": "effect",
            "service_name": "user_crud",
            "domain": "database",
            "description": "User CRUD operations",
        },
        {
            "node_type": "effect",
            "service_name": "auth_client",
            "domain": "api",
            "description": "Authentication API client",
        },
        {
            "node_type": "compute",
            "service_name": "validation",
            "domain": "business_logic",
            "description": "Input validation logic",
        },
    ]

    # Generate all nodes in parallel
    tasks = []
    for spec in node_specs:
        requirements = ModelPRDRequirements(
            node_type=spec["node_type"],
            service_name=spec["service_name"],
            domain=spec["domain"],
            business_description=spec["description"],
            operations=["execute"],  # Simple for demo
        )

        task = service.generate_node(
            requirements=requirements,
            strategy="auto",
        )
        tasks.append(task)

    # Wait for all generations to complete
    results = await asyncio.gather(*tasks)

    print(f"✅ Generated {len(results)} nodes")
    for result in results:
        print(f"   - {result.artifacts.node_name} ({result.generation_time_ms:.0f}ms)")

    return results
```

---

## Advanced Features

### 1. Custom Strategy Selection

```python
# List available strategies
service = CodeGenerationService()
strategies = service.list_strategies()

print("Available strategies:")
for strategy in strategies:
    print(f"  {strategy['name']} ({strategy['type']})")
    if strategy['is_default']:
        print("    ↳ DEFAULT")

# Get detailed strategy info
jinja2_info = service.get_strategy_info("jinja2")
print(f"\nJinja2 Strategy:")
print(f"  Supported node types: {jinja2_info['supported_node_types']}")
print(f"  Requires LLM: {jinja2_info['requires_llm']}")
```

### 2. Validation Levels

```python
# No validation (fastest)
result = await service.generate_node(
    requirements=requirements,
    validation_level="none",
)

# Basic validation (syntax only)
result = await service.generate_node(
    requirements=requirements,
    validation_level="basic",
)

# Standard validation (recommended)
result = await service.generate_node(
    requirements=requirements,
    validation_level="standard",  # Type checking + structure
)

# Strict validation (comprehensive)
result = await service.generate_node(
    requirements=requirements,
    validation_level="strict",  # All checks + quality metrics
)
```

### 3. Test Execution

```python
# Generate code and run tests
result = await service.generate_node(
    requirements=requirements,
    run_tests=True,  # Execute generated tests
    strict_mode=False,  # Don't raise exception on test failure
)

# Check test results
if result.validation_passed:
    print("✅ All tests passed")
else:
    print("❌ Tests failed:")
    for error in result.validation_errors:
        print(f"   - {error}")

# Strict mode (raise exception on failure)
try:
    result = await service.generate_node(
        requirements=requirements,
        run_tests=True,
        strict_mode=True,  # Raise exception if tests fail
    )
except ValueError as e:
    print(f"Generation failed: {e}")
```

### 4. Intelligence Integration

```python
# Enable Archon MCP intelligence
service = CodeGenerationService(
    archon_mcp_url="http://archon:8060",
    enable_intelligence=True,
)

result = await service.generate_node(
    requirements=requirements,
    enable_llm=True,
)

# Check intelligence sources used
print("Intelligence sources:")
for source in result.intelligence_sources:
    print(f"   - {source}")
```

### 5. Correlation and Tracing

```python
from uuid import uuid4

# Use correlation ID for tracing
correlation_id = uuid4()

result = await service.generate_node(
    requirements=requirements,
    correlation_id=correlation_id,
)

print(f"Correlation ID: {result.correlation_id}")
print(f"Generated at: {result.generated_at}")

# Use for distributed tracing
logger.info(
    "Code generation complete",
    extra={
        "correlation_id": str(result.correlation_id),
        "node_name": result.artifacts.node_name,
        "strategy": result.strategy_used.value,
    },
)
```

---

## Performance Tuning

### 1. Fast Generation (< 500ms)

```python
# Optimize for speed
result = await service.generate_node(
    requirements=requirements,
    strategy="jinja2",  # Fastest strategy
    validation_level="basic",  # Minimal validation
    run_tests=False,  # Skip test execution
    enable_llm=False,  # Disable LLM
)

print(f"Generated in {result.generation_time_ms:.0f}ms")
```

### 2. High Quality Generation

```python
# Optimize for quality
result = await service.generate_node(
    requirements=requirements,
    strategy="template_loading",  # LLM-powered
    validation_level="strict",  # Comprehensive validation
    run_tests=True,  # Execute tests
    enable_llm=True,  # Use LLM
)

print(f"Quality score: {result.validation_passed}")
```

### 3. Balanced Configuration

```python
# Balanced speed and quality
result = await service.generate_node(
    requirements=requirements,
    strategy="auto",  # Let service decide
    validation_level="standard",  # Standard validation
    run_tests=False,  # Skip tests (run separately)
    enable_llm=True,  # Allow LLM if beneficial
)
```

### 4. Parallel Generation

```python
import asyncio

# Generate multiple nodes in parallel
async def generate_batch():
    service = CodeGenerationService()

    tasks = [
        service.generate_node(requirements=req1, strategy="jinja2"),
        service.generate_node(requirements=req2, strategy="jinja2"),
        service.generate_node(requirements=req3, strategy="jinja2"),
    ]

    results = await asyncio.gather(*tasks)
    return results

results = await generate_batch()
```

### Performance Benchmarks

| Configuration | Generation Time | Quality Score |
|---------------|----------------|---------------|
| Fast (jinja2, basic) | ~200ms | ⭐⭐⭐ |
| Balanced (auto, standard) | ~800ms | ⭐⭐⭐⭐ |
| Quality (template_loading, strict) | ~3000ms | ⭐⭐⭐⭐⭐ |

---

## Troubleshooting

### Issue: "Validation always fails"

**Solution**: Lower validation level or check requirements
```python
# Try basic validation
result = await service.generate_node(
    requirements=requirements,
    validation_level="basic",
)

# Or disable validation
result = await service.generate_node(
    requirements=requirements,
    validation_level="none",
)
```

### Issue: "Generation is too slow"

**Solution**: Use faster strategy and skip validation
```python
result = await service.generate_node(
    requirements=requirements,
    strategy="jinja2",  # Fast template-based
    validation_level="none",  # Skip validation
    run_tests=False,  # Skip tests
)
```

### Issue: "LLM not working"

**Solution**: Check Archon MCP configuration
```python
service = CodeGenerationService(
    archon_mcp_url="http://archon:8060",  # Verify URL
    enable_intelligence=True,  # Enable intelligence
)

# Or disable LLM
result = await service.generate_node(
    requirements=requirements,
    enable_llm=False,  # Disable LLM
)
```

### Issue: "Output directory permission denied"

**Solution**: Check directory permissions
```python
output_dir = Path("./generated")
output_dir.mkdir(parents=True, exist_ok=True)  # Create with parents

result = await service.generate_node(
    requirements=requirements,
    output_directory=output_dir,
)
```

### Issue: "Strategy not found"

**Solution**: List available strategies
```python
strategies = service.list_strategies()
print("Available strategies:")
for strategy in strategies:
    print(f"  - {strategy['type']}")

# Use available strategy
result = await service.generate_node(
    requirements=requirements,
    strategy="jinja2",  # Use available strategy
)
```

---

## Next Steps

1. ✅ **Review Examples**: See [examples/codegen/](../../examples/codegen/)
2. ✅ **Read Architecture**: See [ARCHITECTURE.md](./ARCHITECTURE.md)
3. ✅ **Check Migration**: See [MIGRATION_GUIDE.md](./MIGRATION_GUIDE.md)
4. ✅ **Try in Development**: Generate your first node
5. ✅ **Provide Feedback**: Report issues or suggestions

---

## Support

- **Documentation**: [docs/codegen/](.)
- **Examples**: [examples/codegen/](../../examples/codegen/)
- **Issues**: GitHub Issues
- **Questions**: Team Slack #code-generation

---

**Status**: ✅ Production ready, comprehensive examples available
