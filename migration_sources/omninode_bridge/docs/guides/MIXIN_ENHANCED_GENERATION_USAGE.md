# Mixin-Enhanced Code Generation - Usage Guide

**Status**: ✅ Wave 2 Integration Complete
**Last Updated**: 2025-11-04
**Components**: CodeGenerationService, YAMLContractParser, MixinInjector, NodeValidator

---

## Overview

The CodeGenerationService now supports **mixin-enhanced code generation** through Wave 2 integration. This enables automatic injection of omnibase_core mixins into generated nodes with comprehensive validation.

### Key Features

- ✅ **Automatic Mixin Injection** - Parse v2.0 contracts and inject mixin code
- ✅ **6-Stage Validation** - Syntax, AST, imports, type checking, ONEX compliance, security
- ✅ **Backward Compatible** - Existing v1.0 workflows unchanged
- ✅ **Comprehensive Metrics** - Track mixin usage and validation results
- ✅ **Error Handling** - Clear error messages with validation suggestions

---

## Quick Start

### Basic Usage (No Mixins)

```python
from omninode_bridge.codegen.service import CodeGenerationService
from omninode_bridge.codegen.prd_analyzer import ModelPRDRequirements

# Initialize service
service = CodeGenerationService()

# Define requirements
requirements = ModelPRDRequirements(
    service_name="postgres_crud",
    node_type="effect",
    domain="database",
    business_description="PostgreSQL CRUD operations",
    operations=["create", "read", "update", "delete"],
    features=["Connection pooling", "Async operations"],
    input_schema={},
    output_schema={},
    performance_requirements={"max_latency_ms": 100},
    error_handling_strategy="retry",
    dependencies={"asyncpg": "^0.27.0"},
)

# Generate node (backward compatible - no mixins)
result = await service.generate_node(
    requirements=requirements,
    strategy="auto",
    enable_llm=True,
    enable_mixins=False,  # Disable mixin enhancement
    validation_level="standard",
)

print(f"Generated: {result.artifacts.node_name}")
```

### Mixin-Enhanced Generation

```python
from pathlib import Path
from omninode_bridge.codegen.service import CodeGenerationService
from omninode_bridge.codegen.prd_analyzer import ModelPRDRequirements

# Initialize service with mixin support
service = CodeGenerationService(
    enable_mixin_validation=True,  # Enable validation (default: True)
    enable_type_checking=False,    # Disable mypy (faster, default: False)
)

# Define requirements
requirements = ModelPRDRequirements(
    service_name="postgres_crud",
    node_type="effect",
    domain="database",
    business_description="PostgreSQL CRUD operations with health checks",
    operations=["create", "read", "update", "delete"],
    features=["Connection pooling", "Health monitoring"],
    input_schema={},
    output_schema={},
    performance_requirements={"max_latency_ms": 100},
    error_handling_strategy="retry",
    dependencies={"asyncpg": "^0.27.0"},
)

# Contract path with mixin declarations
contract_path = Path("examples/contracts/postgres_crud_with_mixins_v2.yaml")

# Generate with mixin enhancement
result = await service.generate_node(
    requirements=requirements,
    contract_path=contract_path,  # Required for mixin parsing
    strategy="auto",
    enable_llm=True,
    enable_mixins=True,  # Enable mixin enhancement (default: True)
    validation_level="strict",  # Strict validation
)

# Access results
print(f"Generated: {result.artifacts.node_name}")
print(f"Mixins Applied: {len(result.artifacts.node_file.count('Mixin'))}")
print(f"Validation Passed: {result.validation_passed}")
```

---

## Contract Format

### v2.0 Contract with Mixins

```yaml
name: postgres_crud_effect
schema_version: v2.0.0
version:
  major: 1
  minor: 0
  patch: 0
description: PostgreSQL CRUD operations with health checks
node_type: EFFECT

# Input/Output models
input_model: ModelPostgresCRUDRequest
output_model: ModelPostgresCRUDResponse

# Mixin declarations (Wave 2 feature)
mixins:
  - name: MixinHealthCheck
    enabled: true
    config:
      health_check_interval_seconds: 60
      failure_threshold: 3

  - name: MixinMetrics
    enabled: true
    config:
      metrics_enabled: true
      collect_latency: true

# Advanced features (optional)
advanced_features:
  circuit_breaker:
    enabled: true
    failure_threshold: 5
    recovery_timeout_ms: 60000

  retry_policy:
    enabled: true
    max_attempts: 3
    backoff_multiplier: 2.0

# Effect-specific configuration
io_operations:
  - operation_type: database_query
    atomic: true
    timeout_seconds: 30
```

---

## Configuration Options

### Service Initialization

```python
service = CodeGenerationService(
    templates_directory=None,          # Path to custom templates (optional)
    archon_mcp_url=None,                # Archon MCP endpoint (optional)
    enable_intelligence=True,            # Enable RAG intelligence (default: True)
    enable_mixin_validation=True,       # Enable mixin validation (default: True)
    enable_type_checking=False,         # Enable mypy type checking (default: False)
)
```

**Parameters**:
- `enable_mixin_validation`: Enable/disable mixin validation (faster if disabled)
- `enable_type_checking`: Enable mypy type checking (adds 2-3s overhead)

### Generation Options

```python
result = await service.generate_node(
    requirements=requirements,           # PRD requirements (required)
    classification=None,                 # Node classification (auto if None)
    output_directory=None,               # Output directory (temp if None)
    strategy="auto",                     # Generation strategy
    enable_llm=True,                     # Enable LLM enhancement
    enable_mixins=True,                  # Enable mixin injection (NEW)
    validation_level="standard",         # Validation level
    run_tests=False,                     # Run generated tests
    strict_mode=False,                   # Raise on test failure
    correlation_id=None,                 # Correlation ID for tracing
    contract_path=None,                  # Path to YAML contract (NEW)
)
```

**New Parameters**:
- `enable_mixins`: Enable/disable mixin-enhanced generation (default: `True`)
- `contract_path`: Path to v2.0 YAML contract file with mixin declarations (required for mixins)

### Validation Levels

- **`none`**: No validation
- **`basic`**: Syntax checking only
- **`standard`**: Syntax + AST + imports + security (default)
- **`strict`**: All stages + type checking + ONEX compliance (raises on failure)

---

## Validation Pipeline

### 6-Stage Validation

When `enable_mixin_validation=True` and `enable_mixins=True`, the following stages run:

1. **Syntax Validation** (<10ms) - Python compile check
2. **AST Validation** (<20ms) - Structure and method signatures
3. **Import Resolution** (<50ms) - Verify imports can be resolved
4. **Type Checking** (1-3s) - Optional mypy integration (if `enable_type_checking=True`)
5. **ONEX Compliance** (<100ms) - Mixin verification and patterns
6. **Security Scanning** (<100ms) - Dangerous pattern detection

**Performance Target**: <200ms without type checking, <3s with type checking

### Validation Results

```python
# Access validation results from metrics
if validation_results:
    for result in validation_results:
        print(result)  # Formatted validation result
        if not result.passed:
            print(f"  Errors: {result.errors}")
            print(f"  Suggestions: {result.suggestions}")
```

---

## Metrics and Logging

### Mixin Metrics

Automatically tracked when mixins are enabled:

```python
logger.info(
    "Node generation complete",
    extra={
        "node_name": "NodePostgresCrudEffect",
        "mixins_applied": 2,
        "mixin_names": ["MixinHealthCheck", "MixinMetrics"],
        "has_advanced_features": True,
        "validation_stages_run": 6,
        "validation_stages_passed": 6,
        "validation_time_ms": 150.5,
        "validation_errors": 0,
        "validation_warnings": 0,
    }
)
```

### Generation Flow

```
1. Parse Contract (if contract_path provided)
   ├─ Parse YAML with YAMLContractParser
   ├─ Validate contract schema
   └─ Extract mixin declarations

2. Generate Base Code
   ├─ Select strategy (auto/jinja2/template_loading/hybrid)
   └─ Generate node code

3. Apply Mixin Enhancement (if enable_mixins=True and mixins present)
   ├─ Convert contract to dict
   ├─ Generate mixin-enhanced code with MixinInjector
   └─ Update artifacts with enhanced code

4. Validate Code (if enable_mixin_validation=True)
   ├─ Run 6-stage validation pipeline
   ├─ Collect validation results
   └─ Raise exception if strict mode and validation fails

5. Log Metrics
   ├─ Generation time
   ├─ Mixin metrics
   ├─ Validation metrics
   └─ Strategy used

6. Distribute Files
   └─ Write to output directories
```

---

## Error Handling

### Contract Validation Errors

```python
try:
    result = await service.generate_node(
        requirements=requirements,
        contract_path=Path("invalid_contract.yaml"),
        enable_mixins=True,
    )
except ValueError as e:
    # Contract validation failed
    print(f"Contract error: {e}")
    # Output: "Contract validation failed: ['Missing required field: name']"
```

### Code Validation Errors

```python
try:
    result = await service.generate_node(
        requirements=requirements,
        contract_path=contract_path,
        enable_mixins=True,
        validation_level="strict",  # Raises on validation failure
    )
except RuntimeError as e:
    # Code validation failed
    print(f"Validation error: {e}")
    # Output includes specific stage errors and suggestions
```

### Mixin-Specific Errors

```python
# Invalid mixin reference
mixins:
  - name: MixinNonExistent  # This mixin doesn't exist
    enabled: true

# Error: "Mixin not found: MixinNonExistent"
```

---

## Backward Compatibility

### v1.0 Contracts (No Mixins)

v1.0 contracts continue to work without modifications:

```python
# No contract_path provided - uses requirements only
result = await service.generate_node(
    requirements=requirements,
    enable_mixins=False,  # Explicitly disable
)
```

### Gradual Migration Path

1. **Keep existing workflows** - No changes required for v1.0 contracts
2. **Add mixins incrementally** - Upgrade contracts to v2.0 one at a time
3. **Enable validation gradually** - Start with `validation_level="basic"`, increase to `"strict"`
4. **Enable type checking last** - Add `enable_type_checking=True` when ready

---

## Performance Considerations

### Generation Performance

| Configuration | Time | Notes |
|---------------|------|-------|
| No mixins | ~500ms | Backward compatible |
| With mixins (no validation) | ~600ms | +100ms for mixin injection |
| With mixins + validation (no type checking) | ~800ms | +200ms for 5 stages |
| With mixins + validation + type checking | ~3.5s | +2.7s for mypy |

### Optimization Tips

1. **Disable type checking in development** - Use `enable_type_checking=False` (default)
2. **Use validation_level="basic"** - For rapid iteration
3. **Enable strict validation in CI/CD** - Catch issues before production
4. **Cache contract parsing** - Reuse parsed contracts when possible

---

## Examples

### Example 1: Simple CRUD with Health Checks

```python
# Contract: postgres_crud_health_v2.yaml
name: postgres_crud_effect
mixins:
  - name: MixinHealthCheck
    enabled: true

# Generate
result = await service.generate_node(
    requirements=crud_requirements,
    contract_path=Path("postgres_crud_health_v2.yaml"),
    enable_mixins=True,
)

# Output includes health check methods
assert "async def get_health_checks(self)" in result.artifacts.node_file
```

### Example 2: Complex Node with Multiple Mixins

```python
# Contract: advanced_service_v2.yaml
name: advanced_service_effect
mixins:
  - name: MixinHealthCheck
    enabled: true
  - name: MixinMetrics
    enabled: true
  - name: MixinCaching
    enabled: true
    config:
      cache_ttl_seconds: 300

# Generate with strict validation
result = await service.generate_node(
    requirements=advanced_requirements,
    contract_path=Path("advanced_service_v2.yaml"),
    enable_mixins=True,
    validation_level="strict",
)

# All 3 mixins injected and validated
assert result.validation_passed
```

### Example 3: Disable Mixins for Specific Generation

```python
# Even if contract has mixins, you can disable them
result = await service.generate_node(
    requirements=requirements,
    contract_path=Path("contract_with_mixins_v2.yaml"),
    enable_mixins=False,  # Generate without mixins
)

# No mixin code injected
assert "Mixin" not in result.artifacts.node_file
```

---

## Troubleshooting

### Issue: Contract validation fails

**Solution**: Check contract YAML syntax and required fields
```bash
# Validate contract manually
python -m omninode_bridge.codegen.yaml_contract_parser contract.yaml
```

### Issue: Mixin not found

**Solution**: Verify mixin name matches catalog
```python
from omninode_bridge.codegen.mixin_injector import MIXIN_CATALOG
print(MIXIN_CATALOG.keys())  # List available mixins
```

### Issue: Type checking too slow

**Solution**: Disable type checking
```python
service = CodeGenerationService(enable_type_checking=False)
```

### Issue: Validation fails in strict mode

**Solution**: Use standard mode during development
```python
result = await service.generate_node(
    ...,
    validation_level="standard",  # Instead of "strict"
)
```

---

## Next Steps

- **[Mixin Enhancement Master Plan](../planning/CODEGEN_MIXIN_ENHANCEMENT_MASTER_PLAN.md)** - Complete implementation roadmap
- **[Wave 2 Quickstart](MIXIN_ENHANCED_GENERATION_QUICKSTART.md)** - Quick start guide
- **[Code Generation Guide](CODEGEN_GUIDE.md)** - General code generation documentation
- **[API Reference](../api/API_REFERENCE.md)** - Complete API documentation

---

**Document Status**: ✅ Complete
**Wave 2 Integration**: ✅ Fully Integrated
**Author**: Claude Code (Polymorphic Agent)
