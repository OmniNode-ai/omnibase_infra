# Template Engine Loader Usage Guide

## Overview

The Template Engine Loader provides an alternative to Jinja2-based code generation. Instead of generating code from templates, it **loads pre-written Python template files**, detects stubs, and prepares them for LLM enhancement via BusinessLogicGenerator.

## Architecture

```
Hand-written Template Files
         ↓
   TemplateEngine (loads & discovers)
         ↓
   ModelTemplateArtifacts (with detected stubs)
         ↓
   BusinessLogicGenerator (LLM enhances stubs)
         ↓
   CodeInjector (injects generated code)
         ↓
   Complete ONEX Node
```

## Basic Usage

### 1. Discover Available Templates

```python
from omninode_bridge.codegen.template_engine_loader import TemplateEngine

# Initialize engine
engine = TemplateEngine()

# Discover all templates
templates = engine.discover_templates()

for template in templates:
    print(f"{template.node_type}/{template.version}: {template.template_name}")
    print(f"  Description: {template.metadata.description}")
    print(f"  Path: {template.template_path}")
```

### 2. Load Specific Template

```python
from omninode_bridge.codegen.template_engine_loader import TemplateEngine

engine = TemplateEngine()

# Load template with stub detection
artifacts = engine.load_template(
    node_type="effect",
    version="v1_0_0",
    template_name="node"
)

print(f"Loaded template: {artifacts.template_path}")
print(f"Detected {artifacts.get_stub_count()} stubs:")

for stub in artifacts.stubs:
    print(f"  - {stub.method_name} (line {stub.line_start}-{stub.line_end})")
    print(f"    Signature: {stub.signature}")
```

### 3. Integration with BusinessLogicGenerator

```python
from omninode_bridge.codegen.template_engine_loader import TemplateEngine
from omninode_bridge.codegen.business_logic import BusinessLogicGenerator
from omninode_bridge.codegen.prd_analyzer import ModelPRDRequirements

# Load template
engine = TemplateEngine()
artifacts = engine.load_template("effect", "v1_0_0")

# Convert to format BusinessLogicGenerator expects
from omninode_bridge.codegen.template_engine import ModelGeneratedArtifacts

# Map TemplateArtifacts → GeneratedArtifacts
generated_artifacts = ModelGeneratedArtifacts(
    node_file=artifacts.template_code,
    contract_file="# Contract placeholder",
    init_file="# Init placeholder",
    node_type=artifacts.node_type,
    node_name="NodeExampleEffect",  # Extract from template
    service_name="example_service",
)

# Create PRD requirements
requirements = ModelPRDRequirements(
    service_name="example_service",
    business_description="Example service for demonstration",
    domain="api_development",
    operations=["create", "read", "update"],
    features=["CRUD operations", "Data validation"],
)

# Enhance with LLM
generator = BusinessLogicGenerator(enable_llm=True)
enhanced = await generator.enhance_artifacts(
    artifacts=generated_artifacts,
    requirements=requirements,
    context_data={
        "patterns": ["Use async/await throughout"],
        "best_practices": ["Add comprehensive error handling"],
    }
)

print(f"Generated {len(enhanced.methods_generated)} method implementations")
print(f"Total cost: ${enhanced.total_cost_usd:.4f}")
print(f"Success rate: {enhanced.generation_success_rate:.1%}")
```

## Directory Structure

Templates should be organized as:

```
templates/
  node_templates/
    effect/
      v1_0_0/
        node.py           # Main template file
        models.py         # Optional models
    compute/
      v1_0_0/
        node.py
    reducer/
      v1_0_0/
        node.py
    orchestrator/
      v1_0_0/
        node.py
```

## Template File Format

Templates should be valid Python files with:

1. **Module docstring** (for description metadata)
2. **Stub markers** for LLM enhancement
3. **Required methods** for the node type

### Example Template

```python
#!/usr/bin/env python3
"""
Example Effect Node Template.

This template demonstrates the structure for Effect nodes
with stubs for LLM enhancement.

Author: OmniNode Team
Tags: effect, crud, database
"""

from typing import Any
from omnibase_core import ModelOnexError, EnumCoreErrorCode
from omnibase_core.nodes.node_effect import NodeEffect
from omnibase_core.models.contracts.model_contract_effect import ModelContractEffect

class NodeExampleEffect(NodeEffect):
    """Example Effect node with stub implementations."""

    async def execute_effect(self, contract: ModelContractEffect) -> Any:
        """
        Execute effect operation.

        Args:
            contract: Effect contract with operation parameters

        Returns:
            Operation result
        """
        # IMPLEMENTATION REQUIRED
        # TODO: Add effect-specific logic here
        pass
```

## Configuration

Customize template paths and behavior:

```python
from pathlib import Path
from omninode_bridge.codegen.template_engine_loader import (
    TemplateEngine,
    TemplateEngineConfig
)

# Custom template directory
custom_engine = TemplateEngine(
    template_root=Path("/path/to/custom/templates"),
    enable_validation=True  # Validate required methods
)

# Access configuration
print(f"Default template root: {TemplateEngineConfig.DEFAULT_TEMPLATE_ROOT}")
print(f"Stub indicators: {TemplateEngineConfig.STUB_INDICATORS}")
```

## Error Handling

```python
from omninode_bridge.codegen.template_engine_loader import (
    TemplateEngine,
    TemplateEngineError
)

engine = TemplateEngine()

try:
    artifacts = engine.load_template("effect", "v99_0_0")
except TemplateEngineError as e:
    print(f"Template loading failed: {e}")
    # Template not found, invalid format, etc.
```

## Key Design Decisions

1. **Separation from Jinja2**: This is a separate module (`template_engine_loader`) to avoid confusion with the existing Jinja2-based `template_engine.py`

2. **CodeInjector Integration**: Reuses existing `CodeInjector.find_stubs()` for consistency in stub detection patterns

3. **Pydantic v2 Models**: All models follow ONEX naming conventions (`ModelTemplateArtifacts`, `ModelStubInfo`)

4. **Metadata Extraction**: Automatic extraction from docstrings and comments for template documentation

5. **Version-aware**: Supports multiple versions of templates in parallel (v1_0_0, v2_0_0, etc.)

## Next Steps

1. **Create Template Files**: Add hand-written template files to `templates/node_templates/`
2. **Test Discovery**: Run `discover_templates()` to verify templates are found
3. **Test Loading**: Load templates and verify stub detection works
4. **Integrate with Pipeline**: Connect to full code generation pipeline with BusinessLogicGenerator

## Comparison: Jinja2 vs Template Loader

| Feature | Jinja2 (template_engine.py) | Template Loader (template_engine_loader) |
|---------|----------------------------|----------------------------------------|
| **Approach** | Generate from templates | Load pre-written files |
| **Flexibility** | High (dynamic generation) | Medium (static templates) |
| **Control** | Template syntax | Direct Python code |
| **Use Case** | Automated generation | Hand-crafted templates |
| **Stubs** | Generated with markers | Detected automatically |

Choose **Jinja2** when you want fully automated code generation.

Choose **Template Loader** when you want to:
- Start with hand-written templates
- Have full control over structure
- Incrementally enhance with LLM
- Version templates independently
