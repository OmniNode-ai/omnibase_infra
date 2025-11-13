# Code Generation Documentation

**Version**: 1.0.0
**Last Updated**: 2025-11-01
**Status**: Production Ready

---

## 📚 Documentation Hub

Welcome to the OmniNode Bridge unified code generation system documentation!

### Quick Navigation

#### 🚀 Getting Started
- **[README.md (Project Root)](../../README.md#-unified-code-generation-service)** - Quick overview and quick start
- **[USAGE_GUIDE.md](./USAGE_GUIDE.md)** - Comprehensive usage guide with examples (START HERE)

#### 🔄 Migration
- **[MIGRATION_GUIDE.md](./MIGRATION_GUIDE.md)** - Migrate from old to new system (fully backward compatible)

#### 🏗️ Architecture
- **[ARCHITECTURE.md](./ARCHITECTURE.md)** - System architecture, design decisions, and patterns

#### 💻 Examples
- **[examples/codegen/basic_usage.py](../../examples/codegen/basic_usage.py)** - Simple node generation
- **[examples/codegen/strategy_selection.py](../../examples/codegen/strategy_selection.py)** - Compare strategies
- **[examples/codegen/custom_strategy.py](../../examples/codegen/custom_strategy.py)** - Custom strategies
- **[examples/codegen/batch_generation.py](../../examples/codegen/batch_generation.py)** - Batch generation

---

## What's New?

### Unified Service (November 2025)

The code generation system has been **unified into a single service** with a strategy pattern architecture:

**Before (Two Parallel Systems)**:
```
TemplateEngine (Jinja2)        TemplateEngineLoader + BusinessLogicGenerator
```

**After (Unified Service)**:
```
CodeGenerationService → Strategy Pattern → Multiple Strategies
```

### Key Benefits

1. ✅ **Single Entry Point**: One service replaces multiple systems
2. ✅ **Automatic Strategy Selection**: Service chooses optimal approach
3. ✅ **Consistent Validation**: Built-in quality gates
4. ✅ **Better Observability**: Rich metrics and tracing
5. ✅ **Backward Compatible**: All old code continues to work

---

## Quick Start (5 Minutes)

```python
from omninode_bridge.codegen import CodeGenerationService, ModelPRDRequirements
from pathlib import Path

# Initialize service
service = CodeGenerationService()

# Define requirements
requirements = ModelPRDRequirements(
    node_type="effect",
    service_name="user_crud",
    domain="database",
    business_description="User CRUD operations",
    operations=["create", "read", "update", "delete"],
)

# Generate code
result = await service.generate_node(
    requirements=requirements,
    output_directory=Path("./generated/user_crud"),
    strategy="auto",  # Automatic strategy selection
)

print(f"✅ Generated: {result.artifacts.node_name}")
```

**Output**:
```
✅ Generated: NodeUserCrudEffect
⏱️  Time: 245ms
🎯 Strategy: jinja2
✨ Validation: PASSED
```

---

## Documentation Structure

### 1. MIGRATION_GUIDE.md

**Purpose**: Help transition from old to new system

**Contents**:
- Why migrate (benefits)
- Breaking changes (none - fully compatible!)
- Migration paths (incremental, immediate, or no migration)
- Step-by-step examples
- Troubleshooting

**Who Should Read**: Anyone using old TemplateEngine or CodeGenerationPipeline

### 2. USAGE_GUIDE.md

**Purpose**: Complete usage documentation

**Contents**:
- Getting started (5-minute quickstart)
- Strategy selection guide
- API reference (all methods documented)
- Code examples (10+ working examples)
- Advanced features
- Performance tuning
- Troubleshooting

**Who Should Read**: All users, especially new users

### 3. ARCHITECTURE.md

**Purpose**: System architecture and design

**Contents**:
- System overview
- Architecture diagrams
- Component relationships
- Strategy pattern explanation
- Quality gates pipeline
- Extension points (custom strategies)
- Design decisions and rationale

**Who Should Read**: Architects, maintainers, contributors

---

## Examples Overview

### Basic Usage (`basic_usage.py`)

Simple node generation with default settings.

**What it demonstrates**:
- Initialize service
- Define requirements
- Generate code
- Access results
- Write files to disk

**Best for**: First-time users

### Strategy Selection (`strategy_selection.py`)

Compare different generation strategies.

**What it demonstrates**:
- Generate with Jinja2 (fast)
- Generate with LLM (high quality)
- Generate with Auto (automatic)
- Compare results
- Use case recommendations

**Best for**: Understanding strategy trade-offs

### Custom Strategy (`custom_strategy.py`)

Implement and register custom strategies.

**What it demonstrates**:
- Create custom strategy class
- Implement required methods
- Register with service
- Use in generation
- Advanced concepts

**Best for**: Extending the system

### Batch Generation (`batch_generation.py`)

Generate multiple nodes efficiently.

**What it demonstrates**:
- Parallel generation
- Sequential generation (comparison)
- Error handling
- Performance metrics
- Best practices

**Best for**: Generating multiple nodes

---

## Common Use Cases

### Use Case 1: Simple CRUD Node
**Strategy**: Jinja2
**Why**: Fast, proven patterns
```python
result = await service.generate_node(
    requirements=requirements,
    strategy="jinja2",
)
```

### Use Case 2: Complex Algorithm
**Strategy**: Template Loading (LLM)
**Why**: Handles sophisticated logic
```python
result = await service.generate_node(
    requirements=requirements,
    strategy="template_loading",
    enable_llm=True,
)
```

### Use Case 3: Production Critical
**Strategy**: Auto or Hybrid
**Why**: Optimizes quality and speed
```python
result = await service.generate_node(
    requirements=requirements,
    strategy="auto",
    validation_level="strict",
)
```

---

## Performance Reference

| Strategy | Time (avg) | Memory | Quality | Best For |
|----------|-----------|--------|---------|----------|
| **jinja2** | ~200ms | ~30MB | ⭐⭐⭐ | Simple CRUD |
| **template_loading** | ~3000ms | ~120MB | ⭐⭐⭐⭐⭐ | Complex logic |
| **hybrid** | ~800ms | ~80MB | ⭐⭐⭐⭐ | Critical features |
| **auto** | Varies | Varies | Optimized | General use |

---

## Backward Compatibility

**All old code continues to work** - the new service is a facade over existing systems:

```python
# ✅ OLD CODE STILL WORKS
from omninode_bridge.codegen import TemplateEngine

engine = TemplateEngine()
artifacts = await engine.generate(...)  # Still works!

# ✅ NEW API (RECOMMENDED)
from omninode_bridge.codegen import CodeGenerationService

service = CodeGenerationService()
result = await service.generate_node(...)  # Better API
```

**Timeline**:
- Old API supported until Q1 2027 (24+ months)
- No forced migration required
- Incremental adoption recommended

---

## Support

- **Documentation**: [docs/codegen/](.)
- **Examples**: [examples/codegen/](../../examples/codegen/)
- **Issues**: GitHub Issues
- **Questions**: Team Slack #code-generation

---

## Next Steps

1. ✅ **Read [USAGE_GUIDE.md](./USAGE_GUIDE.md)** for comprehensive examples
2. ✅ **Try [basic_usage.py](../../examples/codegen/basic_usage.py)** to generate your first node
3. ✅ **Review [MIGRATION_GUIDE.md](./MIGRATION_GUIDE.md)** if migrating from old system
4. ✅ **Explore [ARCHITECTURE.md](./ARCHITECTURE.md)** to understand design
5. ✅ **Build something!** Generate your first production node

---

**Status**: ✅ Production ready, fully documented, backward compatible
