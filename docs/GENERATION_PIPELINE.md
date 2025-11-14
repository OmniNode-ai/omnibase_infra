# ONEX Node Generation Pipeline

**Status:** Migrated and Operational
**Date:** 2025-11-14
**Location:** `src/omnibase_infra/generation/`

---

## 🎯 Overview

The ONEX Node Generation Pipeline provides automated scaffolding generation for all ONEX node types (EFFECT, COMPUTE, REDUCER, ORCHESTRATOR) using comprehensive templates.

This pipeline was migrated from the omninode_bridge archive and enables rapid, consistent node creation following ONEX architectural patterns.

---

## 📦 Components

### 1. NodeGenerator (Main Interface)
**File:** `src/omnibase_infra/generation/node_generator.py`

Primary entry point for node generation with high-level API.

```python
from omnibase_infra.generation import NodeGenerator

generator = NodeGenerator(output_dir="/path/to/output")

result = generator.generate_node(
    node_type="effect",
    repository_name="omnibase_infra",
    domain="infrastructure",
    microservice_name="redis_adapter",
    business_description="Redis caching adapter",
    external_system="Redis",
)
```

### 2. TemplateProcessor
**File:** `src/omnibase_infra/generation/utils/template_processor.py`

Handles template loading, placeholder substitution, and code block extraction.

**Features:**
- Load templates from `generation/templates/`
- Substitute placeholders with actual values
- Extract code blocks from markdown templates
- Generate file mappings for node structure

### 3. NameConverter
**File:** `src/omnibase_infra/generation/utils/name_converter.py`

Converts between naming conventions:
- `snake_case` → `my_service_name`
- `PascalCase` → `MyServiceName`
- `kebab-case` → `my-service-name`
- `SCREAMING_SNAKE_CASE` → `MY_SERVICE_NAME`

### 4. FileWriter
**File:** `src/omnibase_infra/generation/utils/file_writer.py`

Writes generated files to filesystem with directory creation and `__init__.py` management.

**Features:**
- Create directory structures
- Write files with proper encoding
- Generate `__init__.py` files
- Dry-run mode for previewing

---

## 📋 Available Templates

Located in: `src/omnibase_infra/generation/templates/`

1. **EFFECT_NODE_TEMPLATE.md** (39KB)
   - External system integration nodes
   - Message bus bridge pattern
   - Circuit breaker integration
   - Health monitoring

2. **COMPUTE_NODE_TEMPLATE.md** (65KB)
   - Business logic processing nodes
   - Stateless computation
   - Performance optimization patterns

3. **REDUCER_NODE_TEMPLATE.md** (38KB)
   - Data aggregation and consolidation
   - Stream processing
   - Pattern recognition
   - Statistical reduction

4. **ORCHESTRATOR_NODE_TEMPLATE.md** (31KB)
   - Workflow coordination
   - Multi-step process management
   - Node communication orchestration
   - State management

5. **ENHANCED_NODE_PATTERNS.md** (37KB)
   - Advanced patterns and best practices
   - Security patterns
   - Performance optimizations

---

## 🚀 Usage Examples

### Example 1: Generate EFFECT Node

```python
from omnibase_infra.generation import NodeGenerator

generator = NodeGenerator(output_dir=".")

result = generator.generate_node(
    node_type="effect",
    repository_name="omnibase_infra",
    domain="infrastructure",
    microservice_name="vault_adapter",
    business_description="HashiCorp Vault secret management adapter",
    external_system="HashiCorp Vault",
)

print(f"Generated node at: {result['node_directory']}")
```

**Output Structure:**
```
src/omnibase_infra/nodes/node_infrastructure_vault_adapter_effect/v1_0_0/
├── __init__.py
├── node.py
├── models/
│   ├── __init__.py
│   ├── model_vault_adapter_input.py
│   └── model_vault_adapter_output.py
├── enums/
│   └── __init__.py
└── contracts/
    └── __init__.py
```

### Example 2: Generate REDUCER Node

```python
result = generator.generate_node(
    node_type="reducer",
    repository_name="omnibase_infra",
    domain="infrastructure",
    microservice_name="health_aggregator",
    business_description="Aggregate health metrics from multiple infrastructure adapters",
)
```

### Example 3: Generate from Existing Contract

```python
from pathlib import Path

result = generator.generate_from_contract(
    contract_path=Path("src/omnibase_infra/nodes/consul_adapter/v1_0_0/contract.yaml"),
)
```

### Example 4: Dry Run (Preview Only)

```python
result = generator.generate_node(
    node_type="orchestrator",
    repository_name="omnibase_infra",
    domain="infrastructure",
    microservice_name="service_coordinator",
    business_description="Coordinate infrastructure service startup and shutdown",
    dry_run=True,  # Won't write files
)
```

---

## 🖥️ CLI Interface

**File:** `src/omnibase_infra/generation/cli.py`

### Usage

```bash
# Generate EFFECT node
python -m omnibase_infra.generation.cli effect \
    --domain infrastructure \
    --microservice redis_adapter \
    --repository omnibase_infra \
    --description "Redis caching adapter" \
    --external-system Redis

# List available templates
python -m omnibase_infra.generation.cli --list-templates

# Generate from contract
python -m omnibase_infra.generation.cli effect \
    --from-contract path/to/contract.yaml

# Dry run (preview)
python -m omnibase_infra.generation.cli reducer \
    --domain infrastructure \
    --microservice metrics_aggregator \
    --dry-run
```

### CLI Arguments

- `node_type`: Type of node (effect, compute, reducer, orchestrator)
- `--domain`: Domain name (e.g., infrastructure, ai)
- `--microservice`: Microservice name (e.g., postgres_adapter)
- `--repository`: Repository name (default: omnibase_infra)
- `--description`: Business description
- `--external-system`: External system for EFFECT nodes
- `--output-dir`: Output directory (default: current directory)
- `--dry-run`: Preview without writing files
- `--list-templates`: List available templates
- `--from-contract`: Generate from existing contract.yaml

---

## 📐 Template Placeholder System

Templates use placeholders that are automatically replaced:

| Placeholder | Description | Example |
|------------|-------------|---------|
| `{REPOSITORY_NAME}` | Repository name | `omnibase_infra` |
| `{DOMAIN}` | Domain (snake_case) | `infrastructure` |
| `{DOMAIN_PASCAL}` | Domain (PascalCase) | `Infrastructure` |
| `{MICROSERVICE_NAME}` | Microservice (snake_case) | `postgres_adapter` |
| `{MICROSERVICE_NAME_PASCAL}` | Microservice (PascalCase) | `PostgresAdapter` |
| `{BUSINESS_DESCRIPTION}` | Description | `PostgreSQL database adapter` |
| `{EXTERNAL_SYSTEM}` | External system | `PostgreSQL` |

---

## 🔧 Customization

### Adding New Templates

1. Create template file: `src/omnibase_infra/generation/templates/NEW_TYPE_NODE_TEMPLATE.md`
2. Use placeholder syntax: `{PLACEHOLDER_NAME}`
3. Include code blocks with language specifiers:

````markdown
```python
class Node{DOMAIN_PASCAL}{MICROSERVICE_NAME_PASCAL}{NodeType}:
    """Generated node."""
    pass
```
````

### Modifying Existing Templates

Templates are markdown files with embedded code blocks. Edit templates directly in `generation/templates/`.

---

## ✅ Validation

After generation, validate nodes:

```bash
# Run contract validation
python -m omnibase_infra.generation.cli effect \
    --domain infrastructure \
    --microservice test_adapter \
    --dry-run

# Check generated structure
tree src/omnibase_infra/nodes/node_infrastructure_test_adapter_effect/
```

---

## 🎯 Migration Status

### Migrated Components ✅

- ✅ EFFECT node template (39KB)
- ✅ COMPUTE node template (65KB)
- ✅ REDUCER node template (38KB)
- ✅ ORCHESTRATOR node template (31KB)
- ✅ Enhanced patterns template (37KB)
- ✅ NodeGenerator orchestration class
- ✅ TemplateProcessor for placeholder substitution
- ✅ NameConverter for naming conventions
- ✅ FileWriter for filesystem operations
- ✅ CLI interface for command-line usage

### Integration Points

- **CLAUDE.md**: References `agent-contract-driven-generator` and `agent-ast-generator`
- **Templates**: Located in `src/omnibase_infra/generation/templates/`
- **CLI**: Accessible via `python -m omnibase_infra.generation.cli`

---

## 📊 Template Statistics

| Template | Size | Lines | Components |
|----------|------|-------|------------|
| EFFECT | 39KB | ~1000 | Node, Models, Circuit Breaker, Security |
| COMPUTE | 65KB | ~1600 | Node, Models, Processing, Validation |
| REDUCER | 38KB | ~900 | Node, Models, Aggregation, Streaming |
| ORCHESTRATOR | 31KB | ~800 | Node, Models, Workflows, Coordination |
| **Total** | **173KB** | **~4300** | **Complete node scaffolding** |

---

## 🚀 Next Steps

### Immediate

1. ✅ Templates migrated
2. ✅ Generation utilities created
3. ✅ CLI interface implemented
4. ✅ Documentation complete

### Future Enhancements

1. Contract-to-model generation automation
2. AST-based code analysis and generation
3. Integration with agent-contract-driven-generator
4. Template versioning and upgrades
5. Validation hooks for generated code

---

## 🎉 Migration Complete!

The ONEX Node Generation Pipeline has been successfully migrated from the omninode_bridge archive and is now operational in `src/omnibase_infra/generation/`.

**Key Achievements:**
- ✅ 4 comprehensive node templates (173KB total)
- ✅ Full placeholder substitution system
- ✅ CLI and programmatic interfaces
- ✅ Dry-run and validation support
- ✅ Complete documentation

**Ready for use in generating new infrastructure nodes!**
