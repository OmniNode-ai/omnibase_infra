# Mixin-Enhanced Node Examples - Quick Start Guide

## Overview

This directory contains complete, runnable examples demonstrating the ONEX v2.0 mixin-enhanced code generation system.

## Directory Structure

```
examples/mixin_enhanced_nodes/
├── README.md                              # Comprehensive documentation
├── QUICK_START.md                         # This file
├── validate_examples.py                   # Validation script
│
├── basic_effect_with_health_check/        # Example 1: Basic mixin integration
│   ├── contract.yaml                      # Node contract definition
│   ├── node.py                            # Generated node implementation
│   └── __init__.py                        # Package initialization
│
├── advanced_orchestrator_with_metrics/    # Example 2: Advanced observability
│   ├── contract.yaml
│   ├── node.py
│   └── __init__.py
│
└── complete_node_all_mixins/              # Example 3: Comprehensive example
    ├── contract.yaml
    ├── node.py
    └── __init__.py
```

## Prerequisites

```bash
# Ensure you're in the project root
cd /Volumes/PRO-G40/Code/omninode_bridge

# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

## Quick Start

### 1. Validate Examples

Verify all examples are syntactically correct:

```bash
python examples/mixin_enhanced_nodes/validate_examples.py
```

Expected output:
```
✓ All examples are valid!
```

### 2. Run Basic Example

Run the simplest example with health checking:

```bash
python examples/mixin_enhanced_nodes/basic_effect_with_health_check/node.py
```

Expected output:
```
============================================================
Basic Effect with Health Check - Example
============================================================

1. Initializing node...
   ✓ Node initialized

2. Checking health...
   Overall Status: healthy
   Components: 2
     ✓ self: healthy
     ✗ database: degraded

3. Processing data...
   Input: hello world
   Output: PROCESSED: HELLO WORLD
   Processing Time: 12.34ms
   Status: success

4. Shutting down...
   ✓ Node shutdown complete
```

### 3. Run Advanced Example

Run the orchestrator with full observability:

```bash
python examples/mixin_enhanced_nodes/advanced_orchestrator_with_metrics/node.py
```

Expected output shows:
- Multi-component health checks
- Workflow execution with metrics
- Performance tracking

### 4. Run Complete Example

Run the comprehensive example with all mixins:

```bash
python examples/mixin_enhanced_nodes/complete_node_all_mixins/node.py
```

Expected output demonstrates:
- Result caching (cache hit/miss)
- Event publishing
- Comprehensive metrics
- Multiple operations

## Using Examples in Your Code

### Import and Use a Node

```python
from omnibase_core.models.core.model_container import ModelContainer
from examples.mixin_enhanced_nodes.basic_effect_with_health_check.node import (
    NodeDataProcessingEffect
)

# Create container and node
container = ModelContainer()
node = NodeDataProcessingEffect(container)

# Initialize
await node.initialize()

# Check health
health_result = await node.check_health()
print(f"Health: {health_result.overall_status}")

# Process data
result = await node.execute_effect({
    "data": "test input",
    "correlation_id": "test-001"
})
print(f"Result: {result}")

# Shutdown
await node.shutdown()
```

### Generate a Custom Node from Example Contract

```bash
# Copy example contract as starting point
cp examples/mixin_enhanced_nodes/basic_effect_with_health_check/contract.yaml \
   my_custom_node_contract.yaml

# Edit the contract (change name, mixins, etc.)
vim my_custom_node_contract.yaml

# Generate the node using code generator
poetry run python -m omninode_bridge.cli.codegen.main generate \
  --contract my_custom_node_contract.yaml \
  --output-dir ./output/my_custom_node
```

## Example Comparison

| Feature | Basic | Advanced | Complete |
|---------|-------|----------|----------|
| **Mixins** | 1 | 3 | 8 |
| **Health Checks** | ✓ | ✓ | ✓ |
| **Metrics** | - | ✓ | ✓ |
| **Logging** | - | ✓ | ✓ |
| **Event Bus** | - | - | ✓ |
| **Caching** | - | - | ✓ |
| **Service Registry** | - | - | ✓ |
| **Complexity** | Low | Medium | High |
| **Best For** | Learning | Production | Complex Systems |

## Common Tasks

### Check Syntax Without Running

```bash
python -m py_compile examples/mixin_enhanced_nodes/*/node.py
```

### View Contract Schema

```bash
cat examples/mixin_enhanced_nodes/basic_effect_with_health_check/contract.yaml
```

### Test Contract Parsing

```bash
python -c "
import yaml
with open('examples/mixin_enhanced_nodes/basic_effect_with_health_check/contract.yaml') as f:
    contract = yaml.safe_load(f)
    print(f'Node: {contract[\"name\"]}')
    print(f'Mixins: {len(contract.get(\"mixins\", []))}')
"
```

### Compare Examples Side-by-Side

```bash
# View mixin configurations
for dir in examples/mixin_enhanced_nodes/*/; do
    echo "=== $(basename $dir) ==="
    grep -A 20 "^mixins:" "$dir/contract.yaml" | head -20
    echo
done
```

## Troubleshooting

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'omnibase_core'`

**Solution**:
```bash
# Check if omnibase_core is installed
poetry show omnibase_core

# If not installed
poetry add omnibase_core
```

### Health Check Mixin Not Available

**Problem**: Examples run but show "Health check mixin not available"

**Solution**: This is expected behavior. The examples are designed to work with or without the actual mixin implementations. They use protocol stubs for type checking.

### Examples Hang or Timeout

**Problem**: Example execution hangs

**Solution**:
```bash
# Check for infinite loops or blocking operations
# Ctrl+C to interrupt
# Review async operations in the example
```

## Next Steps

1. **Read the comprehensive README**: `cat examples/mixin_enhanced_nodes/README.md`
2. **Study the examples**: Start with basic, progress to complete
3. **Customize a contract**: Copy and modify an example contract
4. **Generate your own node**: Use the code generator CLI
5. **Run tests**: Add tests for your custom nodes

## Additional Resources

- **[README.md](./README.md)** - Comprehensive documentation with best practices
- **[Code Generator Guide](../../docs/codegen/CODE_GENERATOR_GUIDE.md)** - Complete code generation documentation
- **[Mixin Injector](../../src/omninode_bridge/codegen/mixin_injector.py)** - Mixin generation implementation
- **[ONEX v2.0 Specification](../../docs/ONEX_V2_SPECIFICATION.md)** - Full ONEX v2.0 spec

## Support

For questions or issues:
1. Check the README.md in this directory
2. Review the code generator documentation
3. Examine the mixin_injector.py implementation
4. Review existing node implementations in `src/omninode_bridge/nodes/`

---

**Last Updated**: 2025-11-05
**ONEX Version**: v2.0
**Examples Version**: 1.0.0
