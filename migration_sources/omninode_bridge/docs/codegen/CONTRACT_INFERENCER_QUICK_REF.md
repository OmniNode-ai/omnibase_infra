# ContractInferencer Quick Reference

**TL;DR**: Automated ONEX v2.0 contract generation from node.py files using AST + LLM

## Installation

```bash
# Ensure you're in the poetry environment
poetry install

# Set up environment
echo "ZAI_API_KEY=your_api_key" >> .env  # pragma: allowlist secret
```

## Basic Usage

```python
import asyncio
from omninode_bridge.codegen.contract_inferencer import ContractInferencer

async def generate_contract():
    inferencer = ContractInferencer(enable_llm=True)

    contract_yaml = await inferencer.infer_from_node(
        node_path="path/to/node.py",
        output_path="path/to/contract.yaml"  # Optional
    )

    await inferencer.cleanup()
    return contract_yaml

# Run it
result = asyncio.run(generate_contract())
```

## Command Line

```bash
# Test with llm_effect node
poetry run python scripts/test_contract_inferencer.py

# Run examples
poetry run python examples/contract_inferencer_example.py
poetry run python examples/contract_inferencer_example.py --example 1
poetry run python examples/contract_inferencer_example.py --no-llm
```

## Options

```python
# Enable/disable LLM
inferencer = ContractInferencer(enable_llm=True)  # Default
inferencer = ContractInferencer(enable_llm=False)  # Use defaults

# Choose LLM tier
from omninode_bridge.nodes.llm_effect.v1_0_0.models.enum_llm_tier import EnumLLMTier

inferencer = ContractInferencer(
    enable_llm=True,
    llm_tier=EnumLLMTier.CLOUD_FAST  # Default (fast, cheap)
)
inferencer = ContractInferencer(
    enable_llm=True,
    llm_tier=EnumLLMTier.CLOUD_PREMIUM  # Premium (better, slower)
)
```

## What It Detects

### From AST Parsing
- Node name and type (effect/compute/reducer/orchestrator)
- Base class (NodeEffect, NodeCompute, etc.)
- Mixins in inheritance chain
- Method signatures
- Docstrings
- Version from directory structure (v1_0_0 → 1.0.0)

### I/O Operations
| Import | Detected Operation |
|--------|-------------------|
| `httpx`, `requests` | `http_request` |
| `asyncpg`, `sqlalchemy` | `database_query` |
| `aiokafka`, `redis` | `message_queue` |
| `open()`, `Path()` | `file_io` |

### Mixin Configurations (via LLM)
- `MixinMetrics`: Latency, throughput, percentiles
- `MixinHealthCheck`: Check intervals, timeouts, components
- `MixinCaching`: TTL, size, eviction policy
- (Others from MIXIN_CATALOG)

## Generated Contract Structure

```yaml
schema_version: v2.0.0
name: example_node
version: {major: 1, minor: 0, patch: 0}
node_type: effect

capabilities: [...]  # From I/O operations + mixins

mixins:  # If detected in code
  - name: MixinMetrics
    enabled: true
    config: {...}  # LLM-inferred

advanced_features:
  circuit_breaker: {...}  # All nodes
  retry_policy: {...}  # All nodes
  observability: {...}  # All nodes
  dead_letter_queue: {...}  # Effect nodes only
  transactions: {...}  # Database operations only
  security_validation: {...}  # All nodes

subcontracts:
  effect:  # Or compute/reducer/orchestrator
    operations: [initialize, cleanup, execute_effect]
```

## Error Handling

```python
from omnibase_core import ModelOnexError

try:
    contract_yaml = await inferencer.infer_from_node(node_path)
except ModelOnexError as e:
    print(f"Error: {e.message}")
    print(f"Details: {e.details}")
    # Handle specific error codes
    if e.error_code == EnumCoreErrorCode.INVALID_INPUT:
        # Invalid file or missing node class
        pass
    elif e.error_code == EnumCoreErrorCode.EXECUTION_ERROR:
        # LLM or parsing failure
        pass
```

## Integration with Code Generation

```python
# Step 1: Generate node (existing workflow)
artifacts = await template_engine.generate_artifacts(requirements)
enhanced = await business_logic_generator.enhance_artifacts(artifacts)

# Step 2: Generate contract (NEW)
inferencer = ContractInferencer(enable_llm=True)
contract_yaml = await inferencer.infer_from_node(
    node_path=enhanced.enhanced_node_file_path,
    output_path=output_dir / "contract.yaml"
)
await inferencer.cleanup()

# Step 3: Validate
parser = YAMLContractParser()
contract = parser.parse_contract_file(output_dir / "contract.yaml")
assert contract.is_valid
```

## Batch Processing

```python
from pathlib import Path

async def generate_all_contracts():
    inferencer = ContractInferencer(enable_llm=True)

    # Find all node files
    nodes_dir = Path("src/omninode_bridge/nodes")
    node_files = nodes_dir.rglob("*/v*/node.py")

    for node_file in node_files:
        contract_path = node_file.parent / "contract.yaml"

        try:
            await inferencer.infer_from_node(
                node_path=node_file,
                output_path=contract_path
            )
            print(f"✓ {node_file.parent.parent.name}")
        except Exception as e:
            print(f"✗ {node_file.parent.parent.name}: {e}")

    await inferencer.cleanup()

asyncio.run(generate_all_contracts())
```

## Performance

| Operation | Time | Notes |
|-----------|------|-------|
| No LLM | ~25ms | AST parsing + generation |
| With LLM (2 mixins) | ~2-6s | Includes API latency |

## Environment Variables

```bash
# Required for LLM inference
export ZAI_API_KEY="your_api_key"  # pragma: allowlist secret

# Optional
export ZAI_ENDPOINT="https://api.z.ai/api/anthropic"  # Default
```

## Common Issues

### "ZAI_API_KEY not set"
```bash
# Set in .env file
echo "ZAI_API_KEY=your_api_key" >> .env  # pragma: allowlist secret

# Or disable LLM
inferencer = ContractInferencer(enable_llm=False)
```

### "No module named 'omnibase_core'"
```bash
# Run with poetry
poetry run python your_script.py
```

### "No node class found"
Check that your node.py has a class inheriting from NodeEffect/Compute/etc.

### LLM inference fails
- Falls back to default configs automatically
- Check logs for warnings
- Verify ZAI_API_KEY is valid

## Files

| File | Purpose |
|------|---------|
| `src/omninode_bridge/codegen/contract_inferencer.py` | Implementation |
| `scripts/test_contract_inferencer.py` | Test script |
| `examples/contract_inferencer_example.py` | Usage examples |
| `docs/codegen/CONTRACT_INFERENCER.md` | Full documentation |

## API Reference

### ContractInferencer

```python
class ContractInferencer:
    def __init__(
        enable_llm: bool = True,
        llm_tier: EnumLLMTier = EnumLLMTier.CLOUD_FAST
    )

    async def infer_from_node(
        node_path: str | Path,
        output_path: Optional[str | Path] = None
    ) -> str:
        """Generate contract YAML from node.py"""

    async def cleanup() -> None:
        """Cleanup LLM resources"""
```

### Data Models

```python
@dataclass
class ModelNodeAnalysis:
    node_name: str
    node_type: str
    base_class: str
    mixins_detected: list[str]
    imports: dict[str, list[str]]
    methods: list[str]
    docstring: Optional[str]
    version: str
    io_operations: list[str]
    file_path: str

@dataclass
class ModelMixinConfigInference:
    mixin_name: str
    confidence: float
    inferred_config: dict[str, Any]
    reasoning: str
```

## Tips & Tricks

1. **Reuse inferencer instance** for batch processing (saves initialization time)
2. **Use enable_llm=False** for fast default configs during development
3. **Check generated contract** with YAMLContractParser before deployment
4. **Cache LLM results** for common patterns (future enhancement)
5. **Run with --no-llm** flag in CI/CD to avoid API costs

## Next Steps

1. Read full documentation: `docs/codegen/CONTRACT_INFERENCER.md`
2. Run test script: `poetry run python scripts/test_contract_inferencer.py`
3. Try examples: `poetry run python examples/contract_inferencer_example.py`
4. Integrate into your workflow

---

**Need help?** See `docs/codegen/CONTRACT_INFERENCER.md` for detailed documentation.
