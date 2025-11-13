# Quick File Distribution Guide

## ✅ Problem Fixed

Generated code is now **automatically distributed** to multiple target directories:
- `generated_nodes/{service_name}/` (base)
- `generated_nodes/{service_name}_final/` (final version)
- `generated_nodes/{service_name}_llm/` (LLM version)

## 🚀 Usage

### Automatic (Recommended)

Just use `CodeGenerationService` - distribution happens automatically:

```python
from pathlib import Path
from omninode_bridge.codegen.service import CodeGenerationService

service = CodeGenerationService()
result = await service.generate_node(
    requirements=requirements,
    output_directory=Path("generated_nodes/my_service"),
    enable_llm=True,
)

# Files are now in:
# - generated_nodes/my_service/node.py
# - generated_nodes/my_service_final/node.py
# - generated_nodes/my_service_llm/node.py
```

### Test It

```bash
poetry run python test_file_distribution.py
```

## 📁 What Gets Distributed

All generated files:
- `node.py` (implementation)
- `__init__.py` (module init)
- `contract.yaml` (node contract)
- `README.md` (documentation)
- `models/` (data models)
- `tests/` (test files)

## ✨ Key Benefits

- **No manual copying** - automatic distribution
- **Consistency guaranteed** - all versions identical
- **Works with all strategies** - jinja2, template_loading, hybrid
- **Complete structure** - all files and directories copied

## 🔍 Verification

Check files exist and are identical:

```bash
# Check files exist
ls -la generated_nodes/*/node.py

# Verify they're identical
diff generated_nodes/my_service/node.py \
     generated_nodes/my_service_final/node.py
```

## 📝 Modified Files

- `src/omninode_bridge/codegen/service.py` - Added distribution logic
- `test_file_distribution.py` - Test script
- `FILE_DISTRIBUTION_FIX.md` - Detailed documentation

## 💡 Note

Distribution only happens when `output_directory` is specified in `generate_node()`.
