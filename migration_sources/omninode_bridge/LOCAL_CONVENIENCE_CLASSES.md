# Local Convenience Classes - Migration Notice

**Date**: 2025-11-05
**Status**: ✅ ACTIVE (Temporary workaround)

## Quick Reference

### Current Usage (Until omnibase_core v0.2.0+)

```python
# Use LOCAL copies
from omninode_bridge.utils.node_services import ModelServiceOrchestrator
from omninode_bridge.utils.node_services import ModelServiceReducer

class NodeMyOrchestrator(ModelServiceOrchestrator):
    pass

class NodeMyReducer(ModelServiceReducer):
    pass
```

### Future Usage (After omnibase_core v0.2.0+)

```python
# Use OMNIBASE_CORE exports
from omnibase_core.models.nodes.node_services import ModelServiceOrchestrator
from omnibase_core.models.nodes.node_services import ModelServiceReducer

class NodeMyOrchestrator(ModelServiceOrchestrator):
    pass

class NodeMyReducer(ModelServiceReducer):
    pass
```

## Why This Exists

The convenience classes `ModelServiceOrchestrator` and `ModelServiceReducer` exist in omnibase_core source code but are **intentionally disabled** in the released version (v0.1.0).

**Location in omnibase_core**: `src/omnibase_core/models/nodes/node_services/__init__.py`

**Current state**:
```python
# Lines 73-75: Commented out imports
# NOTE: Available after Phase 3 restoration:
# from omnibase_core.models.nodes.node_services.model_service_orchestrator import ModelServiceOrchestrator
# from omnibase_core.models.nodes.node_services.model_service_reducer import ModelServiceReducer

# Lines 77-80: Not exported
__all__ = [
    "ModelServiceEffect",      # ✅ Available
    "ModelServiceCompute",     # ✅ Available
    # ModelServiceOrchestrator  ❌ Missing
    # ModelServiceReducer       ❌ Missing
]
```

## What We Did

1. **Copied** the disabled classes from omnibase_core source to `src/omninode_bridge/utils/node_services/`
2. **Created** local module to export them
3. **Verified** they work with current omnibase_core v0.1.0

## Migration Checklist

When omnibase_core v0.2.0+ is released with these classes enabled:

- [ ] Update `pyproject.toml`: `omnibase-core = "^0.2.0"`
- [ ] Run `poetry update omnibase-core`
- [ ] Find all imports: `grep -r "from omninode_bridge.utils.node_services" src/`
- [ ] Replace with: `from omnibase_core.models.nodes.node_services`
- [ ] Remove local directory: `rm -rf src/omninode_bridge/utils/node_services/`
- [ ] Remove this file: `rm LOCAL_CONVENIENCE_CLASSES.md`
- [ ] Run full test suite: `pytest`
- [ ] Commit changes

## Files to Remove Later

```
src/omninode_bridge/utils/node_services/
├── __init__.py
├── model_service_orchestrator.py
├── model_service_reducer.py
└── README.md

LOCAL_CONVENIENCE_CLASSES.md (this file)
```

## Impact on Code Generation

The upgrade plan for CodeGenerationService now uses these local classes:

```python
# In code generation templates
from omninode_bridge.utils.node_services import ModelServiceOrchestrator

# Generated node
class Node{ServiceName}Orchestrator(ModelServiceOrchestrator):
    """Generated orchestrator with production mixins."""

    async def execute_orchestration(self, contract):
        # Automatically includes:
        # - MixinNodeService (service lifecycle)
        # - MixinHealthCheck (health monitoring)
        # - MixinEventBus (event publishing)
        # - MixinMetrics (performance tracking)
        pass
```

## Verification Commands

```bash
# Verify local classes work
poetry run python -c "
from omninode_bridge.utils.node_services import ModelServiceOrchestrator, ModelServiceReducer
print('✅ Local convenience classes available')
print(f'  Orchestrator: {ModelServiceOrchestrator.__mro__}')
print(f'  Reducer: {ModelServiceReducer.__mro__}')
"

# Check when to migrate (check omnibase_core version)
poetry show omnibase-core | grep version

# When version >= 0.2.0, migrate!
```

## Questions?

See `src/omninode_bridge/utils/node_services/README.md` for detailed migration instructions.
