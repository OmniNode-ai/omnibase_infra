# Local Convenience Classes (Temporary)

**Status**: Temporary workaround until omnibase_core v0.2.0+
**Created**: 2025-11-05
**Remove After**: omnibase_core v0.2.0+ is released

## Purpose

This directory contains convenience classes copied from omnibase_core source that are currently disabled in the released version (v0.1.0). These classes exist in the omnibase_core codebase but are not exported in the `__init__.py`.

## Available Classes

### ModelServiceOrchestrator

Pre-composed orchestrator node with standard production mixins:

```python
from omninode_bridge.utils.node_services import ModelServiceOrchestrator

class NodeMyOrchestrator(ModelServiceOrchestrator):
    """Production-ready orchestrator with automatic capabilities."""

    async def execute_orchestration(self, contract):
        # Health checks, event publishing, metrics - all automatic!
        await self.publish_event("workflow_started", {...})
        result = await self._coordinate_workflow(contract)
        return result
```

**Includes**:
- MixinNodeService (service lifecycle)
- NodeOrchestrator (workflow coordination)
- MixinHealthCheck (health monitoring)
- MixinEventBus (event publishing)
- MixinMetrics (performance tracking)

### ModelServiceReducer

Pre-composed reducer node with standard production mixins:

```python
from omninode_bridge.utils.node_services import ModelServiceReducer

class NodeMyReducer(ModelServiceReducer):
    """Production-ready reducer with automatic capabilities."""

    async def execute_reduction(self, contract):
        # Health checks, caching, metrics - all automatic!
        cached = await self.get_cached(cache_key)
        if cached:
            return cached

        result = await self._aggregate_data(contract)
        await self.set_cached(cache_key, result, ttl_seconds=300)
        return result
```

**Includes**:
- MixinNodeService (service lifecycle)
- NodeReducer (aggregation semantics)
- MixinHealthCheck (health monitoring)
- MixinCaching (result caching)
- MixinMetrics (performance tracking)

## Migration Plan

### When omnibase_core v0.2.0+ is Released

1. **Update pyproject.toml**:
   ```toml
   omnibase-core = "^0.2.0"
   ```

2. **Update all imports**:
   ```python
   # BEFORE (local)
   from omninode_bridge.utils.node_services import ModelServiceOrchestrator

   # AFTER (omnibase_core)
   from omnibase_core.models.nodes.node_services import ModelServiceOrchestrator
   ```

3. **Remove this directory**:
   ```bash
   rm -rf src/omninode_bridge/utils/node_services/
   ```

4. **Run tests** to verify everything still works

## Source Location

Copied from:
- `/Volumes/PRO-G40/Code/omnibase_core/src/omnibase_core/models/nodes/node_services/model_service_orchestrator.py`
- `/Volumes/PRO-G40/Code/omnibase_core/src/omnibase_core/models/nodes/node_services/model_service_reducer.py`

These files exist in omnibase_core but are disabled in `__init__.py` with comment:
```python
# NOTE: Available after Phase 3 restoration:
# from omnibase_core.models.nodes.node_services.model_service_orchestrator import ModelServiceOrchestrator
# from omnibase_core.models.nodes.node_services.model_service_reducer import ModelServiceReducer
```

## Verification

```bash
# Test local imports work
poetry run python -c "from omninode_bridge.utils.node_services import ModelServiceOrchestrator, ModelServiceReducer; print('✅ OK')"
```

## Notes

- These are **exact copies** from omnibase_core source
- No modifications made to the code
- Will be removed once official release includes them
- Track omnibase_core releases: https://github.com/your-org/omnibase_core/releases
