# Design Documents

This directory contains architectural design documents for omnibase_infra.

## PolicyRegistry Dependency Injection Integration

A comprehensive design for migrating PolicyRegistry from singleton pattern to container-based dependency injection using ModelOnexContainer from omnibase_core.

### Documents

1. **[POLICY_REGISTRY_DI_SUMMARY.md](POLICY_REGISTRY_DI_SUMMARY.md)** - Start here
   - Executive summary of the design
   - Current vs. target architecture
   - Implementation phases
   - Migration examples
   - Quick reference

2. **[POLICY_REGISTRY_DI_INTEGRATION.md](POLICY_REGISTRY_DI_INTEGRATION.md)** - Detailed design
   - Full architectural analysis
   - Container API documentation
   - Phase-by-phase implementation plan
   - Testing strategy
   - Documentation requirements
   - Risk mitigation

3. **[POLICY_REGISTRY_DI_EXAMPLES.md](POLICY_REGISTRY_DI_EXAMPLES.md)** - Code examples
   - Container bootstrap examples
   - Node integration patterns
   - Service class patterns
   - Testing patterns
   - Migration patterns
   - Configuration examples

4. **[POLICY_REGISTRY_DI_CHECKLIST.md](POLICY_REGISTRY_DI_CHECKLIST.md)** - Implementation checklist
   - Phase 1 tasks (ONM-812)
   - Phase 2-5 tasks (future)
   - Verification steps
   - Success metrics
   - Risk mitigation checklist

### Quick Navigation

**For developers implementing ONM-812**:
1. Read [POLICY_REGISTRY_DI_SUMMARY.md](POLICY_REGISTRY_DI_SUMMARY.md) for overview
2. Follow [POLICY_REGISTRY_DI_CHECKLIST.md](POLICY_REGISTRY_DI_CHECKLIST.md) Phase 1 tasks
3. Reference [POLICY_REGISTRY_DI_EXAMPLES.md](POLICY_REGISTRY_DI_EXAMPLES.md) for patterns

**For architects and reviewers**:
1. Read [POLICY_REGISTRY_DI_INTEGRATION.md](POLICY_REGISTRY_DI_INTEGRATION.md) for full design
2. Review benefits and risks analysis
3. Verify alignment with ONEX principles

**For users migrating code**:
1. See "Migration Examples" in [POLICY_REGISTRY_DI_SUMMARY.md](POLICY_REGISTRY_DI_SUMMARY.md)
2. Study patterns in [POLICY_REGISTRY_DI_EXAMPLES.md](POLICY_REGISTRY_DI_EXAMPLES.md)
3. Follow migration phases in [POLICY_REGISTRY_DI_CHECKLIST.md](POLICY_REGISTRY_DI_CHECKLIST.md)

### Key Concepts

**Current Architecture (Singleton)**:
```python
from omnibase_infra.runtime.policy_registry import get_policy_registry

registry = get_policy_registry()  # Module singleton
```

**Target Architecture (Container-Based DI)**:
```python
from omnibase_core.models.container.model_onex_container import ModelOnexContainer

class MyService:
    def __init__(self, container: ModelOnexContainer):
        self.policy_registry = container.resolve("policy_registry")
```

**Benefits**:
- ✅ ONEX-compliant dependency injection
- ✅ Better testability (isolated containers)
- ✅ Lifecycle management via container
- ✅ Configuration injection support
- ✅ Consistent with node pattern

### Implementation Status

- ✅ **Design Complete**: All design documents written
- ⏳ **Phase 1 (ONM-812)**: In progress - Container wiring implementation
- ⏳ **Phase 2**: Pending - Integration tests
- ⏳ **Phase 3**: Pending - Consumer migration
- ⏳ **Phase 4**: Pending - Deprecation warnings
- ⏳ **Phase 5**: Pending - Breaking change removal

### Related Documentation

- **CLAUDE.md** - Project rules and naming conventions
- **README.md** - Quick start and usage examples
- **MVP_EXECUTION_PLAN.md** - Infrastructure migration plan
- **CURRENT_NODE_ARCHITECTURE.md** - Node pattern documentation

### Questions and Feedback

For questions about this design:
1. Review the design documents thoroughly
2. Check examples for your specific use case
3. Consult ONEX principles in CLAUDE.md
4. Open discussion in PR review

### Changelog

- **2025-12-16**: Initial design documents created for ONM-812
  - POLICY_REGISTRY_DI_SUMMARY.md
  - POLICY_REGISTRY_DI_INTEGRATION.md
  - POLICY_REGISTRY_DI_EXAMPLES.md
  - POLICY_REGISTRY_DI_CHECKLIST.md
