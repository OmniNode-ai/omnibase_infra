# PolicyRegistry DI Integration - Implementation Checklist

## Phase 1: Container Support (ONM-812)

### 1. Create Container Wiring Module

- [ ] **File**: `src/omnibase_infra/runtime/container_wiring.py`
  - [ ] Import `ModelOnexContainer` from `omnibase_core`
  - [ ] Import `PolicyRegistry`, `ProtocolBindingRegistry`, `EventBusBindingRegistry`
  - [ ] Implement `wire_infrastructure_services(container)` function
    - [ ] Create and register `PolicyRegistry` instance
    - [ ] Create and register `ProtocolBindingRegistry` instance
    - [ ] Create and register `EventBusBindingRegistry` instance
    - [ ] Return summary dict with registered service keys
  - [ ] Implement `get_or_create_policy_registry(container)` helper
    - [ ] Try to resolve from container
    - [ ] Create and register if not found
    - [ ] Return PolicyRegistry instance
  - [ ] Add comprehensive docstrings with examples
  - [ ] Add logging for debugging
  - [ ] Define `__all__` exports

### 2. Update PolicyRegistry Module

- [ ] **File**: `src/omnibase_infra/runtime/policy_registry.py`
  - [ ] Add container-aware convenience functions:
    - [ ] `get_policy_class_from_container(container, policy_id, ...)`
    - [ ] `register_policy_in_container(container, policy_id, policy_class, ...)`
  - [ ] Update existing function docstrings:
    - [ ] Add `.. deprecated::` notice to `get_policy_registry()`
    - [ ] Add `.. deprecated::` notice to `get_policy_class()`
    - [ ] Add `.. deprecated::` notice to `register_policy()`
    - [ ] Show container-based alternative in each docstring
  - [ ] Update `PolicyRegistry` class docstring:
    - [ ] Add "Container Integration" section
    - [ ] Add example of container-based usage
    - [ ] Add example of resolution pattern
  - [ ] Update `__all__` to include new functions
  - [ ] Keep all existing code unchanged (no breaking changes)

### 3. Create Test Fixtures

- [ ] **File**: `tests/conftest.py`
  - [ ] Add `mock_container()` fixture
    - [ ] Mock `resolve()` method
    - [ ] Mock `register()` method
    - [ ] Mock `get_tool()` method
    - [ ] Mock `get_config()` method
  - [ ] Add `policy_registry_in_container()` fixture
    - [ ] Create PolicyRegistry instance
    - [ ] Configure mock container to return it
    - [ ] Return registry for test use

### 4. Add Unit Tests

- [ ] **File**: `tests/unit/runtime/test_container_wiring.py`
  - [ ] Test `wire_infrastructure_services()`
    - [ ] Verify all services registered
    - [ ] Verify summary dict format
    - [ ] Verify logging output
  - [ ] Test `get_or_create_policy_registry()`
    - [ ] Test with existing registry in container
    - [ ] Test with auto-creation
    - [ ] Verify singleton per container
  - [ ] Add docstrings to all tests

- [ ] **File**: `tests/unit/runtime/test_policy_registry.py`
  - [ ] Add container-based tests (new section)
    - [ ] Test policy registration via container
    - [ ] Test policy resolution via container
    - [ ] Test container-aware convenience functions
  - [ ] Keep all existing tests (no changes)

### 5. Update Documentation

- [ ] **File**: `CLAUDE.md`
  - [ ] Add "Container-Based Dependency Injection" section
    - [ ] Document `ModelOnexContainer` usage
    - [ ] Document container.resolve() pattern
    - [ ] Document constructor injection pattern
    - [ ] Document service registration pattern
  - [ ] Update "Registry Naming Conventions" section
    - [ ] Add container registration examples
    - [ ] Document service keys ("policy_registry", etc.)

- [ ] **File**: `README.md`
  - [ ] Add "Container Bootstrap" section to Quick Start
    - [ ] Show container creation
    - [ ] Show service wiring
    - [ ] Show service resolution
  - [ ] Update "Registering Policies" section
    - [ ] Show container-based pattern (preferred)
    - [ ] Show singleton pattern (deprecated)
    - [ ] Mark old pattern as deprecated

- [ ] **File**: `src/omnibase_infra/__init__.py`
  - [ ] Update module docstring
    - [ ] Add container bootstrap example
    - [ ] Update PolicyRegistry usage example
    - [ ] Show both old and new patterns

### 6. Verification

- [ ] **Run tests**:
  ```bash
  pytest tests/unit/runtime/test_container_wiring.py -v
  pytest tests/unit/runtime/test_policy_registry.py -v
  ```

- [ ] **Run type checking**:
  ```bash
  mypy src/omnibase_infra/runtime/container_wiring.py
  mypy src/omnibase_infra/runtime/policy_registry.py
  ```

- [ ] **Run linting**:
  ```bash
  ruff check src/omnibase_infra/runtime/container_wiring.py
  ruff check src/omnibase_infra/runtime/policy_registry.py
  ```

- [ ] **Verify backwards compatibility**:
  - [ ] All existing tests pass unchanged
  - [ ] Old singleton pattern still works
  - [ ] No breaking changes to public API

## Phase 2: Integration Tests (Next Sprint)

- [ ] **File**: `tests/integration/test_policy_registry_container_integration.py`
  - [ ] Test with real ModelOnexContainer
  - [ ] Test full lifecycle (bootstrap → register → resolve → use)
  - [ ] Test multiple containers (isolation)
  - [ ] Test container cleanup

- [ ] **File**: `tests/integration/test_node_with_policy_registry.py`
  - [ ] Create example node using PolicyRegistry via DI
  - [ ] Test node initialization with container
  - [ ] Test policy resolution in node
  - [ ] Test policy usage in node execution

## Phase 3: Consumer Migration (Future Sprints)

### Nodes to Migrate

- [ ] Identify all nodes that use policies
- [ ] Update each node to accept container in `__init__`
- [ ] Update each node to resolve PolicyRegistry from container
- [ ] Update each node's tests to use container fixtures

### Services to Migrate

- [ ] Identify all services that use PolicyRegistry
- [ ] Update each service to accept container in `__init__`
- [ ] Update each service to resolve PolicyRegistry from container
- [ ] Update each service's tests to use container fixtures

### Utilities to Migrate

- [ ] Identify all utilities that use `get_policy_registry()`
- [ ] Refactor to accept container or registry as parameter
- [ ] Update tests to inject registry
- [ ] Add deprecation notices to old functions

## Phase 4: Deprecation Warnings (Future Sprint)

- [ ] **File**: `src/omnibase_infra/runtime/policy_registry.py`
  - [ ] Add `warnings.warn()` to `get_policy_registry()`
  - [ ] Add `warnings.warn()` to `get_policy_class()`
  - [ ] Add `warnings.warn()` to `register_policy()`
  - [ ] Configure warnings to show once per call site
  - [ ] Document deprecation in release notes

- [ ] **Update tests**:
  - [ ] Add tests for deprecation warnings
  - [ ] Use `pytest.warns()` to verify warnings
  - [ ] Verify warning messages are clear

## Phase 5: Removal (Breaking Change - Future Release)

- [ ] **File**: `src/omnibase_infra/runtime/policy_registry.py`
  - [ ] Remove `_policy_registry` module variable
  - [ ] Remove `_singleton_lock`
  - [ ] Remove `get_policy_registry()` function
  - [ ] Remove `get_policy_class()` function
  - [ ] Remove `register_policy()` function
  - [ ] Update `__all__` exports
  - [ ] Update module docstring

- [ ] **Update all consumers**:
  - [ ] Verify all code uses container-based DI
  - [ ] Remove any remaining singleton usage
  - [ ] Update all documentation
  - [ ] Update all examples

- [ ] **Breaking change documentation**:
  - [ ] Add to CHANGELOG.md
  - [ ] Add migration guide
  - [ ] Update semantic version (major bump)

## Integration Checkpoints

### After Phase 1 (ONM-812)

- [ ] PR review completed
- [ ] All tests passing
- [ ] Documentation updated
- [ ] Zero breaking changes confirmed
- [ ] Container wiring working
- [ ] Examples tested

### After Phase 2

- [ ] Integration tests passing
- [ ] Real ModelOnexContainer tested
- [ ] Node integration verified
- [ ] Performance validated

### After Phase 3

- [ ] All consumers migrated
- [ ] Old singleton pattern usage eliminated
- [ ] Test coverage maintained
- [ ] Performance benchmarks passed

### After Phase 4

- [ ] Deprecation warnings in place
- [ ] Users notified of deprecation
- [ ] Migration guide published
- [ ] Timeline communicated

### After Phase 5

- [ ] Breaking changes merged
- [ ] Major version released
- [ ] Singleton pattern removed
- [ ] All tests updated
- [ ] Documentation current

## Risk Mitigation Checklist

- [ ] **Backwards compatibility verified**:
  - [ ] Existing tests pass without changes
  - [ ] Old singleton pattern still works
  - [ ] No changes to public API signatures

- [ ] **Container API verified**:
  - [ ] ModelOnexContainer imports successfully
  - [ ] Required methods available (resolve, register)
  - [ ] Type checking passes

- [ ] **Test isolation verified**:
  - [ ] Container fixtures isolate tests
  - [ ] No cross-test contamination
  - [ ] Tests can run in parallel

- [ ] **Documentation complete**:
  - [ ] All patterns documented
  - [ ] Examples tested and working
  - [ ] Migration path clear

## Success Metrics

- [ ] **Phase 1 Complete**:
  - [ ] Container wiring module created
  - [ ] Tests passing (100% of existing + new)
  - [ ] Type checking passing
  - [ ] Linting passing
  - [ ] Documentation updated
  - [ ] PR approved and merged

- [ ] **Full Migration Complete** (All Phases):
  - [ ] All consumers using container-based DI
  - [ ] Singleton pattern removed
  - [ ] Test coverage maintained or improved
  - [ ] Performance benchmarks met
  - [ ] Zero production incidents

## Notes

- Keep each phase small and independently deliverable
- Maintain backwards compatibility until Phase 5
- Test thoroughly at each phase
- Document as you go
- Get feedback early and often

## Questions to Resolve

1. **ModelOnexContainer API**: Confirm exact method signatures
2. **Container lifecycle**: Who owns container creation?
3. **Configuration**: Does PolicyRegistry need config from container?
4. **Other registries**: Migrate ProtocolBindingRegistry/EventBusBindingRegistry?
5. **Timeline**: When is next major release for breaking changes?

## References

- Design document: `docs/design/POLICY_REGISTRY_DI_INTEGRATION.md`
- Code examples: `docs/design/POLICY_REGISTRY_DI_EXAMPLES.md`
- Summary: `docs/design/POLICY_REGISTRY_DI_SUMMARY.md`
- Current implementation: `src/omnibase_infra/runtime/policy_registry.py`
