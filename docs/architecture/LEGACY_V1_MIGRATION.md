# Legacy v1_0_0 Directory Structure Migration Plan

**Document Version**: 1.0.0
**Status**: Active
**Created**: 2025-12-22
**Related Ticket**: H1 (Legacy Component Refactor Plan)
**Related Design Doc**: `docs/design/ONEX_RUNTIME_REGISTRATION_TICKET_PLAN.md`

---

## Executive Summary

This document describes the migration from legacy versioned directory structures (`nodes/<name>/v1_0_0/`) to the canonical flat structure (`nodes/<name>/`). Versioning in ONEX is **logical** (via `contract_version` fields), not **structural** (via directory hierarchy).

**Current Status**: The omnibase_infra4 codebase has already adopted the flat structure. No active `v1_0_0` directories exist in the source tree. This document serves as:
1. Documentation of the design decision
2. Guide for any remaining migration or new contributors
3. Reference for understanding why versioned directories are prohibited

---

## Background

### Why Versioned Directories Are Problematic

Versioned directory structures like `nodes/<name>/v1_0_0/` create several issues:

1. **Import Path Fragility**: Changing versions requires updating all import paths throughout the codebase
2. **Cognitive Overhead**: Developers must remember which version directory to use
3. **Merge Conflicts**: Parallel development on different versions increases merge complexity
4. **Deployment Complexity**: Multiple versions in the directory tree complicate deployment and dependency resolution
5. **Anti-Pattern**: Directory structure should reflect logical organization, not version history

### The ONEX Approach: Logical Versioning

ONEX uses semantic versioning through contract metadata, not directory structure:

```yaml
# contract.yaml - Canonical versioning approach
meta:
  contract_version: "1.0.0"  # Contract interface version
  node_version: "1.2.3"      # Implementation version
```

This approach provides:
- **Version tracking** without directory changes
- **Clear separation** of contract (interface) vs implementation versions
- **Semantic meaning** (MAJOR.MINOR.PATCH) for compatibility assessment
- **Git history** as the authoritative version timeline

---

## Current State (omnibase_infra4)

### Inventory of Node Directories

The codebase uses the **flat structure** exclusively:

```
src/omnibase_infra/nodes/
├── __init__.py
├── node_registration_orchestrator/
│   ├── __init__.py
│   ├── contract.yaml
│   ├── node.py
│   ├── models/
│   │   ├── model_orchestrator_input.py
│   │   ├── model_orchestrator_output.py
│   │   └── ...
│   └── protocols.py
├── node_registry_effect/
│   └── __init__.py
└── reducers/
    ├── __init__.py
    ├── registration_reducer.py
    └── models/
        ├── model_registration_state.py
        └── model_registration_confirmation.py
```

### Verification

No `v1_0_0` directories exist in the source tree:

```bash
# Verification command
find src/omnibase_infra -type d -name "v1_0_0" 2>/dev/null
# Result: (empty - no matches)
```

---

## Target Structure (Reference)

For new nodes or migration, use this canonical structure:

```
nodes/<node_name>/
├── contract.yaml          # ONEX contract with version metadata
├── node.py                # Node<Name><Type> implementation class
├── __init__.py            # Public exports
├── models/                # Node-specific Pydantic models
│   ├── __init__.py
│   ├── model_<name>_input.py
│   └── model_<name>_output.py
├── protocols.py           # Node-specific protocols (if needed)
└── registry/              # Optional: node-specific registry
    └── registry_infra_<node_name>.py
```

### Contract.yaml Requirements

Every node must have a `contract.yaml` with version metadata:

```yaml
# Minimal contract.yaml structure
name: node-example-effect
type: EFFECT
description: Example effect node for external I/O

meta:
  contract_version: "1.0.0"
  node_version: "1.0.0"

input:
  model: ModelExampleInput
  module: omnibase_infra.nodes.node_example.models.model_example_input

output:
  model: ModelExampleOutput
  module: omnibase_infra.nodes.node_example.models.model_example_output

dependencies: []
```

---

## Migration Steps (If Needed)

If legacy `v1_0_0` directories are discovered in any branch or fork:

### Phase 1: Inventory and Analysis

1. **Identify all v1_0_0 directories**:
   ```bash
   find . -type d -name "v1_0_0" -o -name "v1" -o -name "v2" | grep -v ".git"
   ```

2. **Map dependencies**: Find all imports referencing versioned paths:
   ```bash
   grep -r "v1_0_0" src/ tests/ --include="*.py"
   ```

3. **Document each directory**: Create migration ticket with:
   - Current path
   - Target path
   - Dependent files
   - Risk assessment

### Phase 2: Migration Execution

1. **Move files to flat structure**:
   ```bash
   # Example: Move from versioned to flat
   git mv nodes/example_node/v1_0_0/* nodes/example_node/
   rmdir nodes/example_node/v1_0_0
   ```

2. **Update all import paths**:
   ```python
   # Before
   from omnibase_infra.nodes.example_node.v1_0_0.node import NodeExample

   # After
   from omnibase_infra.nodes.example_node.node import NodeExample
   ```

3. **Update contract.yaml** to reflect version in metadata, not path:
   ```yaml
   meta:
     contract_version: "1.0.0"  # The "v1_0_0" version is now here
   ```

4. **Run tests and linting**:
   ```bash
   poetry run pytest
   poetry run ruff check .
   poetry run mypy .
   ```

### Phase 3: Validation

1. **Verify no versioned directories remain**:
   ```bash
   find . -type d -regex ".*v[0-9].*" | grep -v ".git"
   ```

2. **Verify all imports resolve**:
   ```bash
   poetry run python -c "import omnibase_infra"
   ```

3. **Verify CI passes**:
   - All unit tests pass
   - All integration tests pass
   - Linting clean
   - Type checking clean

---

## Enforcement

### CI Validation

The CI pipeline includes a check for versioned directories:

```yaml
# .github/workflows/validate.yml (example)
- name: Check for versioned directories
  run: |
    if find src/ -type d -regex ".*v[0-9]_[0-9]_[0-9].*" | grep -q .; then
      echo "ERROR: Versioned directories found. Use contract_version instead."
      exit 1
    fi
```

### Pre-commit Hook (Optional)

Add to `.pre-commit-config.yaml`:

```yaml
- repo: local
  hooks:
    - id: no-versioned-dirs
      name: No versioned directories
      entry: bash -c 'if find src/ -type d -regex ".*v[0-9]_[0-9]_[0-9].*" | grep -q .; then exit 1; fi'
      language: system
      pass_filenames: false
```

---

## FAQ

### Q: Why not use versioned directories for major breaking changes?

A: Breaking changes are handled through:
1. **Semantic versioning** in `contract_version` (MAJOR bump)
2. **Git branches/tags** for version history
3. **Migration scripts** for data/schema changes
4. **NO backwards compatibility policy** (remove old patterns immediately)

### Q: What if I need to maintain two versions simultaneously?

A: This should be extremely rare in ONEX due to the "no backwards compatibility" policy. If absolutely required:
1. Use feature flags, not directory versions
2. Use Git branches for development
3. Deploy one version at a time
4. Coordinate cutover with consumers

### Q: How do I version my node's contract?

A: Update the `contract_version` field in `contract.yaml`:
- **MAJOR**: Breaking changes to the contract interface
- **MINOR**: Backward-compatible new features
- **PATCH**: Backward-compatible bug fixes

### Q: Where is the legacy v1_0_0 reference in CLAUDE.md coming from?

A: The CLAUDE.md documentation references legacy patterns that existed in earlier architectural phases. These references serve as:
1. Historical context for the design decision
2. Warning to developers not to reintroduce the pattern
3. Reference to the H1 migration ticket for any remaining instances

---

## Cross-References

- **CLAUDE.md**: See "CRITICAL POLICY: NO VERSIONED DIRECTORIES" section
- **Ticket Plan**: `docs/design/ONEX_RUNTIME_REGISTRATION_TICKET_PLAN.md`, Section H1
- **Global Constraint #6**: "No versioned directories" in ticket plan
- **HANDOFF Document**: `docs/handoffs/HANDOFF_TWO_WAY_REGISTRATION_REFACTOR.md`, Phase 0-4 migration steps

---

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-22 | System | Initial document creation |
