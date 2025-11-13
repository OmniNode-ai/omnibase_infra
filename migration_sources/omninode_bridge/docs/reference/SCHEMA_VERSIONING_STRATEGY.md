# Contract Schema Versioning Strategy

**Version:** 1.0.0
**Last Updated:** 2025-11-04
**Status:** Production-Ready

## Overview

This document defines the versioning strategy for ONEX v2.0 contract schemas, ensuring smooth evolution while maintaining backward compatibility.

## Versioning Principles

### Semantic Versioning

Contract schemas follow semantic versioning: `vMAJOR.MINOR.PATCH`

- **MAJOR**: Breaking changes (incompatible with previous version)
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Backward Compatibility Rules

1. **MUST**: All MINOR and PATCH versions are backward compatible
2. **MUST**: Existing contracts work without modification in newer MINOR versions
3. **SHOULD**: Deprecated features remain functional for at least 2 MAJOR versions
4. **SHOULD**: Migration guides provided for all MAJOR version changes

## Schema Evolution Path

### Version Timeline

```
v1.0.0 (Jan 2024)
  ├─ Initial ONEX v2.0 contract schema
  ├─ Core fields: name, version, node_type, description
  ├─ Subcontracts: FSM, EventType, Aggregation, etc.
  └─ Basic error_handling section

v2.0.0 (Nov 2025) ← CURRENT
  ├─ NEW: schema_version field
  ├─ NEW: mixins array
  ├─ NEW: advanced_features object
  ├─ DEPRECATED: error_handling (use advanced_features instead)
  └─ BACKWARD COMPATIBLE: All v1.0.0 contracts valid

v2.1.0 (Future - Planned Q1 2026)
  ├─ NEW: Additional mixin types
  ├─ NEW: Dependency graph validation
  ├─ ENHANCED: Better error messages
  └─ BACKWARD COMPATIBLE: All v2.0.0 contracts valid

v3.0.0 (Future - Planned Q3 2026)
  ├─ BREAKING: Remove deprecated error_handling
  ├─ BREAKING: Require schema_version field
  ├─ NEW: Contract composition features
  └─ MIGRATION GUIDE: Automated migration tool provided
```

## Schema Version Field

### Format

```yaml
schema_version: "v2.0.0"
```

### Validation Rules

1. **Optional in v2.0.0** - Maintains backward compatibility
2. **Recommended for new contracts** - Explicit version tracking
3. **Required in v3.0.0** - Future enforcement
4. **Must match pattern** - `^v[0-9]+\.[0-9]+\.[0-9]+$`

### Version Resolution

```python
def resolve_schema_version(contract: dict) -> str:
    """Resolve contract schema version."""
    # Explicit version
    if "schema_version" in contract:
        return contract["schema_version"]

    # Infer from features
    if "mixins" in contract or "advanced_features" in contract:
        return "v2.0.0"

    # Default to v1.0.0
    return "v1.0.0"
```

## Breaking vs Non-Breaking Changes

### Non-Breaking Changes (MINOR/PATCH)

✅ **Adding new optional fields**
```yaml
# v2.0.0 → v2.1.0
mixins:  # NEW optional field
  - name: "MixinHealthCheck"
```

✅ **Adding new enum values**
```yaml
# v2.0.0 → v2.1.0
node_type: "hybrid"  # NEW value alongside existing effect/compute/reducer/orchestrator
```

✅ **Relaxing validation constraints**
```yaml
# v2.0.0: min=1000, max=10000
# v2.1.0: min=500, max=20000  # Wider range
```

✅ **Adding new mixin types**
```yaml
# v2.1.0
mixins:
  - name: "MixinCaching"  # NEW mixin type
```

✅ **Deprecating fields (with warning)**
```yaml
# v2.0.0
error_handling:  # DEPRECATED but still works
  retry_policy: {}
```

### Breaking Changes (MAJOR)

❌ **Removing required fields**
```yaml
# v2.x: description required
# v3.0: description optional  # BREAKING
```

❌ **Removing deprecated fields**
```yaml
# v2.x: error_handling deprecated but functional
# v3.0: error_handling removed  # BREAKING
```

❌ **Tightening validation constraints**
```yaml
# v2.x: check_interval_ms min=1000
# v3.0: check_interval_ms min=5000  # BREAKING (rejects previously valid values)
```

❌ **Changing field types**
```yaml
# v2.x: timeout_seconds (number)
# v3.0: timeout_ms (integer)  # BREAKING (different field)
```

❌ **Requiring previously optional fields**
```yaml
# v2.x: schema_version optional
# v3.0: schema_version required  # BREAKING
```

## Deprecation Process

### Phase 1: Deprecation Announcement

**Actions:**
1. Mark field as deprecated in schema
2. Add deprecation warning to documentation
3. Update validation to emit warnings (not errors)
4. Provide migration examples

**Example:**
```json
{
  "error_handling": {
    "type": "object",
    "deprecated": true,
    "description": "DEPRECATED: Use advanced_features instead. Will be removed in v3.0.0"
  }
}
```

### Phase 2: Deprecation Period

**Duration:** Minimum 2 MAJOR versions or 12 months (whichever is longer)

**Actions:**
1. Document in CHANGELOG
2. Emit warnings during validation
3. Track usage metrics
4. Provide automated migration tools
5. Update all examples to use new approach

### Phase 3: Removal

**Actions:**
1. Remove deprecated field from schema
2. Update validator to reject contracts with deprecated field
3. Provide automated migration script
4. Document breaking change in migration guide

**Example Migration Script:**
```python
def migrate_v2_to_v3(contract: dict) -> dict:
    """Migrate v2.x contract to v3.0."""
    # Remove deprecated error_handling
    if "error_handling" in contract:
        # Move to advanced_features
        if "advanced_features" not in contract:
            contract["advanced_features"] = {}

        # Migrate retry_policy
        if "retry_policy" in contract["error_handling"]:
            contract["advanced_features"]["retry_policy"] = \
                contract["error_handling"]["retry_policy"]

        # Remove deprecated section
        del contract["error_handling"]

    # Add required schema_version
    contract["schema_version"] = "v3.0.0"

    return contract
```

## Validation Strategy

### Multi-Version Validation

Validator supports multiple schema versions:

```python
class ContractValidator:
    """Multi-version contract validator."""

    SCHEMAS = {
        "v1.0.0": "schemas/contract_schema_v1.json",
        "v2.0.0": "schemas/contract_schema_v2.json",
        "v2.1.0": "schemas/contract_schema_v2_1.json",
    }

    def validate(self, contract: dict) -> ValidationResult:
        """Validate contract against appropriate schema version."""
        version = self.resolve_version(contract)
        schema = self.load_schema(version)

        # Validate
        errors = jsonschema.validate(contract, schema)

        # Check for deprecations
        warnings = self.check_deprecations(contract, version)

        return ValidationResult(
            valid=(len(errors) == 0),
            errors=errors,
            warnings=warnings,
            schema_version=version
        )
```

### Validation Levels

1. **ERROR**: Contract violates schema (rejected)
2. **WARNING**: Contract uses deprecated features (accepted with warning)
3. **INFO**: Suggestions for improvement (accepted)

**Example Output:**
```
✅ Contract validation successful (v2.0.0)

⚠️  Warnings:
  - error_handling is deprecated, use advanced_features instead
  - Consider adding schema_version field for explicit versioning

ℹ️  Suggestions:
  - Add MixinHealthCheck for production nodes
  - Configure circuit_breaker in advanced_features
```

## Migration Support

### Automated Migration Tools

#### CLI Migration Tool

```bash
# Migrate contract to latest version
python -m omninode_bridge.codegen.migrate_contract \
  --input contract_v1.yaml \
  --output contract_v2.yaml \
  --target-version v2.0.0

# Migrate all contracts in directory
python -m omninode_bridge.codegen.migrate_contracts \
  --directory src/nodes \
  --target-version v2.0.0 \
  --in-place
```

#### Python API

```python
from omninode_bridge.codegen.migration import ContractMigrator

migrator = ContractMigrator()

# Migrate single contract
contract_v2 = migrator.migrate(
    contract_v1,
    from_version="v1.0.0",
    to_version="v2.0.0"
)

# Batch migration
results = migrator.migrate_directory(
    "src/nodes",
    target_version="v2.0.0",
    dry_run=False
)
```

### Migration Reports

```
Contract Migration Report
========================

Source Version: v1.0.0
Target Version: v2.0.0
Contracts Processed: 47

✅ Successful: 45
⚠️  With Warnings: 2
❌ Failed: 0

Changes Applied:
  - Added schema_version field: 45 contracts
  - Migrated error_handling → advanced_features: 12 contracts
  - Added recommended mixins: 0 contracts (manual review required)

Warnings:
  - nodes/custom_effect/v1_0_0/contract.yaml: Custom error_handling config may need manual review
  - nodes/legacy_compute/v1_0_0/contract.yaml: Unusual configuration pattern detected

Manual Review Required: 2 contracts
```

## Schema Storage and Distribution

### Repository Structure

```
src/omninode_bridge/codegen/schemas/
├── v1/
│   └── contract_schema_v1.json
├── v2/
│   ├── contract_schema_v2.json
│   ├── mixins_schema.json
│   └── advanced_features_schema.json
├── v2.1/  (future)
│   └── ...
└── current -> v2  (symlink to latest stable)
```

### Schema URIs

Each schema has a unique URI:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://omninode.dev/schemas/contract/v2.0.0",
  "title": "ONEX v2.0 Contract Schema"
}
```

### Schema Registry

Schemas published to registry for external validation:

```bash
# Download specific version
curl https://schemas.omninode.dev/contract/v2.0.0 > contract_schema_v2.json

# Download latest
curl https://schemas.omninode.dev/contract/latest > contract_schema_latest.json
```

## Version Compatibility Matrix

| Contract Version | Validator v1.x | Validator v2.x | Validator v3.x (future) |
|------------------|---------------|---------------|------------------------|
| **v1.0.0** | ✅ Full support | ✅ Full support | ⚠️ Deprecated warnings |
| **v2.0.0** | ❌ Not supported | ✅ Full support | ✅ Full support |
| **v2.1.0** | ❌ Not supported | ✅ Full support | ✅ Full support |
| **v3.0.0** | ❌ Not supported | ❌ Not supported | ✅ Full support |

## Best Practices

### For Schema Maintainers

✅ **DO:**
- Maintain backward compatibility in MINOR versions
- Provide migration tools for MAJOR versions
- Document all breaking changes clearly
- Test migrations on real-world contracts
- Keep deprecation period for at least 12 months

❌ **DON'T:**
- Break compatibility in MINOR/PATCH versions
- Remove fields without deprecation period
- Change validation rules unexpectedly
- Release MAJOR versions too frequently

### For Contract Authors

✅ **DO:**
- Add `schema_version` to all new contracts
- Migrate from deprecated features proactively
- Test contracts with latest validator
- Review migration guides before upgrading
- Keep contracts in version control

❌ **DON'T:**
- Ignore deprecation warnings
- Skip validation before deployment
- Mix different schema versions inconsistently
- Modify generated code instead of contracts

## Future Considerations

### Planned Features

**v2.1.0 (Q1 2026):**
- Additional mixin types (MixinCaching, MixinRateLimiting)
- Dependency graph validation
- Enhanced error messages
- Contract composition support

**v3.0.0 (Q3 2026):**
- Remove deprecated `error_handling`
- Require `schema_version` field
- Contract inheritance
- Multi-contract validation (cross-references)

### Research Topics

- **Contract Templates**: Reusable contract patterns
- **Contract Composition**: Compose contracts from multiple sources
- **Contract Testing**: Automated contract compliance testing
- **Schema Evolution ML**: Predict breaking changes before release

## Summary

The contract schema versioning strategy ensures:

✅ **Stability** - Backward compatibility in MINOR versions
✅ **Evolution** - Clear path for new features
✅ **Safety** - Automated migration tools
✅ **Clarity** - Explicit version tracking
✅ **Support** - Comprehensive documentation

**Key Takeaways:**
1. Use semantic versioning (vMAJOR.MINOR.PATCH)
2. Maintain backward compatibility in MINOR versions
3. Deprecate before removing (minimum 12 months)
4. Provide automated migration tools for MAJOR versions
5. Add `schema_version` to all new contracts

**Resources:**
- [Contract Schema v2.0 Reference](CONTRACT_SCHEMA_V2.md)
- [Migration Tools](../../src/omninode_bridge/codegen/migration/)
- [Schema Files](../../src/omninode_bridge/codegen/schemas/)
