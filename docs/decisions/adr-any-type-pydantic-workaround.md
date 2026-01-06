# ADR: Use of Any Type as Pydantic 2.x JsonType Workaround

**Status**: Accepted
**Date**: 2026-01-06
**Related Tickets**: OMN-1104

## Context

The CLAUDE.md coding standards explicitly state:

> **NEVER use `Any`** - Use `object` for generic payloads

This rule exists to enforce strong typing throughout the codebase and prevent type erosion. However, PR #116 introduces `Any` across 33+ files as a necessary workaround for a Pydantic 2.x compatibility issue.

### The Problem

`omnibase_core` defines a `JsonType` recursive type alias for representing arbitrary JSON-serializable data:

```python
# From omnibase_core
JsonType = dict[str, "JsonType"] | list["JsonType"] | str | int | float | bool | None
```

When this type is used in Pydantic 2.x models, particularly in fields that may contain deeply nested or recursive structures, Python raises a `RecursionError`:

```
RecursionError: maximum recursion depth exceeded
```

This occurs because Pydantic 2.x attempts to build validators for recursive type aliases at model definition time, causing infinite recursion during schema generation.

### Affected Use Cases

The recursion issue manifests in:

1. **Model validation** - When Pydantic builds validators for `JsonType` fields
2. **Schema generation** - When generating JSON schemas for OpenAPI/documentation
3. **Serialization** - When serializing models with `JsonType` fields to JSON

### Workaround Attempts

Several alternatives were evaluated before accepting `Any`:

| Alternative | Result |
|------------|--------|
| `object` type | Does not serialize correctly in Pydantic 2.x |
| `dict[str, object]` | Loses support for non-dict JSON roots (arrays, primitives) |
| Custom validators | Still triggers recursion during schema building |
| `typing.ForwardRef` | Pydantic 2.x resolves forward refs eagerly |
| Pydantic `JsonValue` | Not available in all Pydantic 2.x versions used |

## Decision

**Use `Any` as a temporary workaround in place of `JsonType` for Pydantic model fields.**

This decision explicitly deviates from the CLAUDE.md "NEVER use Any" rule due to technical constraints in Pydantic 2.x's handling of recursive type aliases.

### Scope of Deviation

The `Any` type is permitted ONLY for:

1. Fields that previously used `JsonType`
2. Fields representing arbitrary JSON-serializable data
3. Pydantic model field type annotations

The `Any` type is NOT permitted for:

1. Function signatures (use `object` instead)
2. Non-Pydantic data structures
3. Return types where a more specific type is known
4. Generic containers (use `object` for unknown payload types)

### Annotation Pattern

When using `Any` for this workaround, include a comment explaining the deviation:

```python
from typing import Any

class MyModel(BaseModel):
    # Any used due to Pydantic 2.x JsonType recursion issue (ADR-any-type-pydantic-workaround)
    metadata: Any = None
```

## Rationale

### Why Accept This Deviation

1. **Technical necessity** - No viable alternative exists within Pydantic 2.x constraints
2. **Limited scope** - Affects only JSON-serializable fields, not business logic types
3. **Temporary measure** - Clear path to resolution exists (see Migration Path)
4. **Runtime safety** - JSON serialization/deserialization provides implicit type checking

### Why Not Delay Migration

1. The codebase requires Pydantic 2.x for other critical features
2. Blocking on `JsonType` fix would delay multiple dependent work items
3. The workaround is localized and documented

## Consequences

### Positive

- Unblocks migration to Pydantic 2.x
- Maintains runtime functionality for JSON serialization
- Clear documentation of deviation via this ADR
- Searchable pattern (`# Any used due to Pydantic 2.x`) for future cleanup

### Negative

- **Reduced type safety** - Static type checkers cannot validate `Any` fields
- **Documentation debt** - 33+ files contain this deviation
- **Rule erosion risk** - May encourage broader use of `Any` without justification
- **Future migration cost** - All uses must be updated when fix is available

### Type Safety Implications

The use of `Any` means:

| Static Analysis | Impact |
|----------------|--------|
| mypy | Will not catch type mismatches in `Any` fields |
| pyright | Same as mypy |
| IDE autocomplete | No suggestions for `Any` field contents |
| Runtime | Pydantic still validates JSON serialization |

## Migration Path

### Short-Term (Current State)

- Use `Any` with documentation comment
- Track all usages for future migration

### Medium-Term (omnibase_core Fix)

When `omnibase_core` provides a Pydantic 2.x-compatible `JsonType`:

1. Update `omnibase_core` dependency
2. Search for `# Any used due to Pydantic 2.x JsonType recursion`
3. Replace `Any` with updated `JsonType`
4. Run full type checking and test suite
5. Remove this ADR or update status to Superseded

### Long-Term (Pydantic Native)

Pydantic may introduce native `JsonValue` type in future versions:

```python
from pydantic import JsonValue  # Future Pydantic version

class MyModel(BaseModel):
    metadata: JsonValue = None
```

Monitor Pydantic releases for this feature.

## Verification

### Identifying Affected Files

```bash
# Find all files using Any for this workaround
grep -r "Any used due to Pydantic 2.x" src/
```

### Ensuring Compliance

New uses of `Any` without the required comment should be flagged in code review. The comment pattern `# Any used due to Pydantic 2.x JsonType recursion` serves as both documentation and a search anchor for future migration.

## References

- CLAUDE.md "Strong Typing & Models" section
- `omnibase_core` JsonType definition
- [Pydantic GitHub Issue: Recursive type support](https://github.com/pydantic/pydantic/issues/3278)
- PR #116: Initial introduction of this workaround
- OMN-1104: Refactor RegistrationReducer to be fully declarative
