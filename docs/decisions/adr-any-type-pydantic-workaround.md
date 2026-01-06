# ADR: Use of Any Type as Pydantic 2.x JsonType Workaround

**Status**: Accepted
**Date**: 2026-01-06
**Related Tickets**: OMN-1104, OMN-1262
**Tracking Issue**: [OMN-1262](https://linear.app/omninode/issue/OMN-1262) - Migration tracking

## Context

The CLAUDE.md coding standards explicitly state:

> **NEVER use `Any`** - Use `object` for generic payloads

This rule exists to enforce strong typing throughout the codebase and prevent type erosion. However, PR #116 introduces `Any` across 33+ files as a necessary workaround for a Pydantic 2.x compatibility issue.

### The Technical Problem: Recursive Type Alias Resolution

`omnibase_core` defines a `JsonType` recursive type alias for representing arbitrary JSON-serializable data:

```python
# From omnibase_core/types/json_types.py
JsonType = dict[str, "JsonType"] | list["JsonType"] | str | int | float | bool | None
```

This is a **recursive type alias** - the type references itself in its definition. This pattern is valid Python typing and works correctly with static type checkers like mypy and pyright. However, it causes severe issues with Pydantic 2.x.

### Why Pydantic 2.x Triggers Infinite Recursion

When Pydantic 2.x processes a model containing a `JsonType` field, it performs the following at **class definition time** (not runtime):

1. **Schema Generation**: Pydantic builds a JSON schema for each field to enable validation
2. **Type Resolution**: For each type annotation, Pydantic resolves forward references and expands type aliases
3. **Validator Construction**: Pydantic generates validators based on the resolved types

For `JsonType`, this process becomes:

```
JsonType
  -> dict[str, JsonType] | list[JsonType] | str | int | float | bool | None
    -> dict[str, (dict[str, JsonType] | list[JsonType] | ...)]
      -> dict[str, dict[str, (dict[str, JsonType] | list[JsonType] | ...)]]
        -> ... (infinite expansion)
```

The result is a `RecursionError`:

```
RecursionError: maximum recursion depth exceeded
  File ".../pydantic/_internal/_generate_schema.py", line 987, in generate_schema
  File ".../pydantic/_internal/_generate_schema.py", line 1342, in _union_schema
  File ".../pydantic/_internal/_generate_schema.py", line 987, in generate_schema
  ...
```

**Key insight**: This error occurs at **model class definition time**, not when creating instances. Simply having `class MyModel(BaseModel): data: JsonType` triggers the recursion before any code using the model runs.

### Pydantic 2.x vs 1.x Behavior

| Behavior | Pydantic 1.x | Pydantic 2.x |
|----------|--------------|--------------|
| Type alias resolution | Lazy (at validation time) | Eager (at class definition) |
| Recursive types | Generally tolerated | Causes `RecursionError` |
| Schema generation | On-demand | At class definition |

This is a known Pydantic 2.x limitation. See [Pydantic GitHub Issue #3278](https://github.com/pydantic/pydantic/issues/3278) for discussion.

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
| `pydantic.Json[Any]` | Different semantics (expects JSON string input) |
| `TypeAdapter` wrapper | Adds complexity; still issues with nested models |

## Decision

**Use `Any` as a temporary workaround in place of `JsonType` for Pydantic model fields.**

This decision explicitly deviates from the CLAUDE.md "NEVER use Any" rule due to technical constraints in Pydantic 2.x's handling of recursive type aliases.

### Scope of Deviation

**CRITICAL: `Any` is ONLY permitted in Pydantic `Field()` definitions.**

The `Any` type is permitted ONLY for:

1. **Pydantic model field type annotations** (the `Field()` definition context)
2. Fields that previously used `JsonType`
3. Fields representing arbitrary JSON-serializable data

**Examples of PERMITTED usage (inside Pydantic models only):**

```python
from typing import Any
from pydantic import BaseModel, Field

# NOTE: Using Any instead of JsonType from omnibase_core to avoid Pydantic 2.x
# recursion issues with recursive type aliases.

class ModelDLQEvent(BaseModel):
    """Dead letter queue event with arbitrary payload."""
    payload: Any = Field(default=None, description="Original event payload")
    metadata: Any = Field(default=None, description="Event metadata")
```

The `Any` type is **STRICTLY FORBIDDEN** for:

1. **Function parameter types** - use `object` instead
2. **Function return types** - use `object` or specific types
3. **Variable type annotations** - use `object` for unknown types
4. **Type aliases** - use `object` in union types
5. **Generic containers** - use `object` for unknown payload types
6. **Non-Pydantic data structures** (dataclasses, TypedDicts, etc.)

**Examples of FORBIDDEN usage:**

```python
# WRONG: Any in function signature
def process_event(payload: Any) -> Any:  # FORBIDDEN
    ...

# CORRECT: Use object for generic payloads
def process_event(payload: object) -> object:  # CORRECT
    ...

# WRONG: Any in variable annotation
result: Any = some_function()  # FORBIDDEN

# CORRECT: Use object or specific type
result: object = some_function()  # CORRECT
result: ModelSpecificType = some_function()  # BEST

# WRONG: Any in type alias
PayloadType = dict[str, Any]  # FORBIDDEN outside Pydantic

# CORRECT: Use object in type aliases
PayloadType = dict[str, object]  # CORRECT
```

**Why This Distinction Matters:**

- Pydantic's `Field()` context has runtime validation that provides type safety
- Function signatures/return types rely solely on static type checking
- Using `Any` outside Pydantic defeats the purpose of type hints entirely
- `object` is the proper "unknown type" marker in Python's type system

### Annotation Pattern

When using `Any` for this workaround, include a comment explaining the deviation:

```python
from typing import Any

# NOTE: Using Any instead of JsonType from omnibase_core to avoid Pydantic 2.x
# recursion issues with recursive type aliases.

class MyModel(BaseModel):
    metadata: Any = None
```

The `NOTE:` comment pattern serves as:
- Documentation explaining the deviation
- Search anchor for future migration
- Code review signal that this is intentional

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
- Searchable pattern (`NOTE: Using Any instead of JsonType`) for future cleanup

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

**Timeline**: Immediate (PR #116)

**Actions**:
- Use `Any` with documentation comment pattern
- Track all usages via the standard `NOTE:` comment
- This ADR serves as the canonical reference

**Comment Pattern** (MUST use exactly):
```python
# NOTE: Using Any instead of JsonType from omnibase_core to avoid Pydantic 2.x
# recursion issues with recursive type aliases.
```

### Medium-Term (omnibase_core Fix)

**Timeline**: Pending omnibase_core 0.7.x or later

**Approach**: Implement `JsonType` using PEP 695 `type` statement

Python 3.12+ introduced [PEP 695](https://peps.python.org/pep-0695/) which provides native syntax for type aliases that Pydantic can handle correctly:

```python
# New PEP 695 syntax (Python 3.12+)
type JsonType = dict[str, JsonType] | list[JsonType] | str | int | float | bool | None
```

Alternatively, use `typing.TypeAlias` with explicit annotation:

```python
from typing import TypeAlias

JsonType: TypeAlias = "dict[str, JsonType] | list[JsonType] | str | int | float | bool | None"
```

**Prerequisites**:
- omnibase_core must be updated to use PEP 695 or TypeAlias pattern
- New omnibase_core release must be published
- Minimum Python version may need to increase to 3.12+ for native syntax

**Migration Steps** (when fix is available):
1. Update `omnibase_core` dependency to version with fix
2. Search for `NOTE: Using Any instead of JsonType` pattern
3. Replace `Any` with `JsonType` import
4. Remove the `NOTE:` comment
5. Run full type checking (`mypy`, `pyright`)
6. Run full test suite
7. Update this ADR status to "Superseded"

### Long-Term (Pydantic Native Support)

**Timeline**: Pydantic 3.x or later

Pydantic may introduce native recursive type support or a built-in `JsonValue` type:

```python
from pydantic import JsonValue  # Future Pydantic version

class MyModel(BaseModel):
    metadata: JsonValue = None
```

**Monitoring**:
- Watch [Pydantic changelog](https://docs.pydantic.dev/latest/changelog/)
- Track [GitHub Issue #3278](https://github.com/pydantic/pydantic/issues/3278)
- Check for `JsonValue` in new Pydantic releases

### Migration Verification

After any migration phase, run:

```bash
# Verify no workaround comments remain
grep -r "NOTE: Using Any instead of JsonType" src/
# Should return empty after full migration

# Verify no unintended Any usage
grep -rn ": Any" src/ | grep -v "test" | grep -v "__pycache__"
# Review any remaining uses

# Run type checking
mypy src/
pyright src/

# Run full test suite
pytest tests/
```

## Affected Files

The following 33 files use `Any` as a workaround for `JsonType`:

### Event Bus (4 files)
- `src/omnibase_infra/event_bus/inmemory_event_bus.py`
- `src/omnibase_infra/event_bus/kafka_event_bus.py`
- `src/omnibase_infra/event_bus/models/model_dlq_event.py`
- `src/omnibase_infra/event_bus/models/model_dlq_metrics.py`

### Handlers (4 files)
- `src/omnibase_infra/handlers/handler_consul.py`
- `src/omnibase_infra/handlers/handler_db.py`
- `src/omnibase_infra/handlers/handler_http.py`
- `src/omnibase_infra/handlers/handler_vault.py`

### Handler Mixins (6 files)
- `src/omnibase_infra/handlers/mixins/mixin_consul_initialization.py`
- `src/omnibase_infra/handlers/mixins/mixin_consul_kv.py`
- `src/omnibase_infra/handlers/mixins/mixin_consul_service.py`
- `src/omnibase_infra/handlers/mixins/mixin_vault_initialization.py`
- `src/omnibase_infra/handlers/mixins/mixin_vault_secrets.py`
- `src/omnibase_infra/handlers/mixins/mixin_vault_token.py`

### Handler Models (4 files)
- `src/omnibase_infra/handlers/models/http/model_http_get_payload.py`
- `src/omnibase_infra/handlers/models/http/model_http_post_payload.py`
- `src/omnibase_infra/handlers/models/model_db_query_payload.py`
- `src/omnibase_infra/handlers/models/vault/model_vault_secret_payload.py`

### Mixins (1 file)
- `src/omnibase_infra/mixins/mixin_envelope_extraction.py`

### Models (2 files)
- `src/omnibase_infra/models/registration/model_node_capabilities.py`
- `src/omnibase_infra/models/registry/model_message_type_entry.py`

### Nodes (1 file)
- `src/omnibase_infra/nodes/node_registration_orchestrator/handlers/handler_node_introspected.py`

### Plugins (3 files)
- `src/omnibase_infra/plugins/examples/plugin_json_normalizer.py`
- `src/omnibase_infra/plugins/examples/plugin_json_normalizer_error_handling.py`
- `src/omnibase_infra/plugins/models/model_plugin_context.py`

### Runtime (8 files)
- `src/omnibase_infra/runtime/envelope_validator.py`
- `src/omnibase_infra/runtime/kernel.py`
- `src/omnibase_infra/runtime/models/model_health_check_response.py`
- `src/omnibase_infra/runtime/models/model_health_check_result.py`
- `src/omnibase_infra/runtime/protocol_policy.py`
- `src/omnibase_infra/runtime/runtime_host_process.py`
- `src/omnibase_infra/runtime/validation.py`
- `src/omnibase_infra/runtime/wiring.py`

## Migration Execution Plan

### Prerequisites for Migration

Before migrating away from `Any`, the following must be complete:

1. **omnibase_core Fix**: New `JsonType` implementation using PEP 695 or TypeAlias
2. **omnibase_core Release**: Published version 0.7.x or later with the fix
3. **Dependency Update**: `omnibase_infra` updated to depend on fixed version

### Migration Steps (Per File)

For each of the 33 affected files:

1. **Update imports**: Replace `from typing import Any` with `from omnibase_core.types import JsonType`
2. **Replace type annotations**: Change `: Any` to `: JsonType` in Pydantic fields
3. **Remove workaround comments**: Delete the `NOTE: Using Any instead of JsonType` comment
4. **Run type checker**: Verify with `mypy` and `pyright`
5. **Run tests**: Ensure all unit and integration tests pass

### Migration Priority Order

**Phase 1 - Core Models** (High Risk):
1. `models/registration/model_node_capabilities.py`
2. `models/registry/model_message_type_entry.py`
3. `event_bus/models/model_dlq_event.py`
4. `event_bus/models/model_dlq_metrics.py`

**Phase 2 - Handlers** (Medium Risk):
1. All files in `handlers/` directory
2. All files in `handlers/mixins/` directory
3. All files in `handlers/models/` directory

**Phase 3 - Runtime** (High Integration Risk):
1. `runtime/envelope_validator.py`
2. `runtime/kernel.py`
3. `runtime/runtime_host_process.py`
4. Remaining runtime files

**Phase 4 - Plugins & Nodes** (Lower Risk):
1. Plugin files (may need separate testing)
2. Node handler files

### Test Coverage Requirements

**Integration tests required before migration is complete:**

| Test Area | Status | Tracking |
|-----------|--------|----------|
| Intent emission from declarative reducer | TODO | OMN-1263 |
| Envelope validation with JsonType fields | TODO | OMN-1263 |
| RuntimeHostProcess with typed payloads | TODO | OMN-1263 |
| DLQ event handling with JsonType | TODO | OMN-1263 |
| Kafka event bus serialization | TODO | OMN-1263 |

**Pre-existing test failures**: See [OMN-1263](https://linear.app/omninode/issue/OMN-1263) for tracking of test failures that existed before this ADR was implemented. These failures are NOT caused by the `Any` workaround but should be resolved as part of the overall migration effort.

### Rollback Plan

If migration causes issues:

1. Revert `omnibase_core` dependency to pre-fix version
2. Re-apply `Any` workaround with comment pattern
3. Document specific failure in this ADR under "Migration Attempts" section

## Verification

### Identifying Affected Files

```bash
# Find all files using Any for this workaround (primary method)
grep -rn "NOTE: Using Any instead of JsonType" src/

# Count affected files
grep -rl "NOTE: Using Any instead of JsonType" src/ | wc -l

# Find Any imports that may need review
grep -rn "from typing import.*Any" src/ | grep -v "__pycache__"
```

### Ensuring Compliance

New uses of `Any` without the required comment should be flagged in code review. The comment pattern `NOTE: Using Any instead of JsonType from omnibase_core to avoid Pydantic 2.x` serves as both documentation and a search anchor for future migration.

### Code Review Checklist

When reviewing PRs with `Any` usage:

- [ ] Is the `NOTE:` comment present exactly as specified?
- [ ] Is `Any` used ONLY for Pydantic model field type annotations?
- [ ] Is `Any` used ONLY for JSON-serializable fields?
- [ ] Is `Any` NOT used in function signatures, return types, or variable annotations?
- [ ] Could a more specific type be used instead?
- [ ] Is this a new occurrence or modification of existing workaround?

**Automatic rejection criteria:**

- `Any` in function parameter types (use `object`)
- `Any` in function return types (use `object` or specific type)
- `Any` in variable annotations outside Pydantic models
- `Any` without the required `NOTE:` comment
- `Any` in non-Pydantic data structures (dataclasses, TypedDicts)

### Known Policy Violations

The following patterns in the codebase may violate the strict `Any` policy and should be reviewed during migration:

1. **Function signatures with `Any`**: Some handler methods may use `Any` for payload parameters
2. **Return types with `Any`**: Some utility functions may return `Any`
3. **Type aliases with `Any`**: Some internal type definitions may use `Any`

These violations are tracked under [OMN-1262](https://linear.app/omninode/issue/OMN-1262) for cleanup.

## References

- CLAUDE.md "Strong Typing & Models" section
- `omnibase_core` JsonType definition
- [PEP 695 - Type Parameter Syntax](https://peps.python.org/pep-0695/)
- [Pydantic GitHub Issue #3278: Recursive type support](https://github.com/pydantic/pydantic/issues/3278)
- [Pydantic Documentation on JSON Types](https://docs.pydantic.dev/latest/concepts/json/)
- PR #116: Initial introduction of this workaround
- OMN-1104: Refactor RegistrationReducer to be fully declarative
- [OMN-1262](https://linear.app/omninode/issue/OMN-1262): Migration tracking issue for Any type cleanup
- [OMN-1263](https://linear.app/omninode/issue/OMN-1263): Pre-existing test failures and integration test coverage
