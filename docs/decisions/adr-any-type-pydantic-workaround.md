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
- [ ] Is `Any` used ONLY for JSON-serializable fields?
- [ ] Could a more specific type be used instead?
- [ ] Is this a new occurrence or modification of existing workaround?

## References

- CLAUDE.md "Strong Typing & Models" section
- `omnibase_core` JsonType definition
- [PEP 695 - Type Parameter Syntax](https://peps.python.org/pep-0695/)
- [Pydantic GitHub Issue #3278: Recursive type support](https://github.com/pydantic/pydantic/issues/3278)
- [Pydantic Documentation on JSON Types](https://docs.pydantic.dev/latest/concepts/json/)
- PR #116: Initial introduction of this workaround
- OMN-1104: Refactor RegistrationReducer to be fully declarative
- [OMN-1262](https://linear.app/omninode/issue/OMN-1262): Migration tracking issue for Any type cleanup
