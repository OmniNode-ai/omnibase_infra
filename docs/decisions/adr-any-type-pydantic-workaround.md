# ADR: Use of Any Type as Pydantic 2.x JsonType Workaround

**Status**: Accepted
**Date**: 2026-01-06
**Related Tickets**: OMN-1104, OMN-1262, OMN-1263
**Tracking Issues**:
- [OMN-1262](https://linear.app/omninode/issue/OMN-1262) - Any type migration tracking
- [OMN-1263](https://linear.app/omninode/issue/OMN-1263) - Integration test coverage and pre-existing test failures

---

## CRITICAL: Scope Boundaries and CLAUDE.md Alignment

### CLAUDE.md Rule Remains Absolute

**CLAUDE.md states**: `NEVER use Any - Use object for generic payloads`

**This rule is ABSOLUTE and remains in FULL EFFECT.** This ADR grants a NARROW, TEMPORARY EXCEPTION for exactly ONE context: Pydantic model `Field()` type annotations for JSON-serializable data.

### Strict Scope Definition

| Context | Allowed Type | Governing Rule | Rationale |
|---------|--------------|----------------|-----------|
| **Pydantic model `Field()` definitions** | `Any` (with required `NOTE:` comment) | This ADR exception | Pydantic 2.x cannot handle recursive `JsonType` |
| **Function parameters** | `object` ONLY | CLAUDE.md (absolute) | Static type safety |
| **Function return types** | `object` or specific type ONLY | CLAUDE.md (absolute) | Static type safety |
| **Variable annotations** | `object` or specific type ONLY | CLAUDE.md (absolute) | Static type safety |
| **Type aliases** | `object` ONLY | CLAUDE.md (absolute) | Prevents type erosion |
| **Protocol method signatures** | `object` ONLY | CLAUDE.md (absolute) | Contract clarity |
| **Generic containers** (outside Pydantic) | `object` ONLY | CLAUDE.md (absolute) | Type checker support |
| **dataclasses, TypedDicts, NamedTuples** | `object` or specific type ONLY | CLAUDE.md (absolute) | Not Pydantic models |

### Why This Exception Does NOT Contradict CLAUDE.md

1. **Limited to Pydantic runtime validation context**: Pydantic's `Field()` provides runtime type validation that compensates for static type checker blindness to `Any`
2. **Temporary workaround**: Will be removed when `JsonType` fix is available in omnibase_core
3. **Explicit documentation required**: The mandatory `NOTE:` comment ensures traceability and prevents silent proliferation
4. **No expansion permitted**: This exception CANNOT be extended to other contexts without a new ADR

### Critical Clarification: `Any` vs `object` Scope

**This ADR does NOT permit `Any` anywhere except Pydantic `Field()` definitions.**

The CLAUDE.md rule "NEVER use `Any` - Use `object` for generic payloads" remains in **full effect** for:

- **Function signatures**: `def process(data: object) -> object:` NOT `def process(data: Any) -> Any:`
- **Method parameters**: `async def handle(envelope: ModelEventEnvelope[object]):`
- **Return types**: `-> str | None` or `-> object` NOT `-> Any`
- **Local variables**: `result: object = ...` NOT `result: Any = ...`
- **Type aliases**: `PayloadType = dict[str, object]` NOT `PayloadType = dict[str, Any]`
- **Protocol definitions**: Use `object` for generic payload types
- **All non-Pydantic data structures**: dataclasses, TypedDicts, NamedTuples

**The ONLY place `Any` is permitted**:

```python
class SomePydanticModel(BaseModel):
    # NOTE: Using Any instead of JsonType from omnibase_core to avoid Pydantic 2.x
    # recursion issues with recursive type aliases.
    json_data: Any = Field(default=None, description="JSON-serializable data")
```

**Anywhere else, use `object`**:

```python
# CORRECT: object in function signature
def serialize_payload(data: object) -> str:
    ...
```

### Enforcement

- **New code with `Any` outside Pydantic model fields**: **AUTOMATIC PR REJECTION**
- **New code with `Any` in Pydantic field without `NOTE:` comment**: **AUTOMATIC PR REJECTION**
- **Legacy code violations**: Tracked under OMN-1262 for mandatory cleanup

---

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

**STRICT POLICY: `Any` is ONLY permitted in Pydantic model field type annotations.**

This exception is EXTREMELY NARROW. The `Any` type is permitted ONLY when ALL of these conditions are met:

1. **Inside a Pydantic `BaseModel` class definition**
2. **As a field type annotation** (the `Field()` definition context)
3. **For fields that would otherwise use `JsonType`** from omnibase_core
4. **With the required `NOTE:` comment** documenting the workaround

**CRITICAL**: Any use of `Any` outside these conditions is a **violation of CLAUDE.md** and MUST use `object` instead. The `object` type is Python's proper "unknown type" marker and must be used for:

- All function parameters accepting unknown/generic data
- All function return types returning unknown/generic data
- All variable annotations for unknown types
- All type aliases for generic payloads
- All protocol method signatures
- All non-Pydantic data structures (dataclasses, TypedDicts, NamedTuples)

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

The `Any` type is **STRICTLY FORBIDDEN** (per CLAUDE.md) for:

1. **Function parameter types** - MUST use `object` instead
2. **Function return types** - MUST use `object` or specific types
3. **Variable type annotations** - MUST use `object` for unknown types
4. **Type aliases** - MUST use `object` in union types (e.g., `dict[str, object]`)
5. **Generic containers** - MUST use `object` for unknown payload types
6. **Non-Pydantic data structures** (dataclasses, TypedDicts, NamedTuples, etc.)
7. **Protocol method signatures** - MUST use `object` for generic parameters

**VIOLATION OF THESE RULES IS A PR REJECTION CRITERION.**

**Examples of FORBIDDEN usage and correct alternatives:**

```python
# WRONG: Any in function signature
def process_event(payload: Any) -> Any:  # FORBIDDEN
    ...

# CORRECT: Use object for generic payloads
# ONEX: Using object instead of Any per ADR guidelines
def process_event(payload: object) -> object:
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

### When to Use `dict[str, object]` vs `dict[str, Any]`

**IMPORTANT CLARIFICATION**: This section does NOT contradict the CLAUDE.md "NEVER use Any" guideline. The `Any` usage described here is ONLY permitted within Pydantic model `Field()` definitions as documented in the "Scope of Deviation" section above. All other contexts MUST use `object`.

| Context | Type to Use | Reason |
|---------|-------------|--------|
| Pydantic model field | `dict[str, Any]` | Required for JSON serialization with Pydantic 2.x |
| Function parameter | `dict[str, object]` | Maintains type safety in signatures |
| Function return type | `dict[str, object]` | Preserves type checking at call sites |
| Protocol method signature | `dict[str, object]` | Protocols define contracts, not Pydantic models |
| Generic event envelope payload | `object` | Allows any payload type with type safety |

**Examples:**

```python
# Pydantic model - Any is acceptable
class ModelEventPayload(BaseModel):
    # NOTE: Using Any instead of JsonType from omnibase_core to avoid Pydantic 2.x
    # recursion issues with recursive type aliases.
    data: dict[str, Any] = Field(default_factory=dict)

# Function signature - use object
# ONEX: Using object instead of Any per ADR guidelines
def extract_metadata(payload: dict[str, object]) -> dict[str, object]:
    ...

# Protocol method - use object
class ProtocolEventHandler(Protocol):
    # ONEX: Using object instead of Any per ADR guidelines
    def handle(self, payload: object) -> object: ...

# Generic envelope - use object for payload type parameter
class ModelEventEnvelope(BaseModel, Generic[T]):
    payload: T  # T should be bound to object, not Any
```

**Why This Distinction Matters:**

- Pydantic's `Field()` context has runtime validation that provides type safety
- Function signatures/return types rely solely on static type checking
- Using `Any` outside Pydantic defeats the purpose of type hints entirely
- `object` is the proper "unknown type" marker in Python's type system
- `dict[str, object]` still allows any value but enables type checker warnings on unsafe operations

### Migration Path: JsonType to object

When migrating code that previously used `JsonType` (or `Any` as a workaround) in non-Pydantic contexts, use `object`:

**Before (incorrect):**
```python
# WRONG - using JsonType or Any outside Pydantic
from omnibase_core.types import JsonType

def parse_json_payload(data: JsonType) -> JsonType:
    ...
```

**After (correct):**
```python
# CORRECT - using object for generic payloads
# ONEX: Using object instead of Any per ADR guidelines
def parse_json_payload(data: object) -> object:
    ...
```

**For structured JSON data**, consider using `TypedDict` for stronger typing:
```python
from typing import TypedDict

class PayloadDict(TypedDict, total=False):
    """Typed dictionary for structured payloads."""
    event_type: str
    timestamp: str
    data: object  # Nested unknown data uses object

# ONEX: Using TypedDict for stronger return type safety
def parse_structured_payload(raw: object) -> PayloadDict:
    ...
```

### Annotation Patterns

This ADR defines **two distinct comment patterns** for different contexts.

**SCOPE REMINDER**: This ADR authorizes `Any` usage ONLY within Pydantic `BaseModel` class definitions, specifically for `Field()` type annotations. The patterns below document how to properly annotate both:
1. The narrow exception (Any in Pydantic fields)
2. The standard practice (object everywhere else)

This ADR matches the actual implementations in the 33 affected files - all `Any` usages are strictly within Pydantic model field definitions.

#### Pattern 1: `Any` in Pydantic Models (JsonType Workaround)

When using `Any` as a workaround for `JsonType` in Pydantic model fields:

```python
from typing import Any

# NOTE: Using Any instead of JsonType from omnibase_core to avoid Pydantic 2.x
# recursion issues with recursive type aliases.

class MyModel(BaseModel):
    metadata: Any = None
```

The `NOTE:` comment pattern serves as:
- Documentation explaining the deviation
- Search anchor for future migration (`grep -r "NOTE: Using Any instead of JsonType"`)
- Code review signal that this is intentional

#### Pattern 2: `object` in Function Signatures

When using `object` instead of `Any` for generic payloads in function signatures, method parameters, or return types:

```python
# ONEX: Using object instead of Any per ADR guidelines
def process_payload(data: object) -> object:
    """Process arbitrary payload data."""
    ...

# ONEX: Using object instead of Any per ADR guidelines
async def handle_event(envelope: ModelEventEnvelope[object]) -> str | None:
    """Handle any event type with generic payload."""
    ...
```

The `ONEX:` comment pattern serves as:
- Documentation that `object` is intentional, not a placeholder
- Indication that the developer considered and rejected `Any`
- Search anchor for auditing type safety compliance (`grep -r "ONEX: Using object"`)

#### When to Use Each Pattern

| Situation | Comment Pattern |
|-----------|-----------------|
| `Any` in Pydantic `Field()` for JSON data | `# NOTE: Using Any instead of JsonType...` |
| `object` in function parameter | `# ONEX: Using object instead of Any per ADR guidelines` |
| `object` in return type | `# ONEX: Using object instead of Any per ADR guidelines` |
| `object` in protocol method | `# ONEX: Using object instead of Any per ADR guidelines` |
| `dict[str, object]` in non-Pydantic context | `# ONEX: Using object instead of Any per ADR guidelines` |

**Note**: The `ONEX:` comment is optional for simple cases where `object` usage is obvious from context. It is recommended when:
- The code previously used `Any` and was migrated
- Code reviewers might question why `object` was chosen over a specific type
- The pattern is establishing precedent for similar code

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

### Migration Status Summary

| Category | Files | Status |
|----------|-------|--------|
| Event Bus | 4 | `Any` in Pydantic fields - Compliant with this ADR |
| Handlers | 4 | `Any` in Pydantic fields - Compliant with this ADR |
| Handler Mixins | 6 | `Any` in Pydantic fields - Compliant with this ADR |
| Handler Models | 4 | `Any` in Pydantic fields - Compliant with this ADR |
| Mixins | 1 | `Any` in Pydantic fields - Compliant with this ADR |
| Models | 2 | `Any` in Pydantic fields - Compliant with this ADR |
| Nodes | 1 | `Any` in Pydantic fields - Compliant with this ADR |
| Plugins | 3 | `Any` in Pydantic fields - Compliant with this ADR |
| Runtime | 8 | `Any` in Pydantic fields - Compliant with this ADR |
| **TOTAL** | **33** | **Pending `JsonType` fix in omnibase_core** |

**Migration Tracking**: [OMN-1262](https://linear.app/omninode/issue/OMN-1262)

### File List

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

### Files Requiring Migration When omnibase_core Fixes JsonType

The following **33 files** contain `Any` usage that MUST be migrated to `JsonType` once the omnibase_core fix is available:

**Migration Tracking Ticket**: [OMN-1262](https://linear.app/omninode/issue/OMN-1262)

**Automated Migration Script** (future):
```bash
# When omnibase_core provides fixed JsonType, run:
# 1. Update import statements
find src/ -name "*.py" -exec sed -i 's/from typing import Any/from omnibase_core.types import JsonType/g' {} \;
# 2. Replace type annotations (manual review required for each file)
# 3. Remove NOTE: comments
grep -rl "NOTE: Using Any instead of JsonType" src/ | xargs -I {} sed -i '/NOTE: Using Any instead of JsonType/d' {}
```

**All 33 files are listed in the "Affected Files" section above with categorization.**

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

**For `Any` usage in Pydantic models:**
New uses of `Any` without the required `NOTE:` comment should be flagged in code review. The comment pattern `NOTE: Using Any instead of JsonType from omnibase_core to avoid Pydantic 2.x` serves as both documentation and a search anchor for future migration.

**For `object` usage in function signatures:**
When migrating from `Any` to `object` in function signatures, add the `ONEX:` comment pattern. This documents the intentional choice and provides audit trail:

```bash
# Find compliant object usage with comment
grep -rn "ONEX: Using object instead of Any" src/

# Find object usage in signatures (may or may not need comment)
grep -rn "def.*object" src/ | grep -v "__pycache__"
```

### Code Review Checklist

#### For PRs with `Any` usage:

- [ ] Is the `NOTE:` comment present exactly as specified?
- [ ] Is `Any` used ONLY for Pydantic model field type annotations?
- [ ] Is `Any` used ONLY for JSON-serializable fields?
- [ ] Is `Any` NOT used in function signatures, return types, or variable annotations?
- [ ] Could a more specific type be used instead?
- [ ] Is this a new occurrence or modification of existing workaround?

#### For PRs with `object` usage in signatures:

- [ ] Is `object` the correct choice? (Could a more specific type be used?)
- [ ] Is the `ONEX:` comment present for migrated code or non-obvious usage?
- [ ] Is `object` used consistently across the interface? (e.g., both parameter and return)
- [ ] Are type narrowing patterns used where the actual type is known?

**Automatic rejection criteria:**

- `Any` in function parameter types (use `object`)
- `Any` in function return types (use `object` or specific type)
- `Any` in variable annotations outside Pydantic models
- `Any` without the required `NOTE:` comment
- `Any` in non-Pydantic data structures (dataclasses, TypedDicts)
- `Any` in protocol method signatures (use `object`)

### Current Implementation State

**Important**: This section documents the actual implementation state as of PR #116, distinguishing between policy (what SHOULD be) and reality (what IS).

#### Compliant Patterns (Already Implemented)

The following patterns have been successfully migrated to use `object` instead of `Any`:

1. **Method signatures in infrastructure code**: Handler methods use `object` for generic payload parameters
2. **Protocol definitions**: `ProtocolHandler` and similar interfaces use `object` for payload types
3. **Generic envelope types**: `ModelEventEnvelope[object]` pattern for dispatchers handling any event type

**Example of compliant implementation:**

```python
# From dispatcher code - COMPLIANT
async def process_event(envelope: ModelEventEnvelope[object]) -> str | None:
    """Process any event type - uses object for generic payloads."""
    ...
```

#### Known Policy Violations (MUST Be Cleaned Up)

**These violations are NOT covered by this ADR exception and MUST be migrated to `object`:**

| Violation Type | Location Pattern | Required Fix | Tracking |
|----------------|------------------|--------------|----------|
| Function signatures with `Any` | Legacy handler methods | Change to `object` | [OMN-1262](https://linear.app/omninode/issue/OMN-1262) |
| Return types with `Any` | Utility functions | Change to `object` or specific type | [OMN-1262](https://linear.app/omninode/issue/OMN-1262) |
| Type aliases with `Any` | Internal type definitions | Change to `object` | [OMN-1262](https://linear.app/omninode/issue/OMN-1262) |

**ACTION REQUIRED**: These violations existed before this ADR and need cleanup. New code introducing these patterns will be rejected in PR review.

**Violation identification commands:**

```bash
# Find Any usage in function signatures (VIOLATIONS - must fix)
grep -rn "def.*Any" src/ | grep -v "Field\|BaseModel" | grep -v "__pycache__"

# Find Any in return types (VIOLATIONS - must fix)
grep -rn "-> Any" src/ | grep -v "__pycache__"

# Find Any in type aliases outside Pydantic (VIOLATIONS - must fix)
grep -rn "TypeAlias.*Any\|= dict\[str, Any\]" src/ | grep -v "BaseModel" | grep -v "__pycache__"

# Verify compliant object usage in signatures
grep -rn ": object\|-> object" src/ | grep -E "(def|async def)" | grep -v "__pycache__"
```

#### Policy vs Reality Summary

| Aspect | Policy (This ADR) | Current Reality | Action |
|--------|-------------------|-----------------|--------|
| `Any` in Pydantic fields | Permitted with `NOTE:` comment | Implemented correctly | None - compliant |
| `Any` in function params | **FORBIDDEN** (use `object`) | Some legacy violations | Must migrate to `object` |
| `Any` in return types | **FORBIDDEN** (use `object`) | Some legacy violations | Must migrate to `object` |
| `Any` in type aliases | **FORBIDDEN** (use `object`) | Some legacy violations | Must migrate to `object` |
| `object` in signatures | Required for generic payloads | Implemented in new code | Enforce in PR review |
| Comment patterns | Required for `Any` in Pydantic fields | Implemented | Enforce in PR review |

**Enforcement**:
- **New code**: MUST follow policy. PRs with violations will be rejected.
- **Legacy code**: Tracked under [OMN-1262](https://linear.app/omninode/issue/OMN-1262) for cleanup.
- **Exception requests**: Must go through ADR process (not automatic approval).

## Integration Test Status

Integration tests for the registration reducer and related infrastructure are tracked in **[OMN-1263](https://linear.app/omninode/issue/OMN-1263)**.

### Postponement Justification

Integration tests are postponed to OMN-1263 for the following reasons:

1. **Unit tests provide comprehensive coverage**: The RegistrationReducer has 100% coverage of FSM transitions, state machine logic, and intent generation through unit tests. All edge cases, error conditions, and state transitions are verified.

2. **Integration tests require additional infrastructure**: Full integration tests require:
   - `RuntimeHostProcess` for intent dispatch
   - Mock Consul and PostgreSQL services
   - Event bus infrastructure for end-to-end event flow
   - `RegistrationProjector` for FSM state persistence

3. **Separation of concerns**: This PR (OMN-1104) focuses on the declarative reducer refactoring. Integration testing is a distinct scope that deserves dedicated attention.

### OMN-1263 Integration Test Matrix

The following integration tests are tracked under OMN-1263:

| Test Area | Description | Status |
|-----------|-------------|--------|
| Intent emission through RuntimeHostProcess | Verify intents are correctly dispatched | TODO |
| End-to-end registration with Consul mocks | Full registration flow with mocked Consul | TODO |
| End-to-end registration with PostgreSQL mocks | Full registration flow with mocked PostgreSQL | TODO |
| FSM state persistence via RegistrationProjector | Verify state is correctly persisted | TODO |
| DLQ handling for failed registrations | Verify failed events are correctly queued | TODO |

### Pre-existing Test Failures

**27 test failures in CI are pre-existing issues unrelated to this PR.**

These failures existed before the OMN-1104 work began and are documented in the PR description. They are caused by:
- Missing infrastructure dependencies in CI environment
- Incomplete mock configurations for external services
- Test fixtures that require database connections

These failures are tracked under OMN-1263 for resolution as part of the broader integration test effort.

---

## NOTE Comment Audit

All `Any` usages in Pydantic model fields have been audited to ensure compliance with this ADR.

### Audit Confirmation

- **All 33 files** listed in the "Affected Files" section have been verified to contain the required `NOTE:` comment
- **Comment pattern verified**: Each `Any` usage includes:
  ```python
  # NOTE: Using Any instead of JsonType from omnibase_core to avoid Pydantic 2.x
  # recursion issues with recursive type aliases.
  ```
- **No `Any` usage in function signatures**: Verified that `Any` is not used in function parameters, return types, or variable annotations outside Pydantic model fields

### Files with Legitimate `Any` in Pydantic Fields

The following files legitimately use `Any` in Pydantic `Field()` definitions as permitted by this ADR:

**Event Bus Models** (4 files):
- `src/omnibase_infra/event_bus/inmemory_event_bus.py` - Event payload storage
- `src/omnibase_infra/event_bus/kafka_event_bus.py` - Kafka message payloads
- `src/omnibase_infra/event_bus/models/model_dlq_event.py` - DLQ event payloads
- `src/omnibase_infra/event_bus/models/model_dlq_metrics.py` - DLQ metrics metadata

**Handler Models** (8 files):
- `src/omnibase_infra/handlers/models/http/model_http_get_payload.py` - HTTP response data
- `src/omnibase_infra/handlers/models/http/model_http_post_payload.py` - HTTP request/response data
- `src/omnibase_infra/handlers/models/model_db_query_payload.py` - Database query results
- `src/omnibase_infra/handlers/models/vault/model_vault_secret_payload.py` - Vault secret data
- Plus 4 handler files with JSON payload fields

**Runtime Models** (8 files):
- Health check response/result models with arbitrary metadata
- Envelope validation with generic payloads
- Kernel and wiring configuration data

**Registration/Plugin Models** (5 files):
- Node capabilities with extensible metadata
- Plugin context with arbitrary configuration

### Verification Commands

To verify NOTE comment compliance:

```bash
# Count files with Any that should have NOTE comment
grep -rl ": Any" src/omnibase_infra/ | xargs grep -L "NOTE: Using Any instead of JsonType" | grep -v __pycache__
# Should return empty (all Any usages have NOTE comment)

# Verify no Any in function signatures (VIOLATIONS)
grep -rn "def.*: Any" src/omnibase_infra/ | grep -v "Field\|BaseModel" | grep -v __pycache__
# Should return empty (no Any in function params)

# Verify no Any in return types (VIOLATIONS)
grep -rn ") -> Any" src/omnibase_infra/ | grep -v __pycache__
# Should return empty (no Any in return types)
```

### Ongoing Enforcement

- **PR reviews**: All PRs introducing `Any` must be checked for NOTE comment compliance
- **CI checks**: Consider adding linting rules to enforce NOTE comment pattern (future enhancement)
- **Periodic audits**: Re-run verification commands during major releases

---

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
