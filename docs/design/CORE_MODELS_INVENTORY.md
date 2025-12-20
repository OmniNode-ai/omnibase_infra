# omnibase_core Models Inventory for Union Pattern Replacement

**Purpose**: Comprehensive inventory of omnibase_core models that can replace union patterns in omnibase_infra.

**Context**:
- omnibase_infra has 305 unions, target is <200
- omnibase_core version 0.5.4

---

## Table of Contents
1. [Value/Flexible Type Models](#1-valueflexible-type-models)
2. [Error Context Models](#2-error-context-models)
3. [Event/Envelope Models](#3-eventenvelope-models)
4. [Type Aliases (JsonSerializable and Domain Types)](#4-type-aliases)
5. [Configuration Models](#5-configuration-models)
6. [Typed Metadata Models](#6-typed-metadata-models)
7. [Union Pattern to Core Model Mapping](#7-union-pattern-to-core-model-mapping)
8. [Already Imported but Underutilized](#8-already-imported-but-underutilized)

---

## 1. Value/Flexible Type Models

### ModelFlexibleValue
**Path**: `omnibase_core.models.common.model_flexible_value`

**Purpose**: Discriminated union for values that can be multiple types. Replaces lazy `Union[str, dict[str, Any], list[Any], int, etc.]` patterns.

**Key Fields**:
- `value_type: EnumFlexibleValueType` - Type discriminator (STRING, INTEGER, FLOAT, BOOLEAN, DICT, LIST, UUID, NONE)
- `string_value: str | None`
- `integer_value: int | None`
- `float_value: float | None`
- `boolean_value: bool | None`
- `dict_value: dict[str, ModelSchemaValue] | None`
- `list_value: list[ModelSchemaValue] | None`
- `uuid_value: UUID | None`
- `source: str | None` - Source of the value
- `is_validated: bool`

**Factory Methods**:
- `from_string()`, `from_integer()`, `from_float()`, `from_boolean()`
- `from_dict_value()`, `from_raw_dict()`, `from_list()`, `from_uuid()`, `from_none()`
- `from_any()` - Automatic type detection

**Replaces**: `dict[str, Any] | None`, `list[Any] | None`, `str | int | float | bool | dict | list`

---

### ModelDiscriminatedValue
**Path**: `omnibase_core.models.common.model_discriminated_value`

**Purpose**: Universal discriminated union for primitive and complex types.

**Key Fields**:
- `value_type: EnumDiscriminatedValueType` - Type discriminator (BOOL, FLOAT, INT, STR, DICT, LIST)
- `bool_value: bool | None`
- `float_value: float | None`
- `int_value: int | None`
- `str_value: str | None`
- `dict_value: dict[str, object] | None` - Must be JSON-serializable
- `list_value: list[object] | None` - Must be JSON-serializable
- `metadata: dict[str, str]`

**Helper Methods**:
- `get_value()` - Get actual value with proper type
- `get_type()` - Get Python type class
- `is_type(expected_type)` - Type checking
- `is_primitive()`, `is_numeric()`, `is_collection()`

**Factory Methods**:
- `from_bool()`, `from_float()`, `from_int()`, `from_str()`
- `from_dict()`, `from_list()`, `from_any()`

**Replaces**: `Union[bool, float, int, str]`, `Union[bool, dict, float, int, list, str]`

---

### ModelValueUnion
**Path**: `omnibase_core.models.common.model_value_union`

**Purpose**: Type-safe wrapper for common primitive value unions with automatic type inference.

**Key Fields**:
- `value: bool | int | float | str | list[object] | dict[str, object]`
- `value_type: Literal["bool", "int", "float", "str", "list", "dict"]`
- `metadata: dict[str, str]`

**Security Features**:
- `MAX_LIST_SIZE: 10000` - DoS prevention
- `MAX_DICT_SIZE: 1000` - DoS prevention
- String key validation for dicts
- NaN and infinity detection for floats

**Replaces**: `Union[bool, float, int, list, str]`, `Union[bool, dict, float, int, list, str]`

---

### ModelMultiTypeValue
**Path**: `omnibase_core.models.common.model_multi_type_value`

**Purpose**: Type-safe wrapper for primitive value unions **without dict support**.

**Key Fields**:
- `value: bool | int | float | str | list[object]`
- `value_type: Literal["bool", "int", "float", "str", "list"]`
- `metadata: dict[str, str]`

**Replaces**: `Union[bool, float, int, list, str]` (when dict support not needed)

---

### ModelDictValueUnion
**Path**: `omnibase_core.models.common.model_dict_value_union`

**Purpose**: Type-safe wrapper for dict-containing union patterns, optimized for extension/plugin systems.

**Key Fields**:
- `value: bool | dict[str, object] | float | int | list[object] | str`
- `value_type: Literal["bool", "dict", "float", "int", "list", "str"]`
- `metadata: dict[str, str]`

**Dict-Specific Helper Methods**:
- `has_key(key: str) -> bool`
- `get_dict_value(key: str, default=None) -> object`
- `get_as_dict() -> dict[str, object]` - Returns empty dict for non-dict values (safe access)

**Type Guards**:
- `is_bool()`, `is_dict()`, `is_float()`, `is_int()`, `is_list()`, `is_string()`

**Type Getters**:
- `get_as_bool()`, `get_as_dict()`, `get_as_float()`, `get_as_int()`, `get_as_list()`, `get_as_str()`

**Replaces**: `Union[bool, dict, float, int, list, str]` in plugin/extension contexts

---

### ModelValueContainer
**Path**: `omnibase_core.models.common.model_value_container`

**Purpose**: Generic container that preserves exact type information for JSON-serializable values.

**Key Fields**:
- `value: JsonSerializable` - The contained value
- `metadata: dict[str, str]` - Optional string metadata

**Properties**:
- `python_type: type` - Get actual Python type
- `type_name: str` - Human-readable type name

**Methods**:
- `is_type(expected_type)` - Runtime type checking
- `is_json_serializable()` - Validation
- `is_valid()`, `get_errors()` - ProtocolValidatable implementation

**Replaces**: Loose `dict[str, Any]` fields that hold JSON data

---

### ModelTypedMapping
**Path**: `omnibase_core.models.common.model_typed_mapping`

**Purpose**: Strongly-typed mapping to replace `Dict[str, Any]` patterns.

**Key Fields**:
- `data: dict[str, ModelValueContainer]` - Mapping of keys to typed containers
- `current_depth: int` - DoS prevention (max 10 levels)

**Type-Safe Setters**:
- `set_string()`, `set_int()`, `set_float()`, `set_bool()`, `set_list()`, `set_dict()`
- `set_value()` - Automatic type detection

**Type-Safe Getters**:
- `get_string()`, `get_int()`, `get_bool()`
- `get_value()` - Generic access

**Security**: `MAX_DEPTH: 10` - Prevents DoS via deep nesting

**Replaces**: `dict[str, Any]`, `dict[str, object]` patterns

---

### ModelSchemaValue
**Path**: `omnibase_core.models.common.model_schema_value`

**Purpose**: Type-safe representation of schema values following JSON Schema types.

**Key Fields**:
- `value_type: str` - Type indicator (string, number, boolean, null, array, object)
- `string_value: str | None`
- `number_value: ModelNumericValue | None`
- `boolean_value: bool | None`
- `null_value: bool | None`
- `array_value: list[ModelSchemaValue] | None` - Recursive
- `object_value: dict[str, ModelSchemaValue] | None` - Recursive

**Factory Methods**:
- `from_value(value: object)` - Create from Python value
- `create_string()`, `create_number()`, `create_boolean()`, `create_null()`
- `create_array()`, `create_object()`

**Type Checks**:
- `is_string()`, `is_number()`, `is_boolean()`, `is_null()`, `is_array()`, `is_object()`

**Type Getters** (raise on mismatch):
- `get_string()`, `get_number()`, `get_boolean()`, `get_array()`, `get_object()`

**Replaces**: `Any` in JSON schema contexts, nested JSON structures

---

### ModelNumericValue
**Path**: `omnibase_core.models.common.model_numeric_value`

**Purpose**: Type-safe numeric value container replacing `int | float` unions.

**Key Fields**:
- `value: float` - The numeric value
- `value_type: EnumNumericType` - INTEGER or FLOAT
- `is_validated: bool`
- `source: str | None`

**Factory Methods**:
- `from_int()`, `from_float()`, `from_numeric()` (auto-detect)

**Accessors**:
- `as_int()`, `as_float()`
- `integer_value`, `float_value` (properties)
- `to_python_value()` - Preserves original type

**Replaces**: `int | float` unions

---

## 2. Error Context Models

### ModelErrorContext
**Path**: `omnibase_core.models.common.model_error_context`

**Purpose**: Type-safe representation of error context, replaces dictionary usage.

**Key Fields**:
- `file_path: str | None`
- `line_number: int | None`
- `column_number: int | None`
- `function_name: str | None`
- `module_name: str | None`
- `stack_trace: str | None`
- `additional_context: dict[str, ModelSchemaValue]` - Strongly typed

**Factory Methods**:
- `with_context(additional_context)` - Create with only additional context

**Conversion**:
- `to_simple_context()` - Convert to TypedDict
- `from_simple_context()` - Create from TypedDict

**Replaces**: `dict[str, Any]` in error handling contexts

---

## 3. Event/Envelope Models

### ModelEventEnvelope[T]
**Path**: `omnibase_core.models.events.model_event_envelope`

**Purpose**: ONEX-compatible envelope wrapper for all events with generic payload support.

**Key Fields**:
- `payload: T` - The wrapped event payload (generic)
- `envelope_id: UUID`
- `envelope_timestamp: datetime`
- `correlation_id: UUID | None`
- `source_tool: str | None`
- `target_tool: str | None`
- `metadata: ModelEnvelopeMetadata`
- `security_context: ModelSecurityContext | None`
- `priority: int` (1-10)
- `timeout_seconds: int | None`
- `retry_count: int`
- `request_id: UUID | None`
- `trace_id: UUID | None`
- `span_id: UUID | None`
- `onex_version: ModelSemVer`
- `envelope_version: ModelSemVer`

**Builder Methods**:
- `with_correlation_id()`, `with_metadata()`, `with_security_context()`
- `set_routing()`, `with_tracing()`, `with_priority()`
- `increment_retry_count()`

**Query Methods**:
- `is_correlated()`, `has_security_context()`, `has_trace_context()`
- `is_high_priority()`, `is_expired()`, `is_retry()`
- `infer_category()` - EVENT, COMMAND, or INTENT

**Factory Methods**:
- `create_broadcast()` - No specific target
- `create_directed()` - Specific target

**Replaces**: Ad-hoc envelope dictionaries, `dict[str, Any]` event wrappers

---

## 4. Type Aliases

### JsonSerializable
**Path**: `omnibase_core.models.types.model_json_serializable`

**Definition** (PEP 695 recursive type):
```python
type JsonSerializable = (
    str | int | float | bool | None
    | dict[str, "JsonSerializable"]
    | list["JsonSerializable"]
)
```

**Purpose**: Represents all valid JSON values per RFC 8259.

**Replaces**: `Any` in JSON serialization contexts, `dict[str, Any]` for JSON data

---

### Domain-Specific Type Aliases
**Path**: `omnibase_core.models.types.model_onex_common_types`

| Type Alias | Definition | Use Case |
|------------|------------|----------|
| `PropertyValue` | `str \| int \| float \| bool \| list[str] \| dict[str, str]` | Generic containers |
| `EnvValue` | `str \| int \| float \| bool \| None` | Environment variables |
| `MetadataValue` | `str \| int \| float \| bool \| list[str] \| dict[str, str] \| None` | Metadata/results |
| `ValidationValue` | Recursive: primitives + nested collections + None | Validation errors |
| `ConfigValue` | `str \| int \| float \| bool \| list[str] \| dict[str, str] \| None` | Config models |
| `CliValue` | `str \| int \| float \| bool \| list[str]` | CLI processing |
| `ParameterValue` | Same as `PropertyValue` | Tool/service params |
| `ResultValue` | Recursive: primitives + nested collections + None | Result/output |

**Replaces**: Various `Any` patterns in specific domains

---

## 5. Configuration Models

### ModelJsonData
**Path**: `omnibase_core.models.configuration.model_json_data`

**Purpose**: ONEX-compliant strongly typed JSON data model.

**Key Fields**:
- `fields: dict[str, ModelJsonField]` - Strongly typed JSON fields
- `schema_version: ModelSemVer` - Required
- `total_field_count: int`

**Methods**:
- `get_field_value(field_name)` - Type-safe accessor
- `has_field(field_name)` - Check existence
- `get_field_type(field_name)` - Get field type enum

**Replaces**: `dict[str, Any]` in JSON config contexts

---

## 6. Typed Metadata Models

**Path**: `omnibase_core.models.common.model_typed_metadata`

All exported from `omnibase_core.models.common`:

| Model | Purpose |
|-------|---------|
| `ModelConfigSchemaProperty` | Config schema properties |
| `ModelCustomHealthMetrics` | Health check metrics |
| `ModelEffectMetadata` | Effect node metadata |
| `ModelEventSubscriptionConfig` | Event subscription config |
| `ModelGraphNodeData` | Graph node data |
| `ModelGraphNodeInputs` | Graph node inputs |
| `ModelIntentPayload` | Intent payloads |
| `ModelIntrospectionCustomMetrics` | Introspection metrics |
| `ModelMixinConfigSchema` | Mixin configuration |
| `ModelNodeCapabilitiesMetadata` | Node capabilities |
| `ModelNodeRegistrationMetadata` | Node registration |
| `ModelOperationData` | Operation data |
| `ModelReducerMetadata` | Reducer metadata |
| `ModelRequestMetadata` | Request metadata |
| `ModelShutdownMetrics` | Shutdown metrics |
| `ModelToolMetadataFields` | Tool metadata |
| `ModelToolResultData` | Tool results |

**Replaces**: Various `dict[str, Any]` metadata patterns

---

## 7. Union Pattern to Core Model Mapping

### High-Impact Replacements (ordered by impact)

| Current Union Pattern | Recommended Core Model | Notes |
|-----------------------|----------------------|-------|
| `dict[str, Any]` | `ModelTypedMapping` | For dynamic key-value stores |
| `dict[str, Any]` | `JsonSerializable` | For JSON data interchange |
| `dict[str, object]` | `JsonSerializable` | When JSON-compatible |
| `str \| int \| float \| bool` | `ModelDiscriminatedValue` | Full discriminated union |
| `str \| int \| float \| bool \| dict \| list` | `ModelValueUnion` | Includes collections |
| `str \| int \| float \| bool \| list` | `ModelMultiTypeValue` | No dict needed |
| `Union[bool, dict, float, int, list, str]` | `ModelDictValueUnion` | Plugin/extension systems |
| `int \| float` | `ModelNumericValue` | Numeric values |
| `Any` (in JSON context) | `ModelSchemaValue` | Full JSON schema support |
| `Any` (error context) | `ModelErrorContext` | Error handling |
| `Any` (metadata) | Use specific typed metadata models | See section 6 |
| `X \| None` (optional values) | Keep as-is | PEP 604 optional is idiomatic |
| `Callable[...] \| None` | Keep as-is | Runtime optional callbacks |
| `Type[X] \| None` | Keep as-is | Optional type references |

### Patterns to Preserve (Not Unions of Concern)

These are **acceptable** union patterns:
- `X | None` - PEP 604 optional values
- `Type[A] | Type[B]` - Type unions for polymorphism
- `str | EnumX` - String or enum (input flexibility)
- `asyncio.Task[X] | None` - Optional async handles
- `Exception | None` - Optional exceptions
- `Callable[..., X] | None` - Optional callbacks

---

## 8. Already Imported but Underutilized

Based on grep analysis of `from omnibase_core` imports:

### Currently Imported in omnibase_infra

| Import | File(s) | Status |
|--------|---------|--------|
| `ModelONEXContainer` | container_wiring.py, kernel.py, runtime_host_process.py | Well-used |
| `EnumCoreErrorCode` | Multiple error handling files | Well-used |
| `ModelOnexError` | Multiple error handling files | Well-used |
| `ModelEventEnvelope` | message_dispatch_engine.py, dispatcher_registry.py | **UNDERUTILIZED** - use for more event handling |
| `EnumNodeKind` | dispatcher_registry.py, validation | Well-used |
| `ProtocolEventBus` | wiring.py, handler_registry.py | Well-used |

### Recommended New Imports for Union Reduction

**Priority 1 - Immediate Impact**:
```python
from omnibase_core.models.types import JsonSerializable
from omnibase_core.models.common import (
    ModelFlexibleValue,
    ModelDiscriminatedValue,
    ModelValueUnion,
    ModelTypedMapping,
    ModelSchemaValue,
)
```

**Priority 2 - Specific Domains**:
```python
from omnibase_core.models.common import (
    ModelErrorContext,  # Replace dict[str, Any] in error handling
    ModelNumericValue,  # Replace int | float
    ModelDictValueUnion,  # For plugin systems
)
```

**Priority 3 - Metadata Standardization**:
```python
from omnibase_core.models.common import (
    ModelNodeCapabilitiesMetadata,
    ModelNodeRegistrationMetadata,
    ModelOperationData,
    ModelRequestMetadata,
)
```

---

## Summary Statistics

| Category | Model Count | Estimated Union Replacements |
|----------|-------------|------------------------------|
| Value/Flexible Types | 9 | ~80-100 unions |
| Error Context | 1 | ~10-15 unions |
| Event/Envelope | 1 | ~15-20 unions |
| Type Aliases | 8 | ~30-40 unions |
| Configuration | 1 | ~10-15 unions |
| Typed Metadata | 17 | ~20-30 unions |
| **Total** | **37 models** | **~165-220 unions** |

**Conclusion**: The omnibase_core models can theoretically replace 165-220 of the 305 unions in omnibase_infra, bringing it well under the 200 target. Key focus areas:
1. Replace `dict[str, object]` with `JsonSerializable` or `ModelTypedMapping`
2. Replace primitive unions with `ModelDiscriminatedValue` or `ModelValueUnion`
3. Replace error context dicts with `ModelErrorContext`
4. Standardize metadata with typed metadata models
