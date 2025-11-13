# JSONB Field Validation Utilities

Pydantic validation utilities for enforcing PostgreSQL JSONB type annotations in entity models.

## Overview

This package provides three approaches to ensure all `dict[str, Any]` fields in Pydantic models have proper JSONB database type annotations (`json_schema_extra={"db_type": "jsonb"}`).

## Why This Matters

PostgreSQL has two JSON types:
- **JSON**: Stores exact text representation, slower queries
- **JSONB**: Stores binary format, faster queries, supports indexing

Without explicit `json_schema_extra={"db_type": "jsonb"}` annotation, SQLAlchemy/asyncpg may use the slower JSON type. These utilities enforce JSONB usage across all entity models.

## Quick Start

### Approach 1: JsonbField() Helper (⭐ Recommended)

```python
from typing import Any
from pydantic import BaseModel, Field
from omninode_bridge.infrastructure.validation import JsonbField

class ModelWorkflow(BaseModel):
    id: UUID = Field(default_factory=uuid4)

    # JSONB fields using helper
    metadata: dict[str, Any] = JsonbField(
        default_factory=dict,
        description="Workflow metadata"
    )

    config: dict[str, Any] = JsonbField(
        default_factory=dict,
        description="Configuration settings"
    )
```

**Pros:**
- ✅ Most concise and readable
- ✅ Automatic JSONB annotation
- ✅ Full IDE support and type hints
- ✅ No boilerplate validators needed

### Approach 2: Manual Field() + Validator

```python
from typing import Any
from pydantic import BaseModel, Field, model_validator
from omninode_bridge.infrastructure.validation import validate_jsonb_fields

class ModelWorkflow(BaseModel):
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Workflow metadata",
        json_schema_extra={"db_type": "jsonb"}  # Manual annotation
    )

    @model_validator(mode="after")
    def _validate_jsonb(self) -> "ModelWorkflow":
        return validate_jsonb_fields(self)
```

**Pros:**
- ✅ Explicit control over field configuration
- ✅ Useful for migrating existing models
- ✅ Validation catches missing annotations

### Approach 3: JsonbValidatedModel Base Class

```python
from typing import Any
from omninode_bridge.infrastructure.validation import (
    JsonbValidatedModel,
    JsonbField
)

class ModelWorkflow(JsonbValidatedModel):
    metadata: dict[str, Any] = JsonbField(
        default_factory=dict,
        description="Workflow metadata"
    )
```

**Pros:**
- ✅ Automatic validation via inheritance
- ✅ No validator decorator needed
- ✅ Clean model definitions

## API Reference

### `JsonbField()`

Wrapper around Pydantic's `Field()` that automatically adds JSONB annotation.

```python
def JsonbField(
    default: Any = ...,
    *,
    default_factory: Optional[Any] = None,
    description: Optional[str] = None,
    # ... all standard Field() parameters
    **extra: Any,
) -> Any:
    """Create a Pydantic Field with automatic JSONB annotation."""
```

**Parameters:**
- All standard `pydantic.Field()` parameters are supported
- Automatically adds `json_schema_extra={"db_type": "jsonb"}`

**Example:**
```python
metadata: dict[str, Any] = JsonbField(
    default_factory=dict,
    description="JSONB metadata field",
    examples=[{"key": "value"}]
)
```

### `validate_jsonb_fields()`

Validator function that checks all dict fields have JSONB annotations.

```python
def validate_jsonb_fields(model: BaseModel) -> BaseModel:
    """
    Validate that all dict[str, Any] fields have proper JSONB annotations.

    Raises:
        ValueError: If a dict field is missing JSONB annotation
    """
```

**Usage:**
```python
@model_validator(mode="after")
def _validate_jsonb(self) -> "MyModel":
    return validate_jsonb_fields(self)
```

### `JsonbValidatedModel`

Base model class with automatic JSONB validation.

```python
class JsonbValidatedModel(BaseModel):
    """Base model with automatic JSONB field validation."""

    @model_validator(mode="after")
    def _validate_jsonb_fields(self) -> "JsonbValidatedModel":
        return validate_jsonb_fields(self)
```

**Usage:**
```python
class MyModel(JsonbValidatedModel):
    data: dict[str, Any] = JsonbField(default_factory=dict)
    # Validation happens automatically
```

## Error Messages

When validation fails, you get clear, actionable error messages:

```python
# Missing annotation
ValueError: Field 'metadata' in ModelWorkflow is typed as dict but missing
required json_schema_extra={'db_type': 'jsonb'}.

Use JsonbField() helper or add json_schema_extra manually:
  # Option 1 (Recommended):
  metadata: dict[str, Any] = JsonbField(default_factory=dict)

  # Option 2:
  metadata: dict[str, Any] = Field(
      default_factory=dict,
      json_schema_extra={'db_type': 'jsonb'}
  )
```

## Migration Guide

### Updating Existing Models

**Before:**
```python
class ModelBridgeState(BaseModel):
    aggregation_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Aggregation statistics and metadata",
        json_schema_extra={"db_type": "jsonb"},  # Manual annotation
    )
```

**After (Recommended):**
```python
from omninode_bridge.infrastructure.validation import JsonbField

class ModelBridgeState(BaseModel):
    aggregation_metadata: dict[str, Any] = JsonbField(
        default_factory=dict,
        description="Aggregation statistics and metadata",
    )
```

### Adding Validation to Existing Models

**Option 1: Add validator decorator**
```python
from omninode_bridge.infrastructure.validation import validate_jsonb_fields

class ExistingModel(BaseModel):
    # ... existing fields ...

    @model_validator(mode="after")
    def _validate_jsonb(self) -> "ExistingModel":
        return validate_jsonb_fields(self)
```

**Option 2: Change base class**
```python
from omninode_bridge.infrastructure.validation import JsonbValidatedModel

class ExistingModel(JsonbValidatedModel):  # Changed from BaseModel
    # ... existing fields ...
    # Validation happens automatically
```

## Examples

See `examples.py` for comprehensive usage examples including:
- ✅ Correct usage patterns
- ❌ Common mistakes and error handling
- 🔧 Migration patterns

Run examples:
```bash
python -m omninode_bridge.infrastructure.validation.examples
```

## Testing

The validation utilities work seamlessly with pytest:

```python
import pytest
from omninode_bridge.infrastructure.validation import JsonbField

def test_jsonb_field_annotation():
    """Verify JsonbField adds correct annotation."""

    class TestModel(BaseModel):
        data: dict[str, Any] = JsonbField(default_factory=dict)

    # Verify annotation is present
    field_info = TestModel.model_fields['data']
    assert field_info.json_schema_extra == {"db_type": "jsonb"}

def test_missing_annotation_raises_error():
    """Verify validator catches missing annotations."""

    with pytest.raises(ValueError, match="missing required json_schema_extra"):
        class BadModel(BaseModel):
            data: dict[str, Any] = Field(default_factory=dict)

            @model_validator(mode="after")
            def _validate(self) -> "BadModel":
                return validate_jsonb_fields(self)

        BadModel()
```

## Best Practices

1. **Use JsonbField() for new models** - Most concise and maintainable
2. **Add validators to existing models** - Catches annotation issues early
3. **Consistent typing** - Always use `dict[str, Any]` for JSONB fields
4. **Document JSONB fields** - Add clear descriptions explaining the data structure
5. **Test validation** - Include tests that verify JSONB annotations

## ONEX v2.0 Compliance

These utilities follow ONEX v2.0 standards:
- ✅ Type-safe field definitions
- ✅ Clear error messaging
- ✅ Comprehensive documentation
- ✅ Pydantic v2 compatibility
- ✅ Zero runtime overhead (validation only at model creation)

## Performance

- **Zero overhead** for JsonbField() - it's just a wrapper around Field()
- **Minimal overhead** for validation - only runs once at model instantiation
- **No runtime cost** - validation happens during Pydantic model creation, not on every access

## Support

For questions or issues, refer to:
- This README for usage guidance
- `examples.py` for code examples
- `jsonb_validators.py` for implementation details
