# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Comprehensive unit tests for Any type validator.

Tests cover:
- Detection of Any in function parameters, return types, variables, type aliases
- Pydantic Field() with and without NOTE comments
- Exemption mechanisms (@allow_any, ONEX_EXCLUDE, file-level NOTE)
- String annotation handling (from __future__ import annotations)
- False positive prevention
- CI integration formatting
- Edge cases (empty files, syntax errors, async functions)

Validation Policy (from ADR):
- Any is BLOCKED in function signatures (parameters, return types)
- Any is ALLOWED only in Pydantic Field() definitions WITH required NOTE comment
- All other Any usage is BLOCKED
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from textwrap import dedent

import pytest

from omnibase_infra.enums import EnumAnyTypeViolation, EnumValidationSeverity
from omnibase_infra.models.validation.model_any_type_violation import (
    ModelAnyTypeViolation,
)
from omnibase_infra.validation.validator_any_type import (
    AnyTypeDetector,
    ModelAnyTypeValidationResult,
    validate_any_types_in_file,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_dir() -> Path:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


def _create_test_file(temp_dir: Path, content: str, filename: str = "test.py") -> Path:
    """Create a test Python file with given content.

    Args:
        temp_dir: Directory to create file in.
        content: Python source code content.
        filename: Name of the file to create.

    Returns:
        Path to created file.
    """
    filepath = temp_dir / filename
    filepath.write_text(dedent(content))
    return filepath


# =============================================================================
# Detection Tests: Function Parameters
# =============================================================================


class TestDetectionFunctionParameter:
    """Test detection of Any in function parameter annotations."""

    def test_any_in_single_parameter(self, temp_dir: Path) -> None:
        """Detect Any in a single function parameter."""
        code = """
        from typing import Any

        def process(data: Any) -> str:
            return str(data)
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumAnyTypeViolation.FUNCTION_PARAMETER
        assert "process(data)" in violations[0].context_name

    def test_any_in_multiple_parameters(self, temp_dir: Path) -> None:
        """Detect Any in multiple function parameters."""
        code = """
        from typing import Any

        def process(data: Any, config: Any, extra: str) -> str:
            return str(data)
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        # Should detect 2 violations (data and config parameters)
        assert len(violations) == 2
        assert all(
            v.violation_type == EnumAnyTypeViolation.FUNCTION_PARAMETER
            for v in violations
        )

    def test_any_in_args(self, temp_dir: Path) -> None:
        """Detect Any in *args annotation."""
        code = """
        from typing import Any

        def process(*args: Any) -> None:
            pass
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumAnyTypeViolation.FUNCTION_PARAMETER
        assert "*args" in violations[0].context_name

    def test_any_in_kwargs(self, temp_dir: Path) -> None:
        """Detect Any in **kwargs annotation."""
        code = """
        from typing import Any

        def process(**kwargs: Any) -> None:
            pass
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumAnyTypeViolation.FUNCTION_PARAMETER
        assert "**kwargs" in violations[0].context_name

    def test_any_in_positional_only_parameter(self, temp_dir: Path) -> None:
        """Detect Any in positional-only parameter (PEP 570)."""
        code = """
        from typing import Any

        def process(data: Any, /) -> str:
            return str(data)
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumAnyTypeViolation.FUNCTION_PARAMETER

    def test_any_in_keyword_only_parameter(self, temp_dir: Path) -> None:
        """Detect Any in keyword-only parameter."""
        code = """
        from typing import Any

        def process(*, data: Any) -> str:
            return str(data)
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumAnyTypeViolation.FUNCTION_PARAMETER


# =============================================================================
# Detection Tests: Return Types
# =============================================================================


class TestDetectionReturnType:
    """Test detection of Any in function return type annotations."""

    def test_any_as_return_type(self, temp_dir: Path) -> None:
        """Detect Any as function return type."""
        code = """
        from typing import Any

        def process(data: str) -> Any:
            return data
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumAnyTypeViolation.RETURN_TYPE
        assert violations[0].context_name == "process"

    def test_any_in_both_parameter_and_return(self, temp_dir: Path) -> None:
        """Detect Any in both parameter and return type."""
        code = """
        from typing import Any

        def process(data: Any) -> Any:
            return data
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 2
        violation_types = {v.violation_type for v in violations}
        assert EnumAnyTypeViolation.FUNCTION_PARAMETER in violation_types
        assert EnumAnyTypeViolation.RETURN_TYPE in violation_types


# =============================================================================
# Detection Tests: Variable Annotations
# =============================================================================


class TestDetectionVariableAnnotation:
    """Test detection of Any in variable annotations."""

    def test_any_in_variable_annotation(self, temp_dir: Path) -> None:
        """Detect Any in a local variable annotation."""
        code = """
        from typing import Any

        def process() -> str:
            data: Any = get_data()
            return str(data)
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumAnyTypeViolation.VARIABLE_ANNOTATION
        assert violations[0].context_name == "data"

    def test_any_in_module_level_variable(self, temp_dir: Path) -> None:
        """Detect Any in module-level variable annotation."""
        code = """
        from typing import Any

        global_data: Any = None
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumAnyTypeViolation.VARIABLE_ANNOTATION
        assert violations[0].context_name == "global_data"


# =============================================================================
# Detection Tests: Class Attributes
# =============================================================================


class TestDetectionClassAttribute:
    """Test detection of Any in class attribute annotations."""

    def test_any_in_class_attribute(self, temp_dir: Path) -> None:
        """Detect Any in class attribute annotation."""
        code = """
        from typing import Any

        class MyClass:
            data: Any = None
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumAnyTypeViolation.CLASS_ATTRIBUTE
        assert violations[0].context_name == "data"

    def test_any_in_class_attribute_no_default(self, temp_dir: Path) -> None:
        """Detect Any in class attribute without default value."""
        code = """
        from typing import Any

        class MyClass:
            data: Any
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumAnyTypeViolation.CLASS_ATTRIBUTE


# =============================================================================
# Detection Tests: Type Aliases
# =============================================================================


class TestDetectionTypeAlias:
    """Test detection of Any in type alias definitions."""

    def test_any_in_explicit_type_alias(self, temp_dir: Path) -> None:
        """Detect Any in explicit TypeAlias definition."""
        code = """
        from typing import Any, TypeAlias

        JsonType: TypeAlias = dict[str, Any]
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumAnyTypeViolation.TYPE_ALIAS
        assert violations[0].context_name == "JsonType"

    def test_any_in_implicit_type_alias_pascal_case(self, temp_dir: Path) -> None:
        """Detect Any in implicit type alias with PascalCase name."""
        code = """
        from typing import Any

        ConfigType = dict[str, Any]
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumAnyTypeViolation.TYPE_ALIAS
        assert violations[0].context_name == "ConfigType"

    def test_any_in_implicit_type_alias_type_suffix(self, temp_dir: Path) -> None:
        """Detect Any in implicit type alias ending with 'Type'."""
        code = """
        from typing import Any

        DataType = list[Any]
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumAnyTypeViolation.TYPE_ALIAS

    def test_any_in_type_alias_union(self, temp_dir: Path) -> None:
        """Detect Any in type alias with union."""
        code = """
        from typing import Any, TypeAlias

        ResultType: TypeAlias = str | Any
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumAnyTypeViolation.TYPE_ALIAS


# =============================================================================
# Detection Tests: Generic Type Arguments
# =============================================================================


class TestDetectionGenericArgument:
    """Test detection of Any in generic type arguments."""

    def test_any_in_list(self, temp_dir: Path) -> None:
        """Detect Any in list type argument."""
        code = """
        from typing import Any

        def process(items: list[Any]) -> int:
            return len(items)
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumAnyTypeViolation.GENERIC_ARGUMENT

    def test_any_in_dict_value(self, temp_dir: Path) -> None:
        """Detect Any in dict value type argument."""
        code = """
        from typing import Any

        def process(data: dict[str, Any]) -> int:
            return len(data)
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumAnyTypeViolation.GENERIC_ARGUMENT

    def test_any_in_optional(self, temp_dir: Path) -> None:
        """Detect Any in Optional type argument."""
        code = """
        from typing import Any, Optional

        def process(data: Optional[Any]) -> int:
            return 0
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumAnyTypeViolation.GENERIC_ARGUMENT

    def test_any_in_union(self, temp_dir: Path) -> None:
        """Detect Any in Union type argument."""
        code = """
        from typing import Any, Union

        def process(data: Union[str, Any]) -> int:
            return 0
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumAnyTypeViolation.GENERIC_ARGUMENT

    def test_any_in_pep604_union(self, temp_dir: Path) -> None:
        """Detect Any in PEP 604 union syntax (X | Any)."""
        code = """
        from typing import Any

        def process(data: str | Any) -> int:
            return 0
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumAnyTypeViolation.GENERIC_ARGUMENT

    def test_any_in_nested_generic(self, temp_dir: Path) -> None:
        """Detect Any in deeply nested generic type."""
        code = """
        from typing import Any

        def process(data: dict[str, list[tuple[int, Any]]]) -> int:
            return 0
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumAnyTypeViolation.GENERIC_ARGUMENT


# =============================================================================
# Detection Tests: typing.Any Attribute Access
# =============================================================================


class TestDetectionTypingAnyAttribute:
    """Test detection of typing.Any attribute access."""

    def test_typing_any_in_parameter(self, temp_dir: Path) -> None:
        """Detect typing.Any in function parameter."""
        code = """
        import typing

        def process(data: typing.Any) -> str:
            return str(data)
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumAnyTypeViolation.FUNCTION_PARAMETER

    def test_typing_any_in_return(self, temp_dir: Path) -> None:
        """Detect typing.Any in return type."""
        code = """
        import typing

        def process(data: str) -> typing.Any:
            return data
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumAnyTypeViolation.RETURN_TYPE


# =============================================================================
# Pydantic Field Tests
# =============================================================================


class TestPydanticFieldWithNote:
    """Test Pydantic Field() with NOTE comment (should be allowed)."""

    def test_field_with_inline_note_allowed(self, temp_dir: Path) -> None:
        """Field with inline NOTE comment with OMN ticket should be allowed."""
        code = """
        from typing import Any
        from pydantic import BaseModel, Field

        class MyModel(BaseModel):
            data: Any = Field(...)  # NOTE: OMN-1234 - Required for JSON payload
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 0

    def test_field_with_preceding_note_comment_allowed(self, temp_dir: Path) -> None:
        """Field with NOTE comment with OMN ticket in preceding line should be allowed."""
        code = """
        from typing import Any
        from pydantic import BaseModel, Field

        class MyModel(BaseModel):
            # NOTE: OMN-1234 - Required for JSON schema dynamic typing
            data: Any = Field(...)
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 0

    def test_field_with_note_block_comment_allowed(self, temp_dir: Path) -> None:
        """Field with NOTE comment with OMN ticket within lookback range should be allowed."""
        code = """
        from typing import Any
        from pydantic import BaseModel, Field

        class MyModel(BaseModel):
            # This field stores arbitrary JSON data
            # NOTE: OMN-5678 - Required for JSON compatibility
            # The validator handles type coercion
            data: Any = Field(...)
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 0


class TestPydanticFieldWithoutNote:
    """Test Pydantic Field() without NOTE comment (should be blocked)."""

    def test_field_without_note_blocked(self, temp_dir: Path) -> None:
        """Field without NOTE comment should be blocked."""
        code = """
        from typing import Any
        from pydantic import BaseModel, Field

        class MyModel(BaseModel):
            data: Any = Field(...)
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumAnyTypeViolation.FIELD_MISSING_NOTE
        assert violations[0].context_name == "data"

    def test_field_with_unrelated_comment_blocked(self, temp_dir: Path) -> None:
        """Field with unrelated comment should be blocked."""
        code = """
        from typing import Any
        from pydantic import BaseModel, Field

        class MyModel(BaseModel):
            # This stores data
            data: Any = Field(...)
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumAnyTypeViolation.FIELD_MISSING_NOTE

    def test_field_with_note_too_far_blocked(self, temp_dir: Path) -> None:
        """Field with NOTE comment beyond lookback range should be blocked.

        Note: The NOTE comment must be placed far from both the import (to avoid
        file-level NOTE detection) and far from the field (to test the lookback
        range limitation).
        """
        code = """
        from typing import Any
        from pydantic import BaseModel, Field


        # Start of class - far from import so no file-level NOTE applies
        class MyModel(BaseModel):
            '''A model with multiple fields.'''

            # NOTE: This comment documents other_field, not the Any field below
            other_field: str = Field(...)

            another_field: int = Field(...)

            third_field: float = Field(...)

            # No NOTE comment here
            data: Any = Field(...)
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        # The NOTE is more than 5 lines away from `data: Any` field
        # and file-level NOTE only applies when NOTE is within 5 lines of import
        assert len(violations) == 1
        assert violations[0].violation_type == EnumAnyTypeViolation.FIELD_MISSING_NOTE


class TestFileLevelNoteComment:
    """Test file-level NOTE comment near Any import."""

    def test_file_level_note_exempts_all_fields(self, temp_dir: Path) -> None:
        """File-level NOTE comment with OMN ticket near import exempts all Field() usages."""
        code = """
        from typing import Any  # NOTE: OMN-1234 - JSON payload fields in this model

        from pydantic import BaseModel, Field

        class MyModel(BaseModel):
            data: Any = Field(...)
            config: Any = Field(...)
            metadata: Any = Field(...)
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        # All Field() usages should be exempted by file-level NOTE
        assert len(violations) == 0

    def test_file_level_note_above_import(self, temp_dir: Path) -> None:
        """File-level NOTE comment with OMN ticket above import should work."""
        code = """
        # NOTE: OMN-1234 - Required for JSON compatibility
        from typing import Any

        from pydantic import BaseModel, Field

        class MyModel(BaseModel):
            data: Any = Field(...)
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 0

    def test_file_level_note_below_import(self, temp_dir: Path) -> None:
        """File-level NOTE comment with OMN ticket below import should work."""
        code = """
        from typing import Any
        # NOTE: OMN-5678 - Dynamic JSON payloads

        from pydantic import BaseModel, Field

        class MyModel(BaseModel):
            data: Any = Field(...)
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 0

    def test_file_level_note_does_not_exempt_non_field(self, temp_dir: Path) -> None:
        """File-level NOTE should not exempt non-Field() Any usages."""
        code = """
        from typing import Any  # NOTE: OMN-1234 - JSON payload fields

        from pydantic import BaseModel, Field

        class MyModel(BaseModel):
            data: Any = Field(...)  # Exempted

        def process(data: Any) -> Any:  # Not exempted
            return data
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        # Function signature violations should still be detected
        assert len(violations) == 2
        assert all(
            v.violation_type
            in (
                EnumAnyTypeViolation.FUNCTION_PARAMETER,
                EnumAnyTypeViolation.RETURN_TYPE,
            )
            for v in violations
        )


# =============================================================================
# Exemption Tests: @allow_any Decorator
# =============================================================================


class TestAllowAnyDecorator:
    """Test @allow_any decorator exemption mechanism."""

    def test_allow_any_exempts_function(self, temp_dir: Path) -> None:
        """@allow_any decorator exempts function from validation."""
        code = """
        from typing import Any

        def allow_any(func):
            return func

        @allow_any
        def process(data: Any) -> Any:
            return data
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 0

    def test_allow_any_with_reason_exempts_function(self, temp_dir: Path) -> None:
        """@allow_any("reason") decorator with argument exempts function."""
        code = """
        from typing import Any

        def allow_any(reason=None):
            def decorator(func):
                return func
            return decorator

        @allow_any("required for legacy API compatibility")
        def process(data: Any) -> Any:
            return data
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 0

    def test_allow_any_type_decorator_alias(self, temp_dir: Path) -> None:
        """@allow_any_type decorator alias also works."""
        code = """
        from typing import Any

        def allow_any_type(func):
            return func

        @allow_any_type
        def process(data: Any) -> Any:
            return data
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 0

    def test_allow_any_exempts_class(self, temp_dir: Path) -> None:
        """@allow_any decorator on class exempts all methods."""
        code = """
        from typing import Any

        def allow_any(cls):
            return cls

        @allow_any
        class MyClass:
            data: Any = None

            def process(self, data: Any) -> Any:
                return data
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 0

    def test_undecorated_function_still_detected(self, temp_dir: Path) -> None:
        """Functions without @allow_any still have violations detected."""
        code = """
        from typing import Any

        def allow_any(func):
            return func

        @allow_any
        def exempted(data: Any) -> Any:
            return data

        def not_exempted(data: Any) -> Any:
            return data
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 2
        assert all(v.context_name.startswith("not_exempted") for v in violations)


# =============================================================================
# Exemption Tests: ONEX_EXCLUDE Comment
# =============================================================================


class TestOnexExcludeComment:
    """Test ONEX_EXCLUDE: any_type comment exemption mechanism."""

    def test_onex_exclude_same_line(self, temp_dir: Path) -> None:
        """ONEX_EXCLUDE comment on same line exempts violation."""
        code = """
        from typing import Any

        def process(data: Any) -> str:  # ONEX_EXCLUDE: any_type
            return str(data)
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 0

    def test_onex_exclude_preceding_line(self, temp_dir: Path) -> None:
        """ONEX_EXCLUDE comment on preceding line exempts following code."""
        code = """
        from typing import Any

        # ONEX_EXCLUDE: any_type
        def process(data: Any) -> Any:
            return data
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 0

    def test_onex_exclude_with_additional_context(self, temp_dir: Path) -> None:
        """ONEX_EXCLUDE with additional context still works."""
        code = """
        from typing import Any

        # ONEX_EXCLUDE: any_type - Legacy API requires dynamic typing
        def process(data: Any) -> str:
            return str(data)
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 0

    def test_onex_exclude_range(self, temp_dir: Path) -> None:
        """ONEX_EXCLUDE exempts lines within its range (5 lines)."""
        code = """
        from typing import Any

        # ONEX_EXCLUDE: any_type
        def process(
            data: Any,
            config: Any,
            extra: Any
        ) -> Any:
            return data
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        # All lines within the 5-line range should be exempted
        assert len(violations) == 0


# =============================================================================
# String Annotation Tests
# =============================================================================


class TestStringAnnotations:
    """Test with `from __future__ import annotations` enabled."""

    def test_future_annotations_still_detected(self, temp_dir: Path) -> None:
        """Any usage detected even with future annotations."""
        code = """
        from __future__ import annotations
        from typing import Any

        def process(data: Any) -> str:
            return str(data)
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        # AST still parses the annotation as Any name
        assert len(violations) == 1
        assert violations[0].violation_type == EnumAnyTypeViolation.FUNCTION_PARAMETER

    def test_future_annotations_return_type(self, temp_dir: Path) -> None:
        """Return type Any detected with future annotations."""
        code = """
        from __future__ import annotations
        from typing import Any

        def process(data: str) -> Any:
            return data
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumAnyTypeViolation.RETURN_TYPE


# =============================================================================
# False Positive Prevention Tests
# =============================================================================


class TestFalsePositivePrevention:
    """Test that valid types are not flagged as violations."""

    def test_object_type_not_flagged(self, temp_dir: Path) -> None:
        """Valid 'object' type should not be flagged."""
        code = """
        def process(data: object) -> object:
            return data
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 0

    def test_generic_types_without_any_not_flagged(self, temp_dir: Path) -> None:
        """Generic types without Any should not be flagged."""
        code = """
        def process(items: list[str], mapping: dict[str, int]) -> tuple[str, int]:
            return items[0], mapping.get("key", 0)
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 0

    def test_union_without_any_not_flagged(self, temp_dir: Path) -> None:
        """Union types without Any should not be flagged."""
        code = """
        from typing import Union

        def process(data: Union[str, int, None]) -> str | int:
            return data or ""
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 0

    def test_optional_without_any_not_flagged(self, temp_dir: Path) -> None:
        """Optional without Any should not be flagged."""
        code = """
        from typing import Optional

        def process(data: Optional[str]) -> str | None:
            return data
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 0

    def test_custom_type_alias_without_any_not_flagged(self, temp_dir: Path) -> None:
        """Custom type alias without Any should not be flagged."""
        code = """
        from typing import TypeAlias

        JsonType: TypeAlias = dict[str, str | int | list[str]]

        def process(data: JsonType) -> JsonType:
            return data
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 0

    def test_protocol_not_flagged(self, temp_dir: Path) -> None:
        """Protocol types should not be flagged."""
        code = """
        from typing import Protocol

        class Processable(Protocol):
            def process(self) -> str: ...

        def handle(item: Processable) -> str:
            return item.process()
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 0

    def test_variable_named_any_not_flagged(self, temp_dir: Path) -> None:
        """Variable named 'any' (lowercase) should not be flagged."""
        code = """
        any_result = [1, 2, 3]
        any_value: int = 42
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 0


# =============================================================================
# CI Integration Tests
# =============================================================================


class TestModelAnyTypeValidationResultFactoryMethod:
    """Test ModelAnyTypeValidationResult.from_violations() factory method."""

    def test_from_violations_with_empty_list(self) -> None:
        """from_violations with empty list creates passed result."""
        result = ModelAnyTypeValidationResult.from_violations([], files_checked=5)

        assert result.passed is True
        assert result.total_violations == 0
        assert result.blocking_count == 0
        assert result.files_checked == 5
        assert result.violations == []

    def test_from_violations_with_blocking_violations(self) -> None:
        """from_violations with error-severity violations creates failed result."""
        violations = [
            ModelAnyTypeViolation(
                file_path=Path("/test/file.py"),
                line_number=10,
                column=5,
                violation_type=EnumAnyTypeViolation.FUNCTION_PARAMETER,
                code_snippet="def test(data: Any)",
                suggestion="Replace Any with specific type",
                severity=EnumValidationSeverity.ERROR,
                context_name="test",
            ),
            ModelAnyTypeViolation(
                file_path=Path("/test/file2.py"),
                line_number=20,
                column=10,
                violation_type=EnumAnyTypeViolation.RETURN_TYPE,
                code_snippet="def test() -> Any:",
                suggestion="Replace Any with specific type",
                severity=EnumValidationSeverity.ERROR,
                context_name="test",
            ),
        ]
        result = ModelAnyTypeValidationResult.from_violations(
            violations, files_checked=10
        )

        assert result.passed is False
        assert result.total_violations == 2
        assert result.blocking_count == 2
        assert result.files_checked == 10

    def test_from_violations_with_warning_only(self) -> None:
        """from_violations with only warnings creates passed result."""
        violations = [
            ModelAnyTypeViolation(
                file_path=Path("/test/file.py"),
                line_number=10,
                column=5,
                violation_type=EnumAnyTypeViolation.FUNCTION_PARAMETER,
                code_snippet="def test(data: Any)",
                suggestion="Replace Any with specific type",
                severity=EnumValidationSeverity.WARNING,
                context_name="test",
            )
        ]
        result = ModelAnyTypeValidationResult.from_violations(
            violations, files_checked=3
        )

        assert result.passed is True
        assert result.total_violations == 1
        assert result.blocking_count == 0


class TestModelAnyTypeValidationResultFormatting:
    """Test ModelAnyTypeValidationResult formatting methods."""

    def test_format_for_ci_empty(self) -> None:
        """format_for_ci with no violations returns empty list."""
        result = ModelAnyTypeValidationResult.from_violations([])

        assert result.format_for_ci() == []

    def test_format_for_ci_with_violations(self) -> None:
        """format_for_ci formats all violations."""
        violations = [
            ModelAnyTypeViolation(
                file_path=Path("/test/file.py"),
                line_number=10,
                column=5,
                violation_type=EnumAnyTypeViolation.FUNCTION_PARAMETER,
                code_snippet="def test(data: Any)",
                suggestion="Replace Any with specific type",
                severity=EnumValidationSeverity.ERROR,
                context_name="test",
            )
        ]
        result = ModelAnyTypeValidationResult.from_violations(violations)
        ci_output = result.format_for_ci()

        assert len(ci_output) == 1
        assert "::error" in ci_output[0]
        assert "file=/test/file.py" in ci_output[0]
        assert "line=10" in ci_output[0]
        assert "col=5" in ci_output[0]

    def test_format_summary_passed(self) -> None:
        """format_summary shows PASSED for no violations."""
        result = ModelAnyTypeValidationResult.from_violations([], files_checked=5)
        summary = result.format_summary()

        assert "PASSED" in summary
        assert "Files checked: 5" in summary
        assert "Total violations: 0" in summary
        assert "Blocking violations: 0" in summary

    def test_format_summary_failed(self) -> None:
        """format_summary shows FAILED for blocking violations."""
        violations = [
            ModelAnyTypeViolation(
                file_path=Path("/test/file.py"),
                line_number=10,
                column=5,
                violation_type=EnumAnyTypeViolation.FUNCTION_PARAMETER,
                code_snippet="def test(data: Any)",
                suggestion="Replace Any with specific type",
                severity=EnumValidationSeverity.ERROR,
                context_name="test",
            )
        ]
        result = ModelAnyTypeValidationResult.from_violations(
            violations, files_checked=10
        )
        summary = result.format_summary()

        assert "FAILED" in summary
        assert "Files checked: 10" in summary
        assert "Total violations: 1" in summary
        assert "Blocking violations: 1" in summary


class TestModelAnyTypeValidationResultPassedProperty:
    """Test the passed property behavior."""

    def test_passed_with_zero_violations(self) -> None:
        """passed is True when blocking_count is 0."""
        result = ModelAnyTypeValidationResult(
            passed=True,
            violations=[],
            files_checked=5,
            total_violations=0,
            blocking_count=0,
        )

        assert result.passed is True

    def test_passed_with_blocking_violations(self) -> None:
        """passed is False when blocking_count > 0."""
        result = ModelAnyTypeValidationResult(
            passed=False,
            violations=[],
            files_checked=5,
            total_violations=1,
            blocking_count=1,
        )

        assert result.passed is False


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCaseEmptyFile:
    """Test handling of empty files."""

    def test_empty_file(self, temp_dir: Path) -> None:
        """Empty file returns no violations."""
        code = ""
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 0

    def test_whitespace_only_file(self, temp_dir: Path) -> None:
        """Whitespace-only file returns no violations."""
        code = "   \n\n   \n"
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 0

    def test_comments_only_file(self, temp_dir: Path) -> None:
        """Comments-only file returns no violations."""
        code = """
        # This is a comment
        # Another comment
        '''
        A docstring
        '''
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 0


class TestEdgeCaseSyntaxErrors:
    """Test handling of files with syntax errors."""

    def test_syntax_error_returns_syntax_error_violation(self, temp_dir: Path) -> None:
        """File with syntax error returns SYNTAX_ERROR violation."""
        code = """
        from typing import Any

        def broken(
            return None
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumAnyTypeViolation.SYNTAX_ERROR
        assert "Syntax error" in violations[0].code_snippet

    def test_syntax_error_includes_line_info(self, temp_dir: Path) -> None:
        """Syntax error violation includes line number."""
        code = """
        def test():
            x = (1 + 2
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumAnyTypeViolation.SYNTAX_ERROR
        assert violations[0].line_number >= 1


class TestEdgeCaseDeeplyNestedTypes:
    """Test handling of deeply nested type expressions."""

    def test_deeply_nested_any_detected(self, temp_dir: Path) -> None:
        """Any in deeply nested type is detected."""
        code = """
        from typing import Any

        def process(
            data: dict[str, list[tuple[int, dict[str, list[Any]]]]]
        ) -> str:
            return ""
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumAnyTypeViolation.GENERIC_ARGUMENT

    def test_multiple_any_in_nested_type(self, temp_dir: Path) -> None:
        """Multiple Any usages in nested type all detected."""
        code = """
        from typing import Any

        def process(data: dict[Any, list[Any]]) -> str:
            return ""
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        # Should detect both Any usages
        assert len(violations) == 2
        assert all(
            v.violation_type == EnumAnyTypeViolation.GENERIC_ARGUMENT
            for v in violations
        )


class TestEdgeCaseAsyncFunctions:
    """Test handling of async function definitions."""

    def test_async_function_parameter(self, temp_dir: Path) -> None:
        """Any in async function parameter is detected."""
        code = """
        from typing import Any

        async def process(data: Any) -> str:
            return str(data)
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumAnyTypeViolation.FUNCTION_PARAMETER
        assert violations[0].context_name == "process(data)"

    def test_async_function_return_type(self, temp_dir: Path) -> None:
        """Any in async function return type is detected."""
        code = """
        from typing import Any

        async def process(data: str) -> Any:
            return data
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumAnyTypeViolation.RETURN_TYPE
        assert violations[0].context_name == "process"


class TestEdgeCaseMethods:
    """Test handling of class methods."""

    def test_method_parameter(self, temp_dir: Path) -> None:
        """Any in method parameter is detected."""
        code = """
        from typing import Any

        class MyClass:
            def process(self, data: Any) -> str:
                return str(data)
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumAnyTypeViolation.FUNCTION_PARAMETER

    def test_classmethod_parameter(self, temp_dir: Path) -> None:
        """Any in classmethod parameter is detected."""
        code = """
        from typing import Any

        class MyClass:
            @classmethod
            def process(cls, data: Any) -> str:
                return str(data)
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumAnyTypeViolation.FUNCTION_PARAMETER

    def test_staticmethod_parameter(self, temp_dir: Path) -> None:
        """Any in staticmethod parameter is detected."""
        code = """
        from typing import Any

        class MyClass:
            @staticmethod
            def process(data: Any) -> str:
                return str(data)
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumAnyTypeViolation.FUNCTION_PARAMETER


# =============================================================================
# Violation Model Tests
# =============================================================================


class TestModelAnyTypeViolation:
    """Test ModelAnyTypeViolation model behavior."""

    def test_is_blocking_error_severity(self) -> None:
        """is_blocking returns True for error severity."""
        violation = ModelAnyTypeViolation(
            file_path=Path("/test/file.py"),
            line_number=10,
            column=5,
            violation_type=EnumAnyTypeViolation.FUNCTION_PARAMETER,
            code_snippet="def test(data: Any)",
            suggestion="Replace Any",
            severity=EnumValidationSeverity.ERROR,
        )

        assert violation.is_blocking() is True

    def test_is_blocking_warning_severity(self) -> None:
        """is_blocking returns False for warning severity."""
        violation = ModelAnyTypeViolation(
            file_path=Path("/test/file.py"),
            line_number=10,
            column=5,
            violation_type=EnumAnyTypeViolation.FUNCTION_PARAMETER,
            code_snippet="def test(data: Any)",
            suggestion="Replace Any",
            severity=EnumValidationSeverity.WARNING,
        )

        assert violation.is_blocking() is False

    def test_format_for_ci_error(self) -> None:
        """format_for_ci produces GitHub Actions error annotation."""
        violation = ModelAnyTypeViolation(
            file_path=Path("/test/file.py"),
            line_number=10,
            column=5,
            violation_type=EnumAnyTypeViolation.FUNCTION_PARAMETER,
            code_snippet="def test(data: Any)",
            suggestion="Replace Any",
            severity=EnumValidationSeverity.ERROR,
        )
        output = violation.format_for_ci()

        assert output.startswith("::error")
        assert "file=/test/file.py" in output
        assert "line=10" in output
        assert "col=5" in output
        assert "function_parameter" in output

    def test_format_for_ci_warning(self) -> None:
        """format_for_ci produces GitHub Actions warning annotation."""
        violation = ModelAnyTypeViolation(
            file_path=Path("/test/file.py"),
            line_number=10,
            column=5,
            violation_type=EnumAnyTypeViolation.FUNCTION_PARAMETER,
            code_snippet="def test(data: Any)",
            suggestion="Replace Any",
            severity=EnumValidationSeverity.WARNING,
        )
        output = violation.format_for_ci()

        assert output.startswith("::warning")

    def test_format_human_readable(self) -> None:
        """format_human_readable produces human-friendly output."""
        violation = ModelAnyTypeViolation(
            file_path=Path("/test/file.py"),
            line_number=10,
            column=5,
            violation_type=EnumAnyTypeViolation.FUNCTION_PARAMETER,
            code_snippet="def test(data: Any)",
            suggestion="Replace Any with specific type",
            severity=EnumValidationSeverity.ERROR,
            context_name="test",
        )
        output = violation.format_human_readable()

        assert "/test/file.py:10:5" in output
        assert "function_parameter" in output
        assert "def test(data: Any)" in output
        assert "Replace Any with specific type" in output
        assert "Context: test" in output

    def test_format_human_readable_no_context(self) -> None:
        """format_human_readable works without context name."""
        violation = ModelAnyTypeViolation(
            file_path=Path("/test/file.py"),
            line_number=10,
            column=5,
            violation_type=EnumAnyTypeViolation.FUNCTION_PARAMETER,
            code_snippet="def test(data: Any)",
            suggestion="Replace Any",
            severity=EnumValidationSeverity.ERROR,
            context_name="",
        )
        output = violation.format_human_readable()

        assert "Context:" not in output


# =============================================================================
# Enum Tests
# =============================================================================


class TestEnumAnyTypeViolation:
    """Test EnumAnyTypeViolation enum behavior."""

    def test_is_exemptable_syntax_error(self) -> None:
        """SYNTAX_ERROR is not exemptable."""
        assert EnumAnyTypeViolation.SYNTAX_ERROR.is_exemptable is False

    def test_is_exemptable_other_types(self) -> None:
        """Other violation types are exemptable."""
        exemptable_types = [
            EnumAnyTypeViolation.FUNCTION_PARAMETER,
            EnumAnyTypeViolation.RETURN_TYPE,
            EnumAnyTypeViolation.FIELD_MISSING_NOTE,
            EnumAnyTypeViolation.VARIABLE_ANNOTATION,
            EnumAnyTypeViolation.TYPE_ALIAS,
            EnumAnyTypeViolation.CLASS_ATTRIBUTE,
            EnumAnyTypeViolation.GENERIC_ARGUMENT,
        ]

        for vtype in exemptable_types:
            assert vtype.is_exemptable is True

    def test_suggestion_not_empty(self) -> None:
        """All violation types have non-empty suggestions."""
        for vtype in EnumAnyTypeViolation:
            assert vtype.suggestion, f"{vtype} has empty suggestion"
            assert len(vtype.suggestion) > 10, f"{vtype} suggestion too short"


# =============================================================================
# AnyTypeDetector Direct Tests
# =============================================================================


class TestAnyTypeDetectorInternals:
    """Test AnyTypeDetector internal methods directly."""

    def test_is_likely_type_alias_name_pascal_case(self) -> None:
        """PascalCase names are recognized as type aliases."""
        detector = AnyTypeDetector("test.py", [])

        assert detector._is_likely_type_alias_name("JsonType") is True
        assert detector._is_likely_type_alias_name("ConfigData") is True
        assert detector._is_likely_type_alias_name("Result") is True

    def test_is_likely_type_alias_name_type_suffix(self) -> None:
        """Names ending with Type are recognized as type aliases."""
        detector = AnyTypeDetector("test.py", [])

        assert detector._is_likely_type_alias_name("JsonType") is True
        assert detector._is_likely_type_alias_name("DataType") is True
        assert detector._is_likely_type_alias_name("ResultTypes") is True

    def test_is_likely_type_alias_name_snake_case(self) -> None:
        """snake_case names are not type aliases."""
        detector = AnyTypeDetector("test.py", [])

        assert detector._is_likely_type_alias_name("json_data") is False
        assert detector._is_likely_type_alias_name("config_value") is False

    def test_is_likely_type_alias_name_all_caps(self) -> None:
        """ALL_CAPS names are not type aliases (constants)."""
        detector = AnyTypeDetector("test.py", [])

        assert detector._is_likely_type_alias_name("JSON") is False
        assert detector._is_likely_type_alias_name("CONFIG") is False

    def test_is_likely_type_alias_name_empty(self) -> None:
        """Empty names are not type aliases."""
        detector = AnyTypeDetector("test.py", [])

        assert detector._is_likely_type_alias_name("") is False

    def test_check_file_level_note_found(self) -> None:
        """File-level NOTE with OMN ticket near import is detected."""
        source_lines = [
            "from typing import Any  # NOTE: OMN-1234 - Required for JSON compatibility",
            "",
            "class Model:",
            "    pass",
        ]
        detector = AnyTypeDetector("test.py", source_lines)

        assert detector.has_file_level_note is True

    def test_check_file_level_note_not_found(self) -> None:
        """Missing file-level NOTE is detected."""
        source_lines = [
            "from typing import Any",
            "",
            "class Model:",
            "    pass",
        ]
        detector = AnyTypeDetector("test.py", source_lines)

        assert detector.has_file_level_note is False

    def test_check_file_level_note_import_typing(self) -> None:
        """File-level NOTE with OMN ticket with 'import typing' is detected."""
        source_lines = [
            "import typing  # NOTE: OMN-1234 - Required for JSON compatibility",
            "",
            "class Model:",
            "    pass",
        ]
        detector = AnyTypeDetector("test.py", source_lines)

        assert detector.has_file_level_note is True

    def test_check_file_level_note_without_ticket_not_valid(self) -> None:
        """File-level NOTE without OMN ticket is NOT valid."""
        source_lines = [
            "from typing import Any  # NOTE: Any required for JSON compatibility",
            "",
            "class Model:",
            "    pass",
        ]
        detector = AnyTypeDetector("test.py", source_lines)

        # Old format without OMN ticket should NOT be accepted
        assert detector.has_file_level_note is False


# =============================================================================
# Directory Validation Tests
# =============================================================================


class TestValidateAnyTypesDirectory:
    """Test directory-level validation functions."""

    def test_validate_any_types_recursive(self, temp_dir: Path) -> None:
        """Validate recursively finds violations in subdirectories."""
        from omnibase_infra.validation.validator_any_type import validate_any_types

        # Create subdirectory with file
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        code = """
        from typing import Any

        def process(data: Any) -> str:
            return str(data)
        """
        _create_test_file(subdir, code, "module.py")

        violations = validate_any_types(temp_dir, recursive=True)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumAnyTypeViolation.FUNCTION_PARAMETER

    def test_validate_any_types_non_recursive(self, temp_dir: Path) -> None:
        """Non-recursive validation only checks immediate directory."""
        from omnibase_infra.validation.validator_any_type import validate_any_types

        # Create file in root
        code_root = """
        from typing import Any

        def root_func(data: Any) -> str:
            return str(data)
        """
        _create_test_file(temp_dir, code_root, "root.py")

        # Create subdirectory with file
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        code_sub = """
        from typing import Any

        def sub_func(data: Any) -> str:
            return str(data)
        """
        _create_test_file(subdir, code_sub, "sub.py")

        violations = validate_any_types(temp_dir, recursive=False)

        # Only root file should be checked
        assert len(violations) == 1
        assert "root.py" in str(violations[0].file_path)

    def test_validate_any_types_ci(self, temp_dir: Path) -> None:
        """CI validation returns structured result."""
        from omnibase_infra.validation.validator_any_type import validate_any_types_ci

        code = """
        from typing import Any

        def process(data: Any) -> Any:
            return data
        """
        _create_test_file(temp_dir, code, "module.py")

        result = validate_any_types_ci(temp_dir)

        assert result.passed is False
        assert result.files_checked >= 1
        assert result.total_violations == 2
        assert result.blocking_count == 2

    def test_validate_any_types_ci_no_violations(self, temp_dir: Path) -> None:
        """CI validation with no violations returns passed result."""
        from omnibase_infra.validation.validator_any_type import validate_any_types_ci

        code = """
        def process(data: object) -> str:
            return str(data)
        """
        _create_test_file(temp_dir, code, "clean.py")

        result = validate_any_types_ci(temp_dir)

        assert result.passed is True
        assert result.files_checked >= 1
        assert result.total_violations == 0


class TestFileSkipping:
    """Test file skipping patterns."""

    def test_skip_test_files(self, temp_dir: Path) -> None:
        """Files in tests/ directories are skipped."""
        from omnibase_infra.validation.validator_any_type import validate_any_types

        # Create tests directory with violation file
        tests_dir = temp_dir / "tests"
        tests_dir.mkdir()
        code = """
        from typing import Any

        def test_func(data: Any) -> Any:
            return data
        """
        _create_test_file(tests_dir, code, "test_module.py")

        violations = validate_any_types(temp_dir, recursive=True)

        # Test files should be skipped
        assert len(violations) == 0

    def test_skip_underscore_prefixed_files(self, temp_dir: Path) -> None:
        """Files starting with underscore are skipped."""
        from omnibase_infra.validation.validator_any_type import validate_any_types

        code = """
        from typing import Any

        def private_func(data: Any) -> Any:
            return data
        """
        _create_test_file(temp_dir, code, "_private.py")

        violations = validate_any_types(temp_dir, recursive=True)

        # Underscore-prefixed files should be skipped
        assert len(violations) == 0

    def test_skip_archive_directory(self, temp_dir: Path) -> None:
        """Files in archive/ directories are skipped."""
        from omnibase_infra.validation.validator_any_type import validate_any_types

        # Create archive directory with violation file
        archive_dir = temp_dir / "archive"
        archive_dir.mkdir()
        code = """
        from typing import Any

        def archived_func(data: Any) -> Any:
            return data
        """
        _create_test_file(archive_dir, code, "old.py")

        violations = validate_any_types(temp_dir, recursive=True)

        # Archive files should be skipped
        assert len(violations) == 0


# =============================================================================
# Additional Edge Cases
# =============================================================================


class TestPydanticFieldAttribute:
    """Test pydantic.Field attribute access pattern."""

    def test_pydantic_field_attribute_with_note(self, temp_dir: Path) -> None:
        """pydantic.Field attribute access with NOTE and OMN ticket is allowed."""
        code = """
        from typing import Any
        import pydantic

        class MyModel(pydantic.BaseModel):
            # NOTE: OMN-1234 - Required for JSON payload
            data: Any = pydantic.Field(...)
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 0

    def test_pydantic_field_attribute_without_note(self, temp_dir: Path) -> None:
        """pydantic.Field attribute access without NOTE is blocked."""
        code = """
        from typing import Any
        import pydantic

        class MyModel(pydantic.BaseModel):
            data: Any = pydantic.Field(...)
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumAnyTypeViolation.FIELD_MISSING_NOTE


class TestAnnotationWithoutValue:
    """Test annotation without assignment value."""

    def test_class_attribute_annotation_only(self, temp_dir: Path) -> None:
        """Class attribute with only annotation (no value) is detected."""
        code = """
        from typing import Any

        class MyClass:
            data: Any
            config: Any
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 2
        assert all(
            v.violation_type == EnumAnyTypeViolation.CLASS_ATTRIBUTE for v in violations
        )


class TestTypingAttributeTypeAlias:
    """Test typing module type alias attribute pattern."""

    def test_typing_type_alias(self, temp_dir: Path) -> None:
        """typing.TypeAlias with Any is detected."""
        code = """
        from typing import Any
        import typing

        JsonType: typing.TypeAlias = dict[str, Any]
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumAnyTypeViolation.TYPE_ALIAS


class TestNestedClass:
    """Test nested class handling."""

    def test_nested_class_attribute(self, temp_dir: Path) -> None:
        """Any in nested class attribute is detected."""
        code = """
        from typing import Any

        class Outer:
            class Inner:
                data: Any = None
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumAnyTypeViolation.CLASS_ATTRIBUTE

    def test_nested_class_method(self, temp_dir: Path) -> None:
        """Any in nested class method is detected."""
        code = """
        from typing import Any

        class Outer:
            class Inner:
                def process(self, data: Any) -> str:
                    return str(data)
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumAnyTypeViolation.FUNCTION_PARAMETER


class TestLambdaAndComprehensions:
    """Test handling of lambda and comprehension patterns."""

    def test_variable_in_comprehension_not_flagged(self, temp_dir: Path) -> None:
        """Variables in comprehensions without Any are not flagged."""
        code = """
        def process() -> list[int]:
            return [x for x in range(10)]
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 0


class TestMultipleImportStyles:
    """Test handling of various import styles."""

    def test_from_typing_import_multiple(self, temp_dir: Path) -> None:
        """Multiple imports from typing with Any."""
        code = """
        from typing import Any, List, Dict

        def process(data: List[Any]) -> Dict[str, Any]:
            return {"result": data[0]}
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 2
        assert all(
            v.violation_type == EnumAnyTypeViolation.GENERIC_ARGUMENT
            for v in violations
        )


# =============================================================================
# Multi-line Import Tests
# =============================================================================


class TestMultiLineImports:
    """Test handling of multi-line import statements."""

    def test_multiline_parenthesized_import_with_note(self, temp_dir: Path) -> None:
        """Multi-line parenthesized import with NOTE should work."""
        code = """
        # NOTE: OMN-1234 - JSON payload fields
        from typing import (
            Any,
            Dict,
            List,
        )

        from pydantic import BaseModel, Field

        class MyModel(BaseModel):
            data: Any = Field(...)
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 0

    def test_multiline_import_any_on_separate_line(self, temp_dir: Path) -> None:
        """Multi-line import with Any on separate line and NOTE should work."""
        code = """
        from typing import (
            Dict,
            Any,  # NOTE: OMN-5678 - Dynamic payloads
            List,
        )

        from pydantic import BaseModel, Field

        class MyModel(BaseModel):
            data: Any = Field(...)
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 0

    def test_multiline_import_without_note_fails(self, temp_dir: Path) -> None:
        """Multi-line import without NOTE should still require field-level NOTE."""
        code = """
        from typing import (
            Any,
            Dict,
        )

        from pydantic import BaseModel, Field

        class MyModel(BaseModel):
            data: Any = Field(...)
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumAnyTypeViolation.FIELD_MISSING_NOTE


class TestOmnTicketValidation:
    """Test that NOTE comments require OMN ticket codes."""

    def test_note_without_omn_ticket_rejected(self, temp_dir: Path) -> None:
        """NOTE comment without OMN ticket should be rejected."""
        code = """
        from typing import Any
        from pydantic import BaseModel, Field

        class MyModel(BaseModel):
            # NOTE: Any required for JSON payload (missing OMN ticket)
            data: Any = Field(...)
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumAnyTypeViolation.FIELD_MISSING_NOTE

    def test_note_with_lowercase_omn_ticket_accepted(self, temp_dir: Path) -> None:
        """NOTE comment with lowercase omn ticket should be accepted (case-insensitive)."""
        code = """
        from typing import Any
        from pydantic import BaseModel, Field

        class MyModel(BaseModel):
            # NOTE: omn-1234 - Required for JSON payload
            data: Any = Field(...)
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 0

    def test_note_with_mixed_case_omn_ticket_accepted(self, temp_dir: Path) -> None:
        """NOTE comment with mixed case OMN ticket should be accepted."""
        code = """
        from typing import Any
        from pydantic import BaseModel, Field

        class MyModel(BaseModel):
            # NOTE: Omn-1234 - Required for JSON payload
            data: Any = Field(...)
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 0

    def test_note_with_long_omn_ticket_accepted(self, temp_dir: Path) -> None:
        """NOTE comment with long OMN ticket number should be accepted."""
        code = """
        from typing import Any
        from pydantic import BaseModel, Field

        class MyModel(BaseModel):
            # NOTE: OMN-12345 - Required for JSON payload
            data: Any = Field(...)
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 0


class TestInlineCommentExtractionPrevention:
    """Test that NOTE matching only inspects the comment portion."""

    def test_note_in_string_literal_not_matched(self, temp_dir: Path) -> None:
        """NOTE in string literal should not be matched."""
        code = """
        from typing import Any
        from pydantic import BaseModel, Field

        class MyModel(BaseModel):
            data: Any = Field(default="NOTE: OMN-1234 - not a real comment")
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        # Should fail because the NOTE is in a string, not a comment
        assert len(violations) == 1
        assert violations[0].violation_type == EnumAnyTypeViolation.FIELD_MISSING_NOTE

    def test_note_in_code_not_matched(self, temp_dir: Path) -> None:
        """NOTE in code (not comment) should not be matched."""
        code = """
        from typing import Any
        from pydantic import BaseModel, Field

        NOTE = "OMN-1234 - this is a variable, not a comment"

        class MyModel(BaseModel):
            data: Any = Field(...)
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        # Should fail because the NOTE is in code, not a comment
        assert len(violations) == 1
        assert violations[0].violation_type == EnumAnyTypeViolation.FIELD_MISSING_NOTE

    def test_note_in_real_comment_matched(self, temp_dir: Path) -> None:
        """NOTE in actual comment (after code) should be matched."""
        code = """
        from typing import Any
        from pydantic import BaseModel, Field

        class MyModel(BaseModel):
            data: Any = Field(default="value")  # NOTE: OMN-1234 - JSON payload
        """
        filepath = _create_test_file(temp_dir, code)
        violations = validate_any_types_in_file(filepath)

        assert len(violations) == 0
