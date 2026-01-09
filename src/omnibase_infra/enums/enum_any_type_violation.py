# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Enum for Any type violation categories.

Defines the specific violation types detected by the Any type validator.
Following the ADR policy:
- Any is BLOCKED in function signatures (parameters, return types)
- Any is ALLOWED only in Pydantic Field() definitions WITH required NOTE comment
- All other Any usage is BLOCKED
"""

from enum import Enum


class EnumAnyTypeViolation(str, Enum):
    """Violation types for Any type usage detection.

    These violation types correspond to the ONEX Any type policy as defined
    in the ADR. The policy aims to maintain strong typing throughout the
    codebase while allowing controlled exceptions for legitimate use cases
    like JSON-like data structures.

    Attributes:
        FUNCTION_PARAMETER: Any used as function parameter type annotation.
        RETURN_TYPE: Any used as function return type annotation.
        FIELD_MISSING_NOTE: Any used in Pydantic Field() without required NOTE comment.
        VARIABLE_ANNOTATION: Any used as variable type annotation.
        TYPE_ALIAS: Any used in type alias definition.
        CLASS_ATTRIBUTE: Any used as class attribute type annotation.
        GENERIC_ARGUMENT: Any used as generic type argument (e.g., list[Any]).
        SYNTAX_ERROR: File has Python syntax error, cannot validate.

    Policy Summary:
        - ALLOWED: ``field: Any = Field(...) # NOTE: Using Any for JSON-like data``
        - BLOCKED: ``def func(data: Any) -> Any:``
        - BLOCKED: ``result: Any = some_call()``
        - BLOCKED: ``JsonType = dict[str, Any]``
    """

    FUNCTION_PARAMETER = "function_parameter"
    RETURN_TYPE = "return_type"
    FIELD_MISSING_NOTE = "field_missing_note"
    VARIABLE_ANNOTATION = "variable_annotation"
    TYPE_ALIAS = "type_alias"
    CLASS_ATTRIBUTE = "class_attribute"
    GENERIC_ARGUMENT = "generic_argument"
    SYNTAX_ERROR = "syntax_error"

    @property
    def is_exemptable(self) -> bool:
        """Check if this violation type can be exempted.

        Some violations like SYNTAX_ERROR cannot be exempted because
        they indicate fundamental issues with the source file.

        Returns:
            True if the violation type can be exempted via decorator or comment.
        """
        return self != EnumAnyTypeViolation.SYNTAX_ERROR

    @property
    def suggestion(self) -> str:
        """Get the suggested fix for this violation type.

        Returns:
            Human-readable suggestion for fixing the violation.
        """
        suggestions = {
            EnumAnyTypeViolation.FUNCTION_PARAMETER: (
                "Replace Any with specific type (e.g., object, Union, Protocol) "
                "or use a typed model for complex data structures."
            ),
            EnumAnyTypeViolation.RETURN_TYPE: (
                "Replace Any with specific type. If the return type varies, "
                "consider using Union types or Protocol for duck typing."
            ),
            EnumAnyTypeViolation.FIELD_MISSING_NOTE: (
                "Add a NOTE comment explaining why Any is needed. "
                "Example: # NOTE: Using Any instead of JsonType for JSON payload"
            ),
            EnumAnyTypeViolation.VARIABLE_ANNOTATION: (
                "Replace Any with specific type. Use object if truly generic, "
                "or create a typed model for structured data."
            ),
            EnumAnyTypeViolation.TYPE_ALIAS: (
                "Replace Any in type alias with specific types. "
                "Consider using TypedDict, Protocol, or explicit Union types."
            ),
            EnumAnyTypeViolation.CLASS_ATTRIBUTE: (
                "Replace Any with specific type annotation. "
                "For Pydantic models, use Field() with a NOTE comment if Any is required."
            ),
            EnumAnyTypeViolation.GENERIC_ARGUMENT: (
                "Replace Any in generic type with specific type. "
                "Example: list[object] instead of list[Any]."
            ),
            EnumAnyTypeViolation.SYNTAX_ERROR: (
                "Fix the Python syntax error before Any type validation can proceed."
            ),
        }
        return suggestions.get(self, "Replace Any with a specific type.")


__all__ = ["EnumAnyTypeViolation"]
