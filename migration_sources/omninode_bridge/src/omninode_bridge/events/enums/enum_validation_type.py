"""Validation type enumeration for code validation requests.

This enum defines the types of validation that can be performed
on generated code during the code generation workflow.
"""

from enum import Enum


class EnumValidationType(str, Enum):
    """
    Types of validation for code quality checking.

    Inherits from str for JSON serialization compatibility with Pydantic v2.

    Values:
        SYNTAX: Basic syntax validation and parsing checks
        SEMANTIC: Semantic analysis including type checking and logic validation
        COMPLIANCE: ONEX architectural compliance and pattern validation
        FULL: Comprehensive validation including all checks above
    """

    SYNTAX = "syntax"
    SEMANTIC = "semantic"
    COMPLIANCE = "compliance"
    FULL = "full"
