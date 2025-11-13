"""Validation rule types."""

from enum import Enum


class EnumValidationRule(str, Enum):
    """
    Validation rules for code validation.

    Defines the types of validation checks performed on generated code.
    """

    SYNTAX = "syntax"
    """Check Python syntax (AST parsing)"""

    ONEX_COMPLIANCE = "onex_compliance"
    """Check ONEX v2.0 compliance (ModelOnexError, emit_log_event, etc.)"""

    TYPE_HINTS = "type_hints"
    """Check for type hints on methods and parameters"""

    DOCSTRINGS = "docstrings"
    """Check for docstrings on classes and methods"""

    SECURITY = "security"
    """Check for security issues (hardcoded secrets, SQL injection, etc.)"""

    IMPORTS = "imports"
    """Check import statements (correct omnibase_core imports)"""

    NAMING = "naming"
    """Check naming conventions (Node prefix, suffix-based naming)"""

    ERROR_HANDLING = "error_handling"
    """Check error handling patterns (OnexError wrapping)"""

    ALL = "all"
    """Run all validation rules"""
