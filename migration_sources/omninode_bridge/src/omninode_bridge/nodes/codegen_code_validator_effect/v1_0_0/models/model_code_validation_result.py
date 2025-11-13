"""Model for code validation results."""

from typing import ClassVar
from pydantic import BaseModel, Field
from .model_validation_error import ModelValidationError
from .model_validation_warning import ModelValidationWarning


class ModelCodeValidationResult(BaseModel):
    """
    Result of code validation operation.

    Contains all validation errors, warnings, and metadata about the validation process.
    """

    is_valid: bool = Field(
        ...,
        description="Whether the code passed all validation checks (no errors)"
    )

    validation_errors: list[ModelValidationError] = Field(
        default_factory=list,
        description="List of validation errors found (critical issues)"
    )

    validation_warnings: list[ModelValidationWarning] = Field(
        default_factory=list,
        description="List of validation warnings found (non-critical issues)"
    )

    validation_time_ms: float = Field(
        ...,
        description="Time taken to validate code in milliseconds",
        ge=0.0
    )

    rules_checked: list[str] = Field(
        default_factory=list,
        description="List of validation rules that were checked"
    )

    file_path: str | None = Field(
        default=None,
        description="Path to the file that was validated"
    )

    file_lines: int = Field(
        default=0,
        description="Total number of lines in the validated file",
        ge=0
    )

    syntax_valid: bool = Field(
        default=True,
        description="Whether Python syntax is valid (AST parsing succeeded)"
    )

    onex_compliant: bool = Field(
        default=True,
        description="Whether code follows ONEX v2.0 patterns"
    )

    security_issues_found: int = Field(
        default=0,
        description="Number of security issues detected",
        ge=0
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra: ClassVar[dict] = {
            "example": {
                "is_valid": False,
                "validation_errors": [
                    {
                        "rule": "syntax",
                        "message": "Invalid syntax: unexpected indent",
                        "line_number": 42,
                        "severity": "error"
                    }
                ],
                "validation_warnings": [
                    {
                        "rule": "type_hints",
                        "message": "Missing type hint on parameter",
                        "line_number": 50,
                        "severity": "warning"
                    }
                ],
                "validation_time_ms": 25.5,
                "rules_checked": ["syntax", "onex_compliance", "type_hints", "security"],
                "file_path": "/path/to/node.py",
                "file_lines": 200,
                "syntax_valid": False,
                "onex_compliant": True,
                "security_issues_found": 0
            }
        }
