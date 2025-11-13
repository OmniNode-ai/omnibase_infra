"""Model for validation errors."""

from typing import Optional
from pydantic import BaseModel, Field

from .enum_validation_rule import EnumValidationRule


class ModelValidationError(BaseModel):
    """
    Validation error found in generated code.

    Represents a critical issue that must be fixed.
    """

    rule: EnumValidationRule = Field(
        ...,
        description="Validation rule that was violated"
    )

    message: str = Field(
        ...,
        description="Error message describing the issue"
    )

    line_number: Optional[int] = Field(
        default=None,
        description="Line number where error was found",
        ge=1
    )

    column: Optional[int] = Field(
        default=None,
        description="Column number where error was found",
        ge=0
    )

    code_snippet: Optional[str] = Field(
        default=None,
        description="Code snippet showing the error"
    )

    severity: str = Field(
        default="error",
        description="Severity level (error, critical)"
    )

    suggested_fix: Optional[str] = Field(
        default=None,
        description="Suggested fix for the error"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "rule": "syntax",
                "message": "Invalid syntax: unexpected indent",
                "line_number": 42,
                "column": 4,
                "code_snippet": "    def foo():",
                "severity": "error",
                "suggested_fix": "Remove extra indentation"
            }
        }
