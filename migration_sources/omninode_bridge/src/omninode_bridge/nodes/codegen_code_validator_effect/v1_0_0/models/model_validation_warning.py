"""Model for validation warnings."""

from typing import Optional
from pydantic import BaseModel, Field


class ModelValidationWarning(BaseModel):
    """
    Validation warning found in generated code.

    Represents a non-critical issue that should be reviewed but doesn't prevent usage.
    """

    rule: str = Field(
        ...,
        description="Validation rule that triggered the warning"
    )

    message: str = Field(
        ...,
        description="Warning message describing the issue"
    )

    line_number: Optional[int] = Field(
        default=None,
        description="Line number where warning was found",
        ge=1
    )

    column: Optional[int] = Field(
        default=None,
        description="Column number where warning was found",
        ge=0
    )

    code_snippet: Optional[str] = Field(
        default=None,
        description="Code snippet showing the warning location"
    )

    severity: str = Field(
        default="warning",
        description="Severity level (warning, info)"
    )

    suggested_fix: Optional[str] = Field(
        default=None,
        description="Suggested fix for the warning"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "rule": "type_hints",
                "message": "Missing type hint on parameter 'data'",
                "line_number": 42,
                "column": 20,
                "code_snippet": "def process(data):",
                "severity": "warning",
                "suggested_fix": "Add type hint: def process(data: dict):"
            }
        }
