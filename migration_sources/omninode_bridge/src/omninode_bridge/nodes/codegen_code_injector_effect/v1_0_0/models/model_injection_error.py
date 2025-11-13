"""Model for code injection errors."""

from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class ModelInjectionError(BaseModel):
    """
    Error encountered during code injection.

    Tracks issues that prevented successful code injection, such as
    method not found, indentation errors, or AST parsing failures.
    """

    method_name: str = Field(
        ...,
        description="Name of the method where injection failed"
    )

    error_type: str = Field(
        ...,
        description="Type of error (e.g., 'method_not_found', 'indentation_error')"
    )

    message: str = Field(
        ...,
        description="Human-readable error message"
    )

    line_number: Optional[int] = Field(
        default=None,
        description="Line number where error occurred (if applicable)",
        ge=1
    )

    suggested_fix: Optional[str] = Field(
        default=None,
        description="Suggested fix for the error"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "method_name": "process_data",
                "error_type": "method_not_found",
                "message": "Method 'process_data' not found at line 42",
                "line_number": 42,
                "suggested_fix": "Verify method name and line number"
            }
        }
    )
