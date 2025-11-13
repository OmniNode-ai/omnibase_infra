"""Model for code injection result."""

from pydantic import BaseModel, Field, ConfigDict
from .model_injection_error import ModelInjectionError


class ModelCodeInjectionResult(BaseModel):
    """
    Result of code injection operation.

    Contains the modified source code, injection metrics, and any errors encountered.
    """

    success: bool = Field(
        ...,
        description="Whether all code injections were successful"
    )

    modified_source: str = Field(
        ...,
        description="The complete source code with injections applied"
    )

    injections_performed: int = Field(
        default=0,
        description="Number of successful code injections",
        ge=0
    )

    injection_errors: list[ModelInjectionError] = Field(
        default_factory=list,
        description="List of errors encountered during injection"
    )

    injection_time_ms: float = Field(
        ...,
        description="Time taken to perform injections in milliseconds",
        ge=0.0
    )

    file_lines_before: int = Field(
        default=0,
        description="Number of lines in source file before injection",
        ge=0
    )

    file_lines_after: int = Field(
        default=0,
        description="Number of lines in source file after injection",
        ge=0
    )

    methods_modified: list[str] = Field(
        default_factory=list,
        description="List of method names that were successfully modified"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "modified_source": "def foo():\n    return 42",
                "injections_performed": 3,
                "injection_errors": [],
                "injection_time_ms": 15.5,
                "file_lines_before": 100,
                "file_lines_after": 150,
                "methods_modified": ["execute_effect", "process_data", "validate_input"]
            }
        }
    )
