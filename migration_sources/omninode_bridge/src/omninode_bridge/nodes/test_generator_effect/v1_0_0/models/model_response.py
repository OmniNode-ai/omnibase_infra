"""
Test Generation Response Model - ONEX v2.0 Compliant.

Response model for test generation operations.
"""

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelGeneratedTestFile(BaseModel):
    """
    Represents a single generated test file.
    """

    file_path: Path = Field(
        ...,
        description="Path to the generated test file",
    )

    file_type: str = Field(
        ...,
        description="Type of test file (unit, integration, contract, performance, conftest)",
    )

    lines_of_code: int = Field(
        ...,
        description="Number of lines of code in the generated file",
        ge=0,
    )

    template_used: str = Field(
        ...,
        description="Name of the Jinja2 template used",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "file_path": "/path/to/tests/test_unit_postgres_crud.py",
                "file_type": "unit",
                "lines_of_code": 150,
                "template_used": "test_unit.py.j2",
            }
        }
    )


class ModelTestGeneratorResponse(BaseModel):
    """
    Response model for test generation operations.

    Contains metadata about generated test files and execution metrics.
    """

    # === GENERATION RESULTS ===

    generated_files: list[ModelGeneratedTestFile] = Field(
        default_factory=list,
        description="List of generated test files",
    )

    file_count: int = Field(
        ...,
        description="Total number of test files generated",
        ge=0,
    )

    total_lines_of_code: int = Field(
        ...,
        description="Total lines of code across all generated files",
        ge=0,
    )

    # === EXECUTION METRICS ===

    duration_ms: float = Field(
        ...,
        description="Total generation duration in milliseconds",
        ge=0.0,
    )

    template_render_ms: float = Field(
        ...,
        description="Time spent rendering templates in milliseconds",
        ge=0.0,
    )

    file_write_ms: float = Field(
        ...,
        description="Time spent writing files in milliseconds",
        ge=0.0,
    )

    # === STATUS ===

    success: bool = Field(
        ...,
        description="Whether all test files were generated successfully",
    )

    warnings: list[str] = Field(
        default_factory=list,
        description="Non-critical warnings during generation",
    )

    # === CORRELATION TRACKING ===

    correlation_id: UUID = Field(
        ...,
        description="UUID for correlation tracking (from request)",
    )

    execution_id: UUID = Field(
        ...,
        description="UUID for this execution instance (from request)",
    )

    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when generation completed",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "generated_files": [
                    {
                        "file_path": "/path/to/tests/test_unit.py",
                        "file_type": "unit",
                        "lines_of_code": 150,
                        "template_used": "test_unit.py.j2",
                    }
                ],
                "file_count": 4,
                "total_lines_of_code": 500,
                "duration_ms": 250.5,
                "template_render_ms": 100.2,
                "file_write_ms": 50.3,
                "success": True,
                "warnings": [],
            }
        }
    )


__all__ = ["ModelGeneratedTestFile", "ModelTestGeneratorResponse"]
