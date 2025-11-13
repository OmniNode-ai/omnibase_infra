"""Model for stub extraction results."""

from typing import ClassVar

from pydantic import BaseModel, Field

from .model_method_stub import ModelMethodStub


class ModelStubExtractionResult(BaseModel):
    """
    Result of stub extraction operation.

    Contains all extracted stubs and metadata about the extraction process.
    """

    success: bool = Field(
        ..., description="Whether stub extraction completed successfully"
    )

    stubs: list[ModelMethodStub] = Field(
        default_factory=list, description="List of extracted method stubs"
    )

    total_stubs_found: int = Field(..., description="Total number of stubs found", ge=0)

    extraction_time_ms: float = Field(
        ..., description="Time taken to extract stubs in milliseconds", ge=0.0
    )

    file_path: str | None = Field(
        default=None, description="Path to the file that was analyzed"
    )

    file_lines: int = Field(
        default=0, description="Total number of lines in the file", ge=0
    )

    extraction_patterns_used: list[str] = Field(
        default_factory=lambda: ["# IMPLEMENTATION REQUIRED", "pass  # Stub"],
        description="Patterns used to identify stubs",
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra: ClassVar[dict] = {
            "example": {
                "success": True,
                "stubs": [
                    {
                        "name": "execute_effect",
                        "signature": "async def execute_effect(self, contract: ModelContractEffect)",
                        "docstring": "Execute the effect.",
                        "line_number": 42,
                    }
                ],
                "total_stubs_found": 1,
                "extraction_time_ms": 15.3,
                "file_path": "/path/to/node.py",
                "file_lines": 200,
                "extraction_patterns_used": ["# IMPLEMENTATION REQUIRED"],
            }
        }
