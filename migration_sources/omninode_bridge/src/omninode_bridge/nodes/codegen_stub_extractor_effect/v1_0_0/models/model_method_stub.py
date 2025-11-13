"""Model for method stub information."""

from typing import Optional
from pydantic import BaseModel, Field


class ModelMethodStub(BaseModel):
    """
    Method stub information extracted from node files.

    Represents a method that requires implementation (stub method).
    """

    name: str = Field(
        ...,
        description="Method name"
    )

    signature: str = Field(
        ...,
        description="Full method signature including parameters and return type"
    )

    docstring: Optional[str] = Field(
        default=None,
        description="Method docstring if available"
    )

    line_number: int = Field(
        ...,
        description="Line number where method is defined",
        ge=1
    )

    context: Optional[str] = Field(
        default=None,
        description="Surrounding context (e.g., class name, decorators)"
    )

    stub_marker: str = Field(
        default="# IMPLEMENTATION REQUIRED",
        description="Marker that identified this as a stub"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "name": "execute_effect",
                "signature": "async def execute_effect(self, contract: ModelContractEffect) -> ModelEffectOutput",
                "docstring": "Execute the effect operation.",
                "line_number": 42,
                "context": "class NodeMyEffect(NodeEffect)",
                "stub_marker": "# IMPLEMENTATION REQUIRED"
            }
        }
