"""Model for code injection request."""

from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class ModelCodeInjectionRequest(BaseModel):
    """
    Request to inject code into a specific method.

    Represents a single code injection operation targeting a method stub
    that needs to be replaced with actual implementation.
    """

    method_name: str = Field(
        ...,
        description="Name of the method to inject code into"
    )

    line_number: int = Field(
        ...,
        description="Line number where the method starts",
        ge=1
    )

    generated_code: str = Field(
        ...,
        description="The validated code to inject into the method"
    )

    preserve_signature: bool = Field(
        default=True,
        description="Whether to preserve the original method signature"
    )

    preserve_docstring: bool = Field(
        default=True,
        description="Whether to preserve the original docstring"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "method_name": "execute_effect",
                "line_number": 42,
                "generated_code": "    return {'result': 'success'}",
                "preserve_signature": True,
                "preserve_docstring": True
            }
        }
    )
