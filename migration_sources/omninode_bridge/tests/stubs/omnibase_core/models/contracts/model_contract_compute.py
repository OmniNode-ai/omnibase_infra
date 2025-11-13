"""Stub for omnibase_core.models.contracts.model_contract_compute"""

from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ModelContractCompute(BaseModel):
    """Stub for ONEX compute contract."""

    name: str = Field(default="", description="Node name")
    version: dict[str, int] = Field(
        default_factory=lambda: {"major": 1, "minor": 0, "patch": 0},
        description="Semantic version",
    )
    description: str = Field(default="", description="Node description")
    node_type: str = Field(default="compute", description="Node type")
    input_model: str = Field(default="", description="Input model class name")
    output_model: str = Field(default="", description="Output model class name")
    algorithm: dict[str, Any] | None = Field(
        default=None, description="Algorithm configuration"
    )
    contract_id: UUID = Field(default_factory=uuid4, description="Contract UUID")

    class Config:
        arbitrary_types_allowed = True


__all__ = ["ModelContractCompute"]
