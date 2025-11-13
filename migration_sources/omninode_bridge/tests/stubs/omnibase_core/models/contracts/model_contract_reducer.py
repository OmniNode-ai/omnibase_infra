"""Stub for omnibase_core.models.contracts.model_contract_reducer"""

from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ModelContractReducer(BaseModel):
    """Stub for ONEX reducer contract."""

    name: str = Field(default="", description="Node name")
    version: dict[str, int] = Field(
        default_factory=lambda: {"major": 1, "minor": 0, "patch": 0},
        description="Semantic version",
    )
    description: str = Field(default="", description="Node description")
    node_type: str = Field(default="reducer", description="Node type")
    input_model: str = Field(default="", description="Input model class name")
    output_model: str = Field(default="", description="Output model class name")
    aggregation: dict[str, Any] | None = Field(
        default=None, description="Aggregation configuration"
    )
    correlation_id: UUID = Field(
        default_factory=uuid4, description="Correlation UUID for tracing"
    )
    contract_id: UUID = Field(
        default_factory=uuid4,
        description="Contract UUID (deprecated, use correlation_id)",
    )

    class Config:
        arbitrary_types_allowed = True


__all__ = ["ModelContractReducer"]
