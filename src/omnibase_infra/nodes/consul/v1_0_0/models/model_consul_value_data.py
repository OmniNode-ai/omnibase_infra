"""Consul Value Data Model.

Typed model for Consul KV value data to replace Dict[str, Any] usage.
"""

from pydantic import BaseModel, Field
from typing import Optional


class ModelConsulValueData(BaseModel):
    """Consul KV value data with strong typing."""
    
    value: str = Field(description="The value to store in Consul KV")
    flags: Optional[int] = Field(default=0, description="Consul KV flags")
    modify_index: Optional[int] = Field(default=None, description="Consul modify index for conditional updates")
    
    class Config:
        validate_assignment = True
        extra = "forbid"