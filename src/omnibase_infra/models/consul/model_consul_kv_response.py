#!/usr/bin/env python3

from pydantic import BaseModel
from typing import Optional, Literal


class ModelConsulKVResponse(BaseModel):
    """Response model for Consul KV operations.
    
    Shared model used across Consul infrastructure nodes for KV store operation responses.
    """
    
    status: Literal["success", "not_found", "failed"]
    key: str
    value: Optional[str] = None
    modify_index: Optional[int] = None