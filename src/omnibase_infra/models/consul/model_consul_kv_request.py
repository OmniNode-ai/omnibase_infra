#!/usr/bin/env python3

from pydantic import BaseModel
from typing import Optional


class ModelConsulKVRequest(BaseModel):
    """Request model for Consul KV operations.
    
    Shared model used across Consul infrastructure nodes for KV store operations.
    """
    
    key: str
    value: Optional[str] = None
    recurse: bool = False