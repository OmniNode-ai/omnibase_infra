#!/usr/bin/env python3

from pydantic import BaseModel, Field
from typing import Optional


class ModelConsulKvDetails(BaseModel):
    """KV key details with strongly typed information."""
    
    key: str = Field(..., description="KV store key")
    value: Optional[str] = Field(None, description="KV store value")
    modify_index: int = Field(..., description="Consul modify index")
    create_index: int = Field(..., description="Consul create index")
    lock_index: int = Field(..., description="Consul lock index")
    flags: int = Field(..., description="Application-specific flags")