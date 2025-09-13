#!/usr/bin/env python3

from pydantic import BaseModel, Field


class ModelConsulKvSummary(BaseModel):
    """KV state summary with strongly typed details."""
    
    total_keys: int = Field(..., description="Total number of keys in KV store")
    total_size_bytes: int = Field(..., description="Total size of all values in bytes")
    key_prefixes: int = Field(..., description="Number of distinct key prefixes")
    last_modified: str = Field(..., description="ISO timestamp of last modification")