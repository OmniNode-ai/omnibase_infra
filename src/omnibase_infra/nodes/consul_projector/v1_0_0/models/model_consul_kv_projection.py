#!/usr/bin/env python3

from pydantic import BaseModel, Field
from typing import List, Optional
from .model_consul_kv_summary import ModelConsulKvSummary
from .model_consul_kv_details import ModelConsulKvDetails


class ModelConsulKVProjection(BaseModel):
    """KV state projection result model."""
    
    key_summary: ModelConsulKvSummary = Field(..., description="Strongly typed KV summary details")
    key_details: Optional[List[ModelConsulKvDetails]] = Field(None, description="List of strongly typed KV key details")