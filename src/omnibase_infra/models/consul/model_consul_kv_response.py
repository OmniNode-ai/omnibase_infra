#!/usr/bin/env python3


from pydantic import BaseModel, Field

from .model_consul_kv_status import ModelConsulKvStatus


class ModelConsulKVResponse(BaseModel):
    """Response model for Consul KV operations.
    
    Shared model used across Consul infrastructure nodes for KV store operation responses.
    """

    status: ModelConsulKvStatus = Field(..., description="KV operation status")
    key: str = Field(..., description="Key that was operated on")
    value: str | None = Field(None, description="Value retrieved or stored")
    modify_index: int | None = Field(None, description="Consul modify index for the key")
