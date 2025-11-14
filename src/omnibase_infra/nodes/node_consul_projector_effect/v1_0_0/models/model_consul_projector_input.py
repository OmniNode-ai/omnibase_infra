#!/usr/bin/env python3


from pydantic import BaseModel, Field

from .model_consul_projection_type import ModelConsulProjectionType


class ModelConsulProjectorInput(BaseModel):
    """Input model for Consul projector operations from event envelopes.
    
    Node-specific model for processing event envelope payloads into projection operations.
    """

    projection_type: ModelConsulProjectionType = Field(..., description="Type of projection to perform")
    target_services: list[str] | None = Field(None, description="List of specific services to include in projection")
    include_metadata: bool = Field(True, description="Whether to include metadata in projection results")
    aggregation_window: int = Field(300, description="Aggregation window in seconds")
