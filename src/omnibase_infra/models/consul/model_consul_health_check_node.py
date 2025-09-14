#!/usr/bin/env python3

from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID
from .model_consul_service_status import ModelConsulServiceStatus


class ModelConsulHealthCheckNode(BaseModel):
    """Health check information for a specific node."""
    
    node: Optional[str] = Field(None, description="Node name")
    service_id: UUID = Field(..., description="Unique service identifier")
    service_name: str = Field(..., description="Service name")
    status: ModelConsulServiceStatus = Field(..., description="Health check status")