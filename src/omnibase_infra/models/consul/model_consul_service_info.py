#!/usr/bin/env python3

from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID


class ModelConsulServiceInfo(BaseModel):
    """Consul service information model."""
    
    service_id: UUID = Field(..., description="Unique service identifier")
    service_name: str = Field(..., description="Service name")
    node: Optional[str] = Field(None, description="Node name hosting the service")