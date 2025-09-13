#!/usr/bin/env python3

from pydantic import BaseModel, Field
from typing import List
from uuid import UUID
from ....models.consul.model_consul_service_info import ModelConsulServiceInfo


class ModelConsulServiceProjection(BaseModel):
    """Service state projection result model."""
    
    services: List[ModelConsulServiceInfo] = Field(..., description="List of services with detailed information")
    total_services: int = Field(..., description="Total number of services in projection")