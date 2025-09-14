#!/usr/bin/env python3

from pydantic import BaseModel, Field
from typing import List
from ....models.consul.model_consul_health_summary import ModelConsulHealthSummary
from ....models.consul.model_consul_health_check_node import ModelConsulHealthCheckNode


class ModelConsulHealthProjection(BaseModel):
    """Health state projection result model."""
    
    health_summary: ModelConsulHealthSummary = Field(..., description="Strongly typed health summary")
    service_health: List[ModelConsulHealthCheckNode] = Field(..., description="List of service health check nodes")