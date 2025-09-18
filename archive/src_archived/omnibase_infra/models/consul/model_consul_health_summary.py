#!/usr/bin/env python3


from pydantic import BaseModel, Field

from .model_consul_service_info import ModelConsulServiceInfo
from .model_consul_service_status import ModelConsulServiceStatus


class ModelConsulHealthSummary(BaseModel):
    """Health summary model with strongly typed details."""

    overall_status: ModelConsulServiceStatus = Field(
        ..., description="Overall health status",
    )
    total_checks: int = Field(..., description="Total number of health checks")
    passing_checks: int = Field(..., description="Number of passing health checks")
    warning_checks: int = Field(..., description="Number of warning health checks")
    critical_checks: int = Field(..., description="Number of critical health checks")
    services: list[ModelConsulServiceInfo] = Field(
        default_factory=list, description="List of services with health information",
    )
