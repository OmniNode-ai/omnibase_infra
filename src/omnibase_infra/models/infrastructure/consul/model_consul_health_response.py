#!/usr/bin/env python3


from pydantic import BaseModel, Field

from omnibase_infra.models.infrastructure.consul.model_consul_health_check_node import (
    ModelConsulHealthCheckNode,
)
from omnibase_infra.models.infrastructure.consul.model_consul_service_status import (
    ModelConsulServiceStatus,
)

from .model_consul_health_summary import ModelConsulHealthSummary


class ModelConsulHealthResponse(BaseModel):
    """Response for Consul health check operations.

    Shared model used across Consul infrastructure nodes for health check responses.
    One model per file following ONEX standards.
    """

    status: ModelConsulServiceStatus = Field(..., description="Overall health status")
    service_name: str | None = Field(None, description="Service name")
    health_checks: list[ModelConsulHealthCheckNode] | None = Field(
        None, description="List of health check nodes",
    )
    health_summary: ModelConsulHealthSummary | None = Field(
        None, description="Strongly typed health summary",
    )
