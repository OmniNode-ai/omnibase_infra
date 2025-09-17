#!/usr/bin/env python3

from uuid import UUID

from pydantic import BaseModel, Field

from omnibase_infra.models.infrastructure.consul.model_consul_service_status import ModelConsulServiceStatus


class ModelConsulHealthCheckNode(BaseModel):
    """Health check information for a specific node."""

    node: str | None = Field(None, description="Node name")
    service_id: UUID = Field(..., description="Unique service identifier")
    service_name: str = Field(..., description="Service name")
    status: ModelConsulServiceStatus = Field(..., description="Health check status")
