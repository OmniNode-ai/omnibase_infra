#!/usr/bin/env python3

from uuid import UUID

from pydantic import BaseModel, Field


class ModelConsulServiceInfo(BaseModel):
    """Consul service information model."""

    service_id: UUID = Field(..., description="Unique service identifier")
    service_name: str = Field(..., description="Service name")
    node: str | None = Field(None, description="Node name hosting the service")
