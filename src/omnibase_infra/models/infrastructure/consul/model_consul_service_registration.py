#!/usr/bin/env python3

from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl

from .model_consul_health_check import ModelConsulHealthCheck


class ModelConsulServiceRegistration(BaseModel):
    """Service registration data for Consul.
    
    Shared model used across Consul infrastructure nodes for service registration.
    One model per file following ONEX standards.
    """

    service_id: UUID = Field(..., description="Unique service identifier")
    name: str = Field(..., description="Service name")
    port: int = Field(..., description="Service port")
    address: HttpUrl = Field(..., description="Service address URL")
    health_check: ModelConsulHealthCheck | None = Field(None, description="Health check configuration")
