#!/usr/bin/env python3

from pydantic import BaseModel
from typing import Optional


class ModelConsulHealthCheck(BaseModel):
    """Health check configuration for Consul service registration."""
    
    url: str
    interval: str
    timeout: str


class ModelConsulServiceRegistration(BaseModel):
    """Service registration data for Consul.
    
    Shared model used across Consul infrastructure nodes for service registration.
    """
    
    service_id: str
    name: str
    port: int
    address: str
    health_check: Optional[ModelConsulHealthCheck] = None