#!/usr/bin/env python3

from pydantic import BaseModel, Field, HttpUrl
from datetime import timedelta


class ModelConsulHealthCheck(BaseModel):
    """Health check configuration for Consul service registration."""
    
    url: HttpUrl = Field(..., description="HTTP URL for health check")
    interval: timedelta = Field(..., description="Health check interval duration")
    timeout: timedelta = Field(..., description="Health check timeout duration")