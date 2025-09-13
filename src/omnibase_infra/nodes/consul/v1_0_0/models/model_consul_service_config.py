"""Consul Service Configuration Model.

Typed model for Consul service configuration to replace Dict[str, Any] usage.
"""

from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, List
from uuid import UUID


class ModelConsulServiceConfig(BaseModel):
    """Consul service configuration with strong typing."""
    
    service_id: Optional[UUID] = Field(default=None, description="Unique service identifier")
    service_name: str = Field(description="Service name for registration")
    address: Optional[str] = Field(default=None, description="Service address")
    port: Optional[int] = Field(default=None, description="Service port")
    tags: Optional[List[str]] = Field(default=None, description="Service tags")
    check_url: Optional[HttpUrl] = Field(default=None, description="Health check URL")
    check_interval: Optional[str] = Field(default=None, description="Health check interval (e.g., '10s')")
    
    class Config:
        validate_assignment = True
        extra = "forbid"