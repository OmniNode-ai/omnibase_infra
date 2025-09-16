"""Consul Cache Entry Models.

Typed models for cache entries to replace Dict[str, Any] usage.
Used by Consul projector for strongly typed cache management.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class ModelConsulServiceCacheEntry(BaseModel):
    """Cache entry for Consul service data."""

    service_id: str = Field(description="Service ID")
    service_name: str = Field(description="Service name")
    instances: int = Field(ge=0, description="Number of instances")
    health_status: str = Field(description="Service health status")
    last_updated: datetime = Field(description="Last update timestamp")
    cached_at: datetime = Field(description="Cache entry timestamp")

    class Config:
        validate_assignment = True
        extra = "forbid"


class ModelConsulHealthCacheEntry(BaseModel):
    """Cache entry for Consul health data."""

    service_name: str = Field(description="Service name")
    check_status: str = Field(description="Health check status")
    check_count: int = Field(ge=0, description="Number of health checks")
    last_check: datetime = Field(description="Last health check timestamp")
    cached_at: datetime = Field(description="Cache entry timestamp")

    class Config:
        validate_assignment = True
        extra = "forbid"


class ModelConsulKVCacheEntry(BaseModel):
    """Cache entry for Consul KV data."""

    key: str = Field(description="KV key")
    value: str | None = Field(default=None, description="KV value")
    modify_index: int = Field(ge=0, description="Consul modify index")
    create_index: int = Field(ge=0, description="Consul create index")
    cached_at: datetime = Field(description="Cache entry timestamp")

    class Config:
        validate_assignment = True
        extra = "forbid"
