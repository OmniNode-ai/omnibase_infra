"""Consul Service Configuration Model.

Typed model for Consul service configuration to replace Dict[str, Any] usage.
"""

from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl


class ModelConsulServiceConfig(BaseModel):
    """Consul service configuration with strong typing."""

    service_id: UUID | None = Field(default=None, description="Unique service identifier")
    service_name: str = Field(description="Service name for registration")
    address: str | None = Field(default=None, description="Service address")
    port: int | None = Field(default=None, description="Service port")
    tags: list[str] | None = Field(default=None, description="Service tags")
    check_url: HttpUrl | None = Field(default=None, description="Health check URL")
    check_interval: str | None = Field(default=None, description="Health check interval (e.g., '10s')")

    class Config:
        validate_assignment = True
        extra = "forbid"
