"""Request context model for strongly typed context information."""

from typing import Optional, List
from pydantic import BaseModel, Field


class ModelRequestContext(BaseModel):
    """Strongly typed request context information."""
    
    request_id: Optional[str] = Field(
        default=None,
        description="Unique request identifier"
    )
    
    user_id: Optional[str] = Field(
        default=None,
        description="User identifier for the request"
    )
    
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant identifier for multi-tenant operations"
    )
    
    source_service: Optional[str] = Field(
        default=None,
        description="Name of the service originating the request"
    )
    
    trace_id: Optional[str] = Field(
        default=None,
        description="Distributed tracing identifier"
    )
    
    span_id: Optional[str] = Field(
        default=None,
        description="Span identifier for distributed tracing"
    )
    
    environment: Optional[str] = Field(
        default=None,
        description="Environment context (dev, staging, prod)"
    )
    
    priority: int = Field(
        default=5,
        description="Request priority (1=highest, 10=lowest)",
        ge=1,
        le=10
    )
    
    tags: List[str] = Field(
        default_factory=list,
        description="Context tags for categorization and filtering"
    )
    
    metadata_flags: List[str] = Field(
        default_factory=list,
        description="Boolean flags as string list (e.g., ['debug_enabled', 'metrics_enabled'])"
    )