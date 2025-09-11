"""PostgreSQL context model for additional request/response context."""

from typing import Optional
from pydantic import BaseModel, Field


class ModelPostgresContext(BaseModel):
    """PostgreSQL context model for additional request/response context."""
    
    request_source: Optional[str] = Field(default=None, description="Source of the request")
    trace_id: Optional[str] = Field(default=None, description="Distributed tracing ID")
    user_id: Optional[str] = Field(default=None, description="User ID associated with request")
    timeout_ms: Optional[int] = Field(default=None, description="Request timeout in milliseconds", ge=0)
    priority: Optional[str] = Field(default="normal", description="Request priority level")