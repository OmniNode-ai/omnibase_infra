"""PostgreSQL context model for additional request/response context."""


from pydantic import BaseModel, Field


class ModelPostgresContext(BaseModel):
    """PostgreSQL context model for additional request/response context."""

    request_source: str | None = Field(default=None, description="Source of the request")
    trace_id: str | None = Field(default=None, description="Distributed tracing ID")
    user_id: str | None = Field(default=None, description="User ID associated with request")
    timeout_ms: int | None = Field(default=None, description="Request timeout in milliseconds", ge=0)
    priority: str | None = Field(default="normal", description="Request priority level")
