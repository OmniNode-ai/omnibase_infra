"""PostgreSQL health check request model."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ModelPostgresHealthRequest(BaseModel):
    """PostgreSQL health check request model."""

    include_performance_metrics: bool = Field(default=True, description="Include performance metrics in response")
    include_connection_stats: bool = Field(default=True, description="Include connection pool statistics")
    include_schema_info: bool = Field(default=True, description="Include schema validation information")
    correlation_id: Optional[str] = Field(default=None, description="Request correlation ID")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional request context")