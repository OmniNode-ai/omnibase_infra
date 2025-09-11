"""PostgreSQL health check response model."""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class ModelPostgresHealthResponse(BaseModel):
    """PostgreSQL health check response model."""

    status: str = Field(description="Health status: healthy, degraded, unhealthy")
    timestamp: float = Field(description="Health check timestamp")
    connection_pool: Optional[Dict[str, Union[str, int, float]]] = Field(
        default=None, description="Connection pool information"
    )
    database_info: Optional[Dict[str, Union[str, int, float]]] = Field(
        default=None, description="Database information"
    )
    schema_info: Optional[Dict[str, Union[str, bool]]] = Field(
        default=None, description="Schema validation information"
    )
    performance: Optional[Dict[str, Union[str, int, float]]] = Field(
        default=None, description="Performance metrics"
    )
    errors: List[str] = Field(default_factory=list, description="List of errors or warnings")
    correlation_id: Optional[str] = Field(default=None, description="Request correlation ID")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional response context")