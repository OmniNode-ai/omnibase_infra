"""PostgreSQL query request model for message bus integration."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ModelPostgresQueryRequest(BaseModel):
    """PostgreSQL query request model."""

    query: str = Field(description="SQL query to execute")
    parameters: List[Any] = Field(default_factory=list, description="Query parameters")
    timeout: Optional[float] = Field(default=None, description="Query timeout in seconds")
    record_metrics: bool = Field(default=True, description="Whether to record query metrics")
    query_type: str = Field(default="general", description="Type of query (select, insert, update, delete, ddl)")
    correlation_id: Optional[str] = Field(default=None, description="Request correlation ID")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional request context")