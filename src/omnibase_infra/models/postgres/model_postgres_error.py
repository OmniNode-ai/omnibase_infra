"""PostgreSQL error model."""

from typing import Optional
from pydantic import BaseModel, Field


class ModelPostgresError(BaseModel):
    """PostgreSQL error model."""
    
    error_code: str = Field(description="PostgreSQL error code")
    error_message: str = Field(description="Human-readable error message")
    severity: str = Field(description="Error severity: ERROR, WARNING, INFO")
    error_context: Optional[str] = Field(default=None, description="Additional error context")
    timestamp: Optional[float] = Field(default=None, description="Error timestamp", ge=0)
    query_id: Optional[str] = Field(default=None, description="Query ID that caused the error")