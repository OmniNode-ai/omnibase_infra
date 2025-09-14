"""Fetch one output model for PostgreSQL EFFECT node."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ModelFetchOneOutput(BaseModel):
    """Output model for fetch_one operation."""

    success: bool = Field(
        description="Whether query executed successfully"
    )
    record: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Single query result record"
    )
    execution_time_ms: float = Field(
        description="Query execution time in milliseconds"
    )