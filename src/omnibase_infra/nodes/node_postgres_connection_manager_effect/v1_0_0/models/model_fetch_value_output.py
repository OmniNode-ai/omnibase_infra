"""Fetch value output model for PostgreSQL EFFECT node."""

from typing import Any, Optional

from pydantic import BaseModel, Field


class ModelFetchValueOutput(BaseModel):
    """Output model for fetch_value operation."""

    success: bool = Field(
        description="Whether query executed successfully"
    )
    value: Optional[Any] = Field(
        default=None,
        description="Single scalar value from query result"
    )
    execution_time_ms: float = Field(
        description="Query execution time in milliseconds"
    )