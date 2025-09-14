"""Fetch one input model for PostgreSQL EFFECT node."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ModelFetchOneInput(BaseModel):
    """Input model for fetch_one operation."""

    query: str = Field(
        description="SQL query to execute"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Query parameters"
    )
    timeout: float = Field(
        default=60.0,
        description="Query timeout in seconds"
    )