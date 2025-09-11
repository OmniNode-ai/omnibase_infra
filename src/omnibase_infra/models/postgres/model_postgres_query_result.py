"""PostgreSQL query result model."""

from typing import Any, Dict, List
from pydantic import BaseModel, Field


class ModelPostgresQueryResult(BaseModel):
    """PostgreSQL query result model."""
    
    rows: List[Dict[str, Any]] = Field(default_factory=list, description="Query result rows")
    column_names: List[str] = Field(default_factory=list, description="Column names in result set")
    row_count: int = Field(description="Number of rows in result", ge=0)
    has_more: bool = Field(default=False, description="Whether there are more rows available")