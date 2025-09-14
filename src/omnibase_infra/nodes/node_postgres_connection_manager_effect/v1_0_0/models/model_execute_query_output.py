"""Execute query output model for PostgreSQL EFFECT node."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ModelExecuteQueryOutput(BaseModel):
    """Output model for execute_query operation."""

    success: bool = Field(
        description="Whether query executed successfully"
    )
    records: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Query result records"
    )
    affected_rows: Optional[int] = Field(
        default=None,
        description="Number of affected rows"
    )
    execution_time_ms: float = Field(
        description="Query execution time in milliseconds"
    )