"""PostgreSQL query result model."""

from typing import Dict, List, Union
from pydantic import BaseModel, Field


class ModelPostgresQueryRowValue(BaseModel):
    """Strongly typed PostgreSQL query row value."""
    
    column_name: str = Field(description="Column name")
    value: Union[str, int, float, bool, None] = Field(description="Column value with proper typing")
    column_type: str = Field(description="PostgreSQL column type")


class ModelPostgresQueryRow(BaseModel):
    """Strongly typed PostgreSQL query row."""
    
    values: Dict[str, Union[str, int, float, bool, None]] = Field(
        default_factory=dict, 
        description="Row values keyed by column name with proper typing"
    )


class ModelPostgresQueryResult(BaseModel):
    """PostgreSQL query result model."""
    
    rows: List[ModelPostgresQueryRow] = Field(default_factory=list, description="Query result rows with strong typing")
    column_names: List[str] = Field(default_factory=list, description="Column names in result set")
    row_count: int = Field(description="Number of rows in result", ge=0)
    has_more: bool = Field(default=False, description="Whether there are more rows available")