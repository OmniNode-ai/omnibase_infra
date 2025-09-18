"""PostgreSQL query result model."""

from pydantic import BaseModel, Field

from .model_postgres_query_row import ModelPostgresQueryRow


class ModelPostgresQueryResult(BaseModel):
    """PostgreSQL query result model."""

    rows: list[ModelPostgresQueryRow] = Field(
        default_factory=list, description="Query result rows with strong typing",
    )
    column_names: list[str] = Field(
        default_factory=list, description="Column names in result set",
    )
    row_count: int = Field(description="Number of rows in result", ge=0)
    has_more: bool = Field(
        default=False, description="Whether there are more rows available",
    )
