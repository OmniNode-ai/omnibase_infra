"""PostgreSQL query row value model."""

from pydantic import BaseModel, Field


class ModelPostgresQueryRowValue(BaseModel):
    """Strongly typed PostgreSQL query row value."""

    column_name: str = Field(description="Column name")
    value: str | int | float | bool | None = Field(
        description="Column value with proper typing",
    )
    column_type: str = Field(description="PostgreSQL column type")