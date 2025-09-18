"""PostgreSQL query row model."""

from pydantic import BaseModel, Field


class ModelPostgresQueryRow(BaseModel):
    """Strongly typed PostgreSQL query row."""

    values: dict[str, str | int | float | bool | None] = Field(
        default_factory=dict,
        description="Row values keyed by column name with proper typing",
    )