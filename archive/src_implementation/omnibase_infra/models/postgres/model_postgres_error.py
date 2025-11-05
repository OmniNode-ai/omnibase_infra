"""PostgreSQL error model."""


from pydantic import BaseModel, Field


class ModelPostgresError(BaseModel):
    """PostgreSQL error model."""

    error_code: str = Field(description="PostgreSQL error code")
    error_message: str = Field(description="Human-readable error message")
    severity: str = Field(description="Error severity: ERROR, WARNING, INFO")
    error_context: str | None = Field(default=None, description="Additional error context")
    timestamp: float | None = Field(default=None, description="Error timestamp", ge=0)
    query_id: str | None = Field(default=None, description="Query ID that caused the error")
