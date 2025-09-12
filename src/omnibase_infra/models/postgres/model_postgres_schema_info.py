"""PostgreSQL schema information model."""

from typing import List, Optional
from pydantic import BaseModel, Field


class ModelPostgresSchemaInfo(BaseModel):
    """PostgreSQL schema information model."""
    
    schema_name: str = Field(description="Name of the schema")
    table_count: int = Field(description="Number of tables in schema", ge=0)
    view_count: int = Field(description="Number of views in schema", ge=0)
    function_count: int = Field(description="Number of functions in schema", ge=0)
    is_valid: bool = Field(default=True, description="Whether schema validation passed")
    validation_errors: List[str] = Field(default_factory=list, description="Schema validation errors")
    last_modified: Optional[str] = Field(default=None, description="Last modification timestamp")