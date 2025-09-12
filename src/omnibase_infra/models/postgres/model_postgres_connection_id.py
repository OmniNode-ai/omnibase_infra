"""PostgreSQL connection identifier model."""

from typing import Optional
from pydantic import BaseModel, Field


class ModelPostgresConnectionId(BaseModel):
    """PostgreSQL connection identifier model."""
    
    connection_id: str = Field(description="Unique connection identifier")
    pool_name: Optional[str] = Field(default=None, description="Connection pool name")
    database_name: str = Field(description="Database name")
    username: str = Field(description="Database username")
    host: str = Field(description="Database host")
    port: int = Field(description="Database port", ge=1, le=65535)