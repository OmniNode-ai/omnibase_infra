"""Input model for PostgreSQL connection manager EFFECT node."""

from enum import Enum
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


class OperationType(str, Enum):
    """PostgreSQL operation types."""
    EXECUTE_QUERY = "execute_query"
    FETCH_ONE = "fetch_one"
    FETCH_VALUE = "fetch_value"
    GET_HEALTH = "get_health"


class ModelPostgresConnectionManagerInput(BaseModel):
    """Input model for PostgreSQL connection manager operations."""

    operation_type: OperationType = Field(
        description="Type of database operation to perform"
    )
    correlation_id: UUID = Field(
        description="Unique identifier for request tracing"
    )