"""Strongly typed PostgreSQL query data model."""

from typing import Optional
from pydantic import BaseModel, Field


class ModelPostgresQueryData(BaseModel):
    """Strongly typed PostgreSQL query data for event publishing."""
    
    query_hash: str = Field(
        description="Hash of the SQL query for identification"
    )
    
    operation_type: str = Field(
        description="Type of database operation (query, transaction, health_check)"
    )
    
    query_length: int = Field(
        description="Length of the SQL query in characters",
        ge=0
    )
    
    parameter_count: int = Field(
        description="Number of query parameters",
        ge=0
    )
    
    status_message: str = Field(
        description="Human-readable status message"
    )
    
    table_name: Optional[str] = Field(
        default=None,
        description="Primary table involved in the operation"
    )
    
    affected_tables: list[str] = Field(
        default_factory=list,
        description="List of tables affected by the operation"
    )
    
    query_complexity_score: Optional[int] = Field(
        default=None,
        description="Complexity score of the query (0-100)",
        ge=0,
        le=100
    )
    
    uses_joins: bool = Field(
        default=False,
        description="Whether the query uses JOIN operations"
    )
    
    uses_subqueries: bool = Field(
        default=False,
        description="Whether the query uses subqueries"
    )
    
    is_transaction: bool = Field(
        default=False,
        description="Whether this is part of a transaction"
    )
    
    transaction_id: Optional[str] = Field(
        default=None,
        description="Transaction identifier if part of a transaction"
    )