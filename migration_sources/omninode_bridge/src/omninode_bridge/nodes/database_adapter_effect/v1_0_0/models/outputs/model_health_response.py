"""
Health Response Model.

Output model for database health check operations.
"""

from datetime import datetime
from typing import Any, ClassVar, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ModelHealthResponse(BaseModel):
    """
    Response from database health check operations.

    This model provides comprehensive database connection health status,
    connection pool metrics, and database server information.

    Attributes:
        success: Whether the health check completed successfully
        correlation_id: UUID preserved from input for end-to-end tracking
        execution_time_ms: Time taken to perform health check in milliseconds
        database_status: Overall database status (HEALTHY, DEGRADED, UNHEALTHY)
        connection_pool_size: Current number of connections in pool
        connection_pool_available: Number of available connections
        connection_pool_in_use: Number of connections currently in use
        database_version: PostgreSQL server version string
        uptime_seconds: Database server uptime in seconds (if available)
        last_check_timestamp: ISO timestamp of this health check
        error_message: Human-readable error description if health check failed
    """

    success: bool = Field(
        ...,
        description="Health check execution success status",
    )

    correlation_id: UUID = Field(
        ...,
        description="Request correlation ID preserved from input",
    )

    execution_time_ms: int = Field(
        ...,
        description="Health check execution time in milliseconds",
        ge=0,
    )

    database_status: str = Field(
        ...,
        description="Overall database status: HEALTHY, DEGRADED, or UNHEALTHY",
    )

    connection_pool_size: int = Field(
        default=0,
        description="Total number of connections in pool",
        ge=0,
    )

    connection_pool_available: int = Field(
        default=0,
        description="Number of available connections in pool",
        ge=0,
    )

    connection_pool_in_use: int = Field(
        default=0,
        description="Number of connections currently in use",
        ge=0,
    )

    database_version: Optional[str] = Field(
        default=None,
        description="PostgreSQL server version string",
    )

    uptime_seconds: Optional[int] = Field(
        default=None,
        description="Database server uptime in seconds",
        ge=0,
    )

    last_check_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="ISO timestamp of this health check",
    )

    error_message: Optional[str] = Field(
        default=None,
        description="Error message if health check failed (None on success)",
    )

    class Config:
        """Pydantic v2 configuration."""

        json_schema_extra: ClassVar[dict[str, Any]] = {
            "examples": [
                {
                    "success": True,
                    "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
                    "execution_time_ms": 5,
                    "database_status": "HEALTHY",
                    "connection_pool_size": 20,
                    "connection_pool_available": 15,
                    "connection_pool_in_use": 5,
                    "database_version": "PostgreSQL 14.5 on x86_64-pc-linux-gnu",
                    "uptime_seconds": 864000,
                    "last_check_timestamp": "2025-10-07T12:34:56.789Z",
                    "error_message": None,
                },
                {
                    "success": True,
                    "correlation_id": "550e8400-e29b-41d4-a716-446655440001",
                    "execution_time_ms": 8,
                    "database_status": "DEGRADED",
                    "connection_pool_size": 20,
                    "connection_pool_available": 2,
                    "connection_pool_in_use": 18,
                    "database_version": "PostgreSQL 14.5 on x86_64-pc-linux-gnu",
                    "uptime_seconds": 864000,
                    "last_check_timestamp": "2025-10-07T12:35:01.123Z",
                    "error_message": "Connection pool near capacity (90% utilization)",
                },
                {
                    "success": False,
                    "correlation_id": "550e8400-e29b-41d4-a716-446655440002",
                    "execution_time_ms": 3,
                    "database_status": "UNHEALTHY",
                    "connection_pool_size": 0,
                    "connection_pool_available": 0,
                    "connection_pool_in_use": 0,
                    "database_version": None,
                    "uptime_seconds": None,
                    "last_check_timestamp": "2025-10-07T12:35:15.456Z",
                    "error_message": "Connection refused: database server not responding",
                },
            ]
        }
