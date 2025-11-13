"""Model for storage operation result."""

from typing import ClassVar, Optional
from pydantic import BaseModel, Field


class ModelStorageResult(BaseModel):
    """
    Result of storage operation.

    Contains information about the storage operation including success status,
    file paths, and optional database storage details.
    """

    success: bool = Field(
        ...,
        description="Whether all storage operations were successful"
    )

    artifacts_stored: int = Field(
        default=0,
        description="Number of artifacts successfully stored",
        ge=0
    )

    storage_errors: list[str] = Field(
        default_factory=list,
        description="List of error messages encountered during storage"
    )

    storage_time_ms: float = Field(
        ...,
        description="Time taken to complete storage operations in milliseconds",
        ge=0.0
    )

    stored_files: list[str] = Field(
        default_factory=list,
        description="List of file paths that were successfully stored"
    )

    total_bytes_written: int = Field(
        default=0,
        description="Total bytes written to disk",
        ge=0
    )

    metrics_stored: bool = Field(
        default=False,
        description="Whether metrics were stored in PostgreSQL"
    )

    database_record_id: Optional[str] = Field(
        default=None,
        description="Database record ID if metrics were stored"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra: ClassVar[dict] = {
            "example": {
                "success": True,
                "artifacts_stored": 3,
                "storage_errors": [],
                "storage_time_ms": 25.5,
                "stored_files": [
                    "/path/to/node.py",
                    "/path/to/test_node.py",
                    "/path/to/models.py"
                ],
                "total_bytes_written": 15840,
                "metrics_stored": True,
                "database_record_id": "550e8400-e29b-41d4-a716-446655440000"
            }
        }
