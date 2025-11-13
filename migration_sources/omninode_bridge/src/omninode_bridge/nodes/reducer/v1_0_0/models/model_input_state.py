"""Input state model for NodeBridgeReducer.

Defines the input state structure for reducer operations,
representing stamp metadata to be aggregated.
"""

from datetime import datetime
from typing import Any, ClassVar
from uuid import UUID

from pydantic import BaseModel, Field


class ModelReducerInputState(BaseModel):
    """
    Input state for reducer operations.

    This model represents a single stamp metadata record
    to be aggregated by the NodeBridgeReducer.
    """

    # Core stamp identification
    stamp_id: str = Field(..., description="Unique stamp identifier")
    file_hash: str = Field(..., description="BLAKE3 hash of the file")
    file_path: str = Field(..., description="Path to the stamped file")
    file_size: int = Field(..., description="Size of the file in bytes", ge=0)

    # Metadata classification
    namespace: str = Field(
        default="omninode.services.metadata",
        description="Namespace for multi-tenant organization",
    )
    content_type: str = Field(
        default="application/octet-stream",
        description="MIME type of the file",
    )

    # Workflow tracking
    workflow_id: UUID = Field(..., description="Associated workflow correlation ID")
    workflow_state: str = Field(
        default="completed",
        description="Current FSM state of the workflow",
    )

    # Temporal tracking
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when stamp was created",
    )

    # Performance metrics
    processing_time_ms: float = Field(
        default=0.0,
        description="Processing time in milliseconds",
        ge=0.0,
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra: ClassVar[dict[str, Any]] = {
            "example": {
                "stamp_id": "550e8400-e29b-41d4-a716-446655440000",
                "file_hash": "abc123def456...",
                "file_path": "/data/documents/report.pdf",
                "file_size": 1024000,
                "namespace": "omninode.services.metadata",
                "content_type": "application/pdf",
                "workflow_id": "660e8400-e29b-41d4-a716-446655440000",
                "workflow_state": "completed",
                "processing_time_ms": 1.5,
            }
        }
