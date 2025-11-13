"""
Stamp Response Output Model for Bridge Orchestrator.

Output state model for stamping workflow results.

ONEX Compliance:
- Suffix-based naming: ModelStampResponseOutput
- Strong typing with Pydantic validation
- Performance metrics tracking
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from .enum_workflow_state import EnumWorkflowState


class ModelStampResponseOutput(BaseModel):
    """
    Output model for stamping workflow results.

    Contains the stamped content, metadata, and workflow
    execution metrics.
    """

    # Stamp results
    stamp_id: str = Field(..., description="Unique identifier for the stamp")
    file_hash: str = Field(..., description="BLAKE3 hash of file content")
    stamped_content: str = Field(
        ..., description="Content with embedded metadata stamp"
    )
    stamp_metadata: dict[str, str] = Field(
        ..., description="Metadata included in stamp"
    )

    # O.N.E. v0.1 compliance fields
    namespace: str = Field(..., description="Namespace for stamp")
    op_id: UUID = Field(..., description="Operation ID for correlation")
    version: int = Field(default=1, description="Stamp version")
    metadata_version: str = Field(default="0.1", description="O.N.E. metadata version")

    # Workflow state
    workflow_state: EnumWorkflowState = Field(..., description="Final workflow state")
    workflow_id: UUID = Field(..., description="Workflow correlation ID")

    # OnexTree intelligence (optional)
    intelligence_data: Optional[dict[str, str]] = Field(
        default=None, description="OnexTree intelligence analysis results"
    )

    # Performance metrics
    processing_time_ms: float = Field(..., description="Total workflow processing time")
    hash_generation_time_ms: float = Field(
        ..., description="BLAKE3 hash generation time"
    )
    workflow_steps_executed: int = Field(..., description="Number of workflow steps")

    # Metadata
    created_at: datetime = Field(..., description="Stamp creation timestamp")
    completed_at: datetime = Field(..., description="Workflow completion timestamp")

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": {
                "stamp_id": "550e8400-e29b-41d4-a716-446655440000",
                "file_hash": "abc123def456...",
                "stamped_content": "Content with embedded stamp...",
                "stamp_metadata": {
                    "file_path": "/data/documents/contract.pdf",
                    "content_type": "application/pdf",
                },
                "namespace": "omninode.services.metadata",
                "workflow_state": "completed",
                "processing_time_ms": 15.5,
                "hash_generation_time_ms": 1.2,
                "workflow_steps_executed": 4,
            }
        },
    )
