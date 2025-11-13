"""Output model for aggregated state from NodeBridgeReducer."""

from datetime import datetime
from typing import Any, ClassVar
from uuid import UUID

from pydantic import BaseModel, Field


class ModelAggregateStateOutput(BaseModel):
    """
    Output model for NodeBridgeReducer aggregation results.

    Contains aggregated metadata across all processed stamps
    with namespace-based grouping and FSM state tracking.
    """

    # Aggregation results
    namespaces: list[str] = Field(
        default_factory=list,
        description="List of unique namespaces processed",
    )
    total_stamps: int = Field(
        default=0,
        description="Total number of stamps aggregated",
        ge=0,
    )

    # Namespace-level aggregations
    aggregations: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Aggregated data by namespace",
    )

    # FSM state tracking
    fsm_states: dict[UUID, str] = Field(
        default_factory=dict,
        description="FSM states by workflow ID",
    )

    # Processing metadata
    items_processed: int = Field(
        default=0,
        description="Number of items processed",
        ge=0,
    )
    processing_time_ms: float = Field(
        default=0.0,
        description="Total processing time in milliseconds",
        ge=0.0,
    )
    batches_processed: int = Field(
        default=0,
        description="Number of batches processed",
        ge=0,
    )

    # Temporal tracking
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When aggregation was completed",
    )

    # Performance statistics
    performance_stats: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional performance statistics",
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra: ClassVar[dict[str, Any]] = {
            "example": {
                "namespaces": [
                    "omninode.services.metadata",
                    "omninode.services.onextree",
                ],
                "total_stamps": 1000,
                "aggregations": {
                    "omninode.services.metadata": {
                        "total_stamps": 800,
                        "total_size_bytes": 1024000000,
                        "file_types": {"application/pdf", "image/png"},
                    }
                },
                "fsm_states": {"660e8400-e29b-41d4-a716-446655440000": "completed"},
                "items_processed": 1000,
                "processing_time_ms": 150.5,
                "batches_processed": 10,
            }
        }
