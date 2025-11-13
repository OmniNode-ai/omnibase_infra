"""Output state model for NodeBridgeReducer.

Defines the output state structure for reducer operations,
representing aggregated metadata results and side effect intents.
"""

from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    pass


class ModelReducerOutputState(BaseModel):
    """
    Output state for NodeBridgeReducer aggregation results.

    Contains aggregated metadata across all processed stamps
    with namespace-based grouping and FSM state tracking.
    """

    # Aggregation type
    aggregation_type: str = Field(
        default="namespace_grouping",
        description="Type of aggregation performed",
    )

    # Aggregation results
    namespaces: list[str] = Field(
        default_factory=list,
        description="List of unique namespaces processed",
    )
    total_items: int = Field(
        default=0,
        description="Total number of items aggregated",
        ge=0,
    )
    total_size_bytes: int = Field(
        default=0,
        description="Total size in bytes across all items",
        ge=0,
    )

    # Namespace-level aggregations
    aggregations: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Aggregated data by namespace",
    )

    # FSM state tracking
    fsm_states: dict[str, str] = Field(
        default_factory=dict,
        description="FSM states by workflow ID",
    )

    # Side effect intents (ONEX v2.0 pure function architecture)
    intents: list[Any] = Field(
        default_factory=list,
        description="Side effect intents to be executed by other nodes",
    )

    # Performance metrics
    aggregation_duration_ms: float = Field(
        default=0.0,
        description="Total aggregation duration in milliseconds",
        ge=0.0,
    )
    items_per_second: float = Field(
        default=0.0,
        description="Processing throughput (items per second)",
        ge=0.0,
    )

    # Temporal tracking
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When aggregation was completed",
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra: ClassVar[dict[str, Any]] = {
            "example": {
                "aggregation_type": "namespace_grouping",
                "namespaces": [
                    "omninode.services.metadata",
                    "omninode.services.onextree",
                ],
                "total_items": 1000,
                "total_size_bytes": 1024000000,
                "aggregations": {
                    "omninode.services.metadata": {
                        "total_stamps": 800,
                        "total_size_bytes": 819200000,
                        "file_types": ["application/pdf", "image/png"],
                    }
                },
                "fsm_states": {"660e8400-e29b-41d4-a716-446655440000": "completed"},
                "intents": [
                    {
                        "intent_type": "PersistState",
                        "target": "store_effect",
                        "payload": {"namespace": "omninode.services.metadata"},
                        "priority": 1,
                    }
                ],
                "aggregation_duration_ms": 150.5,
                "items_per_second": 6644.5,
            }
        }
