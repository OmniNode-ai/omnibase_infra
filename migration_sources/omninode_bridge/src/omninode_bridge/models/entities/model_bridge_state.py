"""Bridge state entity model.

Maps to the bridge_states database table for tracking reducer aggregation state.
"""

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omninode_bridge.schemas import BRIDGE_STATE_METADATA_SCHEMA
from omninode_bridge.utils.metadata_validator import validate_metadata


class ModelBridgeState(BaseModel):
    """
    Bridge state entity model (maps to bridge_states table).

    Tracks aggregation state for NodeBridgeReducer instances, storing
    accumulated workflow processing metrics and FSM state.

    Attributes:
        bridge_id: Primary key UUID for the bridge reducer instance
        namespace: Multi-tenant namespace for aggregation grouping
        total_workflows_processed: Counter of workflows processed
        total_items_aggregated: Counter of total items aggregated
        aggregation_metadata: Additional aggregation metadata as JSONB
        current_fsm_state: Current FSM state of the bridge
        last_aggregation_timestamp: Timestamp of last aggregation window completion
        created_at: Record creation timestamp (auto-managed)
        updated_at: Last update timestamp (auto-managed)

    Example:
        ```python
        from uuid import uuid4
        from datetime import datetime

        bridge_state = ModelBridgeState(
            bridge_id=uuid4(),
            namespace="test_app",
            total_workflows_processed=100,
            total_items_aggregated=500,
            current_fsm_state="ACTIVE",
            aggregation_metadata={"window_size_ms": 5000}
        )
        ```
    """

    # Primary key (NOT auto-generated)
    bridge_id: UUID = Field(
        ...,
        description="Unique identifier for the bridge reducer instance (PRIMARY KEY)",
    )

    # Required fields
    namespace: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Multi-tenant namespace for aggregation grouping",
        examples=["test_app", "production_app"],
    )
    total_workflows_processed: int = Field(
        default=0, ge=0, description="Counter of workflows processed by this bridge"
    )
    total_items_aggregated: int = Field(
        default=0, ge=0, description="Counter of total items aggregated"
    )
    current_fsm_state: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Current FSM state of the bridge",
        examples=["IDLE", "ACTIVE", "PAUSED", "STOPPED"],
    )

    # Optional fields
    aggregation_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional aggregation metadata as JSONB"
    )
    last_aggregation_timestamp: Optional[datetime] = Field(
        default=None, description="Timestamp of last aggregation window completion"
    )

    # Auto-managed timestamps
    created_at: Optional[datetime] = Field(
        default=None, description="Record creation timestamp (auto-managed by DB)"
    )
    updated_at: Optional[datetime] = Field(
        default=None, description="Last update timestamp (auto-managed by DB)"
    )

    @field_validator("current_fsm_state")
    @classmethod
    def validate_current_fsm_state(cls, v: str) -> str:
        """Normalize FSM state to uppercase."""
        return v.upper()

    @field_validator("total_workflows_processed", "total_items_aggregated")
    @classmethod
    def validate_counters(cls, v: int) -> int:
        """Validate counters are non-negative."""
        if v < 0:
            raise ValueError("Counter values must be non-negative")
        return v

    @field_validator("aggregation_metadata")
    @classmethod
    def validate_metadata_field(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate aggregation_metadata against JSON schema."""
        if v:
            validate_metadata(v, BRIDGE_STATE_METADATA_SCHEMA)
        return v

    model_config = ConfigDict(
        from_attributes=True,  # Enable ORM mode for DB row mapping
        json_schema_extra={
            "example": {
                "bridge_id": "550e8400-e29b-41d4-a716-446655440000",
                "namespace": "test_app",
                "total_workflows_processed": 100,
                "total_items_aggregated": 500,
                "aggregation_metadata": {
                    "window_size_ms": 5000,
                    "batch_size": 100,
                    "last_window_items": 25,
                },
                "current_fsm_state": "ACTIVE",
                "last_aggregation_timestamp": "2025-10-08T12:00:00Z",
            }
        },
    )
