#!/usr/bin/env python3
"""
ModelBridgeStateInput - Bridge State Persistence Input (UPSERT).

Input model for persisting bridge aggregation state in PostgreSQL.
Supports UPSERT operations (insert if not exists, update if exists)
for maintaining cumulative bridge statistics across reduction windows.

ONEX v2.0 Compliance:
- Suffix-based naming: ModelBridgeStateInput
- UUID bridge identity tracking
- Namespace-based multi-tenant isolation
- Comprehensive field validation with Pydantic v2
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelBridgeStateInput(BaseModel):
    """
    Input model for bridge state database UPSERT operations.

    This model represents the persistent state of the NodeBridgeReducer,
    tracking cumulative aggregation statistics across multiple reduction
    windows. The database uses UPSERT semantics to update existing state
    or insert new state as needed.

    Database Table: bridge_states
    Primary Key: bridge_id (UUID)
    Unique Constraint: bridge_id
    Indexes: namespace, last_aggregation_timestamp

    UPSERT Logic:
    - If bridge_id exists: UPDATE all fields, set updated_at = NOW()
    - If bridge_id doesn't exist: INSERT new record with all fields

    Event Sources:
    - STATE_AGGREGATION_COMPLETED: Published by NodeBridgeReducer
      after each aggregation window completion

    Persistence Strategy:
    - Incremental updates to counters (total_workflows_processed, etc.)
    - Metadata merging for aggregation statistics
    - FSM state tracking for reducer lifecycle

    Example (New Bridge State):
        >>> from uuid import uuid4
        >>> from datetime import datetime, timezone
        >>> state_data = ModelBridgeStateInput(
        ...     bridge_id=uuid4(),
        ...     namespace="production",
        ...     total_workflows_processed=150,
        ...     total_items_aggregated=750,
        ...     aggregation_metadata={
        ...         "file_type_distribution": {"jpeg": 500, "pdf": 250},
        ...         "avg_file_size_bytes": 102400
        ...     },
        ...     current_fsm_state="aggregating",
        ...     last_aggregation_timestamp=datetime.now(timezone.utc)
        ... )

    Example (State Update):
        >>> updated_state = ModelBridgeStateInput(
        ...     bridge_id=uuid4(),  # Same as original
        ...     namespace="production",
        ...     total_workflows_processed=200,  # Incremented
        ...     total_items_aggregated=1000,    # Incremented
        ...     aggregation_metadata={
        ...         "file_type_distribution": {"jpeg": 650, "pdf": 350},
        ...         "avg_file_size_bytes": 105000
        ...     },
        ...     current_fsm_state="idle",
        ...     last_aggregation_timestamp=datetime.now(timezone.utc)
        ... )
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    # === Bridge Identity ===
    bridge_id: UUID = Field(
        ...,
        description="""
        Unique identifier for this bridge instance.

        This ID is used as the primary key in the bridge_states table.
        UPSERT operations use this ID to determine whether to INSERT
        or UPDATE the record.
        """,
    )

    # === Multi-Tenant Isolation ===
    namespace: str = Field(
        ...,
        description="""
        Namespace for multi-tenant state isolation.

        All aggregation statistics are scoped to this namespace,
        enabling independent state tracking for different tenants
        or environments.
        """,
        min_length=1,
        max_length=255,
    )

    # === Cumulative Statistics ===
    total_workflows_processed: int = Field(
        ...,
        description="""
        Total number of workflows processed across all aggregation windows.

        This counter is incremented with each aggregation cycle.
        Represents the cumulative workflow throughput for this bridge instance.
        """,
        ge=0,
    )

    total_items_aggregated: int = Field(
        ...,
        description="""
        Total number of items aggregated across all windows.

        Tracks the cumulative sum of all items that have been aggregated
        by this bridge instance. Used for throughput analysis.
        """,
        ge=0,
    )

    # === Aggregation Metadata ===
    aggregation_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="""
        Extended aggregation metadata and statistics.

        Example:
        {
            "file_type_distribution": {
                "jpeg": 500,
                "pdf": 300,
                "png": 200
            },
            "size_distribution": {
                "small_files": 400,    // < 1MB
                "medium_files": 300,   // 1-10MB
                "large_files": 100     // > 10MB
            },
            "avg_file_size_bytes": 102400,
            "total_bytes_processed": 819200000,
            "window_statistics": {
                "avg_window_duration_ms": 5000,
                "avg_items_per_window": 100
            },
            "fsm_state_durations": {
                "idle": 10000,
                "active": 15000,
                "aggregating": 25000,
                "persisting": 5000
            }
        }
        """,
    )

    # === FSM State Tracking ===
    current_fsm_state: str = Field(
        ...,
        description="""
        Current FSM state of the bridge reducer.

        Valid states:
        - "idle": No active aggregation
        - "active": Receiving input items
        - "aggregating": Processing aggregation window
        - "persisting": Saving state to database

        This field tracks the operational state of the reducer
        for monitoring and diagnostics.
        """,
        min_length=1,
        max_length=50,
    )

    # === Temporal Tracking ===
    last_aggregation_timestamp: datetime | None = Field(
        default=None,
        description="""
        Timestamp of the most recent aggregation operation.

        Updated each time the reducer completes an aggregation window.
        Used for staleness detection and health monitoring.
        """,
    )
