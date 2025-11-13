#!/usr/bin/env python3
"""
ModelBridgeState - PostgreSQL-Persisted Bridge State.

Persistent bridge state stored in PostgreSQL for the NodeBridgeReducer.
Tracks cumulative aggregation state across multiple reduction windows.

ONEX v2.0 Compliance:
- Suffix-based naming: ModelBridgeState
- O.N.E. v0.1 compliance fields
- PostgreSQL persistence with transaction support
"""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class ModelBridgeState(BaseModel):
    """
    Persistent bridge state for cumulative aggregation tracking.

    This model represents the long-term state of the bridge, persisted
    in PostgreSQL. It tracks cumulative statistics across multiple
    reduction windows and maintains FSM state history.

    Persistence Strategy:
    - Insert new state on first reduction
    - Update existing state on subsequent reductions
    - Maintain versioning for state evolution
    - Track FSM state transitions with history

    Database Mapping:
    - Table: bridge_state
    - Primary Key: state_id
    - Indexes: namespace, last_updated, fsm_state
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    # === State Identity ===
    state_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this bridge state record",
    )

    version: int = Field(
        default=1,
        description="Version number for optimistic locking and state evolution",
        ge=1,
    )

    # === O.N.E. v0.1 Compliance ===
    namespace: str = Field(
        ...,
        description="Namespace for multi-tenant state isolation",
        min_length=1,
        max_length=255,
    )

    metadata_version: str = Field(
        default="0.1",
        description="O.N.E. metadata protocol version",
        pattern=r"^\d+\.\d+$",
    )

    # === Cumulative Aggregation Statistics ===
    total_stamps: int = Field(
        default=0,
        description="Total number of stamps created in this namespace",
        ge=0,
    )

    total_size_bytes: int = Field(
        default=0,
        description="Total size of all stamped files in bytes",
        ge=0,
    )

    unique_file_types: set[str] = Field(
        default_factory=set,
        description="Set of unique content types encountered",
    )

    unique_workflows: set[str] = Field(
        default_factory=set,
        description="Set of unique workflow IDs processed",
    )

    # === FSM State Tracking ===
    current_fsm_state: str = Field(
        default="idle",
        description="Current FSM state of the bridge (idle/active/aggregating/persisting)",
        min_length=1,
        max_length=50,
    )

    fsm_state_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="""
        FSM state transition history.

        Example:
        [
            {"from": "idle", "to": "active", "timestamp": "2025-10-02T10:00:00Z"},
            {"from": "active", "to": "aggregating", "timestamp": "2025-10-02T10:00:05Z"}
        ]
        """,
    )

    # === Temporal Tracking ===
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when this state was first created",
    )

    last_updated: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of the last state update",
    )

    last_aggregation_at: datetime | None = Field(
        default=None,
        description="Timestamp of the most recent aggregation operation",
    )

    # === Window Tracking ===
    last_window_start: datetime | None = Field(
        default=None,
        description="Start timestamp of the last aggregation window",
    )

    last_window_end: datetime | None = Field(
        default=None,
        description="End timestamp of the last aggregation window",
    )

    # === Performance Metrics ===
    total_aggregations: int = Field(
        default=0,
        description="Total number of aggregation operations performed",
        ge=0,
    )

    avg_aggregation_duration_ms: float = Field(
        default=0.0,
        description="Average aggregation duration in milliseconds",
        ge=0.0,
    )

    # === Extended Metadata ===
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="""
        Extended metadata for custom tracking.

        Example:
        {
            "file_type_statistics": {"jpeg": 500, "pdf": 300},
            "size_distribution": {"small": 400, "medium": 300, "large": 100},
            "workflow_statistics": {
                "completed": 750,
                "failed": 50
            }
        }
        """,
    )

    # === Configuration ===
    configuration: dict[str, Any] = Field(
        default_factory=dict,
        description="""
        Bridge-specific configuration.

        Example:
        {
            "aggregation_strategy": "namespace_grouping",
            "window_size_ms": 5000,
            "persistence_interval_ms": 10000
        }
        """,
    )
