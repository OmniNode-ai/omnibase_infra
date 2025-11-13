#!/usr/bin/env python3
"""
ModelProjectionWatermark - Projection Watermark Entity.

Tracks materialization progress per partition for eventual consistency.
Maps 1:1 with projection_watermarks database table.

ONEX v2.0 Compliance:
- Suffix-based naming: ModelProjectionWatermark
- Partition-based progress tracking
- Lag detection support
- Comprehensive field validation with Pydantic v2

Pure Reducer Refactor - Wave 1, Workstream 1B
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ModelProjectionWatermark(BaseModel):
    """
    Entity model for projection_watermarks table.

    This model tracks the progress of projection materialization
    per Kafka partition or shard. Used to detect lag and prevent
    reprocessing of events.

    Database Table: projection_watermarks
    Primary Key: partition_id (TEXT)

    Usage Pattern:
        - One row per Kafka partition or shard
        - Updated atomically with projection writes
        - Used to detect projection lag and staleness

    Example:
        >>> watermark = ModelProjectionWatermark(
        ...     partition_id="kafka-partition-0",
        ...     offset=12345
        ... )
        >>> # Check for lag
        >>> lag_ms = (datetime.now() - watermark.updated_at).total_seconds() * 1000
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        frozen=False,
        extra="forbid",
        populate_by_name=True,
    )

    # Primary key
    partition_id: str = Field(
        ...,
        description="Kafka partition ID or shard identifier",
        min_length=1,
        max_length=255,
    )

    # Offset tracking
    offset: int = Field(
        default=0,
        description="Last successfully processed event offset",
        ge=0,
    )

    # Temporal tracking
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(),
        description="Timestamp of last watermark update (for lag detection)",
    )
