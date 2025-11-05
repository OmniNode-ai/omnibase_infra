"""Consumer group metadata model."""

from datetime import datetime

from pydantic import BaseModel, Field


class ModelConsumerGroupMetadata(BaseModel):
    """Kafka consumer group metadata and tracking information.

    Tracks consumer group state, member information, and health metrics
    for monitoring and debugging consumer group behavior.
    """

    group_id: str = Field(description="Consumer group identifier")
    state: str = Field(
        description="Consumer group state: Stable, PreparingRebalance, CompletingRebalance, Dead, Empty",
    )
    protocol_type: str = Field(default="consumer", description="Group protocol type")
    protocol: str | None = Field(default=None, description="Partition assignment protocol")

    # Member information
    member_count: int = Field(ge=0, description="Number of active members in the group")
    members: list[str] = Field(
        default_factory=list,
        description="List of member IDs in the consumer group",
    )

    # Topic assignments
    assigned_topics: list[str] = Field(
        default_factory=list,
        description="Topics assigned to this consumer group",
    )
    total_partitions: int = Field(ge=0, default=0, description="Total partitions assigned")

    # Lag information
    total_lag: int = Field(ge=0, default=0, description="Total consumer lag across all partitions")
    max_lag: int = Field(ge=0, default=0, description="Maximum lag on any partition")

    # Timestamps
    created_at: datetime = Field(description="When the consumer group was first created")
    last_rebalance: datetime | None = Field(
        default=None,
        description="Last rebalance timestamp",
    )
    last_activity: datetime | None = Field(
        default=None,
        description="Last consumer activity timestamp",
    )

    # Health indicators
    is_healthy: bool = Field(default=True, description="Whether the consumer group is healthy")
    error_count: int = Field(ge=0, default=0, description="Number of errors encountered")
    last_error: str | None = Field(default=None, description="Last error message")

    class Config:
        """Pydantic model configuration."""
        validate_assignment = True
        extra = "forbid"
