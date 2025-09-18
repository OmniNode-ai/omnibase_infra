"""Strongly typed Kafka metadata models."""

from pydantic import BaseModel, Field


class ModelKafkaPartitionInfo(BaseModel):
    """Kafka partition information."""

    partition_id: int = Field(
        description="Partition identifier",
        ge=0,
    )

    leader: int = Field(
        description="Leader broker ID for this partition",
    )

    replicas: list[int] = Field(
        description="List of replica broker IDs",
    )

    in_sync_replicas: list[int] = Field(
        description="List of in-sync replica broker IDs",
    )


class ModelKafkaTopicInfo(BaseModel):
    """Kafka topic metadata information."""

    topic_name: str = Field(
        description="Name of the Kafka topic",
    )

    partition_count: int = Field(
        description="Number of partitions in the topic",
        ge=1,
    )

    replication_factor: int = Field(
        description="Replication factor for the topic",
        ge=1,
    )

    partitions: list[ModelKafkaPartitionInfo] = Field(
        default_factory=list,
        description="Partition information for the topic",
    )

    config: dict[str, str] = Field(
        default_factory=dict,
        description="Topic configuration settings",
    )


class ModelKafkaOffsetInfo(BaseModel):
    """Kafka offset information."""

    topic: str = Field(
        description="Topic name",
    )

    partition: int = Field(
        description="Partition number",
        ge=0,
    )

    offset: int = Field(
        description="Message offset",
        ge=0,
    )

    timestamp: int | None = Field(
        default=None,
        description="Message timestamp in milliseconds since epoch",
    )

    key_size: int | None = Field(
        default=None,
        description="Size of message key in bytes",
    )

    value_size: int | None = Field(
        default=None,
        description="Size of message value in bytes",
    )


class ModelKafkaBrokerInfo(BaseModel):
    """Kafka broker information."""

    broker_id: int = Field(
        description="Unique broker identifier",
    )

    host: str = Field(
        description="Broker hostname or IP address",
    )

    port: int = Field(
        description="Broker port number",
        ge=1,
        le=65535,
    )

    rack: str | None = Field(
        default=None,
        description="Rack identifier for the broker",
    )

    is_controller: bool = Field(
        default=False,
        description="Whether this broker is the cluster controller",
    )

    endpoints: dict[str, str] = Field(
        default_factory=dict,
        description="Protocol endpoint mappings (e.g., PLAINTEXT, SSL)",
    )
