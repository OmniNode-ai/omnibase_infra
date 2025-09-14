"""Send message output model for Kafka producer pool EFFECT node."""

from datetime import datetime

from pydantic import BaseModel, Field


class ModelSendMessageOutput(BaseModel):
    """Output model for send_message operation."""

    success: bool = Field(
        description="Whether message was sent successfully"
    )
    topic: str = Field(
        description="Topic message was sent to"
    )
    partition: int = Field(
        description="Partition message was sent to"
    )
    offset: int = Field(
        description="Offset of the message in the partition"
    )
    timestamp: datetime = Field(
        description="Message timestamp"
    )
    latency_ms: float = Field(
        description="Message send latency in milliseconds"
    )