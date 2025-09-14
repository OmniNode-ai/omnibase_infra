"""Send message input model for Kafka producer pool EFFECT node."""

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ModelSendMessageInput(BaseModel):
    """Input model for send_message operation."""

    topic: str = Field(
        description="Kafka topic to send message to"
    )
    value: str = Field(
        description="Message value/payload"
    )
    key: Optional[str] = Field(
        default=None,
        description="Optional message key for partitioning"
    )
    headers: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional message headers"
    )
    partition: Optional[int] = Field(
        default=None,
        description="Optional specific partition"
    )
    timestamp: Optional[datetime] = Field(
        default=None,
        description="Optional message timestamp"
    )