# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The producer acquired at the bus effect boundary for registration snapshots."""

from aiokafka import AIOKafkaProducer
from pydantic import BaseModel, ConfigDict


class ModelRegistrationSnapshotProducer(BaseModel):
    """The plugin starts the publisher and releases this producer if start fails."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", arbitrary_types_allowed=True, from_attributes=True
    )

    producer: AIOKafkaProducer
