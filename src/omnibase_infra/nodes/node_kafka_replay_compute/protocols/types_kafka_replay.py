# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Type aliases for Kafka replay compute."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.nodes.node_kafka_replay_compute.models import (
    ModelKafkaReplayInput,
    ModelKafkaReplayProgress,
)
from omnibase_infra.nodes.node_kafka_replay_compute.protocols.protocol_kafka_replay_consumer import (
    ProtocolKafkaReplayConsumer,
)

ConsumerFactory = Callable[[ModelKafkaReplayInput], ProtocolKafkaReplayConsumer]
ProgressCallback = Callable[[ModelKafkaReplayProgress], Awaitable[None] | None]
EnvelopeDeserializer = Callable[[bytes], ModelEventEnvelope[object]]


__all__ = ["ConsumerFactory", "EnvelopeDeserializer", "ProgressCallback"]
