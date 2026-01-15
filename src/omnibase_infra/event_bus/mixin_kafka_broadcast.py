# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Broadcast messaging mixin for Kafka event bus.

This module provides broadcast and group messaging functionality that can be
mixed into EventBusKafka to support environment-wide and group-specific
message distribution.

Features:
    - Environment-wide broadcast messaging
    - Group-specific targeted messaging
    - JSON payload serialization
    - Proper header construction with timestamps

Usage:
    ```python
    class EventBusKafka(MixinKafkaBroadcast, MixinKafkaDlq, MixinAsyncCircuitBreaker):
        def __init__(self, config):
            # ... rest of init

        # Broadcast methods are now available:
        # - broadcast_to_environment()
        # - send_to_group()
    ```

Design Note:
    This mixin assumes the parent class has:
    - self._environment: str for environment context
    - self._group: str for consumer group context
    - self.publish(): Async method to publish messages
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

from omnibase_infra.event_bus.models import ModelEventHeaders

logger = logging.getLogger(__name__)


class MixinKafkaBroadcast:
    """Mixin providing broadcast and convenience publishing for Kafka event bus.

    This mixin adds environment-wide messaging, group-specific messaging, and
    envelope publishing capabilities to a Kafka event bus implementation.

    Methods provided by mixin:
        broadcast_to_environment: Broadcast command to all subscribers in environment
        send_to_group: Send command to specific consumer group
        publish_envelope: Publish an OnexEnvelope to a topic (protocol compatibility)

    Required attributes from parent class:
        _environment: str for environment context
        _group: str for consumer group context

    Required methods from parent class:
        publish: Async method to publish messages to a topic
    """

    # Type hints for attributes expected from parent class
    _environment: str
    _group: str

    async def broadcast_to_environment(
        self,
        command: str,
        payload: dict[str, object],
        target_environment: str | None = None,
    ) -> None:
        """Broadcast command to environment.

        Sends a command message to all subscribers in the target environment.

        Args:
            command: Command identifier
            payload: Command payload data
            target_environment: Target environment (defaults to current)
        """
        env = target_environment or self._environment
        topic = f"{env}.broadcast"
        value_dict = {"command": command, "payload": payload}
        value = json.dumps(value_dict).encode("utf-8")

        headers = ModelEventHeaders(
            source=f"{self._environment}.{self._group}",
            event_type="broadcast",
            content_type="application/json",
            timestamp=datetime.now(UTC),
        )

        await self.publish(topic, None, value, headers)

    async def send_to_group(
        self,
        command: str,
        payload: dict[str, object],
        target_group: str,
    ) -> None:
        """Send command to specific group.

        Sends a command message to all subscribers in a specific group.

        Args:
            command: Command identifier
            payload: Command payload data
            target_group: Target group identifier
        """
        topic = f"{self._environment}.{target_group}"
        value_dict = {"command": command, "payload": payload}
        value = json.dumps(value_dict).encode("utf-8")

        headers = ModelEventHeaders(
            source=f"{self._environment}.{self._group}",
            event_type="group_command",
            content_type="application/json",
            timestamp=datetime.now(UTC),
        )

        await self.publish(topic, None, value, headers)

    async def publish_envelope(
        self,
        envelope: object,
        topic: str,
    ) -> None:
        """Publish an OnexEnvelope to a topic.

        Protocol method for ProtocolEventBus compatibility.
        Serializes the envelope to JSON bytes and publishes.

        Args:
            envelope: Envelope object to publish (ModelOnexEnvelope)
            topic: Target topic name
        """
        # Serialize envelope to JSON bytes
        envelope_dict: object
        if hasattr(envelope, "model_dump"):
            envelope_dict = envelope.model_dump(mode="json")
        elif hasattr(envelope, "dict"):
            envelope_dict = envelope.dict()
        elif isinstance(envelope, dict):
            envelope_dict = envelope
        else:
            envelope_dict = envelope

        value = json.dumps(envelope_dict).encode("utf-8")

        headers = ModelEventHeaders(
            source=f"{self._environment}.{self._group}",
            event_type=topic,
            content_type="application/json",
            timestamp=datetime.now(UTC),
        )

        await self.publish(topic, None, value, headers)


__all__: list[str] = ["MixinKafkaBroadcast"]
