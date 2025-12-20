# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol definition for event bus objects.

This protocol defines the interface for event bus dependencies
using duck typing with Python's Protocol class.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ProtocolEventBus(Protocol):
    """Protocol for event bus objects.

    Event bus must implement an async publish method for sending messages
    to topics.

    Future Enhancement (NITPICK):
        Consider replacing the raw `publish(topic, key, value)` signature with a
        Pydantic model-based approach for improved type safety and validation:

        ```python
        class ModelPublishParams(BaseModel):
            topic: str
            key: bytes
            value: bytes
            headers: dict[str, str] = Field(default_factory=dict)
            partition_key: str | None = None

        async def publish(self, params: ModelPublishParams) -> None:
            ...
        ```

        Benefits:
        - Automatic validation of topic format, key/value encoding
        - Easy extension with new fields (headers, partition keys)
        - Self-documenting parameter structure
        - Consistent with ONEX contract-driven patterns

        This is a non-breaking change that can be implemented in a future version
        by adding a new `publish_model` method alongside the existing `publish`.
    """

    async def publish(self, topic: str, key: bytes, value: bytes) -> None:
        """Publish a message to a topic.

        Args:
            topic: The topic name to publish to
            key: Message key as bytes
            value: Message value as bytes (typically JSON-encoded)
        """
        ...


__all__ = [
    "ProtocolEventBus",
]
