# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""aiokafka's ``TopicPartition`` namedtuple, structurally (OMN-16769)."""

from __future__ import annotations

from typing import Protocol


class ProtocolTopicPartition(Protocol):
    """A (topic, partition) pair as aiokafka returns it."""

    @property
    def topic(self) -> str:
        """Topic name."""
        ...

    @property
    def partition(self) -> int:
        """Partition id."""
        ...


__all__ = ["ProtocolTopicPartition"]
