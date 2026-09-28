# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Group-less Kafka operations used to validate one source coordinate."""

from __future__ import annotations

from typing import Protocol

from aiokafka.structs import TopicPartition


class ProtocolExactSimSourceConsumer(Protocol):
    """The group-less Kafka operations needed for one immutable source read."""

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    def assign(self, partitions: list[TopicPartition]) -> None: ...

    def seek(self, partition: TopicPartition, offset: int) -> None: ...

    async def getone(self) -> object: ...


__all__ = ["ProtocolExactSimSourceConsumer"]
