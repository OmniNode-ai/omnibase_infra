# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Minimal consumer protocol for log-end offset lookups (OMN-12632).

Captures only the ``AIOKafkaConsumer.end_offsets`` surface that
:class:`AdapterKafkaAdminLag` needs to serve ``list_offsets``. Declared
structurally so the adapter avoids a hard ``aiokafka`` import at parse time and
so tests can supply a fake mirroring the real 0.13.0 consumer.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Protocol


class ProtocolKafkaLagConsumer(Protocol):
    """Subset of ``AIOKafkaConsumer`` used to read per-partition log-end offsets."""

    async def end_offsets(self, partitions: Iterable[object]) -> Mapping[object, int]:
        """Return the log-end (high-water-mark) offset for each TopicPartition."""
        ...


__all__ = ["ProtocolKafkaLagConsumer"]
