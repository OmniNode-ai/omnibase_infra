# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Structural protocol matching the AIOKafkaAdminClient methods used by the runtime.

Defined here rather than importing ``AIOKafkaAdminClient`` directly so that
modules using only this shape avoid the hard ``aiokafka`` import at module
parse time.

.. versionadded:: 0.39.0
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class ProtocolKafkaAdminLike(Protocol):
    """Minimal protocol for Kafka admin clients.

    Any object exposing ``start``, ``close``, ``list_consumer_groups``, and
    ``describe_consumer_groups`` satisfies this protocol, including
    ``AIOKafkaAdminClient``.
    """

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def list_consumer_groups(
        self, broker_ids: Sequence[int] | None = None
    ) -> Sequence[tuple[str, str | None]]:
        pass

    async def describe_consumer_groups(
        self,
        group_ids: Sequence[str],
        group_coordinator_id: int | None = None,
        include_authorized_operations: bool = False,
    ) -> list[object]:
        pass


__all__: list[str] = ["ProtocolKafkaAdminLike"]
