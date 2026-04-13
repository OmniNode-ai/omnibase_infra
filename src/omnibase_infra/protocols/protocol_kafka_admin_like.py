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

    async def start(self) -> None: ...

    async def close(self) -> None: ...

    async def list_consumer_groups(self) -> Sequence[tuple[str, str | None]]: ...

    async def describe_consumer_groups(
        self, group_ids: Sequence[str]
    ) -> dict[str, object]: ...


__all__: list[str] = ["ProtocolKafkaAdminLike"]
