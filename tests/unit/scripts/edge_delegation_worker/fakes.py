# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Shared in-memory fake channel for worker_cycle / run_worker_loop tests."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from omnibase_core.types import JsonType
from scripts.edge_delegation_worker.models import ModelDelegationEnvelope


@dataclass
class PublishedResult:
    topic: str
    correlation_id: UUID
    event_type: str
    payload: dict[str, JsonType]


class FakeDelegationChannel:
    """In-memory ``ProtocolDelegationChannel`` for worker-cycle tests."""

    def __init__(self, envelopes: list[ModelDelegationEnvelope] | None = None) -> None:
        self._queue: list[ModelDelegationEnvelope] = list(envelopes or [])
        self.published: list[PublishedResult] = []
        self.acked: list[UUID] = []
        self.nacked: list[tuple[UUID, str]] = []

    def push(self, envelope: ModelDelegationEnvelope) -> None:
        self._queue.append(envelope)

    async def claim(self) -> ModelDelegationEnvelope | None:
        if not self._queue:
            return None
        return self._queue.pop(0)

    async def publish_result(
        self,
        *,
        topic: str,
        correlation_id: UUID,
        event_type: str,
        payload: dict[str, JsonType],
    ) -> None:
        self.published.append(
            PublishedResult(
                topic=topic,
                correlation_id=correlation_id,
                event_type=event_type,
                payload=payload,
            )
        )

    async def ack(self, envelope: ModelDelegationEnvelope) -> None:
        self.acked.append(envelope.correlation_id)

    async def nack(self, envelope: ModelDelegationEnvelope, *, reason: str) -> None:
        self.nacked.append((envelope.correlation_id, reason))
