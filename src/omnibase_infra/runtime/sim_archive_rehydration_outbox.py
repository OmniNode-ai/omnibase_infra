# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Durable sim-only archive replay outbox keyed by original broker position."""

from __future__ import annotations

import base64
import json
from collections.abc import Sequence

import asyncpg

from omnibase_infra.runtime.health.runtime_lane_identity import (
    resolve_declared_runtime_lane,
)
from omnibase_infra.runtime.protocols.protocol_raw_replay_publisher import (
    ProtocolRawReplayPublisher,
)
from omnibase_infra.runtime.protocols.protocol_sim_archive_rehydration_plan import (
    ProtocolSimArchiveRehydrationPlan,
)

_INSERT = """
INSERT INTO public.sim_archive_rehydration_outbox (
    source_topic, source_partition, source_offset, target_topic,
    record_key, record_value, headers_json, timestamp_ms
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
ON CONFLICT (source_topic, source_partition, source_offset) DO NOTHING
RETURNING source_topic
"""
_EXISTING = """
SELECT target_topic, record_key, record_value, headers_json, timestamp_ms
FROM public.sim_archive_rehydration_outbox
WHERE source_topic = $1 AND source_partition = $2 AND source_offset = $3
FOR UPDATE
"""
_PENDING = """
SELECT source_topic, source_partition, source_offset, target_topic,
       record_key, record_value, headers_json, timestamp_ms
FROM public.sim_archive_rehydration_outbox
WHERE delivered_at IS NULL
ORDER BY source_topic, source_partition, source_offset
LIMIT $1
FOR UPDATE SKIP LOCKED
"""
_DELIVERED = """
UPDATE public.sim_archive_rehydration_outbox
SET delivered_at = NOW()
WHERE source_topic = $1 AND source_partition = $2 AND source_offset = $3
  AND delivered_at IS NULL
"""


class PostgresSimArchiveRehydrationOutbox:
    """Atomically enqueue a source position and relay it at least once."""

    def __init__(
        self,
        pool: asyncpg.Pool,
        publisher: ProtocolRawReplayPublisher,
        *,
        rehydration_targets: frozenset[str],
    ) -> None:
        if resolve_declared_runtime_lane() != "sim-202":
            raise ValueError("archive rehydration outbox requires sim-202 runtime lane")
        if (
            not isinstance(rehydration_targets, frozenset)
            or not rehydration_targets
            or any(
                not isinstance(topic, str) or not topic or topic != topic.strip()
                for topic in rehydration_targets
            )
        ):
            raise ValueError("verified rehydration_targets must be nonempty")
        self._pool = pool
        self._publisher = publisher
        self._rehydration_targets = rehydration_targets

    @staticmethod
    def encode_headers(headers: Sequence[tuple[str, bytes | None]]) -> str:
        """Store ordered, duplicate-preserving header bytes as canonical JSON."""
        return json.dumps(
            [
                [
                    name,
                    base64.b64encode(value).decode("ascii")
                    if value is not None
                    else None,
                ]
                for name, value in headers
            ],
            separators=(",", ":"),
            ensure_ascii=True,
        )

    @staticmethod
    def decode_headers(raw: str) -> list[tuple[str, bytes | None]]:
        """Rebuild the exact ordered Kafka header sequence from durable storage."""
        parsed: object = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError("invalid stored replay headers")
        result: list[tuple[str, bytes | None]] = []
        for pair in parsed:
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or not isinstance(pair[0], str)
                or (pair[1] is not None and not isinstance(pair[1], str))
            ):
                raise ValueError("invalid stored replay headers")
            encoded = pair[1]
            result.append(
                (
                    pair[0],
                    base64.b64decode(encoded, validate=True)
                    if encoded is not None
                    else None,
                )
            )
        return result

    async def enqueue_once(self, plan: ProtocolSimArchiveRehydrationPlan) -> bool:
        """Insert once or refuse a different record at the same source coordinate."""
        source_topic, source_partition, source_offset = plan.source_key
        if (
            plan.target_topic != source_topic
            or source_topic not in self._rehydration_targets
            or source_partition < 0
            or source_offset < 0
            or plan.timestamp_ms < 0
        ):
            raise ValueError("invalid sim archive rehydration plan")
        headers_json = self.encode_headers(plan.headers)
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                inserted = await connection.fetchrow(
                    _INSERT,
                    source_topic,
                    source_partition,
                    source_offset,
                    plan.target_topic,
                    plan.key,
                    plan.value,
                    headers_json,
                    plan.timestamp_ms,
                )
                if inserted is not None:
                    return True
                existing = await connection.fetchrow(
                    _EXISTING, source_topic, source_partition, source_offset
                )
                if existing is None or (
                    existing["target_topic"] != plan.target_topic
                    or existing["record_key"] != plan.key
                    or existing["record_value"] != plan.value
                    or existing["headers_json"] != headers_json
                    or existing["timestamp_ms"] != plan.timestamp_ms
                ):
                    raise ValueError("sim archive source coordinate collision")
                return False

    async def relay_once(self, *, limit: int = 100) -> int:
        """Publish pending rows under locks; mark delivered after broker ack."""
        if limit < 1:
            raise ValueError("relay limit must be positive")
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                rows = await connection.fetch(_PENDING, limit)
                for row in rows:
                    if (
                        row["source_topic"] not in self._rehydration_targets
                        or row["target_topic"] != row["source_topic"]
                    ):
                        raise ValueError("pending sim archive target is not allowed")
                    await self._publisher.publish(
                        row["target_topic"],
                        key=row["record_key"],
                        value=row["record_value"],
                        headers=self.decode_headers(row["headers_json"]),
                        timestamp_ms=row["timestamp_ms"],
                    )
                    await connection.execute(
                        _DELIVERED,
                        row["source_topic"],
                        row["source_partition"],
                        row["source_offset"],
                    )
                return len(rows)


__all__ = [
    "PostgresSimArchiveRehydrationOutbox",
]
