# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Sim-only source receipt: fresh broker bytes plus full-current owner admission.

An archive or redacted selection manifest is a caller-constructible claim. Only
this adapter can mint an outbox plan, after a trusted owner-first DB read and a
fresh exact-coordinate broker reread. Runtime composition owns both readers;
this module does not open a broker connection or attach a consumer group.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from aiokafka import AIOKafkaConsumer
from aiokafka.structs import TopicPartition

from omnibase_infra.event_bus.kafka_auth import build_aiokafka_auth_kwargs
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.runtime.db.execution_graph_read_adapters import (
    ExecutionGraphCurrentEvidenceReader,
    ExecutionGraphLedgerRecord,
)
from omnibase_infra.runtime.execution_graph_ownership import admit_current_ownership
from omnibase_infra.runtime.execution_graph_read_authority import (
    VerifiedExecutionGraphReadAuthority,
)
from omnibase_infra.runtime.execution_graph_topology_registry import (
    PinnedExecutionGraphTopology,
)
from omnibase_infra.runtime.protocols.protocol_exact_sim_source_consumer import (
    ProtocolExactSimSourceConsumer,
)
from omnibase_infra.runtime.protocols.protocol_fresh_sim_source_reader import (
    ProtocolFreshSimSourceReader,
)
from omnibase_infra.topics.topic_namespace import (
    TOPIC_NAMESPACE_ENV_VAR,
    apply_topic_namespace,
    resolve_topic_namespace,
)

_RECEIPT_MINT = object()


@dataclass(frozen=True, slots=True)
class RawSimArchiveRecord:
    """Original bytes and broker coordinate; untrusted until source comparison."""

    topic: str
    partition: int
    offset: int
    key: bytes | None
    value: bytes
    headers: tuple[tuple[str, bytes | None], ...]
    timestamp_ms: int

    @property
    def source_key(self) -> tuple[str, int, int]:
        return self.topic, self.partition, self.offset


class KafkaSimArchiveSourceReader:
    """Read an exact source coordinate without consumer-group side effects.

    The returned topic is the canonical contract topic requested by the caller.
    The temporary consumer uses the physical namespaced topic only for its
    Kafka assignment; it never subscribes or commits an offset.
    """

    def __init__(
        self,
        config: ModelKafkaEventBusConfig,
        *,
        source_topic_namespace: str,
        consumer_factory: Callable[[], ProtocolExactSimSourceConsumer] | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        if type(config) is not ModelKafkaEventBusConfig:
            raise TypeError("sim source reader requires Kafka configuration")
        if timeout_seconds <= 0:
            raise ValueError("sim source reader timeout must be positive")
        self._config = config
        self._source_topic_namespace = resolve_topic_namespace(
            {TOPIC_NAMESPACE_ENV_VAR: source_topic_namespace}
        )
        self._consumer_factory = consumer_factory or self._build_default_consumer
        self._timeout_seconds = timeout_seconds

    def _build_default_consumer(self) -> ProtocolExactSimSourceConsumer:
        consumer = AIOKafkaConsumer(
            bootstrap_servers=self._config.bootstrap_servers,
            enable_auto_commit=False,
            group_id=None,
            retry_backoff_ms=self._config.reconnect_backoff_ms,
            **build_aiokafka_auth_kwargs(self._config),
        )
        return consumer  # type: ignore[return-value]

    async def read_exact(
        self, topic: str, partition: int, offset: int
    ) -> RawSimArchiveRecord | None:
        """Fetch only the named source coordinate, preserving raw bytes/order."""
        if not topic or topic != topic.strip() or partition < 0 or offset < 0:
            raise ValueError("sim source coordinate is invalid")
        physical_topic = apply_topic_namespace(
            topic, namespace=self._source_topic_namespace
        )
        source_partition = TopicPartition(physical_topic, partition)
        consumer = self._consumer_factory()
        await consumer.start()
        try:
            consumer.assign([source_partition])
            consumer.seek(source_partition, offset)
            record = await asyncio.wait_for(
                consumer.getone(), timeout=self._timeout_seconds
            )
            if (
                getattr(record, "topic", None) != physical_topic
                or getattr(record, "partition", None) != partition
                or getattr(record, "offset", None) != offset
            ):
                return None
            key = getattr(record, "key", None)
            value = getattr(record, "value", None)
            headers = getattr(record, "headers", None)
            timestamp = getattr(record, "timestamp", None)
            if not isinstance(headers, (list, tuple)):
                return None
            normalized_headers = tuple(headers)
            if (
                (key is not None and not isinstance(key, bytes))
                or not isinstance(value, bytes)
                or not isinstance(timestamp, int)
                or timestamp < 0
                or any(
                    not isinstance(header, tuple)
                    or len(header) != 2
                    or not isinstance(header[0], str)
                    or not header[0]
                    or (header[1] is not None and not isinstance(header[1], bytes))
                    for header in normalized_headers
                )
            ):
                return None
            return RawSimArchiveRecord(
                topic=topic,
                partition=partition,
                offset=offset,
                key=key,
                value=value,
                headers=normalized_headers,
                timestamp_ms=timestamp,
            )
        except TimeoutError:
            return None
        finally:
            await consumer.stop()


@dataclass(frozen=True, slots=True, init=False)
class VerifiedSimArchiveRehydrationPlan:
    """In-process enqueue authority, never deserialized from caller input."""

    target_topic: str
    source_key: tuple[str, int, int]
    key: bytes | None
    value: bytes
    headers: tuple[tuple[str, bytes | None], ...]
    timestamp_ms: int

    def __init__(self, raw: RawSimArchiveRecord, *, _mint: object) -> None:
        if _mint is not _RECEIPT_MINT:
            raise TypeError("sim archive enqueue requires a verified source receipt")
        object.__setattr__(self, "target_topic", raw.topic)
        object.__setattr__(self, "source_key", raw.source_key)
        object.__setattr__(self, "key", raw.key)
        object.__setattr__(self, "value", raw.value)
        object.__setattr__(self, "headers", raw.headers)
        object.__setattr__(self, "timestamp_ms", raw.timestamp_ms)


def _parent(row: ExecutionGraphLedgerRecord) -> UUID | None:
    headers = json.loads(row.onex_headers)
    if not isinstance(headers, dict):
        raise ValueError("invalid normalized ledger headers")
    value = headers.get("parent_message_id")
    return UUID(value) if value else None


def _validate_original_headers(
    record: RawSimArchiveRecord,
    row: ExecutionGraphLedgerRecord,
    authority: VerifiedExecutionGraphReadAuthority,
) -> None:
    grouped: dict[str, list[bytes]] = {}
    for name, value in record.headers:
        if not name or value is None:
            raise ValueError("original source header is missing")
        grouped.setdefault(name, []).append(value)
    if any(name.startswith("onex-archive-source-") for name in grouped):
        raise ValueError("archive provenance header cannot be an original header")
    required = {"correlation_id", "message_id", "timestamp", "source", "event_type"}
    if _parent(row) is not None:
        required.add("parent_message_id")
    if any(len(grouped.get(name, ())) != 1 for name in required):
        raise ValueError(
            "required original source identity header is absent or repeated"
        )
    try:
        correlation = UUID(grouped["correlation_id"][0].decode("utf-8"))
        message = UUID(grouped["message_id"][0].decode("utf-8"))
        parent_bytes = grouped.get("parent_message_id")
        parent = UUID(parent_bytes[0].decode("utf-8")) if parent_bytes else None
        datetime.fromisoformat(grouped["timestamp"][0].decode("utf-8"))
        source = grouped["source"][0].decode("utf-8")
        event_type = grouped["event_type"][0].decode("utf-8")
    except (UnicodeDecodeError, ValueError, IndexError) as exc:
        raise ValueError("invalid original source identity header") from exc
    if (
        correlation != authority.correlation_id
        or message != row.envelope_id
        or parent != _parent(row)
        or not source
        or not event_type
    ):
        raise ValueError("original source identity disagrees with admitted ledger")


class SimArchiveSourceReceiptVerifier:
    """Verify a complete selected tree before returning any enqueue authority."""

    def __init__(
        self,
        *,
        current_reader: ExecutionGraphCurrentEvidenceReader,
        source_reader: ProtocolFreshSimSourceReader,
        topology: PinnedExecutionGraphTopology,
    ) -> None:
        if type(current_reader) is not ExecutionGraphCurrentEvidenceReader:
            raise TypeError("sim receipt requires owner-first current reader")
        if type(topology) is not PinnedExecutionGraphTopology:
            raise TypeError("sim receipt requires pinned topology")
        self._current_reader = current_reader
        self._source_reader = source_reader
        self._topology = topology

    async def verify(
        self,
        authority: VerifiedExecutionGraphReadAuthority,
        archive_records: tuple[RawSimArchiveRecord, ...],
    ) -> tuple[VerifiedSimArchiveRehydrationPlan, ...]:
        """Refuse forged archive/selection claims and stale or ambiguous ownership."""
        if type(authority) is not VerifiedExecutionGraphReadAuthority:
            raise TypeError("sim receipt requires signed gateway authority")
        if type(archive_records) is not tuple or any(
            type(record) is not RawSimArchiveRecord for record in archive_records
        ):
            raise TypeError("sim receipt requires typed raw archive records")
        current = await self._current_reader.read_authorized_current(
            authority, self._topology.read_set
        )
        admission = admit_current_ownership(current, self._topology.read_set)
        declared = self._topology.declared_chain
        if len(archive_records) != len(declared):
            raise ValueError("sim selection is not exactly the declared chain")
        selected = {record.source_key: record for record in archive_records}
        if len(selected) != len(archive_records):
            raise ValueError("sim selection repeats a source coordinate")
        owned = {
            (row.topic, row.partition, row.kafka_offset): row
            for row in admission.owned_rows
            if row.topic in {topic for hop in declared for topic in hop.topics}
        }
        if set(selected) != set(owned):
            raise ValueError("sim selection differs from the full current owned chain")
        rows_by_topic = {row.topic: row for row in owned.values()}
        if len(rows_by_topic) != len(declared):
            raise ValueError("sim selection repeats a declared hop")
        ordered: list[RawSimArchiveRecord] = []
        for hop in declared:
            matches = [
                rows_by_topic[topic] for topic in hop.topics if topic in rows_by_topic
            ]
            if len(matches) != 1:
                raise ValueError("sim selection lacks one declared hop")
            row = matches[0]
            if hop.parent is None:
                if (
                    row.envelope_id != admission.head_envelope_id
                    or _parent(row) is not None
                ):
                    raise ValueError("sim selection head conflicts with ownership")
            else:
                parent = rows_by_topic.get(hop.parent)
                if parent is None or _parent(row) != parent.envelope_id:
                    raise ValueError("sim selection parent edge is not recorded")
            raw_archive = selected[(row.topic, row.partition, row.kafka_offset)]
            fresh = await self._source_reader.read_exact(*raw_archive.source_key)
            if fresh is None or type(fresh) is not RawSimArchiveRecord:
                raise ValueError("fresh source receipt is absent")
            if fresh != raw_archive:
                raise ValueError("archive differs from fresh source receipt")
            if fresh.key != row.event_key or fresh.value != row.event_value:
                raise ValueError("fresh source differs from admitted ledger row")
            _validate_original_headers(fresh, row, authority)
            ordered.append(fresh)
        return tuple(
            VerifiedSimArchiveRehydrationPlan(record, _mint=_RECEIPT_MINT)
            for record in ordered
        )


__all__ = [
    "KafkaSimArchiveSourceReader",
    "RawSimArchiveRecord",
    "SimArchiveSourceReceiptVerifier",
    "VerifiedSimArchiveRehydrationPlan",
]
