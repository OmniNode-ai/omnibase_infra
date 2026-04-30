# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Kafka replay compute handler."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast
from uuid import UUID

from aiokafka import AIOKafkaConsumer, TopicPartition

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.projection import ModelSequenceInfo
from omnibase_infra.nodes.node_kafka_replay_compute.models import (
    ModelKafkaReplayInput,
    ModelKafkaReplayOutput,
    ModelKafkaReplayProgress,
)
from omnibase_infra.nodes.node_kafka_replay_compute.protocols import (
    ConsumerFactory,
    EnvelopeDeserializer,
    ProgressCallback,
    ProtocolKafkaMessage,
    ProtocolKafkaReplayConsumer,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelReplayBounds:
    start_offsets: Mapping[TopicPartition, int]
    end_offsets: Mapping[TopicPartition, int]


def _default_consumer_factory(
    command: ModelKafkaReplayInput,
) -> ProtocolKafkaReplayConsumer:
    """Build an AIOKafkaConsumer from only the injected replay command target."""
    consumer = AIOKafkaConsumer(
        bootstrap_servers=command.target_cluster_bootstrap,
        group_id=command.target_consumer_group,
        enable_auto_commit=False,
        auto_offset_reset="earliest",
    )
    return cast("ProtocolKafkaReplayConsumer", consumer)


def _deserialize_event_envelope(value: bytes) -> ModelEventEnvelope[object]:
    """Deserialize canonical event-bus envelope bytes."""
    return ModelEventEnvelope[object].model_validate_json(value)


class HandlerKafkaReplay:
    """Replay Kafka event envelopes from an explicitly injected target cluster."""

    def __init__(
        self,
        consumer_factory: ConsumerFactory | None = None,
        envelope_deserializer: EnvelopeDeserializer = _deserialize_event_envelope,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        """Initialize the replay handler.

        Args:
            consumer_factory: Optional test seam for isolated Kafka fixtures.
            envelope_deserializer: Canonical envelope deserializer.
            progress_callback: Optional progress sink.
        """
        self._consumer_factory = consumer_factory or _default_consumer_factory
        self._envelope_deserializer = envelope_deserializer
        self._progress_callback = progress_callback

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    async def handle(self, command: ModelKafkaReplayInput) -> ModelKafkaReplayOutput:
        """Replay envelopes from the target cluster and return offset evidence."""
        consumer = self._consumer_factory(command)
        started_at = datetime.now(UTC)
        events_replayed = 0
        correlation_id_chain: list[UUID] = []
        failed_event_offsets: list[int] = []
        last_offset_per_topic: dict[str, int] = {}
        sequence_info_per_topic: dict[str, ModelSequenceInfo] = {}

        await consumer.start()
        try:
            partitions = await self._resolve_topic_partitions(consumer, command.topics)
            bounds = await self._resolve_bounds(consumer, command, partitions)
            active_partitions = [
                partition
                for partition in partitions
                if bounds.start_offsets[partition] < bounds.end_offsets[partition]
            ]
            consumer.assign(active_partitions)
            for partition in active_partitions:
                consumer.seek(partition, bounds.start_offsets[partition])

            empty_polls = 0
            completed_partitions: set[TopicPartition] = set()

            while len(completed_partitions) < len(active_partitions):
                if (
                    command.expected_event_count is not None
                    and events_replayed >= command.expected_event_count
                ):
                    break

                records_by_partition = await consumer.getmany(
                    *active_partitions,
                    timeout_ms=command.poll_timeout_ms,
                )
                if not records_by_partition:
                    empty_polls += 1
                    if empty_polls >= command.max_empty_polls:
                        break
                    continue

                empty_polls = 0
                for partition, records in records_by_partition.items():
                    if partition in completed_partitions:
                        continue
                    end_offset = bounds.end_offsets[partition]
                    for record in records:
                        if record.offset >= end_offset:
                            completed_partitions.add(partition)
                            break
                        envelope = self._decode_record(record, failed_event_offsets)
                        if envelope is not None and envelope.correlation_id is not None:
                            correlation_id_chain.append(envelope.correlation_id)

                        events_replayed += 1
                        last_offset_per_topic[record.topic] = record.offset
                        sequence_key = self._sequence_key(
                            record.topic, record.partition
                        )
                        sequence_info_per_topic[sequence_key] = (
                            ModelSequenceInfo.from_kafka(
                                partition=record.partition,
                                offset=record.offset,
                            )
                        )
                        if events_replayed % command.progress_interval_events == 0:
                            await self._emit_progress(record, events_replayed)

                        if record.offset + 1 >= end_offset:
                            completed_partitions.add(partition)
                            break

            if (
                command.expected_event_count is not None
                and events_replayed != command.expected_event_count
            ):
                raise ValueError(
                    "Kafka replay event count mismatch: "
                    f"expected {command.expected_event_count}, got {events_replayed}"
                )

            completed_at = datetime.now(UTC)
            logger.info(
                "Kafka replay complete",
                extra={
                    "topics": command.topics,
                    "events_replayed": events_replayed,
                    "target_consumer_group": command.target_consumer_group,
                },
            )
            return ModelKafkaReplayOutput(
                events_replayed=events_replayed,
                last_offset_per_topic=last_offset_per_topic,
                sequence_info_per_topic=sequence_info_per_topic,
                started_at=started_at,
                completed_at=completed_at,
                correlation_id_chain=correlation_id_chain,
                failed_event_offsets=failed_event_offsets,
            )
        finally:
            await consumer.stop()

    async def _resolve_topic_partitions(
        self, consumer: ProtocolKafkaReplayConsumer, topics: Sequence[str]
    ) -> list[TopicPartition]:
        partitions: list[TopicPartition] = []
        for topic in topics:
            topic_partitions = await consumer.partitions_for_topic(topic)
            if topic_partitions is None:
                raise ValueError(f"Kafka topic not found for replay: {topic}")
            for partition in sorted(topic_partitions):
                partitions.append(TopicPartition(topic, partition))
        return partitions

    async def _resolve_bounds(
        self,
        consumer: ProtocolKafkaReplayConsumer,
        command: ModelKafkaReplayInput,
        partitions: Sequence[TopicPartition],
    ) -> ModelReplayBounds:
        beginning_offsets = await consumer.beginning_offsets(partitions)
        end_offsets = await consumer.end_offsets(partitions)

        resolved_start: dict[TopicPartition, int] = {}
        for partition in partitions:
            sequence_key = self._sequence_key(partition.topic, partition.partition)
            resume_info = command.resume_from.get(sequence_key)
            if resume_info is not None and resume_info.offset is not None:
                candidate = resume_info.offset + 1
            elif command.from_offset == "earliest":
                candidate = beginning_offsets[partition]
            else:
                candidate = int(command.from_offset)
            resolved_start[partition] = max(candidate, beginning_offsets[partition])

        resolved_end: dict[TopicPartition, int]
        if command.to_offset == "latest":
            resolved_end = dict(end_offsets)
        elif not command.to_offset.isdecimal():
            timestamp_ms = int(
                datetime.fromisoformat(command.to_offset).timestamp() * 1000
            )
            offsets_for_times = await consumer.offsets_for_times(
                dict.fromkeys(partitions, timestamp_ms)
            )
            resolved_end = {}
            for partition in partitions:
                offset_for_time = offsets_for_times[partition]
                resolved_end[partition] = (
                    offset_for_time.offset
                    if offset_for_time is not None
                    else end_offsets[partition]
                )
        else:
            end = int(command.to_offset)
            resolved_end = {
                partition: min(end, end_offsets[partition]) for partition in partitions
            }

        return ModelReplayBounds(start_offsets=resolved_start, end_offsets=resolved_end)

    def _decode_record(
        self, record: ProtocolKafkaMessage, failed_event_offsets: list[int]
    ) -> ModelEventEnvelope[object] | None:
        if record.value is None:
            failed_event_offsets.append(record.offset)
            return None
        try:
            return self._envelope_deserializer(bytes(record.value))
        except Exception:
            failed_event_offsets.append(record.offset)
            logger.exception(
                "Failed to deserialize replay event envelope",
                extra={
                    "topic": record.topic,
                    "partition": record.partition,
                    "offset": record.offset,
                },
            )
            return None

    async def _emit_progress(
        self, record: ProtocolKafkaMessage, events_replayed: int
    ) -> None:
        if self._progress_callback is None:
            return
        progress = ModelKafkaReplayProgress(
            topic=record.topic,
            partition=record.partition,
            current_offset=record.offset,
            events_replayed_so_far=events_replayed,
            eta_seconds=None,
        )
        result = self._progress_callback(progress)
        if inspect.isawaitable(result):
            await result

    @staticmethod
    def _sequence_key(topic: str, partition: int) -> str:
        return f"{topic}:{partition}"


__all__ = ["HandlerKafkaReplay"]
