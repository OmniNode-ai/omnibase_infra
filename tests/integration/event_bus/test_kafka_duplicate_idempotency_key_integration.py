# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for duplicate ``idempotency_key`` refusal (OMN-16459).

The unit tests in ``tests/unit/event_bus/`` drive ``KafkaTransport._to_model``
and the ``EventBusKafka`` header conversion directly. This module exercises the
same refusal through the **public consume path** — ``KafkaTransport.poll`` —
because that is the seam the HTTPS ingest leg actually reads through, and a
guard that only holds when the private mapper is called directly would not
protect it.

Kafka headers are a list of ``(key, value)`` pairs and permit repeats, while
``ModelTransportMessage.headers`` is a mapping. Folding one into the other
silently keeps the last occurrence. The ingest leg keys its dedupe decision on
``idempotency_key``, so a collapsed duplicate is not a cosmetic loss — it is a
silently wrong idempotency decision made from a value the producer did not
solely assert.

No broker is required: the consumer is substituted, so this exercises the real
``poll`` code path and the real mapper against records shaped as the broker
delivers them.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from aiokafka.structs import TopicPartition

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.event_bus.kafka_transport import KafkaTransport
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

pytestmark = pytest.mark.integration

TOPIC = "onex.evt.omniclaude.tool-executed.v1"


def _config() -> ModelKafkaEventBusConfig:
    return ModelKafkaEventBusConfig(
        bootstrap_servers="localhost:19092",
        timeout_seconds=10,
    )


def _record(*, offset: int, headers: list[tuple[str, bytes]]) -> SimpleNamespace:
    """A record shaped as aiokafka delivers it, headers as a repeatable list."""
    return SimpleNamespace(
        topic=TOPIC,
        partition=0,
        offset=offset,
        key=None,
        value=b'{"hook_source": "post_tool_use"}',
        headers=headers,
    )


class _FakeConsumer:
    """Stands in for AIOKafkaConsumer, returning one partition's records."""

    def __init__(self, records: list[SimpleNamespace]) -> None:
        self._records = records
        self.getmany_calls = 0

    async def getmany(
        self, *, timeout_ms: int, max_records: int
    ) -> dict[TopicPartition, list[SimpleNamespace]]:
        self.getmany_calls += 1
        return {TopicPartition(TOPIC, 0): self._records[:max_records]}


def _transport_with(records: list[SimpleNamespace]) -> KafkaTransport:
    transport = KafkaTransport(config=_config(), group="onex.test", topics=(TOPIC,))
    transport._consumer = _FakeConsumer(records)
    transport._started = True
    return transport


@pytest.mark.asyncio
async def test_poll_refuses_a_record_carrying_idempotency_key_twice() -> None:
    """A duplicate must surface as a refusal, never as a last-value win."""
    transport = _transport_with(
        [
            _record(
                offset=58861,
                headers=[
                    ("idempotency_key", b"first"),
                    ("idempotency_key", b"second"),
                ],
            )
        ]
    )

    with pytest.raises(ProtocolConfigurationError, match="duplicate idempotency_key"):
        await transport.poll(max_messages=10, timeout_ms=100)


@pytest.mark.asyncio
async def test_poll_maps_a_single_idempotency_key_unchanged() -> None:
    """Positive control: the guard must not disturb the ordinary record."""
    transport = _transport_with(
        [
            _record(
                offset=58862,
                headers=[
                    ("idempotency_key", b"only"),
                    ("correlation_id", b"corr-1"),
                ],
            )
        ]
    )

    messages = await transport.poll(max_messages=10, timeout_ms=100)

    assert len(messages) == 1
    assert messages[0].headers["idempotency_key"] == b"only"
    assert messages[0].headers["correlation_id"] == b"corr-1"
    assert messages[0].offset == 58862


@pytest.mark.asyncio
async def test_poll_refusal_does_not_depend_on_duplicate_adjacency() -> None:
    """The two occurrences are separated by an unrelated header."""
    transport = _transport_with(
        [
            _record(
                offset=58863,
                headers=[
                    ("idempotency_key", b"first"),
                    ("message_id", b"mid-1"),
                    ("idempotency_key", b"second"),
                ],
            )
        ]
    )

    with pytest.raises(ProtocolConfigurationError, match="duplicate idempotency_key"):
        await transport.poll(max_messages=10, timeout_ms=100)
