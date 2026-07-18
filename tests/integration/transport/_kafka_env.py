# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Broker helpers for the Kafka-transport integration tests (OMN-14756, S3).

These tests exercise the real broker face of the unified-runtime transport, so
they need a live Kafka/Redpanda. They are gated on broker reachability (skipped
when unreachable — e.g. in CI without a broker) and are intended to be run
locally / on ``.201`` against the external broker.

Broker selection precedence (first non-empty wins):

1. ``ONEX_TRANSPORT_KAFKA_BOOTSTRAP`` — dedicated override for these tests so a
   ``.201`` run is unambiguous and never collides with the local-docker default
   the shared conftest pins for ``tests/integration/event_bus``.
2. ``KAFKA_BOOTSTRAP_SERVERS`` — the standard runtime bootstrap var.
3. no fallback — tests skip when no configured broker is reachable.

Admin discipline: all metadata reads are **topic-scoped** (``describe_topics``),
never the all-topics ``list_topics``. On a shared broker with hundreds/thousands
of topics, an all-topics metadata fetch over the wire is a large response that
times out intermittently; a per-topic describe is small and deterministic.
"""

from __future__ import annotations

import asyncio
import os
import socket
from collections.abc import Awaitable, Callable
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

__all__ = [
    "committed_offset",
    "kafka_available",
    "recreate_topic",
    "transport_bootstrap",
]

# Kafka protocol error codes used here.
_ERR_NONE = 0
_ERR_UNKNOWN_TOPIC = 3
_ERR_TOPIC_ALREADY_EXISTS = 36
_ERR_INVALID_PARTITIONS = 37


@lru_cache(maxsize=1)
def _load_test_env() -> None:
    load_dotenv(Path.home() / ".omnibase" / ".env", override=False)


def transport_bootstrap() -> str:
    """Resolve the broker bootstrap-servers string for the transport tests."""
    _load_test_env()
    return (
        os.environ.get("ONEX_TRANSPORT_KAFKA_BOOTSTRAP")
        or os.environ.get("KAFKA_BOOTSTRAP_SERVERS")
        or ""
    )


def _can_reach(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def kafka_available() -> bool:
    """True when the resolved bootstrap broker is TCP-reachable."""
    first = transport_bootstrap().split(",")[0].strip()
    host, _, port = first.partition(":")
    if not host or not port.isdigit():
        return False
    return _can_reach(host, int(port))


async def _topic_error_code(admin: object, topic: str) -> int:
    """Topic-scoped describe -> the topic's error code (0 exists, 3 absent)."""
    described = await admin.describe_topics([topic])  # type: ignore[attr-defined]
    for entry in described:
        if entry.get("topic") == topic:
            return int(entry.get("error_code", _ERR_UNKNOWN_TOPIC))
    return _ERR_UNKNOWN_TOPIC


async def _wait_for(
    predicate: Callable[[], Awaitable[bool]],
    *,
    timeout_s: float,
    interval_s: float = 0.2,
) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        if await predicate():
            return True
        await asyncio.sleep(interval_s)
    return False


async def recreate_topic(
    bootstrap: str, topic: str, *, partitions: int = 1, timeout_s: float = 30.0
) -> None:
    """Drop and recreate ``topic`` with ``partitions`` so each test is isolated.

    Delivers the suite's documented "fresh Kafka topic per test": offsets and
    committed cursors reset because the partitions themselves are new. Uses only
    topic-scoped admin calls so it stays deterministic on a busy shared broker.
    """
    from aiokafka.admin import AIOKafkaAdminClient, NewTopic
    from aiokafka.errors import UnknownTopicOrPartitionError

    admin = AIOKafkaAdminClient(bootstrap_servers=bootstrap, request_timeout_ms=15000)
    await admin.start()
    try:
        if await _topic_error_code(admin, topic) == _ERR_NONE:
            try:
                await admin.delete_topics([topic])
            except UnknownTopicOrPartitionError:
                pass

        async def _absent() -> bool:
            return await _topic_error_code(admin, topic) != _ERR_NONE

        if not await _wait_for(_absent, timeout_s=timeout_s):
            raise TimeoutError(f"topic {topic!r} not deleted on {bootstrap}")

        deadline = asyncio.get_running_loop().time() + timeout_s
        while asyncio.get_running_loop().time() < deadline:
            response = await admin.create_topics(
                [NewTopic(name=topic, num_partitions=partitions, replication_factor=1)]
            )
            code = _ERR_NONE
            for created_topic, error_code, *_rest in response.topic_errors:
                if created_topic == topic:
                    code = int(error_code)
            if code == _ERR_NONE:
                break
            if code == _ERR_TOPIC_ALREADY_EXISTS:
                await asyncio.sleep(0.3)
                continue
            if code == _ERR_INVALID_PARTITIONS:
                raise RuntimeError(
                    f"cannot create {topic!r} on {bootstrap}: INVALID_PARTITIONS "
                    "(broker partition/core limit reached). Use a broker with "
                    "topic-create headroom."
                )
            raise RuntimeError(f"create_topics {topic!r} failed error_code={code}")
        else:
            raise TimeoutError(f"topic {topic!r} was not created on {bootstrap}")

        async def _present() -> bool:
            return await _topic_error_code(admin, topic) == _ERR_NONE

        if not await _wait_for(_present, timeout_s=timeout_s):
            raise TimeoutError(f"topic {topic!r} not visible on {bootstrap}")
    finally:
        await admin.close()


async def delete_topic(bootstrap: str, topic: str) -> None:
    """Best-effort topic drop (test cleanup)."""
    from aiokafka.admin import AIOKafkaAdminClient

    admin = AIOKafkaAdminClient(bootstrap_servers=bootstrap, request_timeout_ms=15000)
    await admin.start()
    try:
        await admin.delete_topics([topic])
    except Exception:  # noqa: BLE001 — cleanup is best-effort.
        pass
    finally:
        await admin.close()


async def committed_offset(
    bootstrap: str, group: str, topic: str, partition: int
) -> int | None:
    """Return the group's committed offset for ``(topic, partition)`` or None.

    The committed offset is Kafka's *next* offset to fetch, so after committing a
    message at offset ``k`` this reads ``k + 1``. Group-scoped
    (``list_consumer_group_offsets``), with one retry to absorb a transient.
    """
    from aiokafka.admin import AIOKafkaAdminClient
    from aiokafka.structs import TopicPartition

    admin = AIOKafkaAdminClient(bootstrap_servers=bootstrap, request_timeout_ms=15000)
    await admin.start()
    try:
        last_exc: Exception | None = None
        for _ in range(3):
            try:
                offsets = await admin.list_consumer_group_offsets(group)
                entry = offsets.get(TopicPartition(topic, partition))
                if entry is None:
                    return None
                return int(entry.offset)
            except Exception as exc:  # noqa: BLE001 — retry a transient metadata read.
                last_exc = exc
                await asyncio.sleep(0.3)
        if last_exc is not None:
            raise last_exc
        return None
    finally:
        await admin.close()
