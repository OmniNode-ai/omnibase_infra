# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18631: per-record Kafka consumers commit their own record's coordinate.

A bare ``commit()`` commits the consumer's position for every assigned
partition. These consumers handle one record at a time, so the bounded form is
the record's own ``TopicPartition`` mapped to ``offset + 1``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from aiokafka import AIOKafkaConsumer, TopicPartition
from aiokafka.structs import ConsumerRecord

from omnibase_infra.services.post_merge.config import ConfigPostMergeConsumer
from omnibase_infra.services.post_merge.consumer import PostMergeConsumer
from omnibase_infra.services.session.config_consumer import ConfigSessionConsumer
from omnibase_infra.services.session.consumer import SessionEventConsumer

pytestmark = pytest.mark.unit

_RECORD: ConsumerRecord[object, object] = ConsumerRecord(
    topic="t",
    partition=3,
    offset=41,
    timestamp=0,
    timestamp_type=0,
    key=None,
    value=None,
    checksum=None,
    serialized_key_size=0,
    serialized_value_size=0,
    headers=(),
)


async def test_post_merge_commits_only_the_handled_record() -> None:
    consumer = PostMergeConsumer(
        config=ConfigPostMergeConsumer(kafka_bootstrap_servers="localhost:19092")
    )
    kafka = AsyncMock(spec=AIOKafkaConsumer)
    consumer._consumer = kafka

    await consumer._commit_through(_RECORD)

    kafka.commit.assert_awaited_once_with({TopicPartition("t", 3): 42})


async def test_session_commits_only_the_handled_record() -> None:
    consumer = SessionEventConsumer(
        config=ConfigSessionConsumer(bootstrap_servers="localhost:19092", topics=["t"]),
        aggregator=AsyncMock(),
    )
    kafka = AsyncMock(spec=AIOKafkaConsumer)
    consumer._consumer = kafka

    await consumer._commit_through(_RECORD)

    kafka.commit.assert_awaited_once_with({TopicPartition("t", 3): 42})
