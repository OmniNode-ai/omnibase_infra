# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20318: narrowly recognize gateway quarantine and avoid new side effects."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from aiokafka import AIOKafkaConsumer

from omnibase_infra.dlq import EnumReplayStatus
from omnibase_infra.dlq.service_dlq_tracking import ServiceDlqTracking
from omnibase_infra.nodes.node_dlq_replay_effect.engine_dlq_replay import (
    DLQConsumer,
    DLQProducer,
    DLQQuarantineProducer,
    ModelDlqReplayEngineConfig,
)
from omnibase_infra.nodes.node_dlq_replay_effect.handlers.handler_dlq_replay import (
    HandlerDlqReplay,
)
from omnibase_infra.nodes.node_dlq_replay_effect.models import (
    ModelGatewayQuarantinedDlqRecord,
)

pytestmark = pytest.mark.unit


def _payload() -> dict[str, object]:
    return {
        "direction": "inbound",
        "original_topic": "original-events",
        "failure_class": "gateway_refused_record",
        "error_message": "sensitive-forensic-detail",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("original_message", None),
        ("original_message", {}),
        ("direction", "other"),
        ("direction", []),
        ("original_topic", ""),
        ("original_topic", 1),
        ("failure_class", "other_refused_record"),
        ("failure_class", "gateway_refused"),
        ("failure_class", None),
    ],
)
def test_near_matches_keep_the_existing_path(field: str, value: object) -> None:
    payload = _payload()
    payload[field] = value
    assert (
        ModelGatewayQuarantinedDlqRecord.from_payload(
            payload, dlq_topic="events-dlq", dlq_partition=2, dlq_offset=41
        )
        is None
    )


@pytest.mark.parametrize("payload", [None, [], "gateway_refused_record", 42])
def test_non_mapping_payload_is_not_a_gateway_record(payload: object) -> None:
    assert (
        ModelGatewayQuarantinedDlqRecord.from_payload(
            payload, dlq_topic="events-dlq", dlq_partition=2, dlq_offset=41
        )
        is None
    )


async def test_inbound_gateway_record_advances_without_publish_audit_or_sensitive_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = ModelDlqReplayEngineConfig(
        bootstrap_servers="localhost:9092", dlq_topic="events-dlq"
    )
    kafka = MagicMock(spec=AIOKafkaConsumer)
    kafka.__aiter__.return_value = [
        SimpleNamespace(value=json.dumps(_payload()).encode(), partition=2, offset=41)
    ]
    consumer = DLQConsumer(config)
    consumer._consumer = kafka
    consumer._started = True
    producer = MagicMock(spec=DLQProducer)
    producer._started = True
    quarantine = MagicMock(spec=DLQQuarantineProducer)
    quarantine._started = True
    tracking = MagicMock(spec=ServiceDlqTracking)
    tracking.is_tracking_enabled = True
    handler = HandlerDlqReplay(
        consumers={config.dlq_topic: consumer},
        producer=producer,
        quarantine_producer=quarantine,
        tracking=tracking,
    )

    with caplog.at_level(logging.INFO):
        result = await handler.run()

    assert result.total_processed == 1
    assert result.results[0].status == EnumReplayStatus.SKIPPED
    assert "already quarantined by the gateway" in result.results[0].message
    assert "stays on the DLQ topic events-dlq" in result.results[0].message
    producer.replay_message.assert_not_awaited()
    quarantine.quarantine_message.assert_not_awaited()
    quarantine.quarantine_unparseable_record.assert_not_awaited()
    tracking.record_replay_attempt.assert_not_awaited()
    kafka.commit.assert_awaited_once()
    offsets = kafka.commit.call_args.args[0]
    assert len(offsets) == 1
    coordinate, offset = next(iter(offsets.items()))
    assert coordinate.topic == "events-dlq"
    assert coordinate.partition == 2
    assert offset == 42
    assert "events-dlq/2/41" in caplog.text
    assert "gateway_refused_record" in caplog.text
    assert "sensitive-forensic-detail" not in caplog.text
