# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Kafka consume preserves archive replay bytes beside normalized ONEX headers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka


@pytest.mark.unit
def test_archive_replay_transport_preserves_raw_headers_and_broker_timestamp() -> None:
    bus = object.__new__(EventBusKafka)
    raw_headers = [
        ("source", b"archive"),
        ("event_type", b"topic-archive-replay"),
        ("onex-archive-source-topic", b"onex.cmd.omnimarket.delegate-skill.v1"),
        ("onex-archive-source-partition", b"0"),
        ("onex-archive-source-offset", b"7"),
        ("duplicate", b"\xff"),
        ("duplicate", b"second"),
    ]
    record = SimpleNamespace(
        key=b"\x00key",
        value=b"\x00original-value",
        headers=raw_headers,
        timestamp=1_800_000_000_123,
        offset=42,
        partition=3,
    )

    message = bus._kafka_msg_to_model(
        record, "onex.evt.omnimarket.topic-archive-replay.v1"
    )

    assert message.value == b"\x00original-value"
    assert message.key == b"\x00key"
    assert message.original_kafka_headers == tuple(raw_headers)
    assert message.broker_timestamp_ms == 1_800_000_000_123
    assert message.headers.source == "archive"
    assert not hasattr(message.headers, "onex_archive_source_topic")
    dumped = message.model_dump(mode="json")
    assert "original_kafka_headers" not in dumped
    assert "broker_timestamp_ms" not in dumped
